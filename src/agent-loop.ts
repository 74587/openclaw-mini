/**
 * Agent 主循环
 *
 * 对应 OpenClaw: pi-agent-core → agent-loop.ts — runLoop()
 *
 * 从 Agent 类中提取的纯函数: 接收所有依赖，不访问 Agent 实例状态。
 *
 * 架构对齐（EventStream 模式）:
 * - 同步返回 EventStream<MiniAgentEvent, MiniAgentResult>
 * - 内部 IIFE 异步执行循环，通过 stream.push() 推送类型化事件
 * - 消费方用 for-await 迭代 stream，或用 stream.result() 获取最终结果
 *
 * 双层循环结构 (对齐 openclaw):
 *
 * OUTER LOOP (follow-ups)
 * ├─ INNER LOOP (tools + steering)
 * │  ├─ 注入 pendingMessages（steering 或 follow-up）
 * │  ├─ LLM 流式调用
 * │  ├─ 执行工具（每执行一个后检查 steering）
 * │  ├─ 若 steering: 跳过剩余工具（每个被跳过的工具生成 skipToolCall 结果）
 * │  └─ 循环条件: hasMoreToolCalls || pendingMessages.length > 0
 * ├─ 检查 follow-up 消息
 * └─ 若有 follow-up: 继续外层循环
 */

import type { EventStream } from "@mariozechner/pi-ai";
import type { Tool, ToolContext } from "./tools/types.js";
import type { Message, ContentBlock } from "./session.js";
import type {
  Model,
  StreamFunction,
  SimpleStreamOptions,
  Context as PiContext,
  ThinkingLevel,
  AssistantMessage,
} from "@mariozechner/pi-ai";
import {
  retryAsync,
  isContextOverflowError,
  isRateLimitError,
  describeError,
} from "./provider/errors.js";
import { pruneContextMessages } from "./context/index.js";
import { createMiniAgentStream, type MiniAgentEvent, type MiniAgentResult } from "./agent-events.js";
import { abortable } from "./tools/abort.js";
import { convertMessagesToPi } from "./message-convert.js";

// ============== 类型定义 ==============

export interface AgentLoopParams {
  runId: string;
  sessionKey: string;
  agentId: string;
  /** 可变: 循环中会 push 新消息 */
  currentMessages: Message[];
  compactionSummary: Message | undefined;
  systemPrompt: string;
  toolsForRun: Tool[];
  toolCtx: ToolContext;
  modelDef: Model<any>;
  streamFn: StreamFunction;
  apiKey?: string;
  temperature?: number;
  /** 思考级别: 传入后启用 extended thinking */
  reasoning?: ThinkingLevel;
  maxTurns: number;
  contextTokens: number;
  /**
   * 获取 steering 消息
   *
   * 对应 OpenClaw: pi-agent-core → AgentLoopConfig.getSteeringMessages
   * - 每执行完一个工具后调用
   * - 返回非空数组时跳过剩余工具，注入到下一轮
   */
  getSteeringMessages: () => Promise<Message[]>;
  /**
   * 获取 follow-up 消息
   *
   * 对应 OpenClaw: pi-agent-core → AgentLoopConfig.getFollowUpMessages
   * - 内层循环结束后（agent 本来要停下）调用
   * - 返回非空数组时继续外层循环
   */
  getFollowUpMessages?: () => Promise<Message[]>;
  /** 持久化 */
  appendMessage: (sessionKey: string, msg: Message) => Promise<void>;
  /** Compaction 触发器 */
  prepareCompaction: (params: {
    messages: Message[];
    sessionKey: string;
    runId: string;
  }) => Promise<{
    summary?: string;
    summaryMessage?: Message;
  }>;
  /**
   * 工具审批检查
   *
   * 对应 OpenClaw: bash-tools.exec.ts → requiresExecApproval() + waitForDecision()
   * - 每个工具执行前调用
   * - 返回 null: 无需审批，直接执行
   * - 返回 { approved: true }: 审批通过
   * - 返回 { approved: false }: 审批拒绝
   */
  checkToolApproval?: (call: {
    id: string;
    name: string;
    input: unknown;
  }) => Promise<{ approved: boolean; decision: string } | null>;
  /** 外部 abort 信号 */
  abortSignal: AbortSignal;
}

// ============== skipToolCall (对齐 openclaw) ==============

/**
 * 为被跳过的工具生成占位结果
 *
 * 对应 OpenClaw: pi-agent-core → skipToolCall()
 * - isError: true，标记为错误结果
 * - 消息: "Skipped due to queued user message."
 * - 保持消息结构完整，便于 LLM 理解上下文
 */
function skipToolCall(call: { id: string; name: string }): ContentBlock {
  return {
    type: "tool_result",
    tool_use_id: call.id,
    name: call.name,
    content: "Skipped due to queued user message.",
  };
}

// ============== 主循环 ==============

/**
 * Agent 主循环
 *
 * 对应 pi-agent-core/agent-loop.js → agentLoop()
 * - 同步返回 EventStream（IIFE 模式）
 * - 通过 stream.push() 推送类型化事件
 * - stream.end() 在终止时调用（agent_end / agent_error）
 */
export function runAgentLoop(params: AgentLoopParams): EventStream<MiniAgentEvent, MiniAgentResult> {
  const stream = createMiniAgentStream();

  // 对应 pi-agent-core: IIFE 异步执行循环，同步返回 stream
  (async () => {
    const {
      runId,
      sessionKey,
      agentId,
      currentMessages,
      systemPrompt,
      toolsForRun,
      toolCtx,
      modelDef,
      streamFn,
      apiKey,
      temperature,
      reasoning,
      maxTurns,
      contextTokens,
      getSteeringMessages,
      getFollowUpMessages,
      appendMessage,
      prepareCompaction,
      abortSignal,
    } = params;

    let { compactionSummary } = params;
    let turns = 0;
    let totalToolCalls = 0;
    let finalText = "";
    let overflowCompactionAttempted = false;

    try {
      // 对应 OpenClaw: 循环开始前检查 steering（用户可能在等待期间输入）
      let pendingMessages = await getSteeringMessages();

      // ========== 外层循环 (follow-ups) ==========
      // 对应 OpenClaw: agent-loop.js outer while(true) loop
      outerLoop: while (true) {
        let hasMoreToolCalls = true;

        // ========== 内层循环 (tools + steering) ==========
        // 对应 OpenClaw: inner while (hasMoreToolCalls || pendingMessages.length > 0)
        while (hasMoreToolCalls || pendingMessages.length > 0) {
          if (turns >= maxTurns) break outerLoop;
          if (abortSignal.aborted) break outerLoop;

          turns++;
          stream.push({ type: "turn_start", turn: turns });

          // 注入 pending 消息（steering 或 follow-up）
          if (pendingMessages.length > 0) {
            for (const msg of pendingMessages) {
              await appendMessage(sessionKey, msg);
              currentMessages.push(msg);
            }
            pendingMessages = [];
          }

          // ===== Prune: 每轮都执行 =====
          const pruneResult = pruneContextMessages({
            messages: currentMessages,
            contextWindowTokens: contextTokens,
          });
          let messagesForModel = pruneResult.messages;
          if (compactionSummary) {
            messagesForModel = [compactionSummary, ...messagesForModel];
          }

          // 构造 pi-ai Context
          const piMessages = convertMessagesToPi(messagesForModel, modelDef);
          const piContext: PiContext = {
            systemPrompt,
            messages: piMessages,
            tools: toolsForRun.map((t) => ({
              name: t.name,
              description: t.description,
              parameters: t.inputSchema as any,
            })),
          };

          // ===== 带重试的 LLM 调用 =====
          const assistantContent: ContentBlock[] = [];
          const toolCalls: { id: string; name: string; input: Record<string, unknown> }[] = [];
          const turnTextParts: string[] = [];
          // 最终 assistant message：用于提取带 signature 的 thinking block（流事件不携带）
          let finalAssistantMessage: AssistantMessage | undefined;

          try {
            await retryAsync(
              async () => {
                assistantContent.length = 0;
                toolCalls.length = 0;
                turnTextParts.length = 0;
                finalAssistantMessage = undefined;

                const streamOpts: SimpleStreamOptions = {
                  maxTokens: modelDef.maxTokens,
                  signal: abortSignal,
                  apiKey,
                  ...(temperature !== undefined ? { temperature } : {}),
                  ...(reasoning ? { reasoning } : {}),
                };
                const eventStream = streamFn(modelDef, piContext, streamOpts);

                for await (const event of eventStream) {
                  if (abortSignal.aborted) break;

                  switch (event.type) {
                    case "thinking_delta":
                      stream.push({ type: "thinking_delta", delta: (event as any).delta });
                      break;

                    case "thinking_end":
                      // 不计入 turnTextParts（思考不是最终输出）
                      // thinking block（含 signature）从 eventStream.result() 提取后写入
                      // assistantContent，不在此累积——因为流事件不携带 signature
                      break;

                    case "text_delta":
                      stream.push({ type: "message_delta", delta: event.delta });
                      break;

                    case "text_end":
                      turnTextParts.push(event.content);
                      assistantContent.push({ type: "text", text: event.content });
                      break;

                    case "toolcall_start":
                      break;

                    case "toolcall_end": {
                      const tc = event.toolCall;
                      const tcArgs = tc.arguments as Record<string, unknown>;
                      assistantContent.push({
                        type: "tool_use",
                        id: tc.id,
                        name: tc.name,
                        input: tcArgs,
                      });
                      toolCalls.push({
                        id: tc.id,
                        name: tc.name,
                        input: tcArgs,
                      });
                      break;
                    }

                    // pi-ai 的 error 事件: API 错误、网络错误等
                    // AssistantMessageEventStream 将 error 事件 resolve（非 reject），
                    // 必须在这里显式抛出，否则错误被静默吞掉
                    case "error": {
                      const errObj = (event as any).error;
                      const errMsg =
                        errObj?.errorMessage ??
                        (errObj instanceof Error ? errObj.message : null) ??
                        "unknown stream error";
                      throw new Error(`LLM stream error: ${errMsg}`);
                    }
                  }
                }

                const result = eventStream.result();
                finalAssistantMessage = await abortable(result, abortSignal);
              },
              {
                attempts: 3,
                minDelayMs: 300,
                maxDelayMs: 30_000,
                jitter: 0.1,
                label: "llm-call",
                shouldRetry: (err) => {
                  if (abortSignal.aborted) return false;
                  return isRateLimitError(describeError(err));
                },
                onRetry: ({ attempt, delay, error }) => {
                  stream.push({ type: "retry", attempt, delay, error: describeError(error) });
                },
              },
            );
          } catch (llmError) {
            // Context overflow → auto-compact → 重试一次
            const errorText = describeError(llmError);
            if (isContextOverflowError(errorText) && !overflowCompactionAttempted) {
              overflowCompactionAttempted = true;
              stream.push({ type: "context_overflow_compact", error: errorText });
              const overflowPrep = await prepareCompaction({
                messages: currentMessages,
                sessionKey,
                runId,
              });
              if (overflowPrep.summary && overflowPrep.summaryMessage) {
                compactionSummary = overflowPrep.summaryMessage;
                turns--;
                continue;
              }
            }
            throw llmError;
          }

          // 提取 thinking block（含 signature）放到 content 最前
          // 对齐 pi-agent-core：Anthropic extended thinking + tool use 多轮场景下，
          // 上一条 assistant message 必须原样回传带 signature 的 thinking block，否则 400。
          // signature 只能从 result() 取（流事件不携带），且 thinking 必须在 content 开头。
          if (finalAssistantMessage) {
            const thinkingBlocks: ContentBlock[] = [];
            for (const c of finalAssistantMessage.content) {
              if (c.type === "thinking" && c.thinking) {
                thinkingBlocks.push({
                  type: "thinking",
                  thinking: c.thinking,
                  thinkingSignature: c.thinkingSignature,
                });
              }
            }
            if (thinkingBlocks.length > 0) {
              assistantContent.unshift(...thinkingBlocks);
            }
          }

          // 保存 assistant 消息
          const assistantMsg: Message = {
            role: "assistant",
            content: assistantContent,
            timestamp: Date.now(),
          };
          await appendMessage(sessionKey, assistantMsg);
          currentMessages.push(assistantMsg);

          const turnText = turnTextParts.join("");
          if (turnText) {
            stream.push({ type: "message_end", message: assistantMsg, text: turnText });
          }

          hasMoreToolCalls = toolCalls.length > 0;

          // 没有工具调用 → 内层循环结束条件之一
          if (!hasMoreToolCalls) {
            finalText = turnText;
            stream.push({ type: "turn_end", turn: turns });
            // 检查是否有 steering 消息待处理
            pendingMessages = await getSteeringMessages();
            continue;
          }

          // ===== 执行工具（串行 + steering 中断检测） =====
          // 对应 OpenClaw: executeToolCalls() + getSteeringMessages 检查
          const toolResults: ContentBlock[] = [];
          let steeringMessages: Message[] | null = null;

          for (let i = 0; i < toolCalls.length; i++) {
            const call = toolCalls[i];
            const tool = toolsForRun.find((t) => t.name === call.name);
            let result: string;

            stream.push({
              type: "tool_execution_start",
              toolCallId: call.id,
              toolName: call.name,
              args: call.input,
            });

            if (tool) {
              // 审批检查（对齐 openclaw: exec-approvals → requiresExecApproval + waitForDecision）
              if (params.checkToolApproval) {
                const approval = await params.checkToolApproval(call);
                if (approval !== null) {
                  const decision = approval.decision as "allow-once" | "allow-always" | "deny";
                  stream.push({
                    type: "tool_approval_request",
                    toolCallId: call.id,
                    toolName: call.name,
                    args: call.input,
                  });
                  stream.push({
                    type: "tool_approval_resolved",
                    toolCallId: call.id,
                    toolName: call.name,
                    decision,
                  });
                  if (!approval.approved) {
                    result = "Tool execution denied by user.";
                    totalToolCalls++;
                    stream.push({
                      type: "tool_execution_end",
                      toolCallId: call.id,
                      toolName: call.name,
                      result,
                      isError: true,
                    });
                    toolResults.push({
                      type: "tool_result",
                      tool_use_id: call.id,
                      name: call.name,
                      content: result,
                    });
                    // 审批拒绝后继续检查 steering
                    const steering = await getSteeringMessages();
                    if (steering.length > 0) {
                      steeringMessages = steering;
                      const remaining = toolCalls.slice(i + 1);
                      for (const skipped of remaining) {
                        stream.push({ type: "tool_skipped", toolCallId: skipped.id, toolName: skipped.name });
                        toolResults.push(skipToolCall(skipped));
                      }
                      stream.push({ type: "steering", pendingCount: steering.length });
                      break;
                    }
                    continue;
                  }
                }
              }

              try {
                result = await tool.execute(call.input, toolCtx);
              } catch (err) {
                result = `执行错误: ${(err as Error).message}`;
              }
            } else {
              result = `未知工具: ${call.name}`;
            }

            totalToolCalls++;
            const isError = !tool;
            stream.push({
              type: "tool_execution_end",
              toolCallId: call.id,
              toolName: call.name,
              result: result.length > 500 ? `${result.slice(0, 500)}...` : result,
              isError,
            });
            toolResults.push({
              type: "tool_result",
              tool_use_id: call.id,
              name: call.name,
              content: result,
            });

            // 对应 OpenClaw: 每执行完一个工具检查 steering
            const steering = await getSteeringMessages();
            if (steering.length > 0) {
              steeringMessages = steering;
              // 对应 OpenClaw: skipToolCall() — 跳过剩余工具
              const remaining = toolCalls.slice(i + 1);
              for (const skipped of remaining) {
                stream.push({
                  type: "tool_skipped",
                  toolCallId: skipped.id,
                  toolName: skipped.name,
                });
                toolResults.push(skipToolCall(skipped));
              }
              stream.push({ type: "steering", pendingCount: steering.length });
              break;
            }
          }

          // 添加工具结果（含 skip 结果）
          const resultMsg: Message = {
            role: "user",
            content: toolResults,
            timestamp: Date.now(),
          };
          await appendMessage(sessionKey, resultMsg);
          currentMessages.push(resultMsg);

          stream.push({ type: "turn_end", turn: turns });

          // 对应 OpenClaw: steering 消息设为 pendingMessages，下一轮注入
          if (steeringMessages && steeringMessages.length > 0) {
            pendingMessages = steeringMessages;
          } else {
            pendingMessages = await getSteeringMessages();
          }
        }
        // ========== 内层循环结束 ==========

        // 对应 OpenClaw: 检查 follow-up 消息
        if (getFollowUpMessages) {
          const followUp = await getFollowUpMessages();
          if (followUp.length > 0) {
            pendingMessages = followUp;
            continue;
          }
        }
        break;
      }
      // ========== 外层循环结束 ==========

      stream.push({ type: "agent_end", runId, messages: currentMessages });
      stream.end({ finalText, turns, totalToolCalls, messages: currentMessages });
    } catch (err) {
      stream.push({ type: "agent_error", runId, error: describeError(err) });
      stream.end({ finalText, turns, totalToolCalls, messages: currentMessages });
    }
  })();

  return stream;
}
