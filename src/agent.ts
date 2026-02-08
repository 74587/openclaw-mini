/**
 * Mini Agent 核心
 *
 * 5 大核心子系统:
 * 1. Session Manager - 会话管理 (JSONL 持久化)
 * 2. Memory Manager - 长期记忆 (关键词搜索)
 * 3. Context Loader - 按需上下文加载 (AGENTS/SOUL/TOOLS/IDENTITY/USER/HEARTBEAT/BOOTSTRAP/MEMORY)
 * 4. Skill Manager - 可扩展技能系统
 * 5. Heartbeat Manager - 主动唤醒机制
 *
 * 核心循环:
 *   while (tool_calls) {
 *     response = llm.generate(messages)
 *     for (tool of tool_calls) {
 *       result = tool.execute()
 *       messages.push(result)
 *     }
 *   }
 */

import crypto from "node:crypto";
import type { Tool, ToolContext } from "./tools/types.js";
import { builtinTools } from "./tools/builtin.js";
import { wrapToolWithAbortSignal, abortable } from "./tools/abort.js";
import { SessionManager, type Message, type ContentBlock } from "./session.js";
import { MemoryManager, type MemorySearchResult } from "./memory.js";
import {
  ContextLoader,
  DEFAULT_CONTEXT_WINDOW_TOKENS,
  compactHistoryIfNeeded,
  pruneContextMessages,
  estimateMessagesTokens,
  type PruneResult,
  type SummarizeFn,
} from "./context/index.js";
import {
  CONTEXT_WINDOW_HARD_MIN_TOKENS,
  CONTEXT_WINDOW_WARN_BELOW_TOKENS,
  evaluateContextWindowGuard,
  resolveContextWindowInfo,
} from "./context-window-guard.js";
import { SkillManager, type SkillMatch } from "./skills.js";
import { HeartbeatManager, type HeartbeatTask, type WakeRequest, type HeartbeatResult } from "./heartbeat.js";
import {
  normalizeAgentId,
  resolveAgentIdFromSessionKey,
  resolveSessionKey,
  isSubagentSessionKey,
} from "./session-key.js";
import { enqueueInLane, resolveGlobalLane, resolveSessionLane, setLaneConcurrency } from "./command-queue.js";
import { filterToolsByPolicy, mergeToolPolicies, type ToolPolicy } from "./tool-policy.js";
import { emitAgentEvent } from "./agent-events.js";
import type {
  Model,
  StreamFunction,
  SimpleStreamOptions,
  Context as PiContext,
  ToolCall as PiToolCall,
  Message as PiMessage,
  TextContent as PiTextContent,
} from "@mariozechner/pi-ai";
import { streamSimple, completeSimple, getModel } from "@mariozechner/pi-ai";
import {
  retryAsync,
  isContextOverflowError,
  isRateLimitError,
  describeError,
} from "./provider/errors.js";


// ============== 消息格式转换 ==============

/**
 * 将内部 Message[] 转换为 pi-ai 的 Message[]
 *
 * pi-ai 使用三种 role: "user" / "assistant" / "toolResult"
 * 我们的内部格式: role 只有 "user" / "assistant"，tool_result 嵌在 user 消息的 content 中
 *
 * 转换规则:
 * - user + string content → PiUserMessage
 * - user + ContentBlock[] 含 tool_result → 拆分为独立 PiToolResultMessage
 * - user + ContentBlock[] 含 text → PiUserMessage
 * - assistant + ContentBlock[] → PiAssistantMessage（需转换 tool_use → ToolCall）
 */
function convertMessagesToPi(
  messages: Message[],
  modelInfo: { api: string; provider: string; id: string },
): PiMessage[] {
  // assistant 消息需要填充的元数据（pi-ai 要求完整结构，但这些字段不影响 stream 行为）
  const emptyUsage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const result: PiMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({
          role: "user",
          content: msg.content,
          timestamp: msg.timestamp,
        });
        continue;
      }

      // ContentBlock[] — 分离 text 和 tool_result
      const textParts: PiTextContent[] = [];
      for (const block of msg.content) {
        if (block.type === "text" && block.text) {
          textParts.push({ type: "text", text: block.text });
        } else if (block.type === "tool_result") {
          result.push({
            role: "toolResult",
            toolCallId: block.tool_use_id ?? "",
            toolName: block.name ?? "",
            content: [{ type: "text", text: typeof block.content === "string" ? block.content : "" }],
            isError: false,
            timestamp: msg.timestamp,
          });
        }
      }
      if (textParts.length > 0) {
        result.push({
          role: "user",
          content: textParts,
          timestamp: msg.timestamp,
        });
      }
    } else {
      // assistant
      if (typeof msg.content === "string") {
        result.push({
          role: "assistant",
          content: [{ type: "text", text: msg.content }],
          api: modelInfo.api,
          provider: modelInfo.provider,
          model: modelInfo.id,
          usage: emptyUsage,
          stopReason: "stop",
          timestamp: msg.timestamp,
        });
        continue;
      }

      const piContent: (PiTextContent | PiToolCall)[] = [];
      for (const block of msg.content) {
        if (block.type === "text" && block.text) {
          piContent.push({ type: "text", text: block.text });
        } else if (block.type === "tool_use") {
          piContent.push({
            type: "toolCall",
            id: block.id ?? "",
            name: block.name ?? "",
            arguments: block.input ?? {},
          });
        }
      }

      result.push({
        role: "assistant",
        content: piContent,
        api: modelInfo.api,
        provider: modelInfo.provider,
        model: modelInfo.id,
        usage: emptyUsage,
        stopReason: "stop",
        timestamp: msg.timestamp,
      });
    }
  }

  return result;
}

// ============== 类型定义 ==============

export interface AgentConfig {
  /** API Key（向后兼容：未指定 streamFn 时作为 Anthropic key 使用） */
  apiKey?: string;
  /** 模型 ID（向后兼容：未指定 model 对象时用于创建默认 Anthropic model） */
  model?: string;
  /**
   * Provider 流式调用函数
   *
   * 对应 OpenClaw: pi-agent-core → Agent.streamFn
   * - 不指定则默认使用 pi-ai 的 streamSimple（自动路由到对应 provider）
   * - 可替换为任意自定义 StreamFunction
   */
  streamFn?: StreamFunction;
  /**
   * 模型定义
   *
   * 对应 OpenClaw: pi-ai → Model<TApi>
   * - 不指定则通过 getModel("anthropic", modelId) 获取
   */
  modelDef?: Model<any>;
  /** Agent ID（默认 main） */
  agentId?: string;
  /** 系统提示 */
  systemPrompt?: string;
  /** 工具列表 */
  tools?: Tool[];
  /** 工具策略（allow/deny） */
  toolPolicy?: ToolPolicy;
  /** 沙箱设置（示意版，仅控制工具可用性） */
  sandbox?: {
    enabled?: boolean;
    allowExec?: boolean;
    allowWrite?: boolean;
  };
  /** 温度参数（0-1，对应 OpenClaw: agents.defaults.models[provider/model].params.temperature） */
  temperature?: number;
  /** 最大循环次数 */
  maxTurns?: number;
  /** 会话存储目录 */
  sessionDir?: string;
  /** 工作目录 */
  workspaceDir?: string;
  /** 记忆存储目录 */
  memoryDir?: string;
  /** 是否启用记忆 */
  enableMemory?: boolean;
  /** 是否启用上下文加载 */
  enableContext?: boolean;
  /** 是否启用技能 */
  enableSkills?: boolean;
  /** 是否启用主动唤醒 */
  enableHeartbeat?: boolean;
  /** Heartbeat 检查间隔 (毫秒) */
  heartbeatInterval?: number;
  /** 上下文窗口大小（token 估算） */
  contextTokens?: number;
  /**
   * Global lane 最大并发数（跨 session 的总并行度）
   *
   * 对应 OpenClaw: gateway/server-lanes.ts → resolveAgentMaxConcurrent()
   * - session lane 固定 maxConcurrent=1（同一 session 内串行）
   * - global lane 控制不同 session 间可同时跑几个（默认 2）
   */
  maxConcurrentRuns?: number;
}

export interface AgentCallbacks {
  /** 流式文本增量 */
  onTextDelta?: (delta: string) => void;
  /** 文本完成 */
  onTextComplete?: (text: string) => void;
  /** 工具调用开始 */
  onToolStart?: (name: string, input: unknown) => void;
  /** 工具调用结束 */
  onToolEnd?: (name: string, result: string) => void;
  /** 轮次开始 */
  onTurnStart?: (turn: number) => void;
  /** 轮次结束 */
  onTurnEnd?: (turn: number) => void;
  /** 技能匹配 */
  onSkillMatch?: (match: SkillMatch) => void;
  /** 记忆检索 */
  onMemorySearch?: (results: MemorySearchResult[]) => void;
  /** Heartbeat 任务触发 */
  onHeartbeat?: (tasks: HeartbeatTask[]) => void;
}

export interface RunResult {
  /** 本次运行 ID */
  runId?: string;
  /** 最终文本 */
  text: string;
  /** 总轮次 */
  turns: number;
  /** 工具调用次数 */
  toolCalls: number;
  /** 是否触发了技能 */
  skillTriggered?: string;
  /** 记忆检索结果数（memory_search 返回的条数） */
  memoriesUsed?: number;
}

// ============== 默认系统提示 ==============

const DEFAULT_SYSTEM_PROMPT = `你是一个编程助手 Agent。

## 可用工具
- read: 读取文件内容
- write: 写入文件
- edit: 编辑文件 (字符串替换)
- exec: 执行 shell 命令
- list: 列出目录
- grep: 搜索文件内容

## 原则
1. 修改代码前必须先读取文件
2. 使用 edit 进行小范围修改
3. 保持简洁，不要过度解释
4. 遇到错误时分析原因并重试

## 输出格式
- 简洁的语言
- 代码使用 markdown 格式`;

// ============== Agent 核心类 ==============

export class Agent {
  /**
   * Provider 流式调用函数
   *
   * 对应 OpenClaw: pi-agent-core/agent.d.ts → Agent.streamFn
   * - 可在运行时替换（如 failover 切换 provider）
   */
  streamFn: StreamFunction;
  private modelDef: Model<any>;
  private apiKey?: string;
  private temperature?: number;
  private agentId: string;
  private baseSystemPrompt: string;
  private tools: Tool[];
  private maxTurns: number;
  private workspaceDir: string;
  private toolPolicy?: ToolPolicy;
  private contextTokens: number;
  private sandbox?: {
    enabled: boolean;
    allowExec: boolean;
    allowWrite: boolean;
  };

  // 5 大子系统
  private sessions: SessionManager;
  private memory: MemoryManager;
  private context: ContextLoader;
  private skills: SkillManager;
  private heartbeat: HeartbeatManager;

  // 功能开关
  private enableMemory: boolean;
  private enableContext: boolean;
  private enableSkills: boolean;
  private enableHeartbeat: boolean;

  /**
   * 运行中的 AbortController 映射 (runId → controller)
   *
   * 对应 OpenClaw: pi-embedded-runner/run/attempt.ts
   * - 每次 run() 创建一个 runAbortController
   * - abort() 可从外部取消指定或全部运行
   */
  private runAbortControllers = new Map<string, AbortController>();

  /**
   * Steering 消息队列 (sessionKey → messages[])
   *
   * 对应 OpenClaw: pi-agent-core → Agent.steeringQueue
   * - 用户在工具执行期间发送新消息时入队
   * - 每次工具执行完毕后检查，若非空则跳过剩余工具
   * - 队列中的消息作为下一个 user turn 处理
   */
  private steeringQueues = new Map<string, string[]>();

  constructor(config: AgentConfig) {
    // Provider 初始化（对应 OpenClaw: attempt.ts → activeSession.agent.streamFn）
    // 优先使用显式传入的 streamFn / modelDef，否则通过 pi-ai 获取 Anthropic 模型
    const modelId = config.model ?? "claude-sonnet-4-20250514";
    this.modelDef = config.modelDef ?? getModel("anthropic", modelId as any);
    this.streamFn = config.streamFn ?? streamSimple;
    this.agentId = normalizeAgentId(config.agentId ?? "main");
    this.baseSystemPrompt = config.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;
    this.tools = config.tools ?? builtinTools;
    this.maxTurns = config.maxTurns ?? 20;
    this.workspaceDir = config.workspaceDir ?? process.cwd();
    this.apiKey = config.apiKey;
    this.temperature = config.temperature;
    this.toolPolicy = config.toolPolicy;
    this.contextTokens = Math.max(
      1,
      Math.floor(config.contextTokens ?? DEFAULT_CONTEXT_WINDOW_TOKENS),
    );
    this.sandbox = {
      enabled: config.sandbox?.enabled ?? false,
      allowExec: config.sandbox?.allowExec ?? false,
      allowWrite: config.sandbox?.allowWrite ?? true,
    };

    // 初始化子系统
    this.sessions = new SessionManager(config.sessionDir);
    this.memory = new MemoryManager(config.memoryDir ?? "./.mini-agent/memory");
    this.context = new ContextLoader(this.workspaceDir);
    this.skills = new SkillManager(this.workspaceDir);
    this.heartbeat = new HeartbeatManager(this.workspaceDir, {
      intervalMs: config.heartbeatInterval,
    });

    // 功能开关
    this.enableMemory = config.enableMemory ?? true;
    this.enableContext = config.enableContext ?? true;
    this.enableSkills = config.enableSkills ?? true;
    this.enableHeartbeat = config.enableHeartbeat ?? false; // 默认关闭自动唤醒

    // Global lane 并发数（对应 OpenClaw resolveAgentMaxConcurrent）
    const globalLane = resolveGlobalLane();
    setLaneConcurrency(globalLane, config.maxConcurrentRuns ?? 2);
  }

  /**
   * 创建 SummarizeFn（用于 compaction）
   *
   * 通过 pi-ai 的 completeSimple 实现，与 Agent 当前的 model/apiKey 绑定
   * 对应 OpenClaw: compaction 走独立的 summarization call
   */
  private createSummarizeFn(): SummarizeFn {
    const model = this.modelDef;
    const apiKey = this.apiKey;
    return async (params) => {
      const result = await completeSimple(model, {
        systemPrompt: params.system,
        messages: [{ role: "user", content: params.userPrompt, timestamp: Date.now() }],
      }, { maxTokens: params.maxTokens, apiKey });
      const text = result.content
        .filter((c): c is { type: "text"; text: string } => c.type === "text")
        .map((c) => c.text)
        .join("");
      return text.trim();
    };
  }

  /**
   * 上下文压缩：裁剪 + 可选摘要
   */
  private async prepareMessagesForRun(params: {
    messages: Message[];
    sessionKey: string;
    runId: string;
  }): Promise<{
    pruned: PruneResult;
    summary?: string;
    summaryMessage?: Message;
  }> {
    // compaction 通过 SummarizeFn 抽象，不依赖特定 SDK
    // 底层使用 pi-ai 的 completeSimple，自动路由到当前 model 对应的 provider
    const compacted = await compactHistoryIfNeeded({
      summarize: this.createSummarizeFn(),
      messages: params.messages,
      contextWindowTokens: this.contextTokens,
    });

    if (compacted.summary && compacted.summaryMessage) {
      emitAgentEvent({
        runId: params.runId,
        stream: "lifecycle",
        sessionKey: params.sessionKey,
        agentId: this.agentId,
        data: {
          phase: "compaction",
          summaryChars: compacted.summary.length,
          droppedMessages: compacted.pruneResult.droppedMessages.length,
        },
      });
    }

    return {
      pruned: compacted.pruneResult,
      summary: compacted.summary,
      summaryMessage: compacted.summaryMessage,
    };
  }

  /**
   * 根据策略/沙箱生成最终可用工具集
   */
  private resolveToolsForRun(): Tool[] {
    let tools = [...this.tools];

    if (!this.enableMemory) {
      tools = tools.filter(
        (tool) => tool.name !== "memory_search" && tool.name !== "memory_get" && tool.name !== "memory_save",
      );
    }

    const sandboxPolicy = this.buildSandboxToolPolicy();
    const effectivePolicy = mergeToolPolicies(this.toolPolicy, sandboxPolicy);
    return filterToolsByPolicy(tools, effectivePolicy);
  }

  /**
   * 沙箱策略（示意版）
   * - enable=true 且 allowExec=false 时禁用 exec
   * - allowWrite=false 时禁用 write/edit
   */
  private buildSandboxToolPolicy(): ToolPolicy | undefined {
    if (!this.sandbox?.enabled) {
      return undefined;
    }
    const deny: string[] = [];
    if (!this.sandbox.allowExec) {
      deny.push("exec");
    }
    if (!this.sandbox.allowWrite) {
      deny.push("write", "edit");
    }
    return deny.length > 0 ? { deny } : undefined;
  }

  /**
   * 生成子代理 sessionKey
   */
  private buildSubagentSessionKey(agentId: string): string {
    const id = crypto.randomUUID();
    return `agent:${normalizeAgentId(agentId)}:subagent:${id}`;
  }

  /**
   * 启动子代理（最小版）
   *
   * - 只允许主会话触发
   * - 子代理完成后发出 subagent 事件，并写入父会话记录
   */
  private async spawnSubagent(params: {
    parentSessionKey: string;
    task: string;
    label?: string;
    cleanup?: "keep" | "delete";
  }): Promise<{ runId: string; sessionKey: string }> {
    if (isSubagentSessionKey(params.parentSessionKey)) {
      throw new Error("子代理会话不能再触发子代理");
    }
    const childSessionKey = this.buildSubagentSessionKey(this.agentId);
    const runPromise = this.run(childSessionKey, params.task);
    runPromise
      .then(async (result) => {
        const summary = result.text.slice(0, 600);
        emitAgentEvent({
          runId: result.runId ?? childSessionKey,
          stream: "subagent",
          sessionKey: params.parentSessionKey,
          agentId: this.agentId,
          data: {
            phase: "summary",
            childSessionKey,
            label: params.label,
            task: params.task,
            summary,
          },
        });
        const summaryMsg: Message = {
          role: "user",
          content: `[子代理摘要]\n${summary}`,
          timestamp: Date.now(),
        };
        await this.sessions.append(params.parentSessionKey, summaryMsg);
        if (params.cleanup === "delete") {
          await this.sessions.clear(childSessionKey);
        }
      })
      .catch((err) => {
        emitAgentEvent({
          runId: childSessionKey,
          stream: "subagent",
          sessionKey: params.parentSessionKey,
          agentId: this.agentId,
          data: {
            phase: "error",
            childSessionKey,
            label: params.label,
            task: params.task,
            error: err instanceof Error ? err.message : String(err),
          },
        });
      });
    return {
      runId: childSessionKey,
      sessionKey: childSessionKey,
    };
  }

  /**
   * 构建完整系统提示
   */
  private async buildSystemPrompt(params?: { sessionKey?: string }): Promise<string> {
    let prompt = this.baseSystemPrompt;
    const availableTools = new Set(this.resolveToolsForRun().map((t) => t.name));

    // 注入上下文
    if (this.enableContext) {
      const contextPrompt = await this.context.buildContextPrompt({
        sessionKey: params?.sessionKey,
      });
      if (contextPrompt) {
        prompt += contextPrompt;
      }
    }

    // 注入技能描述
    if (this.enableSkills) {
      const skillsPrompt = await this.skills.buildSkillsPrompt();
      if (skillsPrompt) {
        prompt += skillsPrompt;
      }
    }

    // 注入记忆使用指引（工具化，LLM 主导读写）
    // 对应 OpenClaw: LLM 自行决定何时读取和写入记忆
    if (this.enableMemory && (availableTools.has("memory_search") || availableTools.has("memory_save"))) {
      prompt += `\n\n## 记忆\n- 回答涉及历史、偏好、决定时：先用 memory_search 查找，再用 memory_get 拉取细节\n- 遇到值得长期保存的信息（用户偏好、关键决策、重要事实）：用 memory_save 写入\n- 不要保存日常闲聊或一次性查询`;
    }

    // 注入沙箱约束说明
    if (this.sandbox?.enabled) {
      const writeHint = this.sandbox.allowWrite ? "可写" : "只读";
      const execHint = this.sandbox.allowExec ? "允许" : "禁止";
      prompt += `\n\n## 沙箱\n当前为沙箱模式：工作区${writeHint}，命令执行${execHint}。`;
    }

    return prompt;
  }

  /**
   * 运行 Agent
   */
  async run(
    sessionIdOrKey: string,
    userMessage: string,
    callbacks?: AgentCallbacks,
  ): Promise<RunResult> {
    const sessionKey = resolveSessionKey({
      agentId: this.agentId,
      sessionId: sessionIdOrKey,
      sessionKey: sessionIdOrKey,
    });
    const sessionLane = resolveSessionLane(sessionKey);
    const globalLane = resolveGlobalLane();

    return enqueueInLane(sessionLane, () =>
      enqueueInLane(globalLane, async () => {
        const runId = crypto.randomUUID();
        const startedAt = Date.now();

        // ===== AbortController: 对应 OpenClaw attempt.ts:144 =====
        // 每次 run 创建独立的 controller，外部可通过 agent.abort(runId) 取消
        const runAbortController = new AbortController();
        this.runAbortControllers.set(runId, runAbortController);

        // 初始化 steering 队列（保留已有的 pending 消息）
        if (!this.steeringQueues.has(sessionKey)) {
          this.steeringQueues.set(sessionKey, []);
        }

        emitAgentEvent({
          runId,
          stream: "lifecycle",
          sessionKey,
          agentId: this.agentId,
          data: {
            phase: "start",
            startedAt,
            model: this.modelDef.id,
          },
        });
        try {
        const ctxInfo = resolveContextWindowInfo({
          contextTokens: this.contextTokens,
          defaultTokens: DEFAULT_CONTEXT_WINDOW_TOKENS,
        });
        const ctxGuard = evaluateContextWindowGuard({
          info: ctxInfo,
          warnBelowTokens: CONTEXT_WINDOW_WARN_BELOW_TOKENS,
          hardMinTokens: CONTEXT_WINDOW_HARD_MIN_TOKENS,
        });
        if (ctxGuard.shouldWarn) {
          console.warn(
            `上下文窗口偏小: ctx=${ctxGuard.tokens} (warn<${CONTEXT_WINDOW_WARN_BELOW_TOKENS}) source=${ctxGuard.source}`,
          );
        }
        if (ctxGuard.shouldBlock) {
          throw new Error(
            `上下文窗口过小 (${ctxGuard.tokens} tokens)，最低要求 ${CONTEXT_WINDOW_HARD_MIN_TOKENS} tokens。`,
          );
        }

        // 加载历史
        const history = await this.sessions.load(sessionKey);

        let memoriesUsed = 0;
        const toolCtx: ToolContext = {
          workspaceDir: this.workspaceDir,
          sessionKey,
          sessionId: sessionIdOrKey,
          agentId: resolveAgentIdFromSessionKey(sessionKey),
          memory: this.enableMemory ? this.memory : undefined,
          abortSignal: runAbortController.signal,
          onMemorySearch: (results) => {
            memoriesUsed += results.length;
            callbacks?.onMemorySearch?.(results);
          },
          spawnSubagent: async ({ task, label, cleanup }) =>
            this.spawnSubagent({
              parentSessionKey: sessionKey,
              task,
              label,
              cleanup,
            }),
        };

        let processedMessage = userMessage;
        let skillTriggered: string | undefined;

        // ===== 技能匹配 =====
        if (this.enableSkills) {
          const match = await this.skills.match(userMessage);
          if (match) {
            callbacks?.onSkillMatch?.(match);
            skillTriggered = match.skill.id;
            // 将技能 prompt 注入消息
            const trigger = match.matchedTrigger || "";
            const userPart = userMessage.slice(trigger.length).trim() || userMessage;
            processedMessage = `${match.skill.prompt}\n\n用户请求: ${userPart}`;
          }
        }

        // 记忆检索改为工具化调用，不在此自动注入

        // ===== Heartbeat 任务注入 =====
        if (this.enableHeartbeat) {
          const tasksPrompt = await this.heartbeat.buildTasksPrompt();
          if (tasksPrompt) {
            processedMessage += tasksPrompt;
          }
        }

        // 添加用户消息
        const userMsg: Message = {
          role: "user",
          content: processedMessage,
          timestamp: Date.now(),
        };
        await this.sessions.append(sessionKey, userMsg);

        let turns = 0;
        let totalToolCalls = 0;
        let finalText = "";
        const currentMessages = [...history, userMsg];
        // Context overflow compact 只尝试一次（对应 OpenClaw: overflowCompactionAttempted）
        let overflowCompactionAttempted = false;

        // ===== Compaction: run 开始前做一次 =====
        // 对应 OpenClaw: compaction 仅在 context overflow 时触发
        // 参见 src/agents/pi-embedded-runner/run.ts:374 — isContextOverflowError 分支
        const prep = await this.prepareMessagesForRun({
          messages: currentMessages,
          sessionKey,
          runId,
        });
        let compactionSummary = prep.summaryMessage;
        if (prep.summary) {
          let firstKeptEntryId: string | undefined;
          for (const msg of prep.pruned.messages) {
            const candidate = this.sessions.resolveMessageEntryId(sessionKey, msg);
            if (candidate) {
              firstKeptEntryId = candidate;
              break;
            }
          }
          if (firstKeptEntryId) {
            const tokensBefore = estimateMessagesTokens(currentMessages);
            await this.sessions.appendCompaction(
              sessionKey,
              prep.summary,
              firstKeptEntryId,
              tokensBefore,
            );
          } else {
            console.warn("无法定位 compaction 的 firstKeptEntryId，已跳过记录。");
          }
        }

        // 构建系统提示
        const systemPrompt = await this.buildSystemPrompt({ sessionKey });

        // ===== 工具包装: 注入 run-level abort signal =====
        // 对应 OpenClaw: pi-tools.ts:426 → wrapToolWithAbortSignal()
        // 每个工具执行时会合并 tool-level signal 和 run-level signal
        const rawTools = this.resolveToolsForRun();
        const toolsForRun = rawTools.map((t) => wrapToolWithAbortSignal(t, runAbortController.signal));

        // ===== Agent Loop =====
        while (turns < this.maxTurns) {
          turns++;
          callbacks?.onTurnStart?.(turns);

          // ===== Prune: 每轮都执行 =====
          // 对应 OpenClaw: context-pruning extension 在每次 "context" event 时运行
          // 参见 src/agents/pi-extensions/context-pruning/extension.ts
          // prune 是纯内存操作（不修改 session 文件），开销可忽略
          const pruneResult = pruneContextMessages({
            messages: currentMessages,
            contextWindowTokens: this.contextTokens,
          });
          let messagesForModel = pruneResult.messages;
          if (compactionSummary) {
            messagesForModel = [compactionSummary, ...messagesForModel];
          }

          // ===== 调用 LLM (流式) — 通过 Provider 抽象 + 重试 =====
          // 对应 OpenClaw: pi-agent-core → agent.streamFn(model, context, options)
          // abort 检查: 如果已中止则跳过 LLM 调用
          if (runAbortController.signal.aborted) {
            break;
          }

          // 构造 pi-ai Context（统一格式，不依赖特定 SDK）
          // 对应 OpenClaw: pi-ai/types.d.ts → Context
          const piContext: PiContext = {
            systemPrompt,
            messages: convertMessagesToPi(messagesForModel, this.modelDef),
            // pi-ai Tool.parameters 要求 TypeBox TSchema，用 as any 桥接普通 JSON Schema
            tools: toolsForRun.map((t) => ({
              name: t.name,
              description: t.description,
              parameters: t.inputSchema as any,
            })),
          };

          // ===== 带重试的 LLM 调用 =====
          // 对应 OpenClaw: run.ts 主循环的错误处理分支
          // - rate_limit → 指数退避重试（最多 3 次）
          // - context overflow → auto-compact → 重试一次
          // - 其他错误 → 直接抛出
          const assistantContent: ContentBlock[] = [];
          const toolCalls: { id: string; name: string; input: Record<string, unknown> }[] = [];
          const turnTextParts: string[] = [];

          // ===== Context Overflow → Auto-Compact → Retry =====
          // 对应 OpenClaw: run.ts:372-431 — isContextOverflowError 分支
          // retryAsync 处理 rate_limit，context overflow 在外层单独处理
          try {
            await retryAsync(
              async () => {
                // 清空上次重试的残留
                assistantContent.length = 0;
                toolCalls.length = 0;
                turnTextParts.length = 0;

                const streamOpts: SimpleStreamOptions = {
                  maxTokens: this.modelDef.maxTokens,
                  signal: runAbortController.signal,
                  apiKey: this.apiKey,
                  ...(this.temperature !== undefined ? { temperature: this.temperature } : {}),
                };
                const eventStream = this.streamFn(this.modelDef, piContext, streamOpts);

                for await (const event of eventStream) {
                  if (runAbortController.signal.aborted) break;

                  switch (event.type) {
                    case "text_delta":
                      callbacks?.onTextDelta?.(event.delta);
                      emitAgentEvent({
                        runId,
                        stream: "assistant",
                        sessionKey,
                        agentId: this.agentId,
                        data: { delta: event.delta },
                      });
                      break;

                    case "text_end":
                      turnTextParts.push(event.content);
                      assistantContent.push({ type: "text", text: event.content });
                      break;

                    case "toolcall_start":
                      break;

                    case "toolcall_end": {
                      // pi-ai ToolCall 用 arguments 字段
                      const tc = event.toolCall;
                      const tcArgs = tc.arguments as Record<string, unknown>;
                      callbacks?.onToolStart?.(tc.name, tcArgs);
                      emitAgentEvent({
                        runId,
                        stream: "tool",
                        sessionKey,
                        agentId: this.agentId,
                        data: { phase: "start", name: tc.name, input: tcArgs },
                      });
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
                  }
                }

                await abortable(eventStream.result(), runAbortController.signal);
              },
              {
                attempts: 3,
                minDelayMs: 300,
                maxDelayMs: 30_000,
                jitter: 0.1,
                label: "llm-call",
                shouldRetry: (err) => {
                  if (runAbortController.signal.aborted) return false;
                  const msg = describeError(err);
                  if (isRateLimitError(msg)) return true;
                  return false;
                },
                onRetry: ({ attempt, delay, error }) => {
                  emitAgentEvent({
                    runId,
                    stream: "lifecycle",
                    sessionKey,
                    agentId: this.agentId,
                    data: {
                      phase: "retry",
                      attempt,
                      delay,
                      error: describeError(error),
                    },
                  });
                },
              },
            );
          } catch (llmError) {
            // Context overflow: 自动 compact 后重试一次
            // 对应 OpenClaw: run.ts → overflowCompactionAttempted 标记
            const errorText = describeError(llmError);
            if (isContextOverflowError(errorText) && !overflowCompactionAttempted) {
              overflowCompactionAttempted = true;
              emitAgentEvent({
                runId,
                stream: "lifecycle",
                sessionKey,
                agentId: this.agentId,
                data: { phase: "context_overflow_compact", error: errorText },
              });
              // 触发 compaction
              const overflowPrep = await this.prepareMessagesForRun({
                messages: currentMessages,
                sessionKey,
                runId,
              });
              if (overflowPrep.summary && overflowPrep.summaryMessage) {
                compactionSummary = overflowPrep.summaryMessage;
                // 不 break，让 while 循环继续下一轮（重新 prune + 重新调用 LLM）
                turns--; // 抵消本轮 turns++，此轮不算有效 turn
                continue;
              }
            }
            throw llmError;
          }

          // 保存 assistant 消息
          const assistantMsg: Message = {
            role: "assistant",
            content: assistantContent,
            timestamp: Date.now(),
          };
          await this.sessions.append(sessionKey, assistantMsg);
          currentMessages.push(assistantMsg);

          callbacks?.onTurnEnd?.(turns);

          const turnText = turnTextParts.join("");
          if (turnText) {
            callbacks?.onTextComplete?.(turnText);
            emitAgentEvent({
              runId,
              stream: "assistant",
              sessionKey,
              agentId: this.agentId,
              data: {
                text: turnText,
                final: true,
              },
            });
          }

          // 没有工具调用，结束
          if (toolCalls.length === 0) {
            finalText = turnText;
            break;
          }

          // ===== 执行工具（串行 + steering 中断检测） =====
          // 对应 OpenClaw: pi-agent-core → 工具串行执行，每个工具完成后检查 steeringQueue
          // - 当前工具执行完毕后如果 steering 队列非空，跳过剩余工具
          // - 已执行的工具结果正常保留，未执行的完全跳过
          const toolResults: ContentBlock[] = [];
          let steered = false;

          for (const call of toolCalls) {
            const tool = toolsForRun.find((t) => t.name === call.name);
            let result: string;

            if (tool) {
              try {
                result = await tool.execute(call.input, toolCtx);
              } catch (err) {
                result = `执行错误: ${(err as Error).message}`;
              }
            } else {
              result = `未知工具: ${call.name}`;
            }

            totalToolCalls++;
            callbacks?.onToolEnd?.(call.name, result);
            emitAgentEvent({
              runId,
              stream: "tool",
              sessionKey,
              agentId: this.agentId,
              data: {
                phase: "end",
                name: call.name,
                output: result.length > 500 ? `${result.slice(0, 500)}...` : result,
              },
            });
            toolResults.push({
              type: "tool_result",
              tool_use_id: call.id,
              name: call.name,
              content: result,
            });

            // ===== Steering 检查 =====
            // 对应 OpenClaw: agent loop 在 tool_execution_end 后检查 steeringQueue
            const steeringQueue = this.steeringQueues.get(sessionKey);
            if (steeringQueue && steeringQueue.length > 0) {
              steered = true;
              emitAgentEvent({
                runId,
                stream: "lifecycle",
                sessionKey,
                agentId: this.agentId,
                data: { phase: "steering", pendingMessages: steeringQueue.length },
              });
              break; // 跳过剩余工具
            }
          }

          // 添加已执行的工具结果
          const resultMsg: Message = {
            role: "user",
            content: toolResults,
            timestamp: Date.now(),
          };
          await this.sessions.append(sessionKey, resultMsg);
          currentMessages.push(resultMsg);

          // ===== 处理 steering 消息 =====
          // 对应 OpenClaw: steering 消息作为下一个 user turn
          if (steered) {
            const steeringQueue = this.steeringQueues.get(sessionKey);
            if (steeringQueue && steeringQueue.length > 0) {
              // 取出所有 steering 消息合并为一个 user turn
              const steeringText = steeringQueue.join("\n");
              steeringQueue.length = 0;

              const steeringMsg: Message = {
                role: "user",
                content: steeringText,
                timestamp: Date.now(),
              };
              await this.sessions.append(sessionKey, steeringMsg);
              currentMessages.push(steeringMsg);
              // 不 break，让 while 循环继续下一轮（LLM 会看到 steering 消息）
            }
          }
        }

        // 记忆写入: 不再自动保存每轮对话
        // 对应 OpenClaw 设计: LLM 自行决定何时通过 memory_save 工具写入
        // 参见 src/auto-reply/reply/memory-flush.ts — 仅在 compaction 前由 LLM 主动 flush

        const endedAt = Date.now();
        emitAgentEvent({
          runId,
          stream: "lifecycle",
          sessionKey,
          agentId: this.agentId,
          data: {
            phase: "end",
            startedAt,
            endedAt,
            turns,
            toolCalls: totalToolCalls,
          },
        });

        return {
          runId,
          text: finalText,
          turns,
          toolCalls: totalToolCalls,
          skillTriggered,
          memoriesUsed,
        };
        } catch (err) {
          emitAgentEvent({
            runId,
            stream: "lifecycle",
            sessionKey,
            agentId: this.agentId,
            data: {
              phase: "error",
              startedAt,
              endedAt: Date.now(),
              error: err instanceof Error ? err.message : String(err),
            },
          });
          throw err;
        } finally {
          // 清理 AbortController 引用
          this.runAbortControllers.delete(runId);
          // 不清空 steeringQueues：可能有下一次 run 还在排队的消息
        }
      }),
    );
  }

  /**
   * 中止运行
   *
   * 对应 OpenClaw: pi-embedded-runner/run/attempt.ts → abortRun()
   * - 传 runId 取消特定运行
   * - 不传参数取消所有运行
   * - signal 级联到所有被包装的工具和 LLM 调用
   */
  abort(runId?: string): void {
    if (runId) {
      const controller = this.runAbortControllers.get(runId);
      if (controller) {
        controller.abort();
      }
    } else {
      for (const controller of this.runAbortControllers.values()) {
        controller.abort();
      }
    }
  }

  /**
   * 向运行中的会话注入 steering 消息
   *
   * 对应 OpenClaw: pi-agent-core → session.steer(text) / agent.steeringQueue
   * - 在工具串行执行期间调用，当前工具完成后生效
   * - 剩余未执行的工具被跳过
   * - steering 消息作为下一个 user turn 进入对话
   *
   * 典型场景: 用户在 Agent 执行多个工具时发送了新消息
   */
  steer(sessionKey: string, text: string): void {
    const queue = this.steeringQueues.get(sessionKey);
    if (queue) {
      queue.push(text);
    } else {
      this.steeringQueues.set(sessionKey, [text]);
    }
  }

  /**
   * 启动 Heartbeat 监控
   */
  startHeartbeat(callback?: (tasks: HeartbeatTask[], request: WakeRequest) => void): void {
    if (callback) {
      this.heartbeat.onTasks(async (tasks, request): Promise<HeartbeatResult> => {
        callback(tasks, request);
        return { status: "ok", tasks };
      });
    }
    this.heartbeat.start();
  }

  /**
   * 停止 Heartbeat 监控
   */
  stopHeartbeat(): void {
    this.heartbeat.stop();
  }

  /**
   * 手动触发 Heartbeat 检查
   */
  async triggerHeartbeat(): Promise<HeartbeatTask[]> {
    return this.heartbeat.trigger();
  }

  /**
   * 重置会话
   */
  async reset(sessionIdOrKey: string): Promise<void> {
    const sessionKey = resolveSessionKey({
      agentId: this.agentId,
      sessionId: sessionIdOrKey,
      sessionKey: sessionIdOrKey,
    });
    await this.sessions.clear(sessionKey);
  }

  /**
   * 获取会话历史
   */
  getHistory(sessionIdOrKey: string): Message[] {
    const sessionKey = resolveSessionKey({
      agentId: this.agentId,
      sessionId: sessionIdOrKey,
      sessionKey: sessionIdOrKey,
    });
    return this.sessions.get(sessionKey);
  }

  /**
   * 列出会话
   */
  async listSessions(): Promise<string[]> {
    return this.sessions.list();
  }

  // ===== 子系统访问器 =====

  getMemory(): MemoryManager {
    return this.memory;
  }

  getContext(): ContextLoader {
    return this.context;
  }

  getSkills(): SkillManager {
    return this.skills;
  }

  getHeartbeat(): HeartbeatManager {
    return this.heartbeat;
  }
}
