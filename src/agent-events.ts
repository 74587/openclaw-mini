/**
 * Agent 事件类型定义
 *
 * 对应 OpenClaw:
 * - pi-agent-core/types.d.ts → AgentEvent 判别联合类型
 * - pi-ai/utils/event-stream.js → EventStream<T, R> 泛型事件流
 *
 * 架构对齐:
 * - 全局事件总线 → 已移除（原 emitAgentEvent / onAgentEvent）
 * - 替代方案: Agent 实例级 subscribe()/emit() 模式
 *   （对应 pi-agent-core Agent.listeners + Agent.emit()）
 * - EventStream 从 pi-ai 直接导入（DRY，不重新实现）
 *
 * 事件流向（三层架构）:
 *   Layer 1: agent-loop → stream.push(MiniAgentEvent) → EventStream 队列
 *   Layer 2: Agent.run() → for await (event of stream) → 消费事件
 *   Layer 3: Agent.emit(event) → listeners → 外部订阅者（CLI 等）
 */

import { EventStream } from "@mariozechner/pi-ai";
import type { Message } from "./session.js";

// ============== 事件类型（判别联合） ==============

/**
 * Agent 事件类型
 *
 * 对应 pi-agent-core AgentEvent，适配 mini 的 Message 类型:
 * - 核心生命周期: agent_start → agent_end / agent_error
 * - 轮次: turn_start → turn_end
 * - 消息: message_start → message_delta* → message_end
 * - 工具: tool_execution_start → tool_execution_end / tool_skipped
 * - mini 特有: compaction, retry, steering, subagent, context_overflow_compact
 */
export type MiniAgentEvent =
  // 核心生命周期（对齐 pi-agent-core: agent_start / agent_end）
  | { type: "agent_start"; runId: string; sessionKey: string; agentId: string; model: string }
  | { type: "agent_end"; runId: string; messages: Message[] }
  | { type: "agent_error"; runId: string; error: string }

  // 轮次（对齐 pi-agent-core: turn_start / turn_end）
  | { type: "turn_start"; turn: number }
  | { type: "turn_end"; turn: number }

  // 消息（对齐 pi-agent-core: message_start / message_update / message_end）
  | { type: "message_start"; message: Message }
  | { type: "message_delta"; delta: string }
  | { type: "message_end"; message: Message; text: string }

  // 思考（对齐 pi-agent-core: extended thinking 流式输出）
  | { type: "thinking_delta"; delta: string }

  // 工具执行（对齐 pi-agent-core: tool_execution_start / tool_execution_end）
  | { type: "tool_execution_start"; toolCallId: string; toolName: string; args: unknown }
  | { type: "tool_execution_end"; toolCallId: string; toolName: string; result: string; isError: boolean }
  | { type: "tool_skipped"; toolCallId: string; toolName: string }

  // 工具审批（对齐 openclaw: exec-approvals → approval request/resolved 事件）
  | { type: "tool_approval_request"; toolCallId: string; toolName: string; args: unknown }
  | { type: "tool_approval_resolved"; toolCallId: string; toolName: string; decision: "allow-once" | "allow-always" | "deny" }

  // mini 特有事件
  | { type: "steering"; pendingCount: number }
  | { type: "compaction"; summaryChars: number; droppedMessages: number }
  | { type: "memory_flush"; sessionKey: string; saved: number }
  | { type: "context_overflow_compact"; error: string }
  | { type: "retry"; attempt: number; delay: number; error: string }
  | { type: "subagent_summary"; childSessionKey: string; label?: string; task: string; summary: string }
  | { type: "subagent_error"; childSessionKey: string; label?: string; task: string; error: string };

// ============== 结果类型 ==============

/**
 * EventStream 的最终结果
 *
 * 当 stream 收到终止事件（agent_end / agent_error）时通过 extractResult 提取
 */
export interface MiniAgentResult {
  finalText: string;
  turns: number;
  totalToolCalls: number;
  messages: Message[];
}

// ============== 工厂函数 ==============

/**
 * 创建 Agent 事件流
 *
 * 对应 pi-agent-core/agent-loop.js → createAgentStream()
 * - isComplete: agent_end 或 agent_error 为终止事件
 * - extractResult: 从终止事件中提取 MiniAgentResult
 */
export function createMiniAgentStream(): EventStream<MiniAgentEvent, MiniAgentResult> {
  return new EventStream<MiniAgentEvent, MiniAgentResult>(
    // 不使用 isComplete 自动完成，由 agent-loop 的 stream.end() 显式传入结果
    () => false,
    () => ({ finalText: "", turns: 0, totalToolCalls: 0, messages: [] }),
  );
}
