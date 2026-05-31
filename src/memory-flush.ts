/**
 * Pre-compaction Memory Flush（压缩前记忆刷写）
 *
 * 对应 OpenClaw: src/auto-reply/reply/memory-flush.ts
 *
 * 设计思想：compaction 是有损操作（摘要会丢细节）。在压缩“扣动扳机”之前，
 * 先给模型一次机会，把值得长期保存的信息写入 memory，避免被压缩冲掉。
 * 这是“有损操作前的无损保全”模式（类比 fsync before truncate）。
 *
 * 触发时机：token 接近压缩阈值前 softThreshold（默认 4000）时，比压缩早一步。
 */

import type { Message } from "./session.js";
import { shouldTriggerCompaction, DEFAULT_COMPACTION_SETTINGS } from "./context/compaction.js";

/** 比压缩阈值提前多少 token 触发刷写（对应 OpenClaw: DEFAULT_MEMORY_FLUSH_SOFT_TOKENS） */
export const DEFAULT_MEMORY_FLUSH_SOFT_TOKENS = 4000;

export const MEMORY_FLUSH_SYSTEM_PROMPT =
  "这是压缩前的记忆刷写回合。会话临近自动压缩，请把值得长期保存的信息落盘到记忆。通常不需要面向用户回复。";

export const MEMORY_FLUSH_PROMPT =
  "压缩前记忆刷写：会话即将压缩。如果有值得长期保存的信息（用户偏好、关键决策、重要事实、未完成的待办），现在用 memory_save 写入；如果没有，不要保存任何内容。";

/**
 * 是否应触发记忆刷写
 *
 * 复用 shouldTriggerCompaction：把 reserveTokens 增大 softThreshold，
 * 使刷写阈值比压缩阈值低一个 softThreshold（即“压缩前一步”）。
 * 对应 OpenClaw: shouldRunMemoryFlush() 的 contextWindow - reserve - soft 阈值。
 */
export function shouldRunMemoryFlush(params: {
  messages: Message[];
  contextWindowTokens: number;
  softThresholdTokens?: number;
}): boolean {
  const soft = params.softThresholdTokens ?? DEFAULT_MEMORY_FLUSH_SOFT_TOKENS;
  return shouldTriggerCompaction({
    messages: params.messages,
    contextWindowTokens: params.contextWindowTokens,
    settings: { reserveTokens: DEFAULT_COMPACTION_SETTINGS.reserveTokens + soft },
  });
}
