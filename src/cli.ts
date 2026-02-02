#!/usr/bin/env node
/**
 * Mini Agent CLI
 */

import readline from "node:readline";
import { Agent, onAgentEvent } from "./index.js";
import { resolveSessionKey } from "./session-key.js";

// ============== 颜色输出 ==============

const colors = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
};

function color(text: string, c: keyof typeof colors): string {
  return `${colors[c]}${text}${colors.reset}`;
}

let unsubscribe: (() => void) | null = null;
type RunMeta = {
  startedAt?: number;
  endedAt?: number;
  model?: string;
  turns?: number;
  toolCalls?: number;
  error?: string;
};
const runMetaById = new Map<string, RunMeta>();

// ============== 主函数 ==============

async function main() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("错误: 请设置 ANTHROPIC_API_KEY 环境变量");
    process.exit(1);
  }

  const args = process.argv.slice(2);
  const agentId =
    readFlag(args, "--agent") ??
    process.env.OPENCLAW_MINI_AGENT_ID ??
    "main";
  const sessionId = resolveSessionIdArg(args) || `session-${Date.now()}`;
  const workspaceDir = process.cwd();
  const sessionKey = resolveSessionKey({ agentId, sessionId });

  console.log(color("\n🤖 Mini Agent", "cyan"));
  console.log(color(`会话: ${sessionKey}`, "dim"));
  console.log(color(`Agent: ${agentId}`, "dim"));
  console.log(color(`目录: ${workspaceDir}`, "dim"));
  console.log(color("输入 /help 查看命令，Ctrl+C 退出\n", "dim"));

  const agent = new Agent({
    apiKey,
    agentId,
    workspaceDir,
  });

  // 仅追踪当前会话的运行，避免多会话事件串台
  let activeRunId: string | null = null;
  // 事件流默认开启：用于输出运行生命周期、工具调用与汇总信息
  unsubscribe = onAgentEvent((evt) => {
    if (evt.sessionKey !== sessionKey) {
      return;
    }
    if (activeRunId && evt.runId !== activeRunId && evt.stream !== "lifecycle") {
      return;
    }
    if (evt.stream === "lifecycle") {
      const phase = typeof evt.data?.phase === "string" ? evt.data.phase : undefined;
      if (phase === "start") {
        activeRunId = evt.runId;
        const meta = runMetaById.get(evt.runId) ?? {};
        meta.startedAt =
          typeof evt.data?.startedAt === "number" ? evt.data.startedAt : Date.now();
        if (typeof evt.data?.model === "string") {
          meta.model = evt.data.model;
        }
        runMetaById.set(evt.runId, meta);
        const model = typeof evt.data?.model === "string" ? ` model=${evt.data.model}` : "";
        console.error(color(`\n[event] run start id=${evt.runId}${model}`, "magenta"));
        return;
      }
      if (phase === "end" && (!activeRunId || evt.runId === activeRunId)) {
        activeRunId = null;
        const meta = runMetaById.get(evt.runId) ?? {};
        if (typeof evt.data?.startedAt === "number") {
          meta.startedAt = evt.data.startedAt;
        }
        if (typeof evt.data?.endedAt === "number") {
          meta.endedAt = evt.data.endedAt;
        }
        if (typeof evt.data?.turns === "number") {
          meta.turns = evt.data.turns;
        }
        if (typeof evt.data?.toolCalls === "number") {
          meta.toolCalls = evt.data.toolCalls;
        }
        runMetaById.set(evt.runId, meta);
        const duration =
          typeof evt.data?.startedAt === "number" && typeof evt.data?.endedAt === "number"
            ? ` duration=${Math.max(0, evt.data.endedAt - evt.data.startedAt)}ms`
            : "";
        console.error(color(`[event] run end id=${evt.runId}${duration}\n`, "magenta"));
        return;
      }
      if (phase === "error" && (!activeRunId || evt.runId === activeRunId)) {
        activeRunId = null;
        const meta = runMetaById.get(evt.runId) ?? {};
        meta.endedAt = Date.now();
        if (typeof evt.data?.error === "string") {
          meta.error = evt.data.error;
        }
        runMetaById.set(evt.runId, meta);
        const error = typeof evt.data?.error === "string" ? ` error=${evt.data.error}` : "";
        console.error(color(`[event] run error id=${evt.runId}${error}\n`, "magenta"));
      }
      return;
    }

    // 工具调用事件：仅展示当前运行的开始/结束
    if (evt.stream === "tool" && evt.runId === activeRunId) {
      const phase = typeof evt.data?.phase === "string" ? evt.data.phase : undefined;
      const name = typeof evt.data?.name === "string" ? evt.data.name : "unknown";
      if (phase === "start") {
        const input = evt.data?.input ? safePreview(evt.data.input, 120) : "";
        console.error(color(`[event] tool start ${name}${input ? ` ${input}` : ""}`, "yellow"));
      }
      if (phase === "end") {
        const output = typeof evt.data?.output === "string" ? ` ${evt.data.output}` : "";
        console.error(color(`[event] tool end ${name}${output}`, "yellow"));
      }
      return;
    }

    // assistant 最终回复摘要（避免刷屏）
    if (evt.stream === "assistant" && evt.runId === activeRunId) {
      const isFinal = evt.data?.final === true;
      if (isFinal && typeof evt.data?.text === "string") {
        const length = evt.data.text.length;
        console.error(color(`[event] assistant final chars=${length}`, "magenta"));
      }
    }
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = () => {
    rl.question(color("你: ", "green"), async (input) => {
      const trimmed = input.trim();
      if (!trimmed) {
        prompt();
        return;
      }

      // 命令处理
      if (trimmed.startsWith("/")) {
        await handleCommand(trimmed, agent, sessionKey);
        prompt();
        return;
      }

      // 运行 Agent
      process.stdout.write(color("\nAgent: ", "blue"));

      try {
        const result = await agent.run(sessionKey, trimmed, {
          onTextDelta: (delta) => process.stdout.write(delta),
        });

        // 运行报告：从事件元数据汇总时间、工具次数等
        const meta = result.runId ? runMetaById.get(result.runId) : undefined;
        const duration =
          meta?.startedAt && meta?.endedAt
            ? Math.max(0, meta.endedAt - meta.startedAt)
            : undefined;
        const summaryParts = [
          `id=${result.runId ?? "unknown"}`,
          typeof duration === "number" ? `duration=${duration}ms` : "",
          `turns=${result.turns}`,
          `tools=${result.toolCalls}`,
          typeof result.memoriesUsed === "number" ? `memories=${result.memoriesUsed}` : "",
          `chars=${result.text.length}`,
        ].filter(Boolean);
        console.log(color(`\n\n  [${summaryParts.join(", ")}]`, "dim"));
      } catch (err) {
        console.error(color(`\n错误: ${(err as Error).message}`, "yellow"));
      }

      console.log();
      prompt();
    });
  };

  prompt();
}

function readFlag(args: string[], name: string): string | undefined {
  const idx = args.findIndex((arg) => arg === name);
  if (idx === -1) {
    return undefined;
  }
  const next = args[idx + 1];
  if (!next || next.startsWith("--")) {
    return undefined;
  }
  return next.trim() || undefined;
}

function resolveSessionIdArg(args: string[]): string | undefined {
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "chat") {
      continue;
    }
    if (arg === "--agent") {
      i += 1;
      continue;
    }
    if (arg.startsWith("--")) {
      continue;
    }
    return arg.trim() || undefined;
  }
  return undefined;
}

function safePreview(input: unknown, max = 120): string {
  try {
    const text = JSON.stringify(input);
    if (!text) {
      return "";
    }
    return text.length > max ? `${text.slice(0, max)}...` : text;
  } catch {
    return "";
  }
}

async function handleCommand(cmd: string, agent: Agent, sessionKey: string) {
  const [command, ...args] = cmd.slice(1).split(" ");

  switch (command) {
    case "help":
      console.log(`
命令:
  /help     显示帮助
  /reset    重置当前会话
  /history  显示会话历史
  /sessions 列出所有会话
  /quit     退出
`);
      break;

    case "reset":
      await agent.reset(sessionKey);
      console.log(color("会话已重置", "green"));
      break;

    case "history":
      const history = agent.getHistory(sessionKey);
      if (history.length === 0) {
        console.log(color("暂无历史", "dim"));
      } else {
        for (const msg of history) {
          const role = msg.role === "user" ? "你" : "Agent";
          const content =
            typeof msg.content === "string"
              ? msg.content
              : msg.content.map((c) => c.text || `[${c.type}]`).join(" ");
          console.log(`${color(role + ":", role === "你" ? "green" : "blue")} ${content.slice(0, 100)}...`);
        }
      }
      break;

    case "sessions":
      const sessions = await agent.listSessions();
      if (sessions.length === 0) {
        console.log(color("暂无会话", "dim"));
      } else {
        console.log("会话列表:");
        for (const s of sessions) {
          console.log(`  - ${s}${s === sessionKey ? color(" (当前)", "cyan") : ""}`);
        }
      }
      break;

    case "quit":
    case "exit":
      process.exit(0);

    default:
      console.log(color(`未知命令: ${command}`, "yellow"));
  }
}

  // 处理 Ctrl+C
  process.on("SIGINT", () => {
    console.log(color("\n\n再见! 👋", "cyan"));
    unsubscribe?.();
    process.exit(0);
  });

main().catch((err) => {
  console.error("启动失败:", err);
  process.exit(1);
});
