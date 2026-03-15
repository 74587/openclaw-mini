/**
 * Gateway 协议帧定义
 *
 * 对齐 OpenClaw:
 * - protocol/schema/frames.ts → 三种帧类型（判别联合）
 * - protocol/schema/error-codes.ts → ErrorCodes + errorShape()
 * - protocol/index.ts → 帧验证函数
 *
 * 设计要点:
 * - RequestFrame (req)  — 客户端→服务端的 RPC 调用
 * - ResponseFrame (res) — 服务端→客户端的 RPC 响应（通过 id 关联）
 * - EventFrame (event)  — 服务端→客户端的推送事件（seq 单调递增）
 */

import crypto from "node:crypto";

// ============== 协议版本 ==============

export const PROTOCOL_VERSION = 1;

// ============== 帧类型 ==============

export type RequestFrame = {
  type: "req";
  id: string;
  method: string;
  params?: unknown;
};

export type ResponseFrame = {
  type: "res";
  id: string;
  ok: boolean;
  payload?: unknown;
  error?: ErrorShape;
};

export type EventFrame = {
  type: "event";
  event: string;
  payload?: unknown;
  seq: number;
};

export type GatewayFrame = RequestFrame | ResponseFrame | EventFrame;

// ============== 错误 ==============

export type ErrorShape = { code: string; message: string };

export const ErrorCodes = {
  INVALID_REQUEST: "INVALID_REQUEST",
  UNAUTHORIZED: "UNAUTHORIZED",
  UNAVAILABLE: "UNAVAILABLE",
} as const;

export function errorShape(code: string, message: string): ErrorShape {
  return { code, message };
}

// ============== 握手响应 ==============

export type HelloOk = {
  protocol: number;
  methods: string[];
  events: string[];
  policy: { tickIntervalMs: number; maxPayloadBytes: number };
};

// ============== 帧验证（类型守卫） ==============

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

export function isRequestFrame(f: unknown): f is RequestFrame {
  return isObject(f) && f.type === "req" && typeof f.id === "string" && typeof f.method === "string";
}

export function isResponseFrame(f: unknown): f is ResponseFrame {
  return isObject(f) && f.type === "res" && typeof f.id === "string" && typeof f.ok === "boolean";
}

export function isEventFrame(f: unknown): f is EventFrame {
  return isObject(f) && f.type === "event" && typeof f.event === "string";
}

// ============== 常量 ==============

export const TICK_INTERVAL_MS = 30_000;
export const MAX_PAYLOAD_BYTES = 512 * 1024;
export const MAX_BUFFERED_BYTES = 1.5 * 1024 * 1024;
export const HANDSHAKE_TIMEOUT_MS = 10_000;
export const REQUEST_TIMEOUT_MS = 60_000;

export const GATEWAY_METHODS = [
  "connect", "chat.send", "chat.history",
  "sessions.list", "sessions.reset", "health",
] as const;

export const GATEWAY_EVENTS = [
  "connect.challenge", "tick", "agent", "chat", "shutdown",
] as const;

export function newId(): string {
  return crypto.randomUUID();
}
