/**
 * Provider Failover（故障转移）+ Profile Cooldown（冷却）
 *
 * 对应 OpenClaw:
 * - src/agents/model-fallback.ts — fallback 链切换
 * - src/agents/auth-profiles/usage.ts — profile 冷却状态
 *
 * 设计：主 provider 调用失败（限流/认证/欠费/格式错误，超时除外）时，
 * 自动切换到 fallback 链中下一个可用 profile，并对故障 profile 做指数退避冷却。
 * 与 errors.ts 的错误分类配合：分类决定是否值得切换，cooldown 决定多久不再用它。
 */

import type { Model } from "@mariozechner/pi-ai";

/**
 * 一个可用的调用档位（provider + model + key）
 *
 * key 用于 cooldown 标识，label 用于展示。
 */
export interface ProviderProfile {
  key: string;
  label: string;
  modelDef: Model<any>;
  apiKey?: string;
}

// 冷却时长：5^(errorCount-1) 分钟，封顶 60 分钟
// 对应 OpenClaw: 5^(errorCount-1) with 1h cap
const COOLDOWN_BASE_MS = 60_000;
const COOLDOWN_CAP_MS = 60 * 60_000;

interface CooldownEntry {
  until: number;
  errorCount: number;
  reason: string;
}

/**
 * Profile 冷却状态管理（Agent 实例级，跨 run 保持）
 *
 * 对应 OpenClaw: auth-profiles/usage.ts → cooldown 截止时间 + errorCount
 */
export class ProfileCooldown {
  private state = new Map<string, CooldownEntry>();

  constructor(private now: () => number = () => Date.now()) {}

  /** 记录一次失败：累加 errorCount，按指数退避设置冷却截止 */
  markFailure(key: string, reason: string): void {
    const prev = this.state.get(key);
    const errorCount = (prev?.errorCount ?? 0) + 1;
    const delay = Math.min(COOLDOWN_BASE_MS * 5 ** (errorCount - 1), COOLDOWN_CAP_MS);
    this.state.set(key, { until: this.now() + delay, errorCount, reason });
  }

  /** 成功后清除冷却（profile 恢复健康） */
  markSuccess(key: string): void {
    this.state.delete(key);
  }

  isCoolingDown(key: string): boolean {
    const entry = this.state.get(key);
    return entry ? this.now() < entry.until : false;
  }

  /** 按原顺序过滤掉仍在冷却中的 profile */
  available(profiles: ProviderProfile[]): ProviderProfile[] {
    return profiles.filter((p) => !this.isCoolingDown(p.key));
  }
}
