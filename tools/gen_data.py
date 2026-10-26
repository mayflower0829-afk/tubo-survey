#!/usr/bin/env python3
"""按《土拨购乡橙_市场调查报告》的方案与统计结论，生成校准的模拟问卷数据（400 条）。

诚实声明（同样印在页面上）：数据为按调查方案生成的模拟记录，非真实发放回收的问卷；
生成逻辑严格对齐报告的抽样设计与统计结论（配额结构、关注点分布、价格中位数、
购买意愿比例、量表相关方向与信度），用于交互式分析平台的演示与教学。

校准目标（来自报告）：
  - 4 年龄组 × 线上/线下 各 50 条，共 400
  - 购买用途：日常自用占比最高（约 6 成）
  - 关注点 Top1：口感 / 新鲜度 / 价格 为主
  - 5 斤装最高接受总价：中位数 ≈ 95 元
  - 99 元方案购买意愿比例 ≈ 34.75%
  - 感知价格价值、信息信任与意愿正相关；物流风险负相关（Spearman）
  - 各量表内部一致性 Cronbach's α > 0.7
"""
import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "public"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260101)

N_PER_CELL = 50
AGES = ["18-24岁", "25-34岁", "35-44岁", "45岁及以上"]
CHANNELS = ["线上", "线下"]

USE = ["日常自用", "送礼馈赠", "家庭聚会", "其他"]
USE_P = [0.58, 0.24, 0.13, 0.05]
ATTR = ["口感风味", "新鲜度", "价格", "品牌与产地", "包装设计"]
ATTR_P = [0.30, 0.26, 0.24, 0.11, 0.09]
PACK = ["简易防护包装", "礼盒装", "高端定制装"]
PACK_P = [0.52, 0.33, 0.15]

# ---------- 样本结构 ----------
rows = []
i = 0
for age in AGES:
    for chn in CHANNELS:
        for _ in range(N_PER_CELL):
            r = {"id": i + 1, "age": age, "channel": chn}
            r["use"] = RNG.choice(USE, p=USE_P)
            r["attr"] = RNG.choice(ATTR, p=ATTR_P)
            r["pack"] = RNG.choice(PACK, p=PACK_P)
            # 价格接受度：对数正态，中位数校准到 ≈95（报告口径），线上略高
            pass  # 价格接受度在 W_lat 生成后统一构造（与意愿弱正相关）
            rows.append(r)
            i += 1

# ---------- 潜变量 → 量表题（T 信任 / V 价值 / R 风险 / W 意愿） ----------
n = len(rows)
T_lat = RNG.normal(0, 1, n)
V_lat = 0.3 * T_lat + np.sqrt(1 - 0.09) * RNG.normal(0, 1, n)   # 信任与价值弱正相关
R_lat = 0.25 * T_lat + np.sqrt(1 - 0.0625) * RNG.normal(0, 1, n)

def likert(lat, loading, noise_sd=0.8):
    """潜变量 → 1-5 Likert 题得分（裁剪到 [1,5]，保留小数一位便于 Spearman）"""
    x = loading * lat + noise_sd * RNG.normal(0, 1, len(lat))
    return np.clip(np.round(3 + x), 1, 5)

T1, T2, T3 = likert(T_lat, 0.85), likert(T_lat, 0.82), likert(T_lat, 0.78)
V1, V2, V3 = likert(V_lat, 0.86), likert(V_lat, 0.80), likert(V_lat, 0.83)
R1, R2, R3 = likert(R_lat, 0.84), likert(R_lat, 0.81), likert(R_lat, 0.77)
# 意愿潜变量：价值、信任正向；风险负向
# 类别效应（业务逻辑）：礼赠场景对 99 元礼盒接受度更高；价格敏感者更犹豫
use_eff = np.array([0.14 if r["use"] == "送礼馈赠" else (-0.08 if r["use"] == "其他" else 0.02) for r in rows])
attr_eff = np.array([-0.22 if r["attr"] == "价格" else (0.10 if r["attr"] in ("口感风味", "新鲜度") else 0.0) for r in rows])
W_lat = 0.42 * V_lat + 0.30 * T_lat - 0.33 * R_lat + use_eff + attr_eff + np.sqrt(max(0.0, 1 - 0.42**2 - 0.30**2 - 0.33**2 - 0.02)) * RNG.normal(0, 1, n)

# 价格接受度：对数正态（中位数≈95）叠加与意愿的弱正相关（愿买者给价更高），取整到元
w_centered = W_lat - np.median(W_lat)
for k, r in enumerate(rows):
    base = np.exp(RNG.normal(np.log(95), 0.30)) - 10 * w_centered[k]
    r["price_max"] = int(np.clip(round(base), 40, 220))

W1, W2, W3 = likert(W_lat, 0.87), likert(W_lat, 0.83), likert(W_lat, 0.80)

# 99 元方案购买意愿（二分类）：取意愿潜变量最高的 34.75%（报告口径，139/400），精确校准
k_will = round(0.3475 * n)
order = np.argsort(-W_lat)
willing = np.zeros(n, dtype=int)
willing[order[:k_will]] = 1

# ---------- Cronbach's α ----------
def cronbach(items):
    X = np.column_stack(items).astype(float)
    k = X.shape[1]
    var_sum = X.var(axis=0, ddof=1).sum()
    var_tot = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - var_sum / var_tot))

alpha = {
    "trust": cronbach([T1, T2, T3]),
    "value": cronbach([V1, V2, V3]),
    "risk": cronbach([R1, R2, R3]),
    "willing": cronbach([W1, W2, W3]),
}

# ---------- Spearman 校验（报告口径） ----------
def spear(a, b):
    return float(stats.spearmanr(a, b).statistic)

w_sum = (W1 + W2 + W3) / 3
t_sum = (T1 + T2 + T3) / 3
v_sum = (V1 + V2 + V3) / 3
r_sum = (R1 + R2 + R3) / 3
spear = {
    "value_willing": round(spear(v_sum, w_sum), 3),
    "trust_willing": round(spear(t_sum, w_sum), 3),
    "risk_willing": round(spear(r_sum, w_sum), 3),
}

price_median = float(np.median([r["price_max"] for r in rows]))
willing_rate = float(willing.mean() * 100)

print(json.dumps({
    "n": n, "willing_rate": round(willing_rate, 2), "price_median": price_median,
    "alpha": {k: round(v, 3) for k, v in alpha.items()}, "spearman": spear,
    "use_dist": {u: int(sum(1 for r in rows if r["use"] == u)) for u in USE},
    "attr_dist": {a: int(sum(1 for r in rows if r["attr"] == a)) for a in ATTR},
}, ensure_ascii=False, indent=2))

# ---------- 输出 data.js ----------
records = []
for k, r in enumerate(rows):
    records.append({
        "id": r["id"], "age": r["age"], "channel": r["channel"],
        "use": r["use"], "attr": r["attr"], "pack": r["pack"],
        "price": r["price_max"], "willing": int(willing[k]),
        "T1": int(T1[k]), "T2": int(T2[k]), "T3": int(T3[k]),
        "V1": int(V1[k]), "V2": int(V2[k]), "V3": int(V3[k]),
        "R1": int(R1[k]), "R2": int(R2[k]), "R3": int(R3[k]),
        "W1": int(W1[k]), "W2": int(W2[k]), "W3": int(W3[k]),
    })
out = {
    "count": len(records),
    "willing_rate": round(willing_rate, 2),
    "price_median": price_median,
    "alpha": {k: round(v, 3) for k, v in alpha.items()},
    "spearman": spear,
    "updated": "2026-10-27",
    "disclaimer": "模拟数据：按调查方案生成的模拟问卷记录（非真实发放回收），分布与结论已校准至项目市场调查报告口径。",
    "records": records,
}
(ROOT / "public" / "data.js").write_text(
    "/* 由 tools/gen_data.py 生成：按报告口径校准的模拟问卷数据，勿手改 */\n"
    f"window.SURVEY = {json.dumps(out, ensure_ascii=False, separators=(',', ':'))};\n",
    encoding="utf-8",
)
print("data.js 已生成")
