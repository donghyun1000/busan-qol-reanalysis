"""재분석 (2026.10)

질문: 객관적 QOL과 주민 만족도가 반대로 움직인 것은 '역설'인가, 지표 설계의 결과인가?

1) 주거교통 QOL 분해 — 음(-)의 회귀계수가 주거와 교통 중 어디서 왔나
2) 가중치 민감도 — 가중치를 무작위로 바꿔도 역상관이 유지되나
3) 지표 방향 점검 — 지수에 들어간 방향과 만족도의 방향이 충돌하는 지표는 무엇인가
   + 만족도가 아닌 변수로 충돌 원인 확인 (임대료 ~ 소득·안전, 교육 비율 ~ 학생 수)
4) 한 번에 하나씩 보정 — A 원 지수 / B 임대료→소득 대비 부담 / C 교육 영역 제외 / D B+C
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (BLUE, DOMAINS, FIGURES, INK_2, ORANGE, RESULTS, SEED, TEAM_WEIGHTS,
                    composite, housing_traffic_parts, load, setup_matplotlib)

raw, qol, qol_final, sat = load()
y = sat["거주지역의 전반적 만족도"].to_numpy()
names = raw["구군"].to_numpy()
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
lines = []
log = lines.append


def rho(a, b=y):
    r, p = spearmanr(a, b)
    return r, p


# ---------------------------------------------------------------- 1) 주거교통 분해
housing, traffic = housing_traffic_parts(raw)
log("[1] 주거교통 QOL 분해 (Spearman ρ, 거주지역 만족도와)")
for name, s in [("주거교통 전체", housing + traffic), ("주거(임대료, 낮을수록 높은 점수)", housing), ("교통(역·정류장 수)", traffic)]:
    r, p = rho(s)
    log(f"    {name:24s} ρ {r:+.2f} (p {p:.3f})")

# ---------------------------------------------------------------- 3) 지표 방향 점검
# (지표, 영역, 지수에 들어간 방향: +1 높을수록 좋음 / -1 낮을수록 좋음)
COMPONENTS = [
    ("추정평균소득", "경제", +1), ("1인당_지방세", "경제", +1),
    ("교원당학생수", "교육", -1), ("학생당학교수", "교육", +1),
    ("월세_만원", "주거", -1), ("월세보증금_만원", "주거", -1), ("전세보증금_만원", "주거", -1),
    ("지하철역수", "교통", +1), ("버스정류장수", "교통", +1),
    ("생활인프라_지수", "인프라", +1), ("녹지_지수", "인프라", +1), ("비혼잡_지수", "인프라", +1),
    ("범죄_인구비율", "안전", -1), ("교통사고_인구비율", "안전", -1),
    ("화재사고_인구비율", "안전", -1), ("CCTV밀도", "안전", +1),
]
rows = []
for col, domain, sign in COMPONENTS:
    r, p = rho(raw[col])
    rows.append({"지표": col, "영역": domain, "지수_방향": "높을수록 좋음" if sign > 0 else "낮을수록 좋음",
                 "만족도_ρ": round(r, 3), "지수방향_기준_ρ": round(sign * r, 3), "p": round(p, 4),
                 "판정": "유의한 충돌" if (sign * r < 0 and p < 0.05) else ("충돌(비유의)" if sign * r < 0 else "일치")})
direction = pd.DataFrame(rows)
direction.to_csv(RESULTS / "direction_check.csv", index=False, encoding="utf-8-sig")
log("\n[3] 지표 방향 점검 — 지수방향 기준 ρ가 음수면 그 지표가 지수를 만족도와 반대로 민다")
for _, r in direction.iterrows():
    log(f"    {r['영역']:4s} {r['지표']:12s} {r['지수_방향']:7s} ρ(방향 반영) {r['지수방향_기준_ρ']:+.2f} p {r['p']:.3f}  {r['판정']}")
log("    ※ 16개 지표를 동시에 검정했으므로 개별 p값은 참고용이다.")

checks = []
for a, b, label in [
    ("월세_만원", "추정평균소득", "임대료가 높은 곳은 소득이 높은가"),
    ("월세보증금_만원", "추정평균소득", "임대료가 높은 곳은 소득이 높은가"),
    ("전세보증금_만원", "추정평균소득", "임대료가 높은 곳은 소득이 높은가"),
    ("월세보증금_만원", "안전_QOL", "임대료가 높은 곳은 안전한가"),
    ("전세보증금_만원", "안전_QOL", "임대료가 높은 곳은 안전한가"),
    ("교원당학생수", "학생수", "교육 비율이 학생 규모를 반영하는가"),
    ("학생당학교수", "학생수", "교육 비율이 학생 규모를 반영하는가"),
]:
    other = qol[b] if b in qol.columns else raw[b]
    r, p = spearmanr(raw[a], other)
    checks.append({"질문": label, "x": a, "y": b, "ρ": round(r, 3), "p": round(p, 4)})
checks = pd.DataFrame(checks)
checks.to_csv(RESULTS / "independent_checks.csv", index=False, encoding="utf-8-sig")
log("\n[3-1] 만족도를 쓰지 않은 원인 확인")
for _, r in checks.iterrows():
    log(f"    {r['x']:12s} ~ {r['y']:8s} ρ {r['ρ']:+.2f} (p {r['p']:.3f})")

# ---------------------------------------------------------------- 4) 한 번에 하나씩 보정
housing_b, traffic_b = housing_traffic_parts(raw, burden=True)
qol_b = qol.copy()
qol_b["주거교통_QOL"] = (housing_b + traffic_b).to_numpy()
SPECS = {
    "A 원 지수": composite(qol),
    "B 임대료 → 소득 대비 부담": composite(qol_b),
    "C 교육 영역 제외": composite(qol, drop=["교육_QOL"]),
    "D B + C": composite(qol_b, drop=["교육_QOL"]),
}
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, 16, 16) for _ in range(5000)]
spec_rows = []
for name, s in SPECS.items():
    r, p = rho(s)
    loo = [spearmanr(np.delete(s, i), np.delete(y, i))[0] for i in range(16)]
    boots = [spearmanr(s[ix], y[ix])[0] for ix in boot_idx]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    spec_rows.append({"지수": name, "ρ": round(r, 3), "p": round(p, 4),
                      "LOO_최소": round(min(loo), 3), "LOO_최대": round(max(loo), 3),
                      "boot95_하한": round(lo, 3), "boot95_상한": round(hi, 3)})
specs = pd.DataFrame(spec_rows)
specs.to_csv(RESULTS / "spec_comparison.csv", index=False, encoding="utf-8-sig")
log("\n[4] 한 번에 하나씩 보정 (Spearman ρ, 거주지역 만족도와)")
for _, r in specs.iterrows():
    log(f"    {r['지수']:18s} ρ {r['ρ']:+.2f} (p {r['p']:.3f}) · 한 구씩 빼도 [{r['LOO_최소']:+.2f}, {r['LOO_최대']:+.2f}]"
        f" · 부트스트랩 95% [{r['boot95_하한']:+.2f}, {r['boot95_상한']:+.2f}]")
r_cf, _ = rho(composite(qol_final, drop=["교육_QOL"]))
log(f"    (교육 수정 최종본으로 C를 계산해도 ρ {r_cf:+.2f} — 교육 영역을 빼므로 동일)")

rank = pd.DataFrame({"구군": names, "만족도": y,
                     "A_원지수": SPECS["A 원 지수"].round(3), "D_보정지수": SPECS["D B + C"].round(3)})
for c in ["만족도", "A_원지수", "D_보정지수"]:
    rank[c + "_순위"] = rank[c].rank(ascending=False, method="min").astype(int)
rank.to_csv(RESULTS / "district_ranks.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 2) 가중치 민감도
rng = np.random.default_rng(SEED)
W5 = rng.dirichlet(np.ones(5), 10000)
sens_a = np.array([spearmanr(qol[DOMAINS].to_numpy() @ w, y)[0] for w in W5])
dom4 = [d for d in DOMAINS if d != "교육_QOL"]
W4 = rng.dirichlet(np.ones(4), 10000)
sens_d = np.array([spearmanr(qol_b[dom4].to_numpy() @ w, y)[0] for w in W4])
log("\n[2] 가중치 민감도 (무작위 가중치 10,000개, Dirichlet(1))")
log(f"    A 원 지수 구성: 역상관 비율 {np.mean(sens_a < 0):.1%} · ρ 중앙값 {np.median(sens_a):+.2f}")
log(f"    D 보정 구성:   역상관 비율 {np.mean(sens_d < 0):.1%} · ρ 중앙값 {np.median(sens_d):+.2f}")

out = "\n".join(lines)
print(out)
(RESULTS / "02_reanalysis.txt").write_text(out + "\n", encoding="utf-8")

# ================================================================ 그림
plt = setup_matplotlib()

# 그림 1: 원 지수 vs 보정 지수
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
LABELS = ["중구", "사상구", "해운대구", "수영구", "강서구"]
for ax, key, color, title in [
    (axes[0], "A 원 지수", BLUE, "당시 종합 QOL"),
    (axes[1], "D B + C", ORANGE, "보정 QOL (임대료 부담·교육 제외)"),
]:
    s = SPECS[key]
    r = specs.loc[specs["지수"] == key, "ρ"].item()
    ax.scatter(s, y, s=64, color=color, edgecolor="#fcfcfb", linewidth=2, zorder=3)
    for n in LABELS:
        i = int(np.where(names == n)[0][0])
        left = n == "수영구"  # 해운대구와 겹치지 않도록 왼쪽에 표시
        ax.annotate(n, (s[i], y[i]), xytext=(-6 if left else 6, 4), textcoords="offset points",
                    ha="right" if left else "left", fontsize=9, color=INK_2)
    ax.set_title(f"{title}   ρ = {r:+.2f}", fontsize=11, loc="left")
    ax.set_xlabel("종합 QOL 점수")
axes[0].set_ylabel("거주지역 전반적 만족도 (2025)")
fig.suptitle("지표 방향을 바로잡으면 역상관이 사라진다", x=0.06, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig1_index_vs_satisfaction.png")
plt.close(fig)

# 그림 2: 지표 방향 점검
d = direction.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.4, 6.2))
colors = [ORANGE if j == "유의한 충돌" else BLUE for j in d["판정"]]
ax.barh(range(len(d)), d["지수방향_기준_ρ"], color=colors, height=0.6, zorder=3)
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"{a} · {b}" for a, b in zip(d["영역"], d["지표"])], fontsize=9)
ax.axvline(0, color=INK_2, linewidth=1)
for i, (v, j) in enumerate(zip(d["지수방향_기준_ρ"], d["판정"])):
    if j == "유의한 충돌":
        ax.text(v - 0.02, i, f"{v:+.2f}", va="center", ha="right", fontsize=9, color=INK_2)
ax.set_xlim(-0.95, 0.95)
ax.set_xlabel("지수에 들어간 방향 기준 Spearman ρ  (음수 = 지수를 만족도와 반대로 민다)")
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="5% 수준에서 유의한 충돌"), Patch(color=BLUE, label="그 외")],
          loc="lower right", frameon=False, fontsize=9)
ax.set_title("임대료 3개·교육 비율 2개가 만족도와 반대 방향으로 들어갔다", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig2_direction_check.png")
plt.close(fig)

# 그림 3: 한 번에 하나씩 보정
fig, ax = plt.subplots(figsize=(8.4, 3.4))
sp = specs.iloc[::-1].reset_index(drop=True)
for i, r in sp.iterrows():
    ax.plot([r["boot95_하한"], r["boot95_상한"]], [i, i], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
    ax.scatter(r["ρ"], i, s=72, color=BLUE, edgecolor="#fcfcfb", linewidth=2, zorder=3)
    ax.text(0.72, i, f"ρ {r['ρ']:+.2f}", va="center", fontsize=9, color=INK_2)
ax.axvline(0, color=INK_2, linewidth=1)
ax.set_yticks(range(len(sp)))
ax.set_yticklabels(sp["지수"], fontsize=10)
ax.set_xlim(-1, 1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("거주지역 만족도와의 Spearman ρ  (선: 부트스트랩 95% 구간)")
ax.set_title("교육 영역이 역상관의 대부분을 만들었다", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig3_spec_comparison.png")
plt.close(fig)

# 그림 4: 가중치 민감도
fig, ax = plt.subplots(figsize=(8.4, 3.6))
bins = np.linspace(-1, 1, 41)
ax.hist(sens_a, bins=bins, histtype="step", linewidth=2, color=BLUE, zorder=3)
ax.hist(sens_d, bins=bins, histtype="step", linewidth=2, color=ORANGE, zorder=3)
ax.axvline(0, color=INK_2, linewidth=1)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=BLUE, linewidth=2, label=f"원 지수 구성 (5개 영역) · 역상관 {np.mean(sens_a < 0):.0%}"),
                   Line2D([], [], color=ORANGE, linewidth=2, label=f"보정 구성 (임대료 부담·교육 제외) · 역상관 {np.mean(sens_d < 0):.0%}")],
          loc="upper right", frameon=False, fontsize=9)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.set_xlabel("무작위 가중치 10,000개로 만든 종합 지수와 만족도의 Spearman ρ")
ax.set_ylabel("가중치 조합 수")
ax.set_title("가중치를 어떻게 바꿔도 원 구성에서는 대부분 역상관", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig4_weight_sensitivity.png")
plt.close(fig)
print("\n그림 4개 저장:", FIGURES)
