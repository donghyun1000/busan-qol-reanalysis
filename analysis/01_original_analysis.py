"""당시 분석 재현 (2025.12)

1) 주거교통 QOL을 원 지표에서 다시 계산해 팀 산출값과 같은지 확인
2) 5개 영역 QOL → 거주지역 만족도 다중회귀 (R² 0.609)
3) 가중치를 곱한 변수로 다시 회귀 (당시 노트북의 두 번째 모형)
4) 종합 QOL과 만족도의 상관 (팀 결론: 역상관)
"""
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

from common import DOMAINS, RESULTS, TEAM_WEIGHTS, composite, housing_traffic_parts, load

raw, qol, qol_final, sat = load()
y = sat["거주지역의 전반적 만족도"]
lines = []
log = lines.append

# 1) 주거교통 QOL 재계산
housing, traffic = housing_traffic_parts(raw)
diff = np.abs((housing + traffic) - qol["주거교통_QOL"]).max()
log(f"[1] 주거교통 QOL 재계산 — 팀 산출값과의 최대 차이: {diff:.1e}")

# 2) 다중회귀
X = sm.add_constant(qol[DOMAINS])
ols = sm.OLS(y, X).fit()
log("\n[2] 다중회귀: 거주지역 만족도 ~ 5개 영역 QOL (n=16)")
log(f"    R² {ols.rsquared:.3f} · 수정 R² {ols.rsquared_adj:.3f} · F검정 p {ols.f_pvalue:.4f}")
for name in DOMAINS:
    log(f"    {name:10s} 계수 {ols.params[name]:+.3f}  p {ols.pvalues[name]:.3f}")

# 3) 가중치를 곱한 변수로 회귀
Xw = sm.add_constant(qol[DOMAINS] * TEAM_WEIGHTS)
ols_w = sm.OLS(y, Xw).fit()
log("\n[3] 가중치를 곱한 변수로 회귀")
log(f"    R² {ols_w.rsquared:.3f} (가중치 전과 동일: {np.isclose(ols.rsquared, ols_w.rsquared)})")
log("    → 변수에 상수를 곱하면 계수만 그만큼 커지고 적합도는 그대로다. 회귀로는 가중치의 타당성을 검증할 수 없다.")

# 4) 종합 QOL과 만족도의 상관
weighted = composite(qol)
unweighted = qol[DOMAINS].mean(axis=1)
log("\n[4] 종합 QOL vs 거주지역 만족도")
for name, s in [("가중합(팀 가중치)", weighted), ("등가중", unweighted)]:
    r, p = pearsonr(s, y)
    rho, p_rho = spearmanr(s, y)
    log(f"    {name:12s} Pearson r {r:+.3f} (p {p:.3f}) · Spearman ρ {rho:+.3f} (p {p_rho:.3f})")
log("    → 팀 결론: 객관적 QOL이 높을수록 만족도가 낮다 (역상관)")

# 참고: 교육 영역에 학교생활만족도를 추가한 최종본에서도 방향은 같다
r_f, _ = spearmanr(composite(qol_final), y)
log(f"\n[참고] 교육 지표 수정 최종본의 가중합 Spearman ρ {r_f:+.3f}")

out = "\n".join(lines)
print(out)
RESULTS.mkdir(exist_ok=True)
(RESULTS / "01_original_analysis.txt").write_text(out + "\n", encoding="utf-8")
