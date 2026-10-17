"""공통 설정: 데이터 로드, 정규화, 원 지수 산식."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DOMAINS = ["교육_QOL", "경제_QOL", "주거교통_QOL", "인프라_QOL", "안전_QOL"]
# 팀이 정한 영역별 가중치 (종합 QOL 산정 기준)
TEAM_WEIGHTS = np.array([0.15, 0.25, 0.25, 0.20, 0.15])
SEED = 42


def load():
    """16개 구·군 데이터를 같은 순서로 읽는다."""
    raw = pd.read_csv(DATA / "indicators_raw.csv")
    qol_v1 = pd.read_csv(DATA / "qol_team_v1_20251229.csv")
    qol_final = pd.read_csv(DATA / "qol_team_final.csv")
    sat = pd.read_csv(DATA / "satisfaction_2025.csv")
    for df in (qol_v1, qol_final, sat):
        assert (df["구군"].values == raw["구군"].values).all(), "구군 순서 불일치"
    return raw, qol_v1, qol_final, sat


def minmax(s, reverse=False):
    """MinMax 정규화. reverse=True 이면 '낮을수록 좋은 지표'로 뒤집는다."""
    scaled = (s - s.min()) / (s.max() - s.min())
    return 1 - scaled if reverse else scaled


def housing_traffic_parts(raw, burden=False):
    """주거교통 QOL을 원 산식 그대로 계산해 주거·교통 부분으로 나눠 돌려준다.

    원 산식: 월세 0.20, 월세보증금 0.15, 전세보증금 0.15 (모두 낮을수록 좋음),
            지하철역 수 0.30, 버스정류장 수 0.20 (많을수록 좋음).
    burden=True 이면 임대료 '수준' 대신 '소득 대비 부담'을 쓴다 (가중치·방향은 동일).
    """
    income = raw["추정평균소득"]  # 월평균 가구소득 추정치 (만원)
    rent = raw["월세_만원"]
    rent_dep = raw["월세보증금_만원"]
    jeonse = raw["전세보증금_만원"]
    if burden:
        rent = rent / income                  # 월세 / 월소득
        rent_dep = rent_dep / (income * 12)   # 보증금 / 연소득
        jeonse = jeonse / (income * 12)       # 전세보증금 / 연소득
    housing = (0.20 * minmax(rent, True)
               + 0.15 * minmax(rent_dep, True)
               + 0.15 * minmax(jeonse, True))
    traffic = 0.30 * minmax(raw["지하철역수"]) + 0.20 * minmax(raw["버스정류장수"])
    return housing, traffic


def composite(qol, weights=TEAM_WEIGHTS, drop=()):
    """영역 점수를 가중합한다. drop 에 넣은 영역은 빼고 나머지 가중치를 다시 1로 맞춘다."""
    w = np.array(weights, dtype=float).copy()
    for d in drop:
        w[DOMAINS.index(d)] = 0.0
    w = w / w.sum()
    return qol[DOMAINS].to_numpy() @ w


def setup_matplotlib():
    import logging

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c9c8c3",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "text.color": "#0b0b0b",
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })
    return plt


# 차트 색: 원 지수(파랑), 재분석·충돌 표시(주황)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
