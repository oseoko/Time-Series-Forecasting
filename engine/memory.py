"""
Experience memory — append-only case store + 라벨 기반 검색.

case dict:
  date, lb_profile{trend,level,volatility}, y_lookback[ctx], y_base[H], y_calib[H],
  attribution{ch:label}(추론 시점), corrected(GT 보고 고침), reasoning, verdict, reflect
(GT는 저장하지 않음 — leakage 방지)

검색 2종 — 단계마다 쓰는 키가 다름:
  retrieve_by_lookback     : look-back 시계열 DTW 거리 (B-2a, "상황이 비슷한")
  retrieve_by_attribution  : 5단계 라벨의 ordinal 거리 (B-2b, "진단이 비슷한")
"""

import json
import math
import os

import numpy as np


def dtw(a, b):
    """z-정규화 후 DTW 거리 — 수준이 아니라 모양으로 비교. 작을수록 유사."""
    a = np.asarray(a, dtype="float64"); b = np.asarray(b, dtype="float64")
    if a.size == 0 or b.size == 0:
        return 1e18
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


# 5단계 방향×강도 라벨 (ordinal 0..4)
LABELS = ["strong_down", "weak_down", "neutral", "weak_up", "strong_up"]
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}
SHORT = {"strong_down": "SD", "weak_down": "wd", "neutral": "··", "weak_up": "wu", "strong_up": "SU"}


# ── 라벨 표현 헬퍼 ──────────────────────────────────────────────
# future-known 채널: horizon 길이 라벨 리스트 (날짜별)  |  past-only 채널: 스칼라 라벨 1개.
# 아래 헬퍼들이 두 형태를 모두 받아, 쓰는 쪽이 타입을 신경 쓰지 않게 함.
def label_active(lab):
    """하나라도 non-neutral이면 True = 이 채널이 예측을 움직여야 함."""
    return any(x != "neutral" for x in lab) if isinstance(lab, list) else lab != "neutral"


def label_display(lab):
    return "[" + ",".join(SHORT.get(x, x) for x in lab) + "]" if isinstance(lab, list) else lab


def label_short(lab):
    if isinstance(lab, list):
        ext = max(lab, key=lambda x: abs(LABEL_IDX.get(x, 2) - 2))
        return SHORT.get(ext, "··").lower()
    return SHORT.get(lab, "··").lower()


def _label_prop(c, horizon, future_known):
    if c in future_known:
        return {"type": "array", "items": {"type": "string", "enum": LABELS},
                "minItems": horizon, "maxItems": horizon}
    return {"type": "string", "enum": LABELS}


def _label_obj(channels, horizon, future_known):
    return {"type": "object",
            "properties": {c: _label_prop(c, horizon, future_known) for c in channels},
            "required": list(channels), "additionalProperties": False}


def attribution_schema(channels, horizon, future_known):
    """B-2a structured output 스키마."""
    return {
        "type": "object",
        "properties": {
            "attribution": _label_obj(channels, horizon, future_known),
            "reasoning": {"type": "string"},
        },
        "required": ["attribution", "reasoning"], "additionalProperties": False,
    }


def array_schema(field):
    """B-2b calibration 스키마 — 보정된 숫자 배열 + reasoning."""
    return {"type": "object",
            "properties": {field: {"type": "array", "items": {"type": "number"}},
                           "reasoning": {"type": "string"}},
            "required": [field, "reasoning"], "additionalProperties": False}


def reflect_schema(channels, horizon, future_known):
    """B-3 reflection 스키마 — 회고 텍스트 + GT를 보고 고친 라벨(corrected).

    고친 라벨과 별개로 원본 attribution도 case에 남김. 검색 거리는 원본으로 계산해야 공정함
    — 추론 시점(GT 못 봄)의 라벨끼리 비교해야 하므로. corrected는 프롬프트에서
    "그때 이렇게 봤는데 실제론 이게 맞았다"는 학습 재료로만 쓰임.
    """
    return {
        "type": "object",
        "properties": {
            "reflect": {"type": "string"},
            "corrected": _label_obj(channels, horizon, future_known),
        },
        "required": ["reflect", "corrected"], "additionalProperties": False,
    }


# B-2c 사용자용 설명 — 요약 + 근거 목록 + 주의점
EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "drivers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "direction": {"type": "string"},
                    "factor": {"type": "string"},
                    "magnitude": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["date", "direction", "factor", "magnitude", "why"],
                "additionalProperties": False,
            },
        },
        "caveat": {"type": "string"},
    },
    "required": ["summary", "drivers", "caveat"], "additionalProperties": False,
}


def _repr_ord(v):
    """라벨(스칼라/리스트)을 대표 ordinal 하나로. 리스트는 '가장 극단적인 날'로 요약.

    평균이 아니라 극단값인 이유: hol_kr_wd처럼 7일 중 1일만 강한 신호인 채널을 평균 내면
    뭉개져 검색에서 사라짐. 'window 안에 강한 신호가 있나'를 봐야 하므로.
    """
    if isinstance(v, list):
        return max((LABEL_IDX.get(x, 2) for x in v), key=lambda o: abs(o - 2))
    return LABEL_IDX.get(v, 2)


def attribution_distance(a, b, weight):
    """{ch: label} 두 개의 ordinal 거리 합 (공통 채널만, 작을수록 유사)."""
    chs = set(a) & set(b)
    if not chs:
        return 1e9
    return sum(abs(_repr_ord(a[c]) - _repr_ord(b[c])) * weight.get(c, 1.0) for c in chs)


def idf_weights(cases):
    """채널별 IDF 가중 — 거의 매 window 활성인 채널(hol_cn 등)이 라벨 거리를 지배하지 않도록."""
    n = len(cases)
    if n == 0:
        return {}
    w = {}
    for ch in cases[0]["attribution"]:
        act = sum(label_active(c["attribution"][ch]) for c in cases if ch in c["attribution"])
        w[ch] = math.log((n + 1) / (act + 1)) + 0.1   # 활성 잦으면↓, 최소 0.1 (완전 0 방지)
    return w


class Experience:
    def __init__(self, path):
        self.path = path
        self.cases = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []

    def retrieve_by_lookback(self, y_lookback, k=3):
        cases = [c for c in self.cases if c.get("y_lookback")]
        ranked = sorted(cases, key=lambda c: dtw(y_lookback, c["y_lookback"]))
        return ranked[:k]

    def retrieve_by_attribution(self, attribution, k=3, y_lookback=None, pool=None):
        """진단(라벨)이 비슷한 case top-k.

        라벨이 5단계뿐이라 거리만으로는 동점이 매우 흔함. 그래서 2단계로 나눔 —
        ① 라벨 거리로 pool개를 넓게 추린 뒤 ② 라벨 거리와 look-back DTW를 각각 [0,1]로
        정규화해 가중 합산으로 리랭크. 라벨을 먼저 보고 동점만 DTW로 가르는 게 아니라 둘을
        함께 보므로, '진단도 같고 상황도 비슷한' case가 앞선다. y_lookback 없으면 ①만.
        """
        pool = pool or max(k * 3, 8)
        w = idf_weights(self.cases)
        d_all = {id(c): attribution_distance(attribution, c["attribution"], w) for c in self.cases}
        cand = sorted(self.cases, key=lambda c: d_all[id(c)])[:pool]
        if y_lookback is None or len(cand) < 2:   # 메모리가 비었거나 후보 1개 → 리랭크 불가
            return cand[:k]

        # 정규화는 pool 안에서만 — 탈락한 case까지 넣으면 min/max가 달라져 순위가 바뀐다
        d_lab = {id(c): d_all[id(c)] for c in cand}
        d_lb = {id(c): (dtw(y_lookback, c["y_lookback"]) if c.get("y_lookback") else 1e9) for c in cand}

        def norm(dmap):
            vals = list(dmap.values()); lo, hi = min(vals), max(vals)
            return {key: (v - lo) / (hi - lo) if hi > lo else 0.0 for key, v in dmap.items()}
        nlab, nlb = norm(d_lab), norm(d_lb)

        alpha = 0.6      # 라벨 vs look-back 비중. 튜닝값이 아니라 손으로 정한 기본값.

        def score(c):
            base = alpha * nlab[id(c)] + (1 - alpha) * nlb[id(c)]
            return base - (0.15 if c.get("verdict") == "better" else 0.0)   # 통했던 보정 우대(기본값)
        return sorted(cand, key=score)[:k]

    def add(self, case):
        self.cases.append(case)

    def save(self):
        json.dump(self.cases, open(self.path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
