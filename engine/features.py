"""공용 헬퍼 — 프롬프트 입력 생성·파싱·라벨화. experience.py·evaluate.py 공용."""

import numpy as np

from engine.memory import LABELS, label_active, label_display, label_short


def parse_attribution(raw, chans, future_known, horizon):
    """LLM attribution 출력 정규화. future-known 채널은 horizon 길이 라벨 리스트,
    past-only 채널은 라벨 1개. 형식이 어긋나면 neutral(리스트면 neutral×H)로 폴백."""
    out = {}
    for ch in chans:
        v = raw.get(ch) if isinstance(raw, dict) else None
        if ch in future_known:
            out[ch] = v if (isinstance(v, list) and len(v) == horizon
                            and all(x in LABELS for x in v)) else ["neutral"] * horizon
        else:
            out[ch] = v if v in LABELS else "neutral"
    return out


def neutral_attribution(chans, future_known, horizon):
    """전부 neutral인 attribution (파싱 실패 시 폴백)."""
    return {ch: (["neutral"] * horizon if ch in future_known else "neutral") for ch in chans}


def fmt(a):
    """숫자 배열 → 정수 CSV 문자열 (타깃·예측처럼 크기가 큰 값용)."""
    return ", ".join(str(int(round(float(x)))) for x in np.asarray(a))


def fmtf(a):
    """covariate용 — 작은 값(강수량·0/1 플래그)은 3자리 유지. fmt()면 0으로 뭉개져 상태 구분이 사라진다."""
    out = []
    for x in np.asarray(a, dtype="float64"):
        out.append(str(int(round(x))) if abs(x) >= 100 else f"{x:.3g}")
    return ", ".join(out)


def lookback_profile(y, ref):
    """look-back을 구조화 라벨로: trend / level / volatility (검색 키)."""
    n = len(y)
    slope = np.polyfit(np.arange(n), y, 1)[0]
    drift = slope * n / (y.mean() + 1e-9)
    trend = "rising" if drift > 0.05 else "falling" if drift < -0.05 else "flat"
    q1, q2 = np.quantile(ref, [0.33, 0.66])
    lvl = y.mean()
    level = "high" if lvl > q2 else "low" if lvl < q1 else "mid"
    cv = y.std() / (y.mean() + 1e-9)
    volatility = "high" if cv > 0.15 else "low"
    return {"trend": trend, "level": level, "volatility": volatility}


def covariate_value_lines(w):
    """채널별 실제 값 — look-back 구간 전체 + future-known은 horizon 값 (attribution 입력용)."""
    lines = []
    for ch in w["past"]:
        s = f"- {ch}: last {len(w['past'][ch])} days {fmt(w['past'][ch])}"
        if ch in w["future"]:
            s += f" | next-{len(w['future'][ch])} {fmt(w['future'][ch])}"
        lines.append(s)
    return "\n".join(lines)


def attribution_lines(attribution):
    """attribution → 채널당 한 줄 (future-known은 날짜별 [SD,··], past-only는 스칼라)."""
    return ", ".join(f"{ch}:{label_display(lab)}" for ch, lab in attribution.items())


def active_knowledge(attribution, knowledge):
    """non-neutral 채널의 knowledge만 → calibration 지침 (움직일 채널만 넣어 프롬프트 비대화 방지)."""
    lines = [f"- {ch} ({label_display(lab)}): {knowledge_text(knowledge.get(ch, '(no knowledge)'))}"
             for ch, lab in attribution.items() if label_active(lab)]
    return "\n".join(lines) if lines else "(no channel needs moving — keep Ŷ_base as is)"


def horizon_cov_lines(w):
    """calibration용 — future-known은 horizon 7일 값(그날 배치용), past-only는 미래 미지."""
    lines = []
    for ch in w["past"]:
        if ch in w["future"]:
            lines.append(f"- {ch} (future-known, per-day): {fmt(w['future'][ch])}")
        else:
            lines.append(f"- {ch} (past-only): future unknown — level/direction only")
    return "\n".join(lines)


def knowledge_text(rec):
    """knowledge record → 문단 텍스트. record는 {"knowledge": "..."} 형태."""
    if isinstance(rec, dict):
        return rec.get("knowledge", "")
    return str(rec)


def _label_change(a, b):
    """라벨 a→b 변화를 문자열로. 리스트면 바뀐 날만 '3일:··→sd' 형태로 (며칠째인지 보존)."""
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return " ".join(f"{i+1}일:{label_short(x)}→{label_short(y)}"
                        for i, (x, y) in enumerate(zip(a, b)) if x != y)
    return f"{label_short(a)}→{label_short(b)}"


def correction_lines(case):
    """case의 원본 라벨 → GT 보고 고친(corrected) 라벨 중 **바뀐 것만** 한 줄로 (학습 재료)."""
    corr = case.get("corrected")
    if not corr:
        return ""
    attr = case.get("attribution", {})
    diff = []
    for c, v in corr.items():
        if c not in attr or v == attr[c]:
            continue
        s = _label_change(attr[c], v)
        if s:
            diff.append(f"{c}: {s}")
    if not diff:
        return "    사후 라벨 점검: 고칠 것 없었음(위 라벨이 정답과 부합)"
    return "    사후 라벨 점검(정답 확인 후 — 이렇게 봤어야 함): " + " | ".join(diff)


def _case_block(c, *, lookback=False, forecast=False):
    """검색된 case 하나를 프롬프트용 블록으로. 라벨은 추론 시점 원본 + 사후 수정본을 함께 보여준다."""
    lines = [f"- {c['date']}:"]
    if lookback:
        lines.append(f"    look-back arrivals: {fmt(c.get('y_lookback', []))}")
    lines.append(f"    attribution(그때 판단): {attribution_lines(c['attribution'])}")
    corr = correction_lines(c)
    if corr:
        lines.append(corr)
    if forecast:
        lines.append(f"    Ŷ_base : {fmt(c['y_base'])}")
        lines.append(f"    Ŷ_calib: {fmt(c['y_calib'])}")
    lines.append(f"    verdict: {c.get('verdict','?')} — reflect: {c.get('reflect','')}")
    return "\n".join(lines)


def fmt_retrieved_lb(cases):
    """look-back 검색 결과(→ attribution 입력) → look-back 배열 + 라벨(원본/수정) + verdict + reflect."""
    if not cases:
        return "(none yet)"
    return "\n".join(_case_block(c, lookback=True) for c in cases)


def fmt_retrieved_attr(cases):
    """attribution 검색 결과(→ calibration 입력) → 라벨(원본/수정) + Ŷ_base/Ŷ_calib + verdict + reflect."""
    if not cases:
        return "(none yet)"
    return "\n".join(_case_block(c, forecast=True) for c in cases)
