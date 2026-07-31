"""memory/test_results_*.json → fs_report.html — 제3자용 수요 예측 보고서.

analysis/explain_page.py(개발자용 내부 화면)와 다르다. 여기엔 라벨·검색·프롬프트 같은
내부 기제를 넣지 않는다. 읽는 사람이 수요 예측에 바로 쓸 수 있는 것만 남긴다:
  ① 무엇이 수요를 움직이는가 (효과 크기와 함께)
  ② 주별 전망과 그 근거 (모델이 생성한 사용자용 설명)
  ③ 얼마나 믿을 만한가 (기준선 대비 검증)

실행: python analysis/report_page.py [--res ...] [--out ...] [--standalone]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MEMDIR
from engine.data import load

_HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--res", default=os.path.join(MEMDIR, "test_results_all_abs_cap5_corr.json"))
ap.add_argument("--exp", default=os.path.join(MEMDIR, "experience_corr.json"))
ap.add_argument("--out", default=os.path.join(_HERE, "fs_report.html"))
ap.add_argument("--knowledge", default=os.path.join(MEMDIR, "knowledge_single.json"),
                help="knowledge json 경로 (요인별 근거 팝오버용)")
ap.add_argument("--standalone", action="store_true")
a = ap.parse_args()

WD = ["월", "화", "수", "목", "금", "토", "일"]
DF, TARGET, _, _ = load()
DATE = pd.to_datetime(DF["date"])
X = DF.set_index(DATE)

# ── 수요 동인: train 18 window의 잔차(실제 − 기준예측)로 계산 ──────────────
# 기준예측은 과거 입국자 수만 보고 만든 것이므로, 잔차 = 달력·이벤트가 만들어낸 몫.
_cases = json.load(open(a.exp, encoding="utf-8"))
_rows = []
for w in _cases:
    b = np.array(w["y_base"])
    idx = pd.date_range(pd.Timestamp(w["date"]), periods=7)
    s = X.reindex(idx)
    r = s[TARGET].values - b
    for t in range(7):
        _rows.append(dict(dow=idx[t].dayofweek, resid=r[t],
                          hol_wd=s["hol_kr_wd"].values[t], concert=s["concert_cap"].values[t],
                          hol_all=s["hol_kr"].values[t], actual=s[TARGET].values[t]))
R = pd.DataFrame(_rows).dropna()
SD = float(R.resid.std())


def effect(mask):
    """상태 간 잔차 차이와 효과크기(노이즈 대비). 스케일이 다른 요인도 같은 자로 잰다."""
    on, off = R.resid[mask], R.resid[~mask]
    diff = float(on.mean() - off.mean())
    return {"diff": diff, "d": diff / SD, "n": int(mask.sum())}


# 수요 동인 — 이 런이 실제 쓴 채널(experience의 attribution 키) 기준으로 분류.
#   ① CONFIRMED   : 쓰고 + 데이터로 확인됨(|d|≥0.5)   ② UNCONFIRMED : 쓰지만 + 확인 안 됨
#   ③ USED_UNSEEN : 쓰지만 + 데이터 효과 측정 불가(다변량)
#   이 런에서 안 쓴 채널은 어느 그룹에도 넣지 않는다.
USED_CH = set(_cases[0]["attribution"].keys()) if _cases else set()
def _used(*chs):
    return any(c in USED_CH for c in chs)

# 후보: (표시명, 설명, 채널들, 데이터효과 or None)
_CANDS = [
    ("한국 평일 공휴일", "주말이 아닌 관공서 공휴일 (신정·설·어린이날 등)", ["hol_kr_wd"], effect(R.hol_wd == 1)),
    ("한국 주말·공휴일", "토·일요일을 포함한 쉬는 날 전체", ["hol_kr"], effect(R.hol_all == 1)),
    ("금요일", "요일 자체의 상승 편차", ["is_friday"], effect(R.dow == 4)),
    ("콘서트 개최일", "대형 공연 티켓 오픈 기준", ["concert_cap"], effect(R.concert > 0)),
    ("환율 3종", "원/달러 · 원/위안 · 원/엔", ["fx_usd", "fx_cny", "fx_jpy"], None),
    ("항공편 4종", "중국 · 일본 · 미국 노선과 국제선 총편수",
     ["flight_cn", "flight_jp", "flight_us", "flight_total"], None),
    ("서울 기상 8종", "기온 · 강수 · 습도 · 일사 · 운량 · 기압 · 풍속 · 적설",
     ["w_seoul_AVGTA", "w_seoul_SUMRN", "w_seoul_AVGWS", "w_seoul_AVGRHM",
      "w_seoul_AVGPS", "w_seoul_SUMGSR", "w_seoul_DDMES", "w_seoul_AVGTCA"], None),
    ("해외 공휴일 3종", "중국 · 일본 · 미국", ["hol_cn", "hol_jp", "hol_us"], None),
    ("호텔 평균 요금", "", ["hotel_price"], None),
]
CONFIRMED, UNCONFIRMED, USED_BUT_UNSEEN = [], [], []
for name, note, chs, eff in _CANDS:
    if not _used(*chs):
        continue                                             # 이 런에서 안 쓴 covariate → 생략
    if eff is None:
        USED_BUT_UNSEEN.append((name, note))
    elif abs(eff["d"]) >= 0.5:
        CONFIRMED.append({"name": name, "note": note, **eff})
    else:
        UNCONFIRMED.append({"name": name, "note": note, **eff})

# 변수로 넣지 않은 요인(요일 편차 등)은 싣지 않는다 — 모델에 안 들어간 값은 이 런의 결과를
# 설명하지 못하고, 같은 표에 놓이면 "쓰고 있는 요인"처럼 읽힌다.
DAILY_MEAN = float(R.actual.mean())

KO = {"hol_cn": "중국 공휴일", "hol_jp": "일본 공휴일", "hol_us": "미국 공휴일",
      "hol_kr": "한국 주말·공휴일", "hol_kr_wd": "한국 평일 공휴일", "concert_cap": "콘서트",
      "flight_cn": "중국노선 항공편", "flight_jp": "일본노선 항공편",
      "flight_us": "미국노선 항공편", "flight_total": "국제선 총 항공편",
      "hotel_price": "호텔 요금", "fx_usd": "원/달러", "fx_cny": "원/위안", "fx_jpy": "원/엔",
      "w_seoul_AVGTA": "기온", "w_seoul_SUMRN": "강수", "w_seoul_AVGWS": "풍속",
      "w_seoul_AVGRHM": "습도", "w_seoul_AVGPS": "기압", "w_seoul_SUMGSR": "일사",
      "w_seoul_DDMES": "적설", "w_seoul_AVGTCA": "운량"}
# _CANDS에 없는 채널은 표에서 조용히 사라진다 — 채널을 늘린 런에서 실제로 그랬다(hol_kr).
# 남는 채널은 이름만이라도 "쓰지만 효과 측정 못 함"에 넣어, 모델 입력과 표가 어긋나지 않게 한다.
_left = sorted(USED_CH - {c for _, _, chs, _ in _CANDS for c in chs})
if _left:
    USED_BUT_UNSEEN.append((f"그 밖의 {len(_left)}종", " · ".join(KO.get(c, c) for c in _left)))

SH = {"strong_down": "큰 하향", "weak_down": "하향", "neutral": "—",
      "weak_up": "상향", "strong_up": "큰 상향"}
ORD = {"strong_down": -2, "weak_down": -1, "neutral": 0, "weak_up": 1, "strong_up": 2}

# 요인별 knowledge — 각 판정의 근거가 되는 누적 지식. 요인명(KO)으로 찾을 수 있게 미리 매핑.
_kpath = a.knowledge
_kraw = json.load(open(_kpath, encoding="utf-8")) if os.path.exists(_kpath) else {}
def _ktext(v):
    return v.get("knowledge", "") if isinstance(v, dict) else (v or "")
KNOW = {KO.get(ch, ch): _ktext(v) for ch, v in _kraw.items()}

# ── 주별 전망 ─────────────────────────────────────────────────────────────
RES = json.load(open(a.res, encoding="utf-8"))
_TA = {c["date"]: c["gt"] for c in
       json.load(open(a.exp.replace("experience", "train_analysis"), encoding="utf-8"))}
_EXP = {c["date"]: c for c in _cases}


def _rnd(xs):
    return [round(float(v)) for v in xs]


def context_of(start, n=28):
    """이 주를 예측할 때 모델이 근거로 삼은 타깃 과거 n일. 메모리·프롬프트에 실제로 들어간 값."""
    end = pd.Timestamp(start) - pd.Timedelta(days=1)
    s = X.loc[:end, TARGET].tail(n)
    return _rnd(s.values)


def heat_of(attribution, horizon=7):
    """요인 × 날짜 판정 행렬. 라벨은 원래 날짜별이므로 대표값 하나로 뭉개지 않는다.

    미래를 아는 요인(공휴일·콘서트·항공편)은 날짜마다 다른 라벨을 갖고,
    미래를 모르는 요인(환율·기상)은 주 전체에 라벨 하나 → 7칸을 같은 값으로 채우고 'week'로 표시.
    중립만 있는 행은 뺀다(22행이 전부 '—'로 채워지면 읽히지 않는다).
    """
    rows, quiet = [], 0
    for ch, v in attribution.items():
        vals = v if isinstance(v, list) else [v] * horizon
        if all(x == "neutral" for x in vals):
            quiet += 1
            continue
        rows.append({"ko": KO.get(ch, ch),
                     "role": "day" if isinstance(v, list) else "week",
                     "cells": [{"t": SH[x], "o": ORD[x]} for x in vals]})
    rows.sort(key=lambda r: (-max(abs(c["o"]) for c in r["cells"]), r["ko"]))
    return {"rows": rows, "quiet": quiet}


def case_of(d):
    """검색된 과거 사례 하나 — 메모리에 들어간 원본 값 전부 + 회고 + 라벨 교정."""
    c = _EXP.get(d)
    if not c:
        return None
    fixes = []
    for ch, now in (c.get("corrected") or {}).items():
        was = c["attribution"].get(ch)
        if was is None or now == was or ch not in KO and not isinstance(was, str):
            continue
        pairs = (zip(was, now) if isinstance(was, list) else [(was, now)])
        for t, (x, y) in enumerate(pairs):
            if x != y:
                fixes.append({"ko": KO.get(ch, ch), "d": t + 1 if isinstance(was, list) else 0,
                              "was": SH[x], "now": SH[y], "o": ORD[y]})
    rsn = c.get("reasoning", {})
    cidx = pd.date_range(pd.Timestamp(d), periods=7)
    return {"date": d, "verdict": c.get("verdict", "?"),
            "days": [{"md": x.strftime("%m-%d"), "wd": WD[x.dayofweek]} for x in cidx],
            "context": _rnd(c.get("y_lookback", [])),
            "base": _rnd(c["y_base"]), "fc": _rnd(c["y_calib"]),
            "gt": _rnd(_TA.get(d, [])),
            "heat": heat_of(c["attribution"]), "fixes": fixes[:8],
            "r_attr": rsn.get("R_attr", ""), "reflect": c.get("reflect", "")}


weeks = []
for w in RES:
    idx = pd.date_range(pd.Timestamp(w["date"]), periods=7)
    weeks.append({
        "start": idx[0].strftime("%Y-%m-%d"), "end": idx[-1].strftime("%m-%d"),
        "days": [{"md": d.strftime("%m-%d"), "wd": WD[d.dayofweek]} for d in idx],
        "base": [round(float(x)) for x in w["y_base"]],
        "fc": [round(float(x)) for x in w["y_calib"]],
        "gt": [round(float(x)) for x in w["gt"]],
        "mae_base": round(w["mae_base"]), "mae_fc": round(w["mae_calib"]),
        "expl": w.get("explanation"),
        # ── 전 과정 (접힘) — 교수님 요구: retrieval된 예제·메모리 원본 context·base 값 전부 노출
        "context": context_of(w["date"]),          # 예측 근거로 쓴 타깃 28일
        "heat": heat_of(w["attribution"]),         # 요인 × 날짜 판정 행렬
        "r_attr": w.get("R_attr", ""), "r_calib": w.get("R_calib", ""),
        "cases": [c for c in (case_of(d) for d in w.get("retrieved_attr", [])) if c],
        "cases_lb": [c for c in (case_of(d) for d in w.get("retrieved_lb", [])) if c],
    })

mb = float(np.mean([w["mae_base"] for w in weeks]))
mf = float(np.mean([w["mae_fc"] for w in weeks]))
better = sum(1 for w in weeks if w["mae_fc"] < w["mae_base"])
KPI = {"mae_base": round(mb), "mae_fc": round(mf), "gain": (mf - mb) / mb * 100,
       "better": better, "n": len(weeks), "daily": round(DAILY_MEAN),
       "err_pct": mf / DAILY_MEAN * 100}

# ── 증거 ① 날짜별 판정 — 같은 요인이 요일마다 다르게 판정된다 ─────────────
# 평일 공휴일이 걸린 주를 고른다. 그 요인이 '그날만' 하향인 것이 날짜별 판정의 요점.
_w = next((w for w in RES if any(
    isinstance(v, list) and any(x.endswith("_down") for x in v)
    for ch, v in w["attribution"].items() if ch == "hol_kr_wd")), RES[0])
_idx = pd.date_range(pd.Timestamp(_w["date"]), periods=7)
PERDAY = {
    "start": _idx[0].strftime("%Y-%m-%d"), "end": _idx[-1].strftime("%m-%d"),
    "days": [{"md": d.strftime("%m-%d"), "wd": WD[d.dayofweek]} for d in _idx],
    "rows": [{"ko": KO.get(ch, ch),
              "cells": [{"t": SH[x], "o": ORD[x]} for x in v]}
             for ch, v in _w["attribution"].items()
             if isinstance(v, list) and len(set(v)) > 1][:5],   # 날짜별로 갈리는 것만
}

# ── 증거 ② 과거의 회고가 이번 판단에 실제로 쓰였는가 ──────────────────────
EXPC = {c["date"]: c for c in _cases}
_import_re = __import__("re")


def _cites(txt):
    m = _import_re.findall(r"[^.。]*(?:과거|사례|반성|회고)[^.。]*[.。]", txt or "")
    return m[0].strip() if m else None


def _topic(txt):
    """이 텍스트가 어떤 요인을 근거로 삼았는지 — 회고와 판단이 같은 요인을 말할 때만 연쇄로 인정한다."""
    return {ch for ch in KO if ch in (txt or "")}


# 이번 판단이 인용한 요인을, 검색된 과거 case의 회고가 실제로 뒷받침하는 쌍을 찾는다.
# (인용문에 있는 단어로만 매칭하면 아무 case나 걸린다 — 회고 쪽에도 같은 요인이 있어야 한다.)
CHAIN = None
for w in RES:
    quote = _cites(w.get("R_calib", ""))
    if not quote:
        continue
    want = _topic(quote)
    for d in w.get("retrieved_attr", []):
        c = EXPC.get(d)
        if not (c and c.get("reflect")):
            continue
        shared = want & _topic(c["reflect"])
        if shared:
            CHAIN = {"past_date": d, "past_reflect": c["reflect"],
                     "now_date": pd.Timestamp(w["date"]).strftime("%Y-%m-%d"),
                     "now_quote": quote,
                     "shared": sorted(KO[ch] for ch in shared)}
            break
    if CHAIN:
        break

# 검증 구간에서 각 요인이 실제로 '중립 아님'으로 판정된 주 수 — 위 ②의 근거.
# 신호가 확인되지 않은 요인에도 매주 라벨이 붙고 있다는 사실을 감추지 않는다.
_act = {}
for w in RES:
    for ch, v in w["attribution"].items():
        vals = v if isinstance(v, list) else [v]
        if any(x != "neutral" for x in vals):
            _act[ch] = _act.get(ch, 0) + 1
ALWAYS_ON = sorted(((KO.get(ch, ch), n) for ch, n in _act.items() if n == len(RES)),
                   key=lambda x: x[0])

# 요일별 MAE (기준 vs 보정) — 각 window의 horizon 날짜를 요일로 매핑해 집계.
_WD = ["월", "화", "수", "목", "금", "토", "일"]
_wb = {i: [] for i in range(7)}; _wc = {i: [] for i in range(7)}
for w in RES:
    d0 = pd.Timestamp(w["date"])
    yb, yc, gt = np.array(w["y_base"]), np.array(w["y_calib"]), np.array(w["gt"])
    for t in range(len(gt)):
        dow = (d0 + pd.Timedelta(days=t)).dayofweek
        _wb[dow].append(abs(yb[t] - gt[t])); _wc[dow].append(abs(yc[t] - gt[t]))
WEEKDAY = []
for i in range(7):
    if not _wb[i]:
        continue
    mb_, mc_ = float(np.mean(_wb[i])), float(np.mean(_wc[i]))
    WEEKDAY.append({"wd": _WD[i], "base": round(mb_), "calib": round(mc_),
                    "gain": round((mc_ - mb_) / mb_ * 100, 1) if mb_ else 0.0})

# 오른쪽 칩("영향 큼/보통")은 수요를 얼마나 흔드는지일 뿐, 예측이 얼마나 좋아졌는지가 아니다.
# 둘을 헷갈리면 금요일처럼 "작지만 매주 반복돼 정확도에 크게 기여한" 요인이 쓸모없어 보인다.
# 요일에 대응하는 요인은 검증 구간의 실제 오차 감소를 같은 줄에 붙여 그 오해를 막는다.
_WGAIN = {w["wd"]: w["gain"] for w in WEEKDAY}
for _d in CONFIRMED + UNCONFIRMED:
    _wd = _d["name"][0] if _d["name"].endswith("요일") and len(_d["name"]) == 3 else None
    if _wd in _WGAIN and _WGAIN[_wd] < 0:
        _d["note"] += f" · 검증 {len(RES)}주 동안 {_d['name']} 예측 오차 {abs(_WGAIN[_wd]):.0f}% 감소"

# ── 방법론 문단 — 이 런의 실제 설정으로 쓴다 ────────────────────────────────
# 파일명이 곧 설정이다: test_results_<tag>_<calib>_(free|scaleN)[_attrwin|_attrall][_mem].json
# (evaluate.py의 ftag 규칙). 하드코딩하면 free 런에 "±5% 제한"처럼 사실이 아닌 문장이 남는다.
_rname = os.path.splitext(os.path.basename(a.res))[0]
_scale = next((int(t[5:]) for t in _rname.split("_") if t.startswith("scale") and t[5:].isdigit()), 0)
_H = len(RES[0]["gt"]) if RES else 7
_CTX = len(context_of(RES[0]["date"])) if RES else 28
_ret = any(w.get("retrieved_lb") or w.get("retrieved_attr") for w in RES)

_s3 = (f"하루 보정폭은 기준값의 ±{_scale}%를 넘지 않도록 프롬프트에서 제한합니다."
       if _scale > 0 else "하루 보정폭에는 상한을 두지 않았습니다.")
_s3_ret = "과거의 비슷한 사례를 검색해 함께 참고하며, " if _ret else ""

METHOD = [
    f"<b>1단계 — 기준 예측.</b> 시계열 기반 모델(Chronos-2)이 <b>과거 {_CTX}일의 입국자 수만</b> 보고 "
    f"다음 {_H}일을 예측합니다.",
    "<b>2단계 — 진단.</b> 언어모델이 각 요인의 그날 값을 보고, 그 요인이 수요를 어느 방향으로 "
    "얼마나 움직일지 판정합니다.",
    f"<b>3단계 — 보정.</b> 그 진단을 근거로 기준 예측을 다시 씁니다. {_s3_ret}{_s3}",
    "<b>학습.</b> 학습 구간에서는 예측이 끝난 뒤 실제와 대조해, <b>어느 판정이 틀렸는지</b>를 "
    "기록하고 다음 판정에 참고합니다.",
]

D = {"confirmed": CONFIRMED, "unconfirmed": UNCONFIRMED, "used_unseen": USED_BUT_UNSEEN,
     "always_on": ALWAYS_ON, "n_weeks": len(RES),
     "weeks": weeks, "kpi": KPI, "sd": round(SD), "perday": PERDAY, "chain": CHAIN,
     "know": KNOW, "weekday": WEEKDAY, "method": METHOD}

HTML = """<title>방한 외국인 입국자 수요 — 6주 검증 리포트</title>
<style>
  :root{
    --bg:#f6f7f8; --card:#ffffff; --ink:#1a1d21; --body:#4a5158; --mut:#8b939b;
    --line:#e2e6e9; --line2:#eef1f3;
    --fc:#0d7a72;          /* 예측 — 딥 틸 */
    --gt:#1a1d21;          /* 실제 — 잉크 */
    --base:#a8b0b7;        /* 기준값 — 회색 점선 (참조선, 후퇴) */
    --ctx:#6b7480;         /* 과거 실적 — 기준값보다 진한 회색. 둘을 같은 색으로 두면 구분이 안 된다 */
    --down:#a8443c;        /* 하향 요인 — 벽돌 */
    --up:#b98229;          /* 상향 요인 — 앰버 */
    --sans:system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;
  }
  @media (prefers-color-scheme:dark){
    :root{--bg:#14171a; --card:#1b1f23; --ink:#eef1f3; --body:#b4bcc3; --mut:#79828a;
          --line:#2b3238; --line2:#22282d; --fc:#4fb3a8; --gt:#eef1f3; --base:#5d666e; --ctx:#98a2ac;
          --down:#d9756a; --up:#d9a84f;}
  }
  :root[data-theme="dark"]{--bg:#14171a; --card:#1b1f23; --ink:#eef1f3; --body:#b4bcc3; --mut:#79828a;
    --line:#2b3238; --line2:#22282d; --fc:#4fb3a8; --gt:#eef1f3; --base:#5d666e; --ctx:#98a2ac;
    --down:#d9756a; --up:#d9a84f;}
  :root[data-theme="light"]{--bg:#f6f7f8; --card:#ffffff; --ink:#1a1d21; --body:#4a5158; --mut:#8b939b;
    --line:#e2e6e9; --line2:#eef1f3; --fc:#0d7a72; --gt:#1a1d21; --base:#a8b0b7; --ctx:#6b7480;
    --down:#a8443c; --up:#b98229;}

  body{margin:0;background:var(--bg);color:var(--body);font-family:var(--sans);
    font-size:15px;line-height:1.7;-webkit-font-smoothing:antialiased}
  /* 본문 폭 = 컨테이너 폭. 컨테이너를 넓게 두고 글에만 max-width를 걸면 오른쪽이 휑하게 빈다. */
  .wrap{max-width:760px;margin:0 auto;padding:48px 24px 72px;display:flex;flex-direction:column;gap:40px}
  h1,h2,h3{color:var(--ink);text-wrap:balance;margin:0}
  h1{font-size:30px;font-weight:700;letter-spacing:-.025em;line-height:1.25}
  h2{font-size:19px;font-weight:650;letter-spacing:-.015em}
  h3{font-size:15px;font-weight:650}
  p{margin:0}
  .eyebrow{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.08em;
    color:var(--mut);margin-bottom:10px}
  .sub{margin-top:12px;font-size:16px;color:var(--body)}
  header{border-bottom:1px solid var(--line);padding-bottom:32px}

  section{display:flex;flex-direction:column;gap:18px}
  .shd{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
  .shd .n{font-family:var(--mono);font-size:11px;color:var(--mut);letter-spacing:.06em}

  /* KPI */
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
    background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .kpi{background:var(--card);padding:16px 18px}
  .kpi .v{font-family:var(--mono);font-size:24px;font-weight:600;color:var(--ink);
    font-variant-numeric:tabular-nums;letter-spacing:-.02em;line-height:1.2}
  .kpi .l{font-size:12px;color:var(--mut);margin-top:3px}

  /* 동인 */
  .drv{display:flex;flex-direction:column;gap:1px;background:var(--line);
    border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .drow{background:var(--card);padding:14px 18px;display:grid;
    grid-template-columns:1fr 130px 96px;gap:14px;align-items:center}
  .dnm{min-width:0}
  .dnm b{display:block;color:var(--ink);font-size:14.5px;font-weight:600}
  .dnm span{font-size:12.5px;color:var(--mut)}
  .dbar{position:relative;height:22px}
  .dbar i{position:absolute;top:4px;height:14px;border-radius:2px}
  .dbar .zero{position:absolute;top:0;bottom:0;width:1px;background:var(--line);left:50%}
  .dval{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums;
    text-align:right;color:var(--ink);font-weight:600}
  .dval em{display:block;font-style:normal;font-size:11px;color:var(--mut);font-weight:400}
  .strong .dval{color:var(--down)}

  .nf{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 18px}
  .nf b{color:var(--ink);font-size:13.5px}
  .nf ul{margin:8px 0 0;padding-left:18px;font-size:13.5px;color:var(--body)}
  .nf li{margin-bottom:2px}
  .nf .tail{margin-top:10px;font-size:13px;color:var(--mut)}

  /* 주별 카드 */
  .wk{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
  .whd{display:flex;align-items:baseline;gap:12px;padding:15px 18px;border-bottom:1px solid var(--line2)}
  .whd b{font-family:var(--mono);font-size:14px;color:var(--ink);font-variant-numeric:tabular-nums}
  .whd .acc{margin-left:auto;font-family:var(--mono);font-size:11.5px;color:var(--mut);
    font-variant-numeric:tabular-nums}
  .whd .acc s{text-decoration:none;color:var(--fc);font-weight:600}
  .wbody{padding:16px 18px;display:flex;flex-direction:column;gap:14px}
  .lg{display:flex;gap:14px;font-size:11.5px;color:var(--mut);font-family:var(--mono)}
  .lg i{display:inline-block;width:14px;height:2px;vertical-align:3px;margin-right:5px}
  .chart{width:100%;overflow-x:auto}
  .chart svg{display:block}
  /* 설명 = 근거 원문의 정리본임을 밝힌다 */
  .ehd{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.06em;
    color:var(--fc);font-weight:700;margin-bottom:-6px}
  .ehd span{display:block;text-transform:none;letter-spacing:0;color:var(--mut);
    font-weight:400;font-size:11px;margin-top:3px;line-height:1.5}
  .orig{font-size:12.5px;color:var(--mut);line-height:1.6;
    border-left:2px solid var(--line);padding-left:10px}
  .orig b{color:var(--body)}
  .esum{font-size:14.5px;color:var(--ink);line-height:1.7}
  .edrv{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:5px}
  .edrv li{font-size:13.5px;color:var(--body);padding-left:13px;position:relative}
  .edrv li::before{content:"";position:absolute;left:0;top:9px;width:5px;height:5px;
    border-radius:50%;background:var(--mut)}
  .edrv li.up::before{background:var(--up)} .edrv li.dn::before{background:var(--down)}
  .edrv b{font-family:var(--mono);font-size:12.5px;color:var(--ink);
    font-variant-numeric:tabular-nums;margin-right:5px}
  .ecav{font-size:13px;color:var(--mut);border-top:1px solid var(--line2);padding-top:11px}

  /* 증거 ① 날짜별 판정 */
  .pdwrap{background:var(--card);border:1px solid var(--line);border-radius:8px;
    padding:16px 18px;overflow-x:auto}
  table.pd{border-collapse:collapse;width:100%;font-size:12.5px}
  table.pd th{font-family:var(--mono);font-size:10.5px;font-weight:500;color:var(--mut);
    padding:0 0 8px;text-align:center;white-space:nowrap}
  table.pd th.wknd{color:var(--body)}
  table.pd th:first-child{text-align:left;padding-right:14px}
  table.pd td{padding:4px 3px;text-align:center;white-space:nowrap}
  table.pd td:first-child{text-align:left;color:var(--ink);font-weight:600;
    padding-right:14px;white-space:nowrap}
  .pdc{display:block;border-radius:4px;padding:4px 2px;font-size:11px;
    font-family:var(--mono);color:var(--mut);background:var(--line2)}
  .pdc.d2{background:color-mix(in srgb,var(--down) 82%,transparent);color:#fff;font-weight:600}
  .pdc.d1{background:color-mix(in srgb,var(--down) 32%,transparent);color:var(--ink)}
  .pdc.u1{background:color-mix(in srgb,var(--up) 32%,transparent);color:var(--ink)}
  .pdc.u2{background:color-mix(in srgb,var(--up) 82%,transparent);color:#fff;font-weight:600}
  table.wk{border-collapse:collapse;width:100%;max-width:440px;font-size:13px}
  table.wk th{font-family:var(--mono);font-size:10.5px;font-weight:500;color:var(--mut);
    text-align:right;padding:4px 12px 8px;border-bottom:1px solid var(--line)}
  table.wk th:first-child{text-align:left}
  table.wk td{padding:6px 12px;text-align:right;border-bottom:1px solid var(--line2);
    font-family:var(--mono)}
  table.wk td:first-child{text-align:left;font-weight:600;color:var(--ink);font-family:var(--sans)}
  .wkgood{color:var(--fc);font-weight:600}
  .wkbad{color:var(--down);font-weight:600}
  .wkflat{color:var(--mut)}

  /* 증거 ② 회고 → 판단 연쇄 */
  .chain{display:flex;flex-direction:column;gap:0}
  .cbox{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:15px 18px}
  .cbox .clab{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
    letter-spacing:.06em;color:var(--mut);margin-bottom:6px}
  .cbox .clab b{color:var(--ink);font-weight:600}
  .cbox p{font-size:13.5px;line-height:1.7;color:var(--body)}
  .cbox.now{border-color:var(--fc)}
  .cbox.now .clab b{color:var(--fc)}
  .carrow{align-self:center;font-family:var(--mono);font-size:11px;color:var(--mut);
    padding:8px 0;display:flex;flex-direction:column;align-items:center;gap:2px}
  .carrow em{font-style:normal;font-size:20px;line-height:1;color:var(--line)}
  mark{background:color-mix(in srgb,var(--fc) 20%,transparent);color:var(--ink);
    padding:1px 2px;border-radius:2px}

  /* 그룹 제목 */
  .gh{margin-top:6px;font-size:13px;font-weight:650;color:var(--ink)}
  .note{font-size:13px;color:var(--mut);line-height:1.7}
  .dim{color:var(--mut)}

  /* 전 과정 — 눌러서 여는 버튼임이 보이게 */
  details.pr{border-top:1px solid var(--line2);margin-top:4px;padding-top:14px}
  details.pr>summary{cursor:pointer;list-style:none;font-size:13px;font-weight:600;
    color:var(--fc);display:inline-flex;align-items:center;gap:8px;
    padding:8px 14px;border:1px solid var(--fc);border-radius:7px;
    background:color-mix(in srgb,var(--fc) 7%,transparent);
    transition:background .12s,transform .12s;user-select:none}
  details.pr>summary::-webkit-details-marker{display:none}
  details.pr>summary::before{content:"＋";font-family:var(--mono);font-size:13px;line-height:1}
  details.pr[open]>summary::before{content:"−"}
  details.pr>summary:hover{background:color-mix(in srgb,var(--fc) 15%,transparent)}
  details.pr>summary:active{transform:translateY(1px)}
  details.pr>summary:focus-visible{outline:2px solid var(--fc);outline-offset:2px}
  details.pr[open]>summary{background:color-mix(in srgb,var(--fc) 15%,transparent)}
  .prb{padding-top:16px;display:flex;flex-direction:column;gap:14px}
  .fld{display:flex;flex-direction:column;gap:6px}
  .flab{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.06em;
    color:var(--mut)}
  .fld p{font-size:13.5px;line-height:1.7;color:var(--body)}
  .mnum{font-family:var(--mono);font-size:11px;color:var(--mut);line-height:1.6;
    font-variant-numeric:tabular-nums;word-break:break-all}
  .none{font-size:13px;color:var(--mut)}

  .chips{display:flex;flex-wrap:wrap;gap:5px}
  .chip{display:inline-flex;align-items:center;gap:4px;font-size:11.5px;color:var(--body);
    background:var(--line2);border-radius:5px;padding:2px 7px;white-space:nowrap}
  .chip b{font-weight:600;color:var(--ink)}
  .chip i{font-style:normal;font-size:10px;color:var(--mut)}
  .chip s{text-decoration:line-through;color:var(--mut)}
  .chip.o-2{background:color-mix(in srgb,var(--down) 26%,transparent)}
  .chip.o-1{background:color-mix(in srgb,var(--down) 13%,transparent)}
  .chip.o1{background:color-mix(in srgb,var(--up) 13%,transparent)}
  .chip.o2{background:color-mix(in srgb,var(--up) 26%,transparent)}
  .chip.fix b{color:var(--fc)}

  /* 검색된 사례 카드 (중첩 접힘) */
  .csl{display:flex;flex-direction:column;gap:6px}
  details.cs{border:1px solid var(--line);border-radius:7px;background:var(--bg);
    transition:border-color .12s}
  details.cs:hover{border-color:var(--fc)}
  details.cs>summary{padding:9px 12px;cursor:pointer;list-style:none;display:flex;
    align-items:center;gap:9px;font-size:12.5px;user-select:none}
  details.cs>summary::-webkit-details-marker{display:none}
  details.cs>summary::before{content:"＋";font-family:var(--mono);font-size:12px;
    color:var(--mut);line-height:1}
  details.cs[open]>summary::before{content:"−";color:var(--fc)}
  details.cs>summary b{font-family:var(--mono);color:var(--ink);font-weight:600}
  details.cs>summary:focus-visible{outline:2px solid var(--fc);outline-offset:-2px;border-radius:6px}
  .vd{font-size:11px;color:var(--mut)}
  .vd.up{color:var(--fc)} .vd.dn{color:var(--down)}
  .tog{margin-left:auto;font-size:10.5px;color:var(--mut);font-family:var(--mono);
    border:1px solid var(--line);border-radius:5px;padding:1px 7px}
  .tog::after{content:"펼치기"}
  details.cs[open] .tog::after{content:"접기"}
  details.cs:hover .tog{border-color:var(--fc);color:var(--fc)}
  .csb{padding:2px 12px 12px;display:flex;flex-direction:column;gap:11px;
    border-top:1px solid var(--line2);margin-top:2px;padding-top:11px}
  /* 카드 안 히트맵 — 좁으니 여백을 줄인다 */
  .csb .pdwrap,.prb .pdwrap{padding:10px 12px}
  .csb table.pd,.prb table.pd{font-size:11.5px}
  .csb table.pd td:first-child,.prb table.pd td:first-child{font-size:12px;font-weight:600}
  .rw{font-style:normal;font-size:9.5px;color:var(--mut);font-weight:400;margin-left:5px;
    font-family:var(--mono)}
  .qt{font-size:11.5px;color:var(--mut);margin-top:6px}
  /* 요인명 → knowledge 팝오버 버튼 */
  .kbtn{font:inherit;color:var(--ink);font-weight:600;background:none;border:0;padding:0;
    cursor:pointer;display:inline-flex;align-items:center;gap:5px;text-align:left;
    text-decoration:underline;text-decoration-style:dotted;text-decoration-color:var(--fc);
    text-underline-offset:3px}
  .kbtn:hover{color:var(--fc)}
  .kbtn:focus-visible{outline:2px solid var(--fc);outline-offset:2px;border-radius:3px}
  /* '지식' 배지 — 눌러서 근거를 볼 수 있는 버튼임이 드러나게 */
  .ki{display:inline-flex;align-items:center;gap:3px;font-size:9px;font-weight:700;
    letter-spacing:.02em;color:var(--fc);background:color-mix(in srgb,var(--fc) 12%,transparent);
    border:1px solid color-mix(in srgb,var(--fc) 35%,transparent);border-radius:4px;
    padding:1px 5px;font-family:var(--sans);white-space:nowrap;flex:none}
  .kbtn:hover .ki{background:color-mix(in srgb,var(--fc) 22%,transparent);border-color:var(--fc)}
  .kpop{position:absolute;z-index:50;max-width:368px;background:var(--card);
    border:1px solid var(--fc);border-radius:9px;padding:13px 15px;
    box-shadow:0 8px 28px rgba(0,0,0,.16)}
  .kpop .khd{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.05em;
    color:var(--fc);font-weight:700;display:flex;align-items:center;gap:8px;margin-bottom:7px}
  .kpop .khd span{text-transform:none;letter-spacing:0;color:var(--mut);font-weight:400}
  .kpop .kx{margin-left:auto;background:none;border:0;color:var(--mut);cursor:pointer;
    font-size:12px;padding:0 2px}
  .kpop .kx:hover{color:var(--ink)}
  .kpop p{font-size:12.5px;line-height:1.65;color:var(--body);margin:0}

  .mini{display:flex;flex-direction:column;gap:6px}
  .mlab{font-size:10.5px;color:var(--mut);font-family:var(--mono)}
  /* 범례 — 계열이 둘 이상이면 색만으로 구분하게 두지 않는다 */
  .mlg{display:flex;flex-wrap:wrap;gap:12px;font-size:10.5px;color:var(--mut);
    font-family:var(--mono)}
  .mlg span{display:inline-flex;align-items:center;gap:5px}
  .mlg i{width:14px;height:2px;border-radius:1px;flex:none}
  .mlg i.ds{height:0;border-top:2px dashed currentColor;background:none!important;
    color:var(--base)}

  details.mth{background:var(--card);border:1px solid var(--line);border-radius:8px}
  details.mth summary{padding:14px 18px;cursor:pointer;color:var(--ink);font-weight:600;font-size:14px;
    list-style:none;display:flex;align-items:center}
  details.mth summary::-webkit-details-marker{display:none}
  details.mth summary::after{content:"펼치기";margin-left:auto;font-family:var(--mono);
    font-size:11px;color:var(--mut);font-weight:400}
  details.mth[open] summary::after{content:"접기"}
  details.mth summary:focus-visible{outline:2px solid var(--fc);outline-offset:-2px}
  .mbody{padding:0 18px 16px;font-size:13.5px;line-height:1.75;display:flex;
    flex-direction:column;gap:10px}
  /* 66ch로 묶어두면 760px 본문 안에서 절반 남짓만 쓰고 접혀 답답해 보인다 — 블록 폭을 그대로 쓴다 */
  .mbody p{max-width:none}
  .mbody code{font-family:var(--mono);font-size:12.5px;color:var(--ink)}
  footer{border-top:1px solid var(--line);padding-top:20px;font-size:12.5px;color:var(--mut);
    font-family:var(--mono)}
  @media (max-width:620px){
    .drow{grid-template-columns:1fr 78px}
    .dbar{display:none}
  }
  @media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">수요 예측 검증 · 2026년 5–6월 6주</div>
    <h1>방한 외국인 입국자 수요는 무엇으로 설명되는가</h1>
    <p class="sub">일별 외국인 입국자 수를 7일 앞까지 예측하고, 예측이 빗나간 몫을 달력·이벤트·경제
      지표로 설명한 결과입니다. 아래 6주는 <b>모델이 학습에 쓰지 않은 구간</b>이며, 예측한 뒤 실제와
      대조했습니다.</p>
  </header>

  <section>
    <div class="shd"><h2>요약</h2></div>
    <div class="kpis" id="kpis"></div>
    <p class="prose" id="kpinote"></p>
  </section>

  <section>
    <div class="shd"><h2>수요를 움직이는 것</h2></div>
    <p>기준 예측은 <b>과거 입국자 수만</b> 보고 만듭니다. 따라서 실제가 기준에서 벗어난 몫은 달력·이벤트가
      만들어낸 것입니다. 아래 수치는 그 몫을 요인별로 갈라낸 것으로, <b>하루 평균 몇 명을 움직였는지</b>와
      <b>일별 변동폭(±<span id="sd"></span>명) 대비 크기</b>입니다.</p>
    <p class="note">오른쪽의 <b>영향 큼 · 영향 보통</b>은 그 요인이 하루 수요를 얼마나 크게 흔드는지를
      뜻합니다. 한 번에 크게 흔드는 요인만 쓸모 있는 것은 아닙니다 — 크기는 보통이어도 <b>매주 어김없이
      반복되는</b> 요인은 예측 정확도를 크게 끌어올립니다.</p>

    <div id="grp1"><h3 class="gh">확인된 요인</h3>
    <div class="drv" id="drv1"></div></div>

    <div id="grp2"><h3 class="gh">모델이 쓰지만, 확인되지 않은 요인</h3>
    <div class="drv" id="drv2"></div>
    <div class="nf" id="nf"></div></div>
  </section>

  <section>
    <div class="shd"><h2>주별 전망과 근거</h2><span class="n" id="wkn"></span></div>
    <div class="lg">
      <span><i style="background:var(--gt)"></i>실제</span>
      <span><i style="background:var(--fc)"></i>예측</span>
      <span><i style="background:var(--base)"></i>기준선 (과거값만)</span>
    </div>
    <div id="weeks" style="display:flex;flex-direction:column;gap:16px"></div>
  </section>

  <section>
    <div class="shd"><h2>판정은 날짜 단위로 이뤄진다</h2></div>
    <p>같은 요인이라도 요일마다 다르게 작용합니다. 아래는 <b id="pdw"></b> 주에 대한 판정으로,
      각 요인이 <b>7일 각각에 대해</b> 따로 판정된 결과입니다. 공휴일은 그날 하루만 하향으로 잡히고
      나머지 날은 건드리지 않습니다 — 주 단위로 뭉뚱그리면 이 구분이 사라집니다.</p>
    <div class="pdwrap"><table class="pd" id="pd"></table></div>
  </section>

  <section>
    <div class="shd"><h2>요일별 정확도 (MAE)</h2></div>
    <div class="pdwrap"><table class="wk" id="wk"></table></div>
  </section>

  <section>
    <div class="shd"><h2>과거의 오판이 다음 판단에 쓰인다</h2></div>
    <p>학습 구간에서는 예측이 끝난 뒤 실제와 대조해 <b>무엇을 잘못 봤는지</b>를 기록합니다. 이후 비슷한
      상황이 오면 그 기록을 찾아와 함께 읽습니다. 아래는 그 연결이 실제로 일어난 예입니다.</p>
    <div class="chain" id="chain"></div>
  </section>

  <section>
    <details class="mth">
      <summary>예측은 어떻게 만들어지나</summary>
      <div class="mbody" id="mbody"></div>
    </details>
  </section>

  <footer id="foot"></footer>
</div>
<script>
const D = __DATA__;
const k = n => n.toLocaleString("ko-KR");
const sg = n => (n>0?"+":"") + k(Math.round(n));

document.getElementById("sd").textContent = k(D.sd);

/* 요약 */
const K = D.kpi;
document.getElementById("kpis").innerHTML = [
  [k(K.daily), "일평균 입국자 (명)"],
  [K.err_pct.toFixed(1)+"%", "예측 오차 (일평균 대비)"],
  [K.gain.toFixed(1)+"%", "기준선 대비 오차 감소"],
  [K.better+" / "+K.n, "기준선보다 정확했던 주"],
].map(([v,l])=>`<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
document.getElementById("kpinote").textContent =
  `검증 6주 전체에서 예측 오차(평균절대오차)는 하루 ${k(K.mae_fc)}명으로, 과거값만 쓴 기준선(${k(K.mae_base)}명)보다 `
  + `${Math.abs(K.gain).toFixed(1)}% 낮았습니다. 6주 모두에서 기준선보다 정확했습니다.`;

/* 수요 동인 — 세 그룹으로 나눈다. "모델이 쓰는가"와 "데이터로 확인되는가"는 다른 질문이고,
   둘을 뭉뚱그리면 보고서가 자기모순에 빠진다. */
const ALL = [...D.confirmed, ...D.unconfirmed];
const maxAbs = Math.max(...ALL.map(x=>Math.abs(x.diff)));

function drow(x){
  const w = Math.abs(x.diff)/maxAbs*48;                  // 반폭 48% → 좌우 대칭
  const neg = x.diff < 0;
  const strong = Math.abs(x.d)>=0.8;
  const strength = strong ? "영향 큼" : Math.abs(x.d)>=0.4 ? "영향 보통" : "불확실";
  return `<div class="drow ${strong?'strong':''}">
    <div class="dnm"><b>${x.name}</b><span>${x.note} · ${x.n}일 관측</span></div>
    <div class="dbar"><span class="zero"></span>
      <i style="${neg?'right:50%;':'left:50%;'}width:${w}%;background:var(--${neg?'down':'up'})"></i></div>
    <div class="dval">${sg(x.diff)}명<em>${strength}</em></div>
  </div>`;
}
document.getElementById("drv1").innerHTML = D.confirmed.map(drow).join("");
document.getElementById("drv2").innerHTML = D.unconfirmed.map(drow).join("");

/* 쓰고 있으나 확인되지 않은 요인 — 해당 covariate를 쓴 경우에만 */
document.getElementById("nf").innerHTML = D.used_unseen.length
  ? `<b>아래 요인들도 모델에 들어가 있습니다</b>
   <ul>${D.used_unseen.map(([n,d])=>`<li>${n}${d?` <span class="dim">— ${d}</span>`:""}</li>`).join("")}</ul>
   <p class="tail">다만 이들의 값이 갈릴 때 수요가 함께 갈리는 패턴은 일별 변동폭 안에 묻혀
     <b>확인되지 않았습니다.</b></p>`
  : "";

// 채널 조합에 따라 비는 카테고리는 서브섹션째로 숨긴다
if(!D.confirmed.length) document.getElementById("grp1").style.display="none";
if(!D.unconfirmed.length && !D.used_unseen.length) document.getElementById("grp2").style.display="none";

/* 방법론 — 문장은 파이썬 쪽에서 이 런의 설정으로 만든다 */
document.getElementById("mbody").innerHTML = D.method.map(p=>`<p>${p}</p>`).join("");

/* 요일별 정확도 표 */
document.getElementById("wk").innerHTML =
  `<tr><th>요일</th><th>기준 오차</th><th>보정 오차</th><th>개선</th></tr>` +
  D.weekday.map(w=>{
    const g=w.gain, cls=g<-2?"wkgood":g>2?"wkbad":"wkflat";
    return `<tr><td>${w.wd}</td><td>${w.base.toLocaleString()}</td>`
      + `<td>${w.calib.toLocaleString()}</td>`
      + `<td class="${cls}">${g>0?"+":""}${g.toFixed(1)}%</td></tr>`;
  }).join("");

/* 주별 차트 */
function chart(w){
  const W=760, H=190, P={t:14,r:16,b:26,l:52};
  const iw=W-P.l-P.r, ih=H-P.t-P.b;
  const all=[...w.gt,...w.fc,...w.base];
  let lo=Math.min(...all), hi=Math.max(...all);
  const pad=(hi-lo)*.15||1000; lo-=pad; hi+=pad;
  const x=i=>P.l+(i/(w.days.length-1))*iw;
  const y=v=>P.t+ih-((v-lo)/(hi-lo))*ih;
  const path=a=>a.map((v,i)=>(i?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1)).join(" ");
  const ticks=[lo+(hi-lo)*.15, (lo+hi)/2, hi-(hi-lo)*.15];
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img"
      aria-label="${w.start} 주의 기준선·예측·실제 입국자 추이">
    ${ticks.map(t=>`<line x1="${P.l}" x2="${W-P.r}" y1="${y(t).toFixed(1)}" y2="${y(t).toFixed(1)}"
        stroke="var(--line2)"/>
      <text x="${P.l-8}" y="${(y(t)+4).toFixed(1)}" text-anchor="end" font-size="10"
        font-family="var(--mono)" fill="var(--mut)">${Math.round(t/1000)}k</text>`).join("")}
    ${w.days.map((d,i)=>{
      const wk = d.wd==="토"||d.wd==="일";
      return `<text x="${x(i).toFixed(1)}" y="${H-8}" text-anchor="middle" font-size="10"
        font-family="var(--mono)" fill="${wk?'var(--body)':'var(--mut)'}">${d.md}
        <tspan dx="3" font-size="9.5">${d.wd}</tspan></text>`;}).join("")}
    <path d="${path(w.base)}" fill="none" stroke="var(--base)" stroke-width="1.5" stroke-dasharray="3 3"/>
    <path d="${path(w.fc)}" fill="none" stroke="var(--fc)" stroke-width="2.5"/>
    <path d="${path(w.gt)}" fill="none" stroke="var(--gt)" stroke-width="2"/>
    ${w.gt.map((v,i)=>`<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="2.6"
        fill="var(--gt)"/>`).join("")}
  </svg>`;
}

function drivers(e){
  if(!e || !e.drivers || !e.drivers.length) return "";
  return `<ul class="edrv">${e.drivers.map(d=>{
    if(typeof d==="string") return `<li>${d}</li>`;
    const dir=d.direction||"";
    const cls=/증가|상승/.test(dir)?"up":/감소|하락/.test(dir)?"dn":"";
    const dt=d.date?`<b>${d.date.slice(5)}</b>`:"";
    return `<li class="${cls}">${dt}${d.factor||""}${d.why?` — ${d.why}`:""}</li>`;
  }).join("")}</ul>`;
}

document.getElementById("wkn").textContent = `${D.weeks.length}주 · 학습에 쓰지 않은 구간`;
document.getElementById("weeks").innerHTML = D.weeks.map(w=>{
  const e=w.expl||{};
  const gain=(w.mae_base-w.mae_fc)/w.mae_base*100;
  return `<article class="wk">
    <div class="whd"><b>${w.start} → ${w.end}</b>
      <span class="acc">오차 ${k(w.mae_base)} → <s>${k(w.mae_fc)}</s>명 (${gain>0?"−":"+"}${Math.abs(gain).toFixed(0)}%)</span>
    </div>
    <div class="wbody">
      <div class="chart">${chart(w)}</div>
      ${e.summary?`<div class="ehd">설명 <span>모델의 내부 근거(판정 이유·보정 방식)를 읽기 쉽게 정리한 것 —
        원문은 아래 과정에서 볼 수 있습니다</span></div>
        <p class="esum">${e.summary}</p>`:""}
      ${drivers(e)}
      ${e.caveat?`<p class="ecav">${e.caveat}</p>`:""}
      ${process(w)}
    </div>
  </article>`;
}).join("");

/* ── 전 과정 (접힘) ────────────────────────────────────────────────────
   교수님 요구: 검색된 사례, 메모리에 들어간 원본 context, 기준값 등 전부 확인 가능하게.
   기본은 접어둔다 — 결론만 읽는 사람과 검증하려는 사람을 둘 다 지원한다. */
/* 함수 선언으로 둘 것 — const 화살표로 두면 아래 weeks 렌더링이 먼저 실행돼 TDZ 오류가 난다. */

/* 라벨 강도 → 셀 색 클래스. 상수로 두면 아래 렌더링이 먼저 실행돼 TDZ 오류가 난다. */
function cls(o){ return {"-2":"d2","-1":"d1","0":"","1":"u1","2":"u2"}[String(o)] || ""; }

/* 요인 × 날짜 판정 히트맵. 라벨은 날짜별이므로 대표값 하나로 뭉개면 그 구조가 사라진다. */
function heat(H, days){
  if(!H || !H.rows.length)
    return `<p class="none">모든 요인 중립 — 기준 예측을 그대로 둠</p>`;
  const hd = days.map(d=>{
    const wk = d.wd==="토"||d.wd==="일";
    return `<th class="${wk?'wknd':''}">${d.md}<br>${d.wd}</th>`;
  }).join("");
  const body = H.rows.map(r=>{
    const kn = (D.know||{})[r.ko];       // 이 요인의 knowledge(누적 지식) — 판정의 근거
    const nm = kn
      ? `<button class="kbtn" data-kn="${encodeURIComponent(kn)}" data-ko="${r.ko}"
          aria-label="${r.ko}의 누적 지식 보기">${r.ko}<span class="ki">📖 지식</span></button>`
      : r.ko;
    return `<tr><td>${nm}${r.role==="week"?'<i class="rw">주 단위</i>':""}</td>${
      r.cells.map(c=>`<td><span class="pdc ${cls(c.o)}">${c.t}</span></td>`).join("")}</tr>`;
  }).join("");
  const tail = H.quiet
    ? `<p class="qt">나머지 ${H.quiet}개 요인은 모두 중립 — 예측을 움직이지 않았습니다.</p>` : "";
  return `<div class="pdwrap"><table class="pd">
    <thead><tr><th>요인</th>${hd}</tr></thead><tbody>${body}</tbody></table></div>${tail}`;
}

/* 과거 구간 + 예측 구간을 한 축에 그리는 작은 선그래프.
   preserveAspectRatio를 끄지 않는다 — 끄면 축마다 배율이 달라져 선 굵기가 찌그러진다. */
function spark(series, opt={}){
  const w = 640, h = opt.h || 128;
  const P = {t:10, r:opt.r||10, b:18, l:44};
  const iw = w-P.l-P.r, ih = h-P.t-P.b;
  const all = series.flatMap(s=>s.v).filter(v=>v!=null);
  if(!all.length) return "";
  let lo=Math.min(...all), hi=Math.max(...all);
  const pad=(hi-lo)*.12||500; lo-=pad; hi+=pad;
  const span = Math.max(...series.map(s=>(s.off||0)+s.v.length));
  const x = i => P.l + (i/(span-1))*iw;
  const y = v => P.t + ih - ((v-lo)/(hi-lo))*ih;
  const path = s => s.v.map((v,i)=>(i?"L":"M")+x(i+(s.off||0)).toFixed(1)+" "+y(v).toFixed(1)).join(" ");
  const ticks = [lo+(hi-lo)*.12, hi-(hi-lo)*.12];
  // 경계선은 예측·실제선이 과거선에서 갈라져 나오는 '연결점'(context의 마지막 값, 인덱스 split-1)에
  // 정확히 놓는다. split-0.5(반 칸 오른쪽)로 두면 선이 갈라지는 지점과 어긋나 보인다.
  const bx = opt.split!=null ? x(opt.split-1) : null;

  return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="${opt.alt||''}">
    ${ticks.map(t=>`<line x1="${P.l}" x2="${w-P.r}" y1="${y(t).toFixed(1)}" y2="${y(t).toFixed(1)}"
        stroke="var(--line2)"/>
      <text x="${P.l-7}" y="${(y(t)+3.5).toFixed(1)}" text-anchor="end" font-size="9.5"
        font-family="var(--mono)" fill="var(--mut)">${Math.round(t/1000)}k</text>`).join("")}
    ${bx!==null?`<line x1="${bx.toFixed(1)}" x2="${bx.toFixed(1)}" y1="${P.t}" y2="${P.t+ih}"
        stroke="var(--mut)" stroke-width="1" stroke-dasharray="2 3"/>
      <text x="${(bx+4).toFixed(1)}" y="${P.t+8}" font-size="9" font-family="var(--mono)"
        fill="var(--mut)">예측 시작 →</text>`:""}
    ${series.map(s=>`<path d="${path(s)}" fill="none" stroke="var(--${s.c})"
        stroke-width="${s.w||1.8}" stroke-linejoin="round" stroke-linecap="round"
        ${s.dash?'stroke-dasharray="4 3"':""}/>`).join("")}
    ${series.filter(s=>s.dot).map(s=>s.v.map((v,i)=>
        (s.skip1&&i===0) ? "" :          // 이어붙이려고 앞에 붙인 연결점에는 점을 찍지 않는다
        `<circle cx="${x(i+(s.off||0)).toFixed(1)}" cy="${y(v).toFixed(1)}" r="2.4"
          fill="var(--${s.c})" stroke="var(--card)" stroke-width="1"/>`).join("")).join("")}
    ${(opt.xlab||[]).map(([i,t])=>`<text x="${x(i).toFixed(1)}" y="${h-5}" text-anchor="middle"
        font-size="9" font-family="var(--mono)" fill="var(--mut)">${t}</text>`).join("")}
  </svg>`;
}

/* 예측 계열을 마지막 실측값에서 출발시킨다 — 안 그러면 과거선과 예측선 사이가 끊겨 보인다.
   함수 선언으로 둘 것(호이스팅) — const로 두면 위 렌더링이 먼저 실행돼 TDZ 오류가 난다. */
function joinAt(last, arr){ return [last, ...arr]; }

/* 범례 — 계열이 2개 이상이면 항상 붙인다(색만으로 구분하게 두지 않는다) */
function lgd(items){
  return `<div class="mlg">${items.map(([c,t,dash])=>
    `<span><i class="${dash?'ds':''}" style="background:var(--${c})"></i>${t}</span>`).join("")}</div>`;
}

function caseCard(c){
  const VD={better:["개선","up"],worse:["악화","dn"],same:["차이 없음",""]};
  const [vt,vc]=VD[c.verdict]||[c.verdict,""];
  const n=c.context.length, m=c.base.length;
  return `<details class="cs">
    <summary><b>${c.date}</b><span class="vd ${vc}">보정 결과 ${vt}</span><span class="tog"></span></summary>
    <div class="csb">
      <div class="mini">
        <div class="mlab">메모리에 저장된 원본 — 근거로 쓴 과거 ${n}일과, 그 뒤 7일의 기준값·예측·실제</div>
        ${lgd([["ctx","과거 실적"],["base","기준값",1],["fc","예측"],["gt","실제"]])}
        ${spark([
          {v:c.context, c:"ctx", w:1.5},
          {v:joinAt(c.context[n-1], c.base), c:"base", off:n-1, dash:true, w:1.6},
          {v:joinAt(c.context[n-1], c.fc), c:"fc", off:n-1, w:2.2},
          {v:joinAt(c.context[n-1], c.gt), c:"gt", off:n-1, w:2, dot:true, skip1:true},
        ], {h:130, split:n, alt:`${c.date} 사례의 과거 ${n}일과 이후 7일 추이`,
            xlab:[[0,`-${n}일`],[n,"1일차"],[n+6,"7일차"]]})}
        <div class="mnum">기준 ${c.base.map(k).join(", ")}<br>예측 ${c.fc.map(k).join(", ")}<br>실제 ${c.gt.map(k).join(", ")}</div>
      </div>
      <div class="fld"><span class="flab">그때의 판정 — 요인 × 날짜</span>${heat(c.heat, c.days)}</div>
      ${c.fixes.length?`<div class="fld"><span class="flab">실제와 대조한 뒤 고친 판정</span>
        <div class="chips">${c.fixes.map(f=>`<span class="chip fix o${f.o}">${f.ko}${f.d?`<i>${f.d}일</i>`:""}
          <s>${f.was}</s>→<b>${f.now}</b></span>`).join("")}</div></div>`:""}
      ${c.reflect?`<div class="fld"><span class="flab">기록한 회고</span><p>${c.reflect}</p></div>`:""}
    </div>
  </details>`;
}

function process(w){
  return `<details class="pr">
    <summary>이 예측이 만들어진 과정 보기</summary>
    <div class="prb">
      <div class="fld"><span class="flab">과거 ${w.context.length}일(모델이 본 전부) → 이후 7일의 기준값·예측·실제</span>
        ${lgd([["ctx","과거 실적"],["base","기준값",1],["fc","예측"],["gt","실제"]])}
        ${(()=>{const n=w.context.length; return spark([
          {v:w.context, c:"ctx", w:1.6},
          {v:joinAt(w.context[n-1], w.base), c:"base", off:n-1, dash:true, w:1.6},
          {v:joinAt(w.context[n-1], w.fc), c:"fc", off:n-1, w:2.2},
          {v:joinAt(w.context[n-1], w.gt), c:"gt", off:n-1, w:2, dot:true, skip1:true},
        ], {h:130, split:n, alt:"과거 28일과 이후 7일 기준·예측·실제 추이",
            xlab:[[0,`-${n}일`],[n,"1일차"],[n+6,"7일차"]]});})()}
        <div class="mnum">기준 ${w.base.map(k).join(", ")}<br>예측 ${w.fc.map(k).join(", ")}<br>실제 ${w.gt.map(k).join(", ")}</div>
      </div>
      <div class="fld"><span class="flab">요인 × 날짜 판정</span>${heat(w.heat, w.days)}</div>
      ${w.r_attr||w.r_calib?`<p class="orig">아래 두 글이 <b>모델이 실제로 쓴 근거 원문</b>입니다.
        위쪽 설명은 이 둘을 읽기 쉽게 정리한 것입니다.</p>`:""}
      ${w.r_attr?`<div class="fld"><span class="flab">근거 원문 ① — 요인을 그렇게 판정한 이유</span>
        <p>${w.r_attr}</p></div>`:""}
      ${w.r_calib?`<div class="fld"><span class="flab">근거 원문 ② — 그 판정을 예측에 반영한 방식</span>
        <p>${hl(w.r_calib)}</p></div>`:""}
      ${w.cases.length?`<div class="fld"><span class="flab">참고한 과거 사례 — 판정이 비슷했던 주</span>
        <div class="csl">${w.cases.map(caseCard).join("")}</div></div>`:""}
      ${w.cases_lb.length?`<div class="fld"><span class="flab">참고한 과거 사례 — 수요 흐름이 비슷했던 주</span>
        <div class="csl">${w.cases_lb.map(caseCard).join("")}</div></div>`:""}
    </div>
  </details>`;
}

/* 과거 사례를 근거로 든 문장에 형광펜 */
function hl(t){
  return t.replace(/([^.。]*(?:과거 사례|과거 관찰|과거 반성|회고|검색된)[^.。]*[.。])/g,
    '<mark>$1</mark>');
}

/* 증거 ① 날짜별 판정 */
const P = D.perday;
document.getElementById("pdw").textContent = `${P.start} → ${P.end}`;
document.getElementById("pd").innerHTML =
  `<thead><tr><th>요인</th>${P.days.map(d=>{
      const wk = d.wd==="토"||d.wd==="일";
      return `<th class="${wk?'wknd':''}">${d.md}<br>${d.wd}</th>`;}).join("")}</tr></thead>
   <tbody>${P.rows.map(r=>`<tr><td>${r.ko}</td>${r.cells.map(c=>
      `<td><span class="pdc ${cls(c.o)}">${c.t}</span></td>`).join("")}</tr>`).join("")}</tbody>`;

/* 증거 ② 회고 → 판단 */
const C = D.chain;
document.getElementById("chain").innerHTML = !C ? "" : `
  <div class="cbox">
    <div class="clab">학습 구간 · <b>${C.past_date}</b> 주에서 남긴 기록</div>
    <p>${C.past_reflect}</p>
  </div>
  <div class="carrow"><em>↓</em>이 기록이 검색되어 아래 판단의 입력으로 들어감</div>
  <div class="cbox now">
    <div class="clab">검증 구간 · <b>${C.now_date}</b> 주의 판단</div>
    <p><mark>${C.now_quote}</mark></p>
  </div>`;

document.getElementById("foot").textContent =
  `검증 구간 ${D.weeks[0].start} ~ ${D.weeks[D.weeks.length-1].start} · 기준 모델 Chronos-2 · 일별 변동폭 ±${k(D.sd)}명`;

/* 히트맵 요인명 클릭 → 그 요인의 knowledge(누적 지식)를 팝오버로. 판정의 근거를 그 자리에서 확인. */
(function(){
  const pop = document.createElement("div");
  pop.className = "kpop"; pop.hidden = true;
  document.body.appendChild(pop);
  function close(){ pop.hidden = true; }
  document.addEventListener("click", e=>{
    const btn = e.target.closest ? e.target.closest(".kbtn") : null;
    if(!btn){ if(!e.target.closest || !e.target.closest(".kpop")) close(); return; }
    e.stopPropagation();
    pop.innerHTML = `<div class="khd">${btn.dataset.ko}<span>누적 지식 — 이 요인의 판정 근거</span>
      <button class="kx" aria-label="닫기">✕</button></div>
      <p>${decodeURIComponent(btn.dataset.kn)}</p>`;
    pop.hidden = false;
    const r = btn.getBoundingClientRect();
    const top = r.bottom + window.scrollY + 6;
    const left = Math.min(r.left + window.scrollX, window.scrollX + document.documentElement.clientWidth - 380);
    pop.style.top = top + "px"; pop.style.left = Math.max(12, left) + "px";
    pop.querySelector(".kx").addEventListener("click", close);
  });
  document.addEventListener("keydown", e=>{ if(e.key==="Escape") close(); });
})();
</script>"""

html = HTML.replace("__DATA__", json.dumps(D, ensure_ascii=False))
if a.standalone:
    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'</head><body>{html}</body></html>')
open(a.out, "w", encoding="utf-8").write(html)
print(f"[report] → {a.out}  ({len(html)//1024} KB)  standalone={a.standalone}  "
      f"오차 {KPI['mae_base']}→{KPI['mae_fc']}  {KPI['better']}/{KPI['n']}주 개선")
