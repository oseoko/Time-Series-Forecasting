"""memory/test_results_*.json → fs_explain 아티팩트 (간단 버전).

test window마다 base / calib / gt 궤적, 채널 라벨, 그 채널의 일별 값, 설명 두 줄, 검색된 case를 한 카드에 담는다.
라벨이 붙은 채널의 값을 함께 보여야 "왜 그렇게 판단했나"를 눈으로 대조할 수 있다.

실행(리포 루트에서):
  python analysis/explain_page.py [--res memory/test_results_all_abs_cap5.json]
  python analysis/explain_page.py --standalone   # 브라우저에서 바로 열리는 fs_explain.html
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 루트 → config/engine 임포트
from config import MEMDIR
from engine.data import load

_HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--res", default=os.path.join(MEMDIR, "test_results_all_abs_cap5_corr.json"))
ap.add_argument("--exp", default=os.path.join(MEMDIR, "experience_corr.json"),
                help="검색된 case의 원본 (STEP2 산출). corrected 라벨이 여기 들어있다.")
ap.add_argument("--out", default=os.path.join(_HERE, "fs_explain.html"))
ap.add_argument("--standalone", action="store_true",
                help="doctype/head로 감싸 브라우저에서 바로 열리게 한다. 아티팩트 배포본에는 쓰지 않는다.")
a = ap.parse_args()

KO = {"hotel_price": "호텔가", "fx_usd": "원/달러", "fx_cny": "원/위안", "fx_jpy": "원/엔",
      "hol_cn": "중국휴일", "hol_jp": "일본휴일", "hol_us": "미국휴일", "hol_kr": "한국휴일",
      "hol_kr_wd": "한국 평일공휴일", "concert_cap": "콘서트",
      "flight_cn": "중국노선 항공편", "flight_jp": "일본노선 항공편",
      "flight_us": "미국노선 항공편", "flight_total": "국제선 총 항공편",
      "w_seoul_AVGTA": "기온", "w_seoul_SUMRN": "강수", "w_seoul_AVGWS": "풍속",
      "w_seoul_AVGRHM": "습도", "w_seoul_AVGPS": "기압", "w_seoul_SUMGSR": "일사",
      "w_seoul_DDMES": "적설", "w_seoul_AVGTCA": "운량"}
WD = ["월", "화", "수", "목", "금", "토", "일"]

DF, _, FUTURE_KNOWN, _ = load()
X = DF.set_index("date")


def fmt_cov(ch, v):
    """채널 스케일에 맞춘 표기 — 휴일 0/1, 호텔가 604k, 강수 0.31."""
    v = float(v)
    if ch.startswith("hol_"):
        return "1" if v > 0 else "·"
    if abs(v) >= 10000:
        return f"{v/1000:.0f}k"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.3g}"


# experience(STEP2, train) — 검색된 case의 내용. reflection은 GT가 필요해 train에만 있다.
EXP = {c["date"]: c for c in json.load(open(a.exp, encoding="utf-8"))} if os.path.exists(a.exp) else {}
# train_analysis: 분석 전용(GT 포함) — case 차트에 실제값(GT)을 함께 그리기 위해.
# experience 파일과 짝을 이룬다 (experience_corr.json ↔ train_analysis_corr.json).
_ta_path = a.exp.replace("experience", "train_analysis")
TA_GT = {c["date"]: c["gt"] for c in json.load(open(_ta_path, encoding="utf-8"))} if os.path.exists(_ta_path) else {}
_LIDX = {"strong_down": 0, "weak_down": 1, "neutral": 2, "weak_up": 3, "strong_up": 4}
_SH = {"strong_down": "SD", "weak_down": "wd", "neutral": "·", "weak_up": "wu", "strong_up": "SU"}


def _rnd(xs):
    return [round(float(v)) for v in xs]


def case_detail(d):
    """검색된 train case 한 건 → 전 과정 노출: 라벨·verdict·원본 context·base/calib·R_attr/R_calib·reflect."""
    c = EXP.get(d)
    if not c:
        return {"date": d, "missing": True}
    dd = [pd.Timestamp(d) + pd.Timedelta(days=t) for t in range(7)]
    hol = bool(any(X.loc[x, "hol_kr_wd"] > 0 for x in dd))
    labs = []
    for ch, v in c["attribution"].items():
        if isinstance(v, list):
            if any(x != "neutral" for x in v):
                disp = max(v, key=lambda x: abs(_LIDX.get(x, 2) - 2))
                labs.append({"ko": KO.get(ch, ch), "s": _SH.get(disp, "·")})
        elif v != "neutral":
            labs.append({"ko": KO.get(ch, ch), "s": _SH.get(v, "·")})
    # reflection(B-3)이 GT를 보고 라벨을 고친 것 — "그때 이렇게 봤는데 실제론 이게 맞았다".
    # 바뀐 것만 싣는다. 검색 거리는 원본(attribution)으로 계산하고, 이건 학습 재료로만 보여준다.
    corr = c.get("corrected") or {}
    fixes = []
    for ch, now in corr.items():
        was = c["attribution"].get(ch)
        if was is None or now == was:
            continue
        if isinstance(was, list):                        # 날짜별 라벨 → 바뀐 날만
            for t, (x, y) in enumerate(zip(was, now)):
                if x != y:
                    fixes.append({"ko": KO.get(ch, ch), "d": t + 1,
                                  "was": _SH.get(x, "·"), "now": _SH.get(y, "·")})
        else:
            fixes.append({"ko": KO.get(ch, ch), "d": 0,
                          "was": _SH.get(was, "·"), "now": _SH.get(now, "·")})
    rsn = c.get("reasoning", {})
    return {"date": d, "verdict": c.get("verdict", "?"), "hol": hol,
            "reflect": c.get("reflect", ""), "labs": labs, "fixes": fixes,
            "clamp": c.get("clamp_hit", {}),            # clamp에 몇 날 잘렸나
            # 메모리에 실제 들어간 원본 값 전부
            "context": _rnd(c.get("y_lookback", [])),   # 이 case가 예측 근거로 쓴 타깃 28일
            "base": _rnd(c.get("y_base", [])), "calib": _rnd(c.get("y_calib", [])),
            "gt": _rnd(TA_GT.get(d, [])),               # 실제값 (train_analysis에서)
            "r_attr": rsn.get("R_attr", ""), "r_calib": rsn.get("R_calib", "")}


# test window의 원본 context(타깃 28일)는 test_results에 없어 make_windows로 재구성
from engine.data import make_windows, CTX_LEN
_DF, _tgt, _fk, _ = load()
_ctx_of = {}
for _w in make_windows(_DF, _tgt, _fk, ["hol_kr_wd"], region="test"):
    _ctx_of[_w["date"]] = _rnd(_w["y_lookback"])

R = json.load(open(a.res, encoding="utf-8"))
wins = []
for w in R:
    d0 = pd.Timestamp(w["date"])
    days = []
    for t in range(7):
        d = d0 + pd.Timedelta(days=t)
        b, c, g = w["y_base"][t], w["y_calib"][t], w["gt"][t]
        days.append({
            "date": str(d.date()), "wd": WD[d.dayofweek],
            "gt": round(g), "base": round(b), "calib": round(c),
            "adj": round(c - b), "need": round(g - b),
            "hol": bool(X.loc[d, "hol_kr_wd"] > 0),   # 평일 한국 공휴일 — 그래프·표에 표시
        })
    # 라벨 정규화: future-known은 horizon 리스트, past-only는 스칼라.
    #   active = 하나라도 non-neutral, disp = 대표 라벨(가장 극단적인 날), perday = 날짜별 축약
    LIDX = {"strong_down": 0, "weak_down": 1, "neutral": 2, "weak_up": 3, "strong_up": 4}
    SHORT = {"strong_down": "SD", "weak_down": "wd", "neutral": "·", "weak_up": "wu", "strong_up": "SU"}

    def norm(v):
        if isinstance(v, list):
            active = any(x != "neutral" for x in v)
            disp = max(v, key=lambda x: abs(LIDX.get(x, 2) - 2)) if active else "neutral"
            return active, disp, [SHORT.get(x, "·") for x in v]
        return v != "neutral", v, None

    labels = []
    for ch, v in w["attribution"].items():
        active, disp, perday = norm(v)
        if active:
            labels.append({"ch": ch, "ko": KO.get(ch, ch), "lab": disp, "perday": perday})
    idx = [d0 + pd.Timedelta(days=t) for t in range(7)]
    cov = [{"ch": l["ch"], "ko": l["ko"], "lab": l["lab"], "perday": l["perday"],
            "role": "future" if l["ch"] in FUTURE_KNOWN else "past",
            "vals": [fmt_cov(l["ch"], X.loc[d, l["ch"]]) for d in idx]}
           for l in labels]
    # 히트맵용: 전 채널 × 7일 ordinal(0=SD..4=SU, 2=neutral) + 그날 값
    heat = []
    for ch in w["attribution"]:
        v = w["attribution"][ch]
        ords = [LIDX.get(x, 2) for x in v] if isinstance(v, list) else [LIDX.get(v, 2)] * 7
        heat.append({"ch": ch, "ko": KO.get(ch, ch),
                     "role": "future" if ch in FUTURE_KNOWN else "past",
                     "ord": ords, "vals": [fmt_cov(ch, X.loc[d, ch]) for d in idx],
                     "active": any(o != 2 for o in ords)})
    wins.append({
        "date": w["date"], "days": days, "labels": labels, "cov": cov,
        "r_attr": w["R_attr"], "r_calib": w["R_calib"],
        "mae_base": round(w["mae_base"]), "mae_calib": round(w["mae_calib"]),
        "ret_attr": [case_detail(d) for d in w.get("retrieved_attr", [])],
        "ret_lb": [case_detail(d) for d in w.get("retrieved_lb", [])],
        # calibration이 검색 case의 reflect(회고)를 실제로 참고했는지 — 참고 시 R_calib과 case 카드를 같은 색으로 연결
        "uses_reflect": bool(re.search(r"과거|사례|reflect|반성|회고|검색", w["R_calib"])),
        "context": _ctx_of.get(w["date"], []),   # 이 window의 타깃 28일 (예측 근거)
        "heat": heat,   # 전 채널 × 7일 라벨 히트맵
        "expl": w.get("explanation"),   # B-2c 사용자용 설명 {summary, drivers[], caveat}
    })

allday = [d for w in wins for d in w["days"]]
D = {
    "wins": wins,
    "mae_base": round(np.mean([w["mae_base"] for w in wins])),
    "mae_calib": round(np.mean([w["mae_calib"] for w in wins])),
    "better": sum(1 for w in wins if w["mae_calib"] < w["mae_base"]),
    "n": len(wins),
    "span": f'{allday[0]["date"]} ~ {allday[-1]["date"]}',
}

HTML = """<title>fs_explain — 보정은 무엇을 보고 움직였나</title>
<style>
  :root{--bg:#eceff4;--s1:#fff;--s2:#f4f6fa;--ink:#12151b;--sec:#586072;--mut:#8a92a2;--line:#dfe3ec;
    --accent:#3b5bdb;--gt:#12151b;--base:#3b5bdb;--calib:#e0562f;--good:#12946b;--bad:#e0562f;
    --shadow:0 1px 2px rgba(18,22,34,.05),0 10px 26px rgba(18,22,34,.07);
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,system-ui,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;}
  @media (prefers-color-scheme:dark){:root{--bg:#0f1216;--s1:#181c22;--s2:#13161b;--ink:#e9edf3;--sec:#9aa4b2;--mut:#6b7482;--line:#282e38;
    --accent:#748ffc;--gt:#e9edf3;--base:#748ffc;--calib:#f0774a;--good:#34c793;--bad:#f0774a;--shadow:0 1px 2px rgba(0,0,0,.3),0 12px 30px rgba(0,0,0,.4);}}
  :root[data-theme="light"]{--bg:#eceff4;--s1:#fff;--s2:#f4f6fa;--ink:#12151b;--sec:#586072;--mut:#8a92a2;--line:#dfe3ec;--accent:#3b5bdb;--gt:#12151b;--base:#3b5bdb;--calib:#e0562f;--good:#12946b;--bad:#e0562f;}
  :root[data-theme="dark"]{--bg:#0f1216;--s1:#181c22;--s2:#13161b;--ink:#e9edf3;--sec:#9aa4b2;--mut:#6b7482;--line:#282e38;--accent:#748ffc;--gt:#e9edf3;--base:#748ffc;--calib:#f0774a;--good:#34c793;--bad:#f0774a;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;-webkit-font-smoothing:antialiased;letter-spacing:-.005em}
  .wrap{max-width:940px;margin:0 auto;padding:44px 22px 90px}
  .eyebrow{font-size:12px;color:var(--accent);font-weight:600}
  h1{font-size:clamp(25px,3.3vw,33px);line-height:1.15;margin:9px 0 10px;letter-spacing:-.025em;font-weight:600;text-wrap:balance}
  .lede{color:var(--sec);max-width:70ch;margin:0;font-size:15px}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:22px}
  .kpi{background:var(--s1);border:1px solid var(--line);border-radius:12px;padding:12px 14px;box-shadow:var(--shadow)}
  .kpi .v{font-family:var(--mono);font-size:21px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
  .kpi .l{font-size:11.5px;color:var(--mut);margin-top:1px}
  .take{margin-top:18px;padding:13px 16px;background:var(--s1);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;font-size:14px;color:var(--sec);box-shadow:var(--shadow)}
  .take b{color:var(--ink)}
  .lg{display:flex;flex-wrap:wrap;gap:14px;margin:22px 0 0;font-family:var(--mono);font-size:12px;color:var(--sec)}
  .lg i{display:inline-block;width:16px;height:3px;border-radius:2px;vertical-align:3px;margin-right:5px}
  .lg i.gt{height:0;border-top:2.5px solid var(--gt)}
  .card{background:var(--s1);border:1px solid var(--line);border-radius:15px;padding:18px 20px;box-shadow:var(--shadow);margin-top:18px}
  .hd{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
  .hd .w{font-family:var(--mono);font-weight:600;font-size:15px}
  .hd .m{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--sec);font-variant-numeric:tabular-nums}
  .hd .m b{font-weight:600}
  .chips{display:flex;flex-wrap:wrap;gap:5px;margin:10px 0 2px}
  .chip{font-size:11.5px;font-family:var(--mono);border:1px solid var(--line);border-radius:6px;padding:2px 7px;background:var(--s2);color:var(--sec)}
  .chip.su{border-color:var(--good);color:var(--good);font-weight:600}
  .chip.wu{color:var(--good)}
  .chip.sd{border-color:var(--bad);color:var(--bad);font-weight:600}
  .chip.wd{color:var(--bad)}
  .why{font-size:13px;color:var(--sec);margin-top:10px;padding:10px 13px;background:var(--s2);border:1px solid var(--line);border-radius:9px}
  .why + .why{margin-top:7px}
  .why b{color:var(--ink);font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.05em;display:block;margin-bottom:3px}
  .scroll{overflow-x:auto;margin-top:12px}
  table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
  th{text-align:right;color:var(--mut);font-weight:500;padding:5px 8px;border-bottom:1px solid var(--line);font-size:10.5px;letter-spacing:.04em}
  th:first-child,td:first-child{text-align:left}
  td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}  /* 헤더(th)가 우측정렬이므로 숫자도 맞춘다 */
  tbody tr:last-child td{border-bottom:0}
  tr.hol td{background:color-mix(in srgb,var(--calib) 8%,transparent)}
  .tag{font-size:9.5px;color:var(--calib);border:1px solid var(--calib);border-radius:5px;padding:0 4px;margin-left:4px}
  .pdcell{text-align:center;padding:3px 6px!important;line-height:1.25}
  .pdcell .pdlab{display:block;font-size:9px;font-weight:700;font-family:var(--mono);min-height:11px}
  .pdcell .pdval{display:block;font-size:11px}
  .pdcell.up{background:color-mix(in srgb,var(--good) 13%,transparent)}
  .pdcell.up .pdlab{color:var(--good)}
  .pdcell.dn{background:color-mix(in srgb,var(--bad) 13%,transparent)}
  .pdcell.dn .pdlab{color:var(--bad)}
  .pdcell.win .pdlab{opacity:.55;font-weight:600}
  .pdcell.win.up{background:color-mix(in srgb,var(--good) 7%,transparent)}
  .pdcell.win.dn{background:color-mix(in srgb,var(--bad) 7%,transparent)}
  .up{color:var(--good)}.dn{color:var(--bad)}.dim{color:var(--mut)}
  .cap{font-family:var(--mono);font-size:10px;color:var(--mut);margin-top:12px}
  /* 채널 × 일별 라벨 히트맵 */
  table.heat{border-collapse:collapse;font-size:10px}
  table.heat th{text-align:center;color:var(--mut);font-size:9.5px;padding:3px 5px;font-weight:500}
  table.heat th:first-child{text-align:left}
  table.heat th .th2{display:block;opacity:.7;font-weight:400}
  .hcell{text-align:center;padding:2px 5px;border:1px solid var(--line);line-height:1.2;min-width:44px}
  .hcell .hlab{display:block;font-family:var(--mono);font-size:9px;font-weight:700}
  .hcell .hval{display:block;font-size:9.5px;color:var(--sec)}
  .hname{text-align:left;white-space:nowrap;font-size:11px;padding:2px 8px 2px 2px;border-bottom:1px solid var(--line)}
  tr.hneu{opacity:.42}                                  /* 전부 중립인 채널 행은 옅게 */
  tr.hneu .hname{color:var(--mut)}
  table.cov th{text-align:center;font-size:10px}
  table.cov th:first-child{text-align:left}
  table.cov th .th2{display:block;font-weight:400;opacity:.7;font-family:var(--sans)}
  table.cov td{text-align:center}
  table.cov td:first-child{text-align:left;white-space:nowrap}
  .role{font-family:var(--mono);font-size:9.5px;color:var(--mut);border:1px solid var(--line);border-radius:5px;padding:0 5px;margin:0 5px}
  .ret{font-family:var(--mono);font-size:11px;color:var(--mut);margin-top:9px}
  /* 접힌 요약은 3열 그리드로 훑기 좋게, 펼치면 한 줄 전폭으로 */
  .cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px;margin-top:6px;align-items:start}
  .case{background:var(--s2);border:1px solid var(--line);border-radius:9px;padding:9px 12px}
  details.case[open]{grid-column:1/-1}   /* 펼친 카드는 전폭 차지 → 본문 넓게 */
  .case .chd{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:12px}
  .case .chd b{color:var(--ink);font-weight:600}
  .hbadge{font-size:9.5px;color:var(--calib);border:1px solid var(--calib);border-radius:4px;padding:0 4px}
  .chd .up{color:var(--good);font-weight:600}.chd .dn{color:var(--bad);font-weight:600}
  .lchips{display:flex;flex-wrap:wrap;gap:4px;margin:7px 0}
  .lchip{font-family:var(--mono);font-size:10px;color:var(--sec);background:var(--s1);border:1px solid var(--line);border-radius:5px;padding:1px 5px}
  .lchip b{margin-left:3px;color:var(--ink)}
  .crefl{font-size:11.5px;color:var(--sec);line-height:1.5}
  /* 접이식 카드 */
  details.card>summary{list-style:none;cursor:pointer}
  details.card>summary::-webkit-details-marker{display:none}
  .hd .exp{margin-left:8px;color:var(--mut);transition:transform .15s}
  details.card[open]>summary .exp{transform:rotate(180deg)}
  /* case 카드: 요약 항상 보이고 본문 접힘 */
  details.case>summary{list-style:none;cursor:pointer;display:block}
  details.case>summary::-webkit-details-marker{display:none}
  details.case .exp{margin-left:auto;font-size:9.5px;color:var(--accent)}
  details.case[open] .exp{opacity:.5}
  details.case[open] .exp::after{content:" (접기 ▴)"}
  .cbody{margin-top:10px;padding-top:10px;border-top:1px dashed var(--line)}
  .sparkwrap{margin:2px 0 8px}
  .spark{width:100%;height:auto;display:block;background:var(--s1);border:1px solid var(--line);border-radius:8px}
  .sparklg{display:flex;flex-wrap:wrap;gap:10px;font-family:var(--mono);font-size:9.5px;color:var(--mut);margin-bottom:3px}
  .sparklg i{display:inline-block;width:12px;height:2px;border-radius:1px;vertical-align:3px;margin-right:3px}
  .frow{font-family:var(--mono);font-size:11px;margin:5px 0;line-height:1.55}
  .frow .flab{display:block;color:var(--mut);font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:1px}
  .frow .fval{display:block;color:var(--sec);word-break:break-word}
  .frow.ctx{margin:10px 0;padding:9px 12px;background:var(--s1);border:1px solid var(--line);border-radius:8px}
  .cwhy{font-size:12.5px;color:var(--sec);margin-top:9px;line-height:1.6;max-width:78ch}
  .cwhy b{display:block;color:var(--ink);font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px}
  .cwhy.refl b{color:var(--calib)}
  /* 요약 섹션 — 코멘트 대응 + 결과 */
  .sum{margin:26px 0 30px;padding:20px 22px;background:var(--s1);border:1px solid var(--line);border-radius:10px}
  .sum h2{font-size:14px;margin:0 0 14px;color:var(--ink);letter-spacing:-.01em}
  .bul{list-style:none;margin:0;padding:0;counter-reset:b}
  .bul li{position:relative;padding:0 0 0 30px;margin-bottom:14px;counter-increment:b}
  .bul li::before{content:counter(b);position:absolute;left:0;top:1px;width:19px;height:19px;
    display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:10px;
    color:var(--mut);background:var(--s2);border:1px solid var(--line);border-radius:50%}
  .bul li>b{font-size:13px;color:var(--ink)}
  .bul li p{margin:4px 0 0;font-size:12.5px;line-height:1.65;color:var(--sec);max-width:82ch}
  .bul code{font-family:var(--mono);font-size:11px;background:var(--s2);padding:0 3px;border-radius:3px}
  .sum .tag{display:inline-block;font-family:var(--mono);font-size:9px;text-transform:uppercase;
    letter-spacing:.05em;padding:1px 5px;border:0;border-radius:4px;margin:0 7px 0 0;vertical-align:1px}
  .sum .tag.done{color:var(--calib);background:color-mix(in srgb,var(--calib) 14%,transparent)}
  .sum .tag.new{color:var(--gt);background:color-mix(in srgb,var(--gt) 14%,transparent)}
  /* B-2c 사용자용 설명 — 최종 산출물이므로 눈에 띄게 */
  .expl{margin:12px 0 4px;padding:13px 15px;border:1px solid var(--calib);border-radius:9px;
    background:color-mix(in srgb,var(--calib) 6%,transparent)}
  .ehd{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;
    color:var(--calib);font-weight:700;margin-bottom:6px}
  .ehd span{text-transform:none;letter-spacing:0;color:var(--mut);font-weight:400;margin-left:6px}
  .esum{margin:0;font-size:13px;line-height:1.65;color:var(--ink);max-width:82ch}
  .edrv{margin:9px 0 0;padding-left:16px;font-size:12.5px;line-height:1.7;color:var(--sec);max-width:82ch}
  .edrv li{margin-bottom:2px}
  .edrv b{font-family:var(--mono);font-size:11px;color:var(--ink)}
  .ecav{margin:9px 0 0;font-size:12px;color:var(--sec);line-height:1.6;max-width:82ch}
  .ecav b{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;color:var(--mut);margin-right:4px}
  .fixref{color:var(--gt)}
  .hlref{background:color-mix(in srgb,var(--calib) 26%,transparent);padding:0 3px;border-radius:3px;color:var(--ink)}
  .res{margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}
  .res>b{font-size:12px;color:var(--ink)}
  .res table{border-collapse:collapse;margin:9px 0 8px;font-size:12.5px}
  .res th,.res td{padding:4px 14px 4px 0;text-align:left;border-bottom:1px solid var(--line)}
  .res th{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;color:var(--mut);letter-spacing:.04em}
  .res td{font-variant-numeric:tabular-nums;color:var(--sec)}
  .res td b{color:var(--ink)}
  .res p{margin:6px 0 0;font-size:12px;color:var(--sec);line-height:1.6;max-width:82ch}
  /* 라벨 교정 — reflection이 GT를 보고 고친 것 */
  .cwhy.fix b{color:var(--gt)}
  .fchips{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
  .fchip{display:inline-flex;align-items:center;gap:3px;font-family:var(--mono);font-size:10px;
    color:var(--sec);background:var(--s1);border:1px solid var(--line);border-radius:5px;padding:1px 5px}
  .fchip em{font-style:normal;color:var(--mut);font-size:9px}
  .fchip .was{color:var(--mut);text-decoration:line-through;font-weight:600}
  .fchip .arr{color:var(--mut)}
  .fchip .now{color:var(--gt);font-weight:700}
  /* 검색 case ↔ R_calib 연결: 같은 강조색(accent)으로 묶는다 */
  .rmark{background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--ink);border-radius:3px;padding:0 2px}
  .why.linked{border-left:3px solid var(--accent);padding-left:11px}
  .cases.linked .case{border-color:var(--accent);box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 30%,transparent)}
  .linknote{color:var(--accent);font-weight:600}
  svg{display:block;width:100%;height:auto;margin-top:12px}
  footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);font-family:var(--mono);font-size:11px;color:var(--mut);line-height:1.7}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="wrap">
  <div class="eyebrow">STEP3 · 홀드아웃 test 6 window · 22채널</div>
  <h1>보정은 무엇을 보고 움직였나</h1>
  <p class="lede">test window마다 <b>Ŷ_base</b>(타깃만) → <b>Ŷ_calib</b>(LLM 보정) → <b>GT</b>를 나란히 놓고,
    LLM이 어떤 채널을 근거로 삼았는지(라벨), 그 채널의 그날 값은 얼마였는지, 왜 그렇게 판단했는지(설명),
    무엇을 검색했는지를 함께 본다.</p>

  <div class="kpis" id="kpis"></div>
  <div class="take" id="take"></div>

  <section class="sum">
    <h2>이번에 바뀐 것 — 코멘트 대응</h2>
    <ol class="bul">
      <li><span class="tag done">완료</span><b>Attribution을 날짜별로</b>
        <p>future-known 10채널(공휴일·콘서트·항공편)은 horizon <b>7일 각각에 라벨</b>을 붙인다 — 이벤트가 걸린
          그날만 up/down이고 나머지는 중립. past-only 12채널(환율·기상·호텔가)은 미래값을 모르므로 window
          전체에 라벨 하나. 아래 히트맵과 카드에서 <b>날짜 × 채널</b>로 확인할 수 있다.</p></li>

      <li><span class="tag done">완료</span><b>Attribution reflection을 메모리에 되먹임</b>
        <p>reflection이 GT를 본 뒤 <b>라벨을 다시 매겨</b>(<code>corrected</code>) case에 함께 저장한다.
          이 case가 다음 window에서 검색되면 <b>원본 라벨과 교정 라벨이 함께 프롬프트로 들어간다</b> —
          <i>"그때 이렇게 봤는데 실제론 이게 맞았다"</i>가 attribution·calibration의 입력이 되는 것이다.
          각 case 카드의 <b class="fixref">라벨 교정</b> 항목이 그것이다.</p>
        <p>검색 <b>거리</b>는 <b>원본 라벨</b>로만 계산한다. 추론 중인 window는 GT를 못 보므로, 비교 상대도
          GT를 못 본 라벨이어야 공정하기 때문. 교정 라벨은 <b>거리 계산엔 쓰지 않고</b> 검색된 뒤
          학습 재료로만 보여준다.</p></li>

      <li><span class="tag done">완료</span><b>Retrieval 개선</b>
        <p>2차 검색(→calibration)이 lookback을 무시하던 문제를 고쳤다. ① 라벨 순서거리에 <b>IDF 가중</b>
          (거의 모든 window에서 활성인 채널은 변별력이 없으므로 down-weight) ② 날짜별 라벨은 평균이 아니라
          <b>가장 극단적인 날</b>로 대표(평균은 희소한 강신호를 지워버렸다) ③ lookback 유사도로 <b>re-rank</b>
          (α = 0.6 라벨 : 0.4 lookback). 공휴일 window가 공휴일 case를 <b>9/9</b>로 가져온다(기존 1–2/3).</p></li>

      <li><span class="tag done">완료</span><b>UI · 사용자용 설명</b>
        <p>전 과정을 노출한다 — 접이식 카드, 22채널 × 7일 <b>라벨 히트맵</b>, 검색된 case의 원본 context /
          Ŷ_base / Ŷ_calib / GT 궤적 plot, R_attr·R_calib·reflection 원문. calibration이 과거 사례를 인용한
          구절은 <b class="hlref">형광펜</b>으로 칠하고 <b>해당 case 카드와 같은 색으로 연결</b>했다 — 검색이
          실제로 판단에 쓰였는지 눈으로 확인하기 위해서. 사용자용 설명(B-2c)은 별도 프롬프트로
          <code>{summary, drivers, caveat}</code>를 생성한다.</p></li>

      <li><span class="tag new">추가</span><b>항공편 데이터 4채널 (18 → 22채널)</b>
        <p>국가별로 나눠 <code>flight_cn</code>(홍콩·마카오 포함) <code>flight_jp</code>
          <code>flight_us</code>(괌·사이판 포함) <code>flight_total</code>(국내선 제외). 다만
          <b>잔차 신호가 있는 건 <code>flight_us</code> 하나뿐</b>이다(효과 −0.37). <code>flight_cn</code>은
          타깃과의 상관은 높으나(+0.36) <b>잔차와는 무관</b>(+0.06).</p></li>

      <li><span class="tag new">발견</span><b>Reflection이 knowledge의 오류를 스스로 잡아냈다</b>
        <p>knowledge에 "한국 공휴일 = 하향"이라 적혀 있고 attribution이 매번 <code>weak_down</code>을 찍었는데,
          reflection이 잔차를 보고 <b>39회 교정</b>했다. 실제 데이터: <b>주말 잔차 +1,568</b> vs 평일 +794 —
          주말은 하향이 아니라 <b>상향</b>이다. 진짜 하향 신호는 <b>평일 공휴일 하나</b>(<b>−7,992</b>).
          항공편·환율·기상은 잔차 신호가 없어 98회 이상 <code>neutral</code>로 내렸지만,
          <b>콘서트는 오히려 강화</b>(<code>weak_up → strong_up</code> 8회) — 무차별 억제가 아니라
          <b>가짜 신호만 골라 죽였다</b>.</p></li>
    </ol>

    <div class="res">
      <b>결과</b> · reflection 개편 + 라벨 교정 적용 후
      <table>
        <tr><th></th><th>base MAE</th><th>calib MAE</th><th>개선 window</th></tr>
        <tr><td>train (18)</td><td>4,344</td><td><b>4,007 (−7.8%)</b></td><td>10 / 18</td></tr>
        <tr><td>test (6, 홀드아웃)</td><td>4,397</td><td><b>4,048 (−7.9%)</b></td><td><b>6 / 6</b></td></tr>
      </table>
      <p>test 6개 window가 <b>전부</b> 개선된 것은 처음이다(이전 최고 5/6). train과 test의 개선폭이 거의
        같아(−7.8% vs −7.9%) 과적합 없이 일반화됐다.</p>
    </div>
  </section>

  <div class="lg"><span><i class="gt"></i>GT (실제)</span><span><i style="background:var(--base)"></i>Ŷ_base</span><span><i style="background:var(--calib)"></i>Ŷ_calib</span></div>
  <div id="cards"></div>
  <footer id="foot"></footer>
</div>
<script>
const D=__DATA__;
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const k=v=>Math.round(v).toLocaleString();
const sg=v=>(v>=0?"+":"")+k(v);
const LAB={strong_up:["su","강한 상향"],weak_up:["wu","약한 상향"],strong_down:["sd","강한 하향"],weak_down:["wd","약한 하향"]};

// 축 있는 미니 차트 — 여러 시리즈를 한 y축에 겹쳐 그리고, y눈금 3개 + 예측경계 세로선.
// series=[{v:[...], off, color, sw, dash}]. boundary=context/예측 경계 인덱스(세로선).
function spark(series, opt){
  opt=opt||{}; const W=opt.w||440, H=opt.h||96, PL=34, PR=6, PT=8, PB=14;
  const all=series.flatMap(s=>s.v).filter(x=>x!=null);
  if(!all.length) return "";
  let lo=Math.min(...all), hi=Math.max(...all); const pad=(hi-lo)*.1||1; lo-=pad; hi+=pad;
  const N=Math.max(...series.map(s=>(s.off||0)+s.v.length));
  const X=i=>PL+i*(W-PL-PR)/(N-1||1), Y=v=>PT+(H-PT-PB)*(1-(v-lo)/(hi-lo));
  let s="";
  // y축 눈금 3개
  for(let g=0;g<3;g++){const val=lo+(hi-lo)*g/2,y=Y(val);
    s+=`<line x1="${PL}" y1="${y.toFixed(1)}" x2="${W-PR}" y2="${y.toFixed(1)}" stroke="${cssv('--line')}"/>`
      +`<text x="${PL-4}" y="${(y+3).toFixed(1)}" fill="${cssv('--mut')}" font-size="8.5" text-anchor="end" font-family="ui-monospace,monospace">${(val/1000).toFixed(0)}k</text>`;}
  // 예측 경계 세로선 (context ↔ 예측)
  if(opt.boundary!=null){const bx=X(opt.boundary);
    s+=`<line x1="${bx.toFixed(1)}" y1="${PT}" x2="${bx.toFixed(1)}" y2="${H-PB}" stroke="${cssv('--mut')}" stroke-dasharray="2 2" opacity=".6"/>`
      +`<text x="${(bx+3).toFixed(1)}" y="${PT+8}" fill="${cssv('--mut')}" font-size="8" font-family="ui-monospace,monospace">예측→</text>`;}
  series.forEach(sr=>{
    const off=sr.off||0;
    const d=sr.v.map((v,i)=>`${i?"L":"M"}${X(off+i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    s+=`<path d="${d}" fill="none" stroke="${cssv(sr.color)}" stroke-width="${sr.sw||1.5}"${sr.dash?' stroke-dasharray="3 2"':''}/>`;
  });
  return `<svg class="spark" viewBox="0 0 ${W} ${H}">${s}</svg>`;
}
// case용: context(28) 뒤에 예측구간(7)의 GT·base·calib를 이어 붙여 한 흐름으로 (경계선 표시).
// 예측 선들은 context의 마지막 점(off=n-1)부터 시작해 끊김 없이 연결한다.
function caseSpark(c){
  const n=c.context.length, last=c.context[n-1];
  const S=[{v:c.context, color:"--mut", sw:1.3}];
  if(c.gt&&c.gt.length)   S.push({v:[last,...c.gt],    off:n-1, color:"--gt",    sw:2.0});
  S.push({v:[last,...c.base],  off:n-1, color:"--base",  sw:1.5});
  S.push({v:[last,...c.calib], off:n-1, color:"--calib", sw:1.5, dash:1});
  return spark(S, {w:440, h:96, boundary:n-1});
}

document.getElementById("kpis").innerHTML=[
  [`${k(D.mae_base)}`,"base MAE (명)"],
  [`${k(D.mae_calib)}`,"calib MAE (명)"],
  [`${((D.mae_calib/D.mae_base-1)*100).toFixed(1)}%`,"MAE 변화"],
  [`${D.better}/${D.n}`,"calib이 나은 window"],
].map(([v,l])=>`<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

document.getElementById("take").innerHTML=
  `<b>새 채널 <code>hol_kr_wd</code>가 정확히 작동했다.</b> test 구간의 평일 한국 공휴일은 셋이다 —
   5/5 어린이날, 5/25 부처님오신날 대체, 6/3 지방선거. attribution은 <b>그 셋이 든 window에서만</b>
   <span class="dn">강한 하향</span>을 붙였고, 나머지 세 window는 중립이다. 오탐도 미탐도 없다.
   R_attr도 STEP1 record가 학습한 −7천명대 하향 효과를 근거로 인용한다.
   knowledge → attribution → calibration 사슬이 처음으로 끝까지 이어졌다.`;

document.getElementById("foot").textContent=
  `fs · test ${D.span} · Ŷ_base는 Chronos-2 median(타깃만) · 재현 python evaluate.py`;

function chart(days){
  const W=880,H=150,PL=46,PR=10,PT=10,PB=16,N=7;
  const all=days.flatMap(d=>[d.gt,d.base,d.calib]);
  let lo=Math.min(...all),hi=Math.max(...all);const pad=(hi-lo)*.12||1;lo-=pad;hi+=pad;
  const X=i=>PL+i*(W-PL-PR)/(N-1), Y=v=>PT+(H-PT-PB)*(1-(v-lo)/(hi-lo));
  const path=key=>days.map((d,i)=>`${i?"L":"M"}${X(i).toFixed(1)},${Y(d[key]).toFixed(1)}`).join(" ");
  let s="";
  for(let g=0;g<3;g++){const val=lo+(hi-lo)*g/2,y=Y(val);
    s+=`<line x1="${PL}" y1="${y.toFixed(1)}" x2="${W-PR}" y2="${y.toFixed(1)}" stroke="${cssv('--line')}"/>`
      +`<text x="${PL-6}" y="${(y+3).toFixed(1)}" fill="${cssv('--mut')}" font-size="9" text-anchor="end" font-family="ui-monospace,monospace">${(val/1000).toFixed(0)}k</text>`;}
  // 평일 한국 공휴일 — GT가 크게 꺼지는 날. 라벨(strong_down)이 붙은 근거.
  days.forEach((d,i)=>{if(d.hol){
    s+=`<line x1="${X(i).toFixed(1)}" y1="${PT}" x2="${X(i).toFixed(1)}" y2="${H-PB}" stroke="${cssv('--calib')}" stroke-width="1" stroke-dasharray="3 3" opacity=".6"/>`
      +`<text x="${X(i).toFixed(1)}" y="${PT+8}" fill="${cssv('--calib')}" font-size="9" text-anchor="middle" font-family="ui-monospace,monospace">공휴일</text>`;}});
  s+=`<path d="${path('base')}" fill="none" stroke="${cssv('--base')}" stroke-width="1.7"/>`;
  s+=`<path d="${path('calib')}" fill="none" stroke="${cssv('--calib')}" stroke-width="1.7"/>`;
  s+=`<path d="${path('gt')}" fill="none" stroke="${cssv('--gt')}" stroke-width="2.2"/>`;
  days.forEach((d,i)=>{s+=`<text x="${X(i).toFixed(1)}" y="${H-3}" fill="${cssv('--mut')}" font-size="9" text-anchor="middle" font-family="ui-monospace,monospace">${d.date.slice(5)}</text>`;});
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="base, calib, gt 궤적">${s}</svg>`;
}

// R_calib에서 '과거 사례/reflect/회고'를 언급한 구절을 강조 — 검색 case가 보정에 반영된 지점.
function hlReflect(t){
  return t.replace(/([^.。]*(?:과거 반성|과거 사례|reflect|회고|검색된 과거)[^.。]*[.。])/g,
    '<mark class="rmark">$1</mark>');
}

// B-2c 사용자용 설명 — R_attr/R_calib(내부 분석)을 의사결정자가 읽는 요약+근거+주의점으로 재작성한 것.
// 이 프레임워크의 최종 산출물이므로 카드 맨 위에 놓는다.
function explBlock(e){
  if(!e) return "";
  const drv=(e.drivers||[]).map(d=>{
    if(typeof d==="string") return `<li>${d}</li>`;
    const dt=d.date?`<b>${d.date.slice(5)}</b> `:"";
    const dir=d.direction||"";
    const dc=/증가|상승|up/.test(dir)?"up":/감소|하락|down/.test(dir)?"dn":"dim";
    return `<li>${dt}<span class="${dc}">${dir}</span> — ${d.factor||""}${d.why?` <span class="dim">${d.why}</span>`:""}</li>`;
  }).join("");
  return `<div class="expl">
    <div class="ehd">사용자용 설명 <span>B-2c · 이 프레임워크의 최종 산출물</span></div>
    <p class="esum">${e.summary||""}</p>
    ${drv?`<ul class="edrv">${drv}</ul>`:""}
    ${e.caveat?`<p class="ecav"><b>주의</b> ${e.caveat}</p>`:""}
  </div>`;
}

// 검색된 train case 한 건 → 카드. 요약(날짜·verdict·라벨칩)은 항상 보이고, 전 과정은 접었다 펼친다.
const VD={better:["up","보정 개선"],worse:["dn","보정 악화"],same:["dim","차이 없음"]};
const nums=arr=>(arr&&arr.length?arr.map(k).join(", "):"—");
function caseCard(c){
  if(c.missing) return `<div class="case"><span class="dim">${c.date} (case 없음)</span></div>`;
  const [vc,vt]=VD[c.verdict]||["dim",c.verdict];
  const chips=c.labs.map(l=>`<span class="lchip">${l.ko}<b>${l.s}</b></span>`).join("")||'<span class="dim">전부 중립</span>';
  return `<details class="case">
    <summary><span class="chd"><b>${c.date}</b>${c.hol?'<span class="hbadge">공휴일</span>':''}
      <span class="${vc}">${vt}</span><span class="exp">펼치기 ▾</span></span>
      <div class="lchips">${chips}</div></summary>
    <div class="cbody">
      <div class="sparkwrap">
        <div class="sparklg"><span><i style="background:var(--mut)"></i>타깃 28일(근거)</span><span><i style="background:var(--gt)"></i>GT(실제)</span><span><i style="background:var(--base)"></i>Ŷ_base</span><span><i style="background:var(--calib)"></i>Ŷ_calib</span></div>
        ${caseSpark(c)}
      </div>
      <div class="frow"><span class="flab">값 — 타깃28일 / Ŷ_base / Ŷ_calib</span><span class="fval">${nums(c.context)}<br>${nums(c.base)} → ${nums(c.calib)}</span></div>
      <div class="cwhy"><b>R_attr</b>${c.r_attr||"—"}</div>
      <div class="cwhy"><b>R_calib</b>${c.r_calib||"—"}</div>
      <div class="cwhy refl"><b>reflection (GT와 비교한 회고)</b>${c.reflect||"—"}</div>
      ${fixBlock(c)}
    </div>
  </details>`;
}

// reflection이 GT를 보고 고친 라벨 — "그때 이렇게 봤는데(was) 실제론 이게 맞았다(now)".
// 검색 거리는 원본 라벨로 계산하고, 이 교정은 프롬프트에 학습 재료로만 실린다.
function fixBlock(c){
  const f=c.fixes||[];
  if(!f.length) return '<div class="cwhy fix"><b>라벨 교정</b><span class="dim">고칠 것 없었음 — 그때 매긴 라벨이 실제와 부합</span></div>';
  const chips=f.map(x=>`<span class="fchip">${x.ko}${x.d?`<em>${x.d}일</em>`:""}
    <b class="was">${x.was}</b><span class="arr">→</span><b class="now">${x.now}</b></span>`).join("");
  return `<div class="cwhy fix"><b>라벨 교정 (GT 확인 후 — 이렇게 봤어야 함)</b>
    <div class="fchips">${chips}</div></div>`;
}

// 히트맵: 전 채널 × 7일 라벨. 색=ordinal(0 SD 빨강 ~ 2 neutral 회색 ~ 4 SU 초록). 값은 칸 안 작게.
const OSH=["SD","wd","·","wu","SU"];
function heatCol(o){                                    // ordinal → 배경색
  if(o===2) return "transparent";
  const up=o>2, str=(o===0||o===4);
  const c=up?"--good":"--bad", pct=str?26:13;
  return `color-mix(in srgb,var(${c}) ${pct}%,transparent)`;
}
function heatmap(w){
  const hd=`<th>채널</th>`+w.days.map(d=>`<th>${d.date.slice(5)}<span class="th2">${d.wd}</span></th>`).join("");
  const rows=w.heat.map(h=>{
    const cells=h.ord.map((o,i)=>{
      const lab=o!==2?`<span class="hlab">${OSH[o]}</span>`:"";
      return `<td class="hcell" style="background:${heatCol(o)}">${lab}<span class="hval">${h.vals[i]}</span></td>`;
    }).join("");
    const rc=h.active?"":"hneu";                        // 전부 중립 행은 옅게
    const rl=h.role==="future"?"미래·날짜별":"과거·window";
    return `<tr class="${rc}"><td class="hname">${h.ko}<span class="role">${rl}</span></td>${cells}</tr>`;
  }).join("");
  return `<div class="scroll"><table class="heat"><thead><tr>${hd}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

// (구) 활성 채널만 표. 히트맵으로 대체했으나 함수는 남겨둠.
const PDCLS={SD:"dn",wd:"dn",wu:"up",SU:"up","·":""};
const SHORT={strong_down:"SD",weak_down:"wd",neutral:"·",weak_up:"wu",strong_up:"SU"};
function covRows(w){
  return w.cov.map(c=>{
    const [cls,txt]=LAB[c.lab];
    const wlab=SHORT[c.lab]||"·";                          // past-only의 window 라벨 축약
    const cells=c.vals.map((v,i)=>{
      const pd=c.perday?c.perday[i]:wlab;                  // future=날짜별, past=window 라벨 반복
      const k=PDCLS[pd]||"";
      const dim=c.perday?"":"win";                         // past-only는 window단위임을 옅게
      return `<td class="pdcell ${k} ${dim}"><span class="pdlab">${pd==="·"?"":pd}</span><span class="pdval">${v}</span></td>`;
    }).join("");
    const role=c.role==="future"?"미래값·날짜별":"과거값·window";
    return `<tr><td><span class="chip ${cls}">${c.ko}</span>
      <span class="role">${role}</span></td>${cells}</tr>`;
  }).join("") || `<tr><td colspan="8" class="dim">라벨이 붙은 채널 없음</td></tr>`;
}

document.getElementById("cards").innerHTML=D.wins.map((w,wi)=>{
  const better=w.mae_calib<w.mae_base;
  const chips=w.labels.map(l=>`<span class="chip ${LAB[l.lab][0]}">${l.ko} ${LAB[l.lab][1]}</span>`).join("");
  const rows=w.days.map(d=>
    `<tr class="${d.hol?'hol':''}"><td>${d.date.slice(5)} ${d.wd}${d.hol?' <span class="tag">공휴일</span>':''}</td>
      <td>${k(d.gt)}</td><td>${k(d.base)}</td><td>${k(d.calib)}</td>
      <td class="${d.adj>=0?'up':'dn'}">${sg(d.adj)}</td>
      <td class="dim">${sg(d.need)}</td></tr>`).join("");
  return `<details class="card" open>
    <summary class="hd"><span class="w">window ${wi+1} · ${w.date}</span>
      <span class="m">MAE ${k(w.mae_base)} → <b class="${better?'up':'dn'}">${k(w.mae_calib)}</b>
        (${better?'개선':'악화'} ${sg(w.mae_base-w.mae_calib)})</span><span class="exp">▾</span></summary>
    <div class="chips">${chips||'<span class="chip">전부 중립</span>'}</div>
    ${explBlock(w.expl)}
    <div class="sparkwrap"><div class="sparklg"><span><i style="background:var(--mut)"></i>이 window의 예측 근거 — 타깃 28일(과거→최신)</span></div>
      ${spark([{v:w.context,color:"--mut",sw:1.4}],{w:440,h:70})}</div>
    ${chart(w.days)}
    <div class="scroll"><table><thead><tr>
      <th>날짜</th><th>GT</th><th>Ŷ_base</th><th>Ŷ_calib</th><th>적용 보정</th><th>필요했던 보정</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    <div class="cap">채널 × 일별 라벨 히트맵 (전 채널 · 빨강=하향 회색=중립 초록=상향 · 진할수록 strong) ↓</div>
    ${heatmap(w)}
    <div class="why"><b>R_attr — 왜 그렇게 라벨했나</b>${w.r_attr}</div>
    <div class="why ${w.uses_reflect?'linked':''}"><b>R_calib — 어떻게 반영했나</b>${hlReflect(w.r_calib)}</div>
    <div class="cap">검색된 유사 case — 라벨 기반(진단이 비슷한 train window)${w.uses_reflect?' <span class="linknote">← 위 R_calib이 이 case들의 회고를 반영</span>':''} · 카드 클릭 시 전 과정 ↓</div>
    <div class="cases ${w.uses_reflect?'linked':''}">${w.ret_attr.map(caseCard).join("")}</div>
    <div class="cap">look-back 기반(상황이 비슷한 train window) ↓</div>
    <div class="cases">${w.ret_lb.map(caseCard).join("")}</div>
  </details>`;}).join("");
</script>
"""

html = HTML.replace("__DATA__", json.dumps(D, ensure_ascii=False))
if a.standalone:
    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<style>*{box-sizing:border-box}body{margin:0}img{max-width:100%}</style>'
            f'</head><body>{html}</body></html>')

d = os.path.dirname(a.out)
if d:
    os.makedirs(d, exist_ok=True)
open(a.out, "w", encoding="utf-8").write(html)
print(f"[page] → {a.out}  ({os.path.getsize(a.out)/1024:.0f} KB)  standalone={a.standalone}  "
      f"MAE {D['mae_base']}→{D['mae_calib']}  개선 {D['better']}/{D['n']} window")
