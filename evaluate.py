"""
STEP3 : test(홀드아웃) 구간 추론 + 채점.

STEP2와의 차이 — GT를 추론에 쓰지 않고(평가에만), experience memory는 read-only.
test window마다 B-1(base) → B-2a(attribution) → B-2b(calibration) → Ŷ_calib,
그다음 GT와 base vs calib를 MAE/RMSE/horizon별로 비교한다.
--explain을 주면 B-2c로 의사결정자용 설명을 한 번 더 생성한다.

실행: python evaluate.py --channels sig2 --exp memory/run_sig2/experience.json --explain
"""

import argparse
import json
import os

import numpy as np

from config import CHANNEL_SETS, resolve_channels, run_tag, TARGET_DESC_KO, MEMDIR
from engine.data import load, make_windows, HORIZON, TEST_LEN
from engine.tsfm import TSFM
from engine.llm import LLM
from engine.memory import (Experience, LABELS, attribution_schema, array_schema, label_short,
                           EXPLAIN_SCHEMA)
from engine.features import (fmt, lookback_profile, covariate_value_lines,
                             horizon_cov_lines, fmt_retrieved_lb, fmt_retrieved_attr,
                             active_knowledge, knowledge_text, attribution_lines,
                             parse_attribution, neutral_attribution)
from prompt.attribution import attribution_system, attribution_user
from prompt.calibration import calibration_system, calibration_user, calibration_scale_note
from prompt.explain import explain_system, explain_user


def mae(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def rmse(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=str, default="sig2",
                    help=f"{list(CHANNEL_SETS)} 중 하나 또는 쉼표구분 채널명 (STEP2와 같은 값)")
    ap.add_argument("--exp", type=str, default=None,
                    help="experience 읽을 경로 (기본: memory/experience_<tag>.json)")
    ap.add_argument("--explain", action="store_true",
                    help="B-2c: window마다 의사결정자용 설명(요약+근거목록)을 추가 생성")
    ap.add_argument("--scale-pct", type=float, default=0.0,
                    help="프롬프트 내 보정 크기 상한 ±N퍼센트; 0=제한 없음(기본)")
    ap.add_argument("--knowledge", type=str, default="knowledge.json", help="memory/ 아래 knowledge 파일명")
    ap.add_argument("--no-knowledge", action="store_true", help="knowledge 없이 데이터만으로 (ablation)")
    ap.add_argument("--no-experience", action="store_true", help="메모리 검색 없이 (ablation)")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--k", type=int, default=3, help="검색 top-k")
    ap.add_argument("--model", type=str, default="gpt-5-mini")
    ap.add_argument("--effort", type=str, default="low", help="gpt-5 reasoning effort")
    args = ap.parse_args()

    chans = resolve_channels(args.channels)
    tag = run_tag(args.channels, args.no_knowledge, args.no_experience)
    # experience 경로는 build 태그(채널+knowledge)로 — no_experience는 읽지 않으므로 무관
    exp_path = args.exp or os.path.join(
        MEMDIR, f"experience_{run_tag(args.channels, args.no_knowledge)}.json")

    df, target, future_known, past_only = load()
    idx_of = {str(d.date()): i for i, d in enumerate(df["date"])}   # explain 날짜 역참조
    train_end = len(df) - TEST_LEN
    ref = df[target].values[:train_end].astype("float64")
    windows = make_windows(df, target, future_known, chans,
                           horizon=args.horizon, region="test")     # ← 홀드아웃
    tsfm = TSFM()
    llm = LLM(model=args.model, reasoning_effort=args.effort)

    if args.no_knowledge:
        knowledge = {}
        knowledge_lines = "\n".join(f"- {ch}: (no knowledge — judge only from the actual values above)"
                                    for ch in chans)
    else:
        knowledge = json.load(open(os.path.join(MEMDIR, args.knowledge), encoding="utf-8"))
        knowledge_lines = "\n".join(f"- {ch}: {knowledge_text(knowledge.get(ch, '(no knowledge)'))}"
                                    for ch in chans)

    attr_sys = attribution_system.format(target=TARGET_DESC_KO, labels=", ".join(LABELS),
                                         horizon=args.horizon)
    attr_schema = attribution_schema(chans, args.horizon, future_known)
    cal_sys = calibration_system.format(target=TARGET_DESC_KO, horizon=args.horizon)
    if args.scale_pct > 0:
        cal_sys += calibration_scale_note.format(pct=int(args.scale_pct))
    expl_sys = explain_system.format(target=TARGET_DESC_KO, horizon=args.horizon)

    if args.no_experience:
        exp = Experience(os.path.join(MEMDIR, "__no_experience__.json"))  # 없는 경로 → cases=[]
    else:
        if not os.path.exists(exp_path):
            raise SystemExit(f"[infer] {exp_path} 없음 — 먼저 "
                             f"`python experience.py --channels {args.channels}` 을 돌리세요")
        exp = Experience(exp_path)                                # read-only (train에서 구축)

    print(f"[infer] test windows={len(windows)}  channels={len(chans)}({args.channels})  tag={tag}  "
          f"scale-pct={args.scale_pct}  memory cases={len(exp.cases)}  "
          f"device={tsfm.device}  llm={llm.model}")

    results = []
    hb = np.zeros(args.horizon); hc = np.zeros(args.horizon)   # horizon별 절대오차 누적
    for i, w in enumerate(windows):
        gt = w["y_true"]                                       # 채점에만 사용
        y_base = tsfm.forecast(w["y_lookback"], args.horizon)  # B-1
        prof = lookback_profile(w["y_lookback"], ref)

        # B-2a attribution (look-back DTW 검색, GT 없음)
        ret_a = exp.retrieve_by_lookback(w["y_lookback"], k=args.k)
        ua = attribution_user.format(
            horizon=args.horizon, lb_values=fmt(w["y_lookback"]),
            y_base=fmt(y_base), cov_values=covariate_value_lines(w),
            knowledge=knowledge_lines, retrieved=fmt_retrieved_lb(ret_a))
        try:
            a = llm.parse(attr_sys, ua, schema=attr_schema, schema_name="attribution")
            attribution = parse_attribution(a.get("attribution", {}), chans, future_known, args.horizon)
            r_attr = a.get("reasoning", "")
        except Exception as e:
            attribution = neutral_attribution(chans, future_known, args.horizon)
            r_attr = f"(fail: {e})"

        # B-2b calibration (라벨 검색 + look-back 리랭크 + knowledge 지침 + horizon covariate)
        ret_b = exp.retrieve_by_attribution(attribution, k=args.k, y_lookback=w["y_lookback"])
        uc = calibration_user.format(
            y_base=fmt(y_base), horizon=args.horizon,
            attribution=attribution_lines(attribution),
            r_attr=r_attr, knowledge_active=active_knowledge(attribution, knowledge),
            future_cov=horizon_cov_lines(w), retrieved=fmt_retrieved_attr(ret_b))
        try:
            c = llm.parse(cal_sys, uc, schema=array_schema("forecast"), schema_name="calibration")
            y_calib = np.array(c["forecast"], dtype="float64")
            if len(y_calib) != args.horizon:
                raise ValueError("calib length mismatch")
            r_calib = c.get("reasoning", "")
        except Exception as e:
            y_calib = y_base.copy(); r_calib = f"(fail: {e})"

        # B-2c 사용자용 설명 (선택) — 내부 분석을 의사결정자용 요약+근거로 재작성
        explanation = None
        if args.explain:
            c0 = idx_of[w["date"]]
            dates = ", ".join(str(df["date"].iloc[c0 + t].date()) for t in range(args.horizon))
            ue = explain_user.format(
                dates=dates, horizon=args.horizon, y_calib=fmt(y_calib), y_base=fmt(y_base),
                attribution=attribution_lines(attribution), r_attr=r_attr, r_calib=r_calib,
                cov_values=horizon_cov_lines(w))
            try:
                explanation = llm.parse(expl_sys, ue, schema=EXPLAIN_SCHEMA, schema_name="explain")
            except Exception as e:
                explanation = {"summary": f"(explain fail: {e})", "drivers": [], "caveat": ""}

        mb, mc = mae(y_base, gt), mae(y_calib, gt)
        hb += np.abs(y_base - gt); hc += np.abs(y_calib - gt)
        results.append({
            "date": w["date"], "attribution": attribution,
            "y_base": [float(x) for x in y_base], "y_calib": [float(x) for x in y_calib],
            "gt": [float(x) for x in gt], "R_attr": r_attr, "R_calib": r_calib,
            "mae_base": mb, "mae_calib": mc,
            "rmse_base": rmse(y_base, gt), "rmse_calib": rmse(y_calib, gt),
            "retrieved_lb": [c2["date"] for c2 in ret_a],
            "retrieved_attr": [c2["date"] for c2 in ret_b],
            **({"explanation": explanation} if explanation is not None else {}),
        })
        if explanation is not None:
            print(f"    설명: {explanation.get('summary','')}")
        print(f"  win {i+1}/{len(windows)} {w['date']}  MAE base={mb:.0f} calib={mc:.0f}  "
              f"{'calib↓' if mc < mb else 'base↓'}  attr=[{','.join(label_short(v) for v in attribution.values())}]")

    ftag = tag + (f"_scale{int(args.scale_pct)}" if args.scale_pct > 0 else "_free")
    # 어떤 experience 메모리를 읽었는지도 파일명에 넣는다 — 안 그러면 --exp로 다른 메모리를 써도
    # 같은 파일에 덮어써서 이전 결과가 조용히 사라진다.
    _mem = os.path.splitext(os.path.basename(exp_path))[0].replace("experience_", "")
    if _mem and _mem != tag:
        ftag += f"_{_mem}"
    # 결과는 exp 메모리와 같은 폴더에 (--exp가 하위폴더면 결과도 거기로 모인다)
    out_dir = os.path.dirname(exp_path) if args.exp else MEMDIR
    fname = os.path.join(out_dir, f"test_results_{ftag}.json")
    json.dump(results, open(fname, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 집계 ----
    MB = np.mean([r["mae_base"] for r in results]); MC = np.mean([r["mae_calib"] for r in results])
    RB = np.mean([r["rmse_base"] for r in results]); RC = np.mean([r["rmse_calib"] for r in results])
    win = sum(1 for r in results if r["mae_calib"] < r["mae_base"])
    n = len(results)
    print(f"\n===== TEST 결과 ({n} windows) =====")
    print(f"  MAE   base={MB:.0f}  calib={MC:.0f}  ({(MC-MB):+.0f}, {100*(MC-MB)/MB:+.1f}%)")
    print(f"  RMSE  base={RB:.0f}  calib={RC:.0f}  ({100*(RC-RB)/RB:+.1f}%)")
    print(f"  calib이 나은 window: {win}/{n}")
    print(f"  horizon별 MAE (h=1..{args.horizon}):")
    print(f"    base : {[int(x/n) for x in hb]}")
    print(f"    calib: {[int(x/n) for x in hc]}")
    print(f"[store] → {fname}  (scale-pct={args.scale_pct})")
    u = llm.usage
    print(f"[usage] calls={u['calls']}  in={u['in']:,} tok  out={u['out']:,} tok")


if __name__ == "__main__":
    main()
