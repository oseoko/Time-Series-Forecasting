"""
STEP2 : B 루프를 train window에 돌려 experience memory를 사전 구축.

한 window당 5단계:
  B-1  base   : Ŷ_base = TSFM(타깃만)                              [LLM 없음]
  B-2a attrib : look-back 검색 → LLM이 {A_Cᵢ}(5단계 라벨) + R_attr  [GT 없음]
  B-2b calib  : {A_Cᵢ} 검색 → LLM이 Ŷ_calib + R_calib             [GT 없음]
  B-3  reflect: Ŷ_base, Ŷ_calib, GT 비교 → reflect + corrected     [GT 사용(train만)]
  B-4  update : case를 experience memory에 append (GT 미저장)

실행: python experience.py --channels sig2 --exp memory/run_sig2/experience.json
"""

import argparse
import json
import os

import numpy as np

from config import CHANNEL_SETS, resolve_channels, run_tag, TARGET_DESC_KO, MEMDIR
from engine.data import load, make_windows, HORIZON, TEST_LEN
from engine.tsfm import TSFM
from engine.llm import LLM
from engine.memory import (Experience, LABELS, attribution_schema, array_schema, reflect_schema,
                           label_short)
from engine.features import (fmt, lookback_profile, covariate_value_lines,
                             horizon_cov_lines, fmt_retrieved_lb, fmt_retrieved_attr,
                             active_knowledge, knowledge_text, attribution_lines,
                             parse_attribution, neutral_attribution, _label_change)
from prompt.attribution import attribution_system, attribution_user
from prompt.calibration import calibration_system, calibration_user, calibration_scale_note
from prompt.reflection import reflection_system, reflection_user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=str, default="sig2",
                    help=f"{list(CHANNEL_SETS)} 중 하나 또는 쉼표구분 채널명")
    ap.add_argument("--exp", type=str, default=None,
                    help="experience 저장 경로 (기본: memory/experience_<tag>.json)")
    ap.add_argument("--scale-pct", type=float, default=0.0,
                    help="프롬프트 내 보정 크기 상한 ±N퍼센트; 0=제한 없음(기본)")
    ap.add_argument("--knowledge", type=str, default="knowledge.json", help="memory/ 아래 knowledge 파일명")
    ap.add_argument("--no-knowledge", action="store_true", help="knowledge 없이 데이터만으로 (ablation)")
    ap.add_argument("--no-reflection", action="store_true", help="B-3 생략, 원본 라벨만 저장 (ablation)")
    ap.add_argument("--windows", type=int, default=None, help="앞에서 N개 window만 (디버그)")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--k", type=int, default=3, help="검색 top-k")
    ap.add_argument("--model", type=str, default="gpt-5-mini")
    ap.add_argument("--effort", type=str, default="low", help="gpt-5 reasoning effort")
    args = ap.parse_args()

    chans = resolve_channels(args.channels)
    tag = run_tag(args.channels, no_knowledge=args.no_knowledge)
    exp_path = args.exp or os.path.join(MEMDIR, f"experience_{tag}.json")

    os.makedirs(MEMDIR, exist_ok=True)
    df, target, future_known, past_only = load()
    train_end = len(df) - TEST_LEN
    ref = df[target].values[:train_end].astype("float64")      # lookback_profile의 level 기준
    windows = make_windows(df, target, future_known, chans,
                           horizon=args.horizon, max_windows=args.windows)
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
    refl_sys = reflection_system.format(labels=", ".join(LABELS), horizon=args.horizon)
    refl_schema = reflect_schema(chans, args.horizon, future_known)

    os.makedirs(os.path.dirname(exp_path) or ".", exist_ok=True)   # --exp가 하위폴더면 만든다
    if os.path.exists(exp_path):
        print(f"[build] 경고: {os.path.basename(exp_path)} 가 이미 있음 → 새로 만듭니다(append 아님)")
        os.remove(exp_path)
    exp = Experience(exp_path)

    print(f"[build] windows={len(windows)}  channels={len(chans)}({args.channels})  tag={tag}  "
          f"device={tsfm.device}  llm={llm.model}")
    print(f"[build] → {os.path.basename(exp_path)}")
    print(f"[build] labels={LABELS}\n")

    rows = []
    analysis = []          # 보고용 덤프 (GT 포함 — experience.json과 별개 파일)
    for i, w in enumerate(windows):
        gt = w["y_true"]

        # B-1 base (타깃만)
        y_base = tsfm.forecast(w["y_lookback"], args.horizon)
        prof = lookback_profile(w["y_lookback"], ref)

        # B-2a look-back DTW 검색 → attribution
        ret_a = exp.retrieve_by_lookback(w["y_lookback"], k=args.k)
        u = attribution_user.format(
            horizon=args.horizon, lb_values=fmt(w["y_lookback"]),
            y_base=fmt(y_base), cov_values=covariate_value_lines(w),
            knowledge=knowledge_lines, retrieved=fmt_retrieved_lb(ret_a))
        try:
            a = llm.parse(attr_sys, u, schema=attr_schema, schema_name="attribution")
            attribution = parse_attribution(a.get("attribution", {}), chans, future_known, args.horizon)
            r_attr = a.get("reasoning", "")
        except Exception as e:
            attribution = neutral_attribution(chans, future_known, args.horizon)
            r_attr = f"(attr parse fail: {e})"

        # B-2b 라벨 검색(look-back으로 리랭크) → calibration
        ret_b = exp.retrieve_by_attribution(attribution, k=args.k, y_lookback=w["y_lookback"])
        u = calibration_user.format(
            y_base=fmt(y_base), horizon=args.horizon,
            attribution=attribution_lines(attribution),
            r_attr=r_attr, knowledge_active=active_knowledge(attribution, knowledge),
            future_cov=horizon_cov_lines(w), retrieved=fmt_retrieved_attr(ret_b))
        try:
            c = llm.parse(cal_sys, u, schema=array_schema("forecast"), schema_name="calibration")
            y_calib = np.array(c["forecast"], dtype="float64")
            if len(y_calib) != args.horizon:
                raise ValueError("calib length mismatch")
            r_calib = c.get("reasoning", "")
        except Exception as e:
            y_calib = y_base.copy(); r_calib = f"(calib parse fail: {e})"

        # B-3 reflection (GT 사용). verdict는 산술이므로 코드가 정하고 프롬프트에는 넣지 않는다
        # — 결과를 알려주면 LLM이 거기 맞춰 "더 세게/약하게"만 쓰고 라벨 분석을 하지 않는다.
        mae_b = float(np.abs(y_base - gt).mean()); mae_c = float(np.abs(y_calib - gt).mean())
        verdict = "better" if mae_c < mae_b else "worse" if mae_c > mae_b else "same"
        if args.no_reflection:
            reflect = "(reflection off)"
            corrected = dict(attribution)
        else:
            u = reflection_user.format(
                horizon=args.horizon, y_base=fmt(y_base), y_calib=fmt(y_calib), gt=fmt(gt),
                resid=fmt(np.asarray(gt, dtype="float64") - y_base),
                attribution=attribution_lines(attribution), cov_values=covariate_value_lines(w),
                reasoning=f"R_attr: {r_attr}\nR_calib: {r_calib}")
            try:
                rf = llm.parse(refl_sys, u, schema=refl_schema, schema_name="reflection")
                reflect = rf.get("reflect", "")
                corrected = parse_attribution(rf.get("corrected", {}), chans, future_known, args.horizon)
            except Exception as e:
                reflect = f"(reflect parse fail: {e})"
                corrected = dict(attribution)      # 실패 시 원본 유지 = "고칠 것 없음"
        changed = {c: {"was": attribution[c], "now": corrected[c]}
                   for c in chans if corrected[c] != attribution[c]}

        # B-4 update — append-only, GT는 저장하지 않는다(leakage 차단선)
        case = {"date": w["date"], "lb_profile": prof,
                "y_lookback": [float(x) for x in w["y_lookback"]],   # DTW 검색용
                "y_base": [float(x) for x in y_base], "y_calib": [float(x) for x in y_calib],
                "attribution": attribution,          # 추론 시점 판단 → 검색 거리 계산용
                "corrected": corrected,              # GT를 본 뒤 고친 라벨 → 학습 재료용
                "reasoning": {"R_attr": r_attr, "R_calib": r_calib},
                "verdict": verdict, "reflect": reflect}
        exp.add(case)
        rows.append((mae_b, mae_c, verdict))
        analysis.append({
            "date": w["date"], "lb_profile": prof,
            "attribution": attribution, "R_attr": r_attr,
            "y_base": [float(x) for x in y_base], "y_calib": [float(x) for x in y_calib],
            "gt": [float(x) for x in gt], "R_calib": r_calib,
            "mae_base": mae_b, "mae_calib": mae_c, "verdict": verdict, "reflect": reflect,
            "corrected": corrected, "label_changed": changed,
            "retrieved_lb": [c["date"] for c in ret_a],
            "retrieved_attr": [c["date"] for c in ret_b],
        })

        print(f"===== window {i+1}/{len(windows)} ({w['date']}) profile={prof} =====")
        print("  attribution: " + " | ".join(f"{k}={label_short(v)}" for k, v in attribution.items()))
        print(f"  R_attr : {r_attr}")
        print(f"  Ŷ_base mean={np.mean(y_base):.0f}  Ŷ_calib mean={np.mean(y_calib):.0f}  gt={gt.mean():.0f}")
        print(f"  R_calib: {r_calib}")
        print(f"  MAE base={mae_b:.0f} → calib={mae_c:.0f}  verdict={verdict}")
        print(f"  retrieved(lb)={[c['date'] for c in ret_a]}  retrieved(attr)={[c['date'] for c in ret_b]}")
        print(f"  R_reflect: {reflect}")
        if changed:
            print("  라벨 수정: " + " | ".join(
                f"{c}: {_label_change(v['was'], v['now'])}" for c, v in changed.items()))
        print()

    exp.save()
    # train_analysis(GT 포함 보고용 덤프)는 experience 파일과 같은 위치·이름을 따른다.
    ana_path = os.path.join(os.path.dirname(exp_path),
                            os.path.basename(exp_path).replace("experience", "train_analysis", 1))
    if ana_path == exp_path:                 # 파일명에 'experience'가 없으면
        ana_path = os.path.splitext(exp_path)[0] + "_train_analysis.json"
    json.dump(analysis, open(ana_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[store] → {ana_path}")
    mb = np.mean([r[0] for r in rows]); mc = np.mean([r[1] for r in rows])
    better = sum(1 for r in rows if r[2] == "better")
    print(f"[dist] 평균 MAE  base={mb:.0f}  calib={mc:.0f}  ({(mc-mb):+.0f}, {100*(mc-mb)/mb:+.1f}%)")
    print(f"[dist] calib이 나은 window: {better}/{len(rows)}")
    print(f"[store] → memory/{os.path.basename(exp_path)} ({len(exp.cases)} cases)")
    u = llm.usage
    print(f"[usage] calls={u['calls']}  in={u['in']:,} tok  out={u['out']:,} tok")


if __name__ == "__main__":
    main()
