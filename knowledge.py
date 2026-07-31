"""
STEP1 : 채널별 knowledge record 생성 (피처의 기여 관점). 출력 = {knowledge: 한 덩어리 문단}.

채널마다 train 관찰 전체를 한 프롬프트에 넣어 record를 1회 생성한다(채널당 LLM 1콜).
관찰 = 이전28일(타깃) + 피처값 + 예측(타깃만) + 실제 + 잔차.
학습 대상 = 잔차가 피처 상태에 따라 갈리는가 = 이 피처의 기여. 기여가 없으면 "관련 없음"도 결론.

base 예측은 채널과 무관하므로 window당 1회만 계산해 전 채널이 재사용한다.

실행: python knowledge.py [--channels sig2] [--out knowledge.json]
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import CHANNELS, TARGET_DESC_KO, MEMDIR, DATASET, resolve_channels
from engine.data import load, make_windows, HORIZON, STEP
from engine.features import fmt, fmtf
from engine.tsfm import TSFM
from engine.llm import LLM
from prompt.covariate_knowledge import knowledge_system, knowledge_user

KNOWLEDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "knowledge": {"type": "string",
                      "description": "ONE self-contained paragraph a forecaster could act on: what this feature does "
                                     "to arrivals; in which of its observable states actual arrivals run "
                                     "systematically above the target-only forecast (the feature contributes upward) "
                                     "or below it (downward); roughly how large that contribution is and where its "
                                     "cap lies; how confident the pattern is; and when its contribution is "
                                     "effectively nil. Flowing prose — no bullets, no headings, no repetition."},
    },
    "required": ["knowledge"],
    "additionalProperties": False,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=str, default=None,
                    help="all/sig4/sig2 또는 쉼표구분 채널명 (기본: 23채널 전부)")
    ap.add_argument("--out", type=str, default="knowledge.json", help="memory/ 아래 산출 파일명")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--model", type=str, default="gpt-5-mini")
    ap.add_argument("--effort", type=str, default="low", help="gpt-5 reasoning effort")
    ap.add_argument("--workers", type=int, default=8, help="채널 병렬 수")
    args = ap.parse_args()

    channels = resolve_channels(args.channels) if args.channels else list(CHANNELS)
    desc = json.load(open(os.path.join(DATASET, "channel_desc.json"), encoding="utf-8"))
    os.makedirs(MEMDIR, exist_ok=True)

    df, target, future_known, past_only = load()
    windows = make_windows(df, target, future_known, channels, horizon=args.horizon, step=STEP)
    tsfm = TSFM()
    llm = LLM(model=args.model, reasoning_effort=args.effort)
    total = len(windows)
    print(f"[knowledge] channels={channels}  situations={total}  device={tsfm.device}  "
          f"llm={llm.model} effort={args.effort}")
    base = [tsfm.forecast(w["y_lookback"], args.horizon) for w in windows]
    print(f"[knowledge] base forecasts cached ({len(base)} windows)\n")

    def obs_block(j, ch):
        """관찰 하나 — 이전28일 / 피처값 / 예측 / 실제 / 잔차."""
        w = windows[j]
        feat = f"이력 {fmtf(w['past'][ch])}"
        if ch in future_known and ch in w["future"]:
            feat += f" | horizon {fmtf(w['future'][ch])}"
        resid = fmt(np.asarray(w["y_true"]) - np.asarray(base[j]))
        return (f"[관찰 {j+1}]\n"
                f"  이전28일: {fmt(w['y_lookback'])}\n"
                f"  피처값: {feat}\n"
                f"  예측: {fmt(base[j])}\n"
                f"  실제: {fmt(w['y_true'])}\n"
                f"  잔차: {resid}")

    def build(ch):
        """관찰 전체를 한 프롬프트에 넣어 이 채널의 record를 생성."""
        obs = "\n".join(obs_block(j, ch) for j in range(len(windows)))
        sys_p = knowledge_system.format(target=TARGET_DESC_KO, channel=ch,
                                        description=desc.get(ch, ""))
        u = knowledge_user.format(total=total, observations=obs)
        try:
            rec = llm.parse(sys_p, u, schema=KNOWLEDGE_SCHEMA, schema_name="covariate_knowledge")
        except Exception as e:
            rec = {"knowledge": f"(fail: {e})"}
            print(f"  [warn] {ch}: {e}", flush=True)
        print(f"  [done] {ch}", flush=True)
        return ch, rec

    path = os.path.join(MEMDIR, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)      # --out 이 하위폴더면 만든다
    cur = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(channels))) as ex:
        for ch, rec in ex.map(build, channels):
            cur[ch] = rec
    for ch in channels:
        print(f"===== K[{ch}] =====")
        print(json.dumps(cur[ch], ensure_ascii=False, indent=2))
        print()
    json.dump(cur, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[store] → memory/{args.out} (누적 {len(cur)} channels)")


if __name__ == "__main__":
    main()
