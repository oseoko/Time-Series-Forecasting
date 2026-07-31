"""요일별 MAE 메트릭 — test_results의 horizon 날짜를 요일로 매핑해 base vs calib 오차를 요일별로 집계.

'요일별로 얼마나 달라지는가'를 본다: 어느 요일에서 예측이 크게 틀리고(base MAE), 보정이 그걸 얼마나
고치는지(calib MAE, 개선%). 참고로 요일별 실제 수요(GT 평균)도 함께 — 수요 자체의 요일 편차.

실행: python analysis/weekday_mae.py memory/ablation/all_base/test_results_*.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.data import load

WD = ["월", "화", "수", "목", "금", "토", "일"]


def weekday_table(res_path):
    r = json.load(open(res_path, encoding="utf-8"))
    df, _, _, _ = load()
    dts = pd.to_datetime(df["date"])
    idx = {str(d.date()): i for i, d in enumerate(dts)}
    # 요일별 누적: 절대오차(base/calib), GT
    ab = {i: [] for i in range(7)}   # base abs err
    ac = {i: [] for i in range(7)}   # calib abs err
    ag = {i: [] for i in range(7)}   # gt
    for w in r:
        c = idx[w["date"]]
        yb, yc, gt = np.array(w["y_base"]), np.array(w["y_calib"]), np.array(w["gt"])
        for t in range(len(gt)):
            d = int(dts.iloc[c + t].weekday())
            ab[d].append(abs(yb[t] - gt[t])); ac[d].append(abs(yc[t] - gt[t])); ag[d].append(gt[t])
    rows = []
    for d in range(7):
        if not ab[d]:
            continue
        mb, mc = np.mean(ab[d]), np.mean(ac[d])
        rows.append((WD[d], len(ab[d]), mb, mc, 100 * (mc - mb) / mb if mb else 0, np.mean(ag[d])))
    return rows


def main():
    paths = sys.argv[1:] or ["memory/v3_wx_pastonly/test_results_all_abs_cap5_experience.json"]
    for p in paths:
        rows = weekday_table(p)
        print(f"\n=== {p} ===")
        print(f"{'요일':<4}{'n':>4}{'base MAE':>11}{'calib MAE':>11}{'Δ%':>8}{'수요(GT평균)':>14}")
        for wd, n, mb, mc, dp, g in rows:
            print(f"{wd:<4}{n:>4}{mb:>11.0f}{mc:>11.0f}{dp:>+8.1f}{g:>14.0f}")
        # 전체
        allb = np.mean([mb for _, _, mb, _, _, _ in rows])
        allc = np.mean([mc for _, _, _, mc, _, _ in rows])
        print(f"{'평균':<4}{'':>4}{allb:>11.0f}{allc:>11.0f}{100*(allc-allb)/allb:>+8.1f}")


if __name__ == "__main__":
    main()
