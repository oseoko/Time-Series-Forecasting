"""
데이터 로더 + 학습 구간 sliding window (lean, 함수형).

master_daily.parquet + covariate_roles.json → 학습 window 리스트(dict).
leakage 방지: 뒤쪽 홀드아웃(TEST_LEN)은 제외하고 학습 구간에서만 window 생성.
"""

import json
import os

import numpy as np
import pandas as pd

from config import DATASET   # fs/dataset

CTX_LEN = 28       # context 길이
HORIZON = 7        # 예측 길이
STEP = 7           # window 간격 (비겹침)
TEST_LEN = 42      # 뒤쪽 홀드아웃 (6주 = 7의 배수; train:test ≈ 8:2)


def load():
    """(df, target, future_known, past_only) 반환."""
    df = pd.read_parquet(os.path.join(DATASET, "master_daily.parquet")).reset_index()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    roles = json.load(open(os.path.join(DATASET, "covariate_roles.json"), encoding="utf-8"))
    future_known = [c.strip() for c in roles["FUTURE_KNOWN"].split(",")]
    past_only = [c.strip() for c in roles["PAST_ONLY"].split(",")]
    return df, roles["TARGET"], future_known, past_only


def make_windows(df, target, future_known, channels,
                 ctx_len=CTX_LEN, horizon=HORIZON, step=STEP,
                 test_len=TEST_LEN, max_windows=None, region="train"):
    """region='train' → [ctx_len, n-test_len), 'test' → 홀드아웃 [n-test_len, n).
    window dict: date, y_lookback(ctx,), y_true(H,), past{ch:(ctx,)}, future{ch:(H,)} — covariate 원값."""
    n = len(df)
    train_end = n - test_len
    y = df[target].values.astype("float64")
    if region == "train":
        # train cutoff을 train_end에 뒤맞춤: 나머지를 맨 앞 lookback 구간으로 밀어 예측구간 구멍 방지
        cutoffs = list(range(train_end - horizon, ctx_len - 1, -step))[::-1]
    else:
        cutoffs = list(range(train_end, n - horizon + 1, step))               # 홀드아웃 cutoff
    windows = []
    for c in cutoffs:
        ctx, fut = slice(c - ctx_len, c), slice(c, c + horizon)
        past = {ch: df[ch].values[ctx].astype("float64") for ch in channels}
        future = {ch: df[ch].values[fut].astype("float64")
                  for ch in channels if ch in future_known}
        windows.append({
            "date": str(df["date"].iloc[c].date()),
            "y_lookback": y[ctx],
            "y_true": y[fut],
            "past": past,
            "future": future,
        })
        if max_windows and len(windows) >= max_windows:
            break
    return windows
