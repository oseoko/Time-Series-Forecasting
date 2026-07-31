"""
Chronos-2 frozen TSFM 얇은 래퍼.

한 window에 대해 median point forecast 하나를 돌려준다. 학습은 없다(frozen).

이 모델에는 covariate를 넣지 않는다. base 예측 Ŷ_base(B-1)는 타깃 과거값만으로 구하고,
공변량 반영은 전적으로 LLM 단계(B-2)가 맡는다. 그래서 잔차 GT − Ŷ_base 가
'공변량이 만든 몫'이 된다.
"""

import os
import numpy as np

MODEL_NAME = "amazon/chronos-2"


class TSFM:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None,
                 gpu_id: int | None = None):
        # GPU 선택은 torch import 이전에 확정해야 함
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        import torch  # noqa: E402
        from chronos import Chronos2Pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipe = Chronos2Pipeline.from_pretrained(model_name, device_map=device)

    def forecast(self, y_lookback, horizon: int = 7) -> np.ndarray:
        """
        y_lookback : (ctx,) 타깃 과거값 — lookback 구간
        return     : (horizon,) median point forecast = Ŷ_base
        """
        item = {"target": np.asarray(y_lookback, dtype="float32")}
        q, _ = self.pipe.predict_quantiles([item], prediction_length=horizon,
                                            quantile_levels=[0.5])
        # q[0] : (H, nq=1), n_variates=1 축 제거 후 median 열
        return q[0].cpu().numpy()[0][:, 0].astype("float64")
