"""
얇은 LLM 래퍼.
  parse(system, user, schema) -> dict    structured output — 스키마로 필드·타입 강제

키는 환경변수 또는 패키지 루트의 .env 에서 읽음:
  OPENAI_API_KEY(필수), OPENAI_MODEL(선택, 기본 gpt-5-mini).
"""

import json
import os

from config import HERE   # 패키지 루트 (.env 위치)


def _load_env():
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LLM:
    def __init__(self, model: str | None = None, reasoning_effort: str | None = None,
                 seed: int | None = 42):
        _load_env()
        self.reasoning_effort = reasoning_effort   # gpt-5 계열: minimal/low/medium/high
        self.seed = seed                           # 재현성(best-effort) — 같은 입력이면 변동 최소화
        self.usage = {"in": 0, "out": 0, "calls": 0}   # 토큰 실측
        from openai import OpenAI  # lazy

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 없음 — 패키지 루트의 .env 에 OPENAI_API_KEY를 설정. "
                ".env.example 참고."
            )
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        self.client = OpenAI(api_key=api_key)

    def _extra(self):
        extra = {}
        if self.seed is not None:
            extra["seed"] = self.seed
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        return extra

    def _tally(self, resp):
        u = getattr(resp, "usage", None)
        if u:
            self.usage["in"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["out"] += getattr(u, "completion_tokens", 0) or 0
            self.usage["calls"] += 1

    def parse(self, system: str, user: str, schema: dict | None = None,
              schema_name: str = "record") -> dict:
        rf = ({"type": "json_schema",
               "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
              if schema else {"type": "json_object"})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format=rf,
            **self._extra(),
        )
        self._tally(resp)
        return json.loads(resp.choices[0].message.content or "{}")
