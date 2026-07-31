"""
B-2b calibration prompt — attribution을 반영해 base forecast를 보정 → Ŷ_calib + R_calib.

LLM 출력이 곧 Ŷ_calib. 보정 크기는 calibration_scale_note로 제어한다.
"""

_CALIB_RULES = """# 보정 방법
attribution 라벨은 각 채널이 입국자 수에 기여하는 **방향과 상대적 강도**를 나타냅니다. 다만 이 라벨은
그 채널의 **전체 기여**를 뜻하며, 기준 예측 Ŷ_base는 타깃의 과거값을 통해 이미 그 기여의 일부를 간접적으로
머금고 있습니다. 따라서 라벨의 강도를 그대로 옮기지 말고, base가 아직 반영하지 못한 **추가분만큼만**
보수적으로 조정하세요. covariate knowledge가 알려주는 크기와 상한(cap)을 크기의 기준으로 삼되,
방향은 라벨로 정합니다. 살짝 밀되(nudge) 과잉 반응하지 말고, knowledge가 명시한 상한을 절대 넘지 마세요.
`neutral`이거나 입국자 수와 무관하다고 판정된 채널은 예측을 **전혀 움직여선 안 됩니다**.

모든 날을 같은 폭으로 옮기지 마세요. 각 채널의 조정은 그 공변량이 실제로 활성/상승한 **바로 그 날**에
배치하세요:
- future-known 채널(일별 값이 주어짐): 값이 0이 아니거나 높은 날만 조정하세요
  (3일차에 콘서트나 휴일이 있으면 3일차를 움직이고, 조용한 날은 base 그대로 두세요).
- past-only 채널(미래 미지): 특정 날짜가 아니라 전체 수준/추세만 살짝 밀 수 있습니다.

조정은 완만하게, base의 궤적 모양을 따르도록 유지하세요.

**검색된 과거 사례의 reflect(회고)를 반드시 읽고 그 교훈을 이번 보정에 반영하세요.** reflect는 그때 무엇이
틀렸는지를 GT와 대조해 기록한 것입니다 — 예컨대 "hotel_price를 strong_up으로 과대반영해 예측이 GT와
멀어졌다"고 적혀 있고 이번에도 같은 상황이면, 그 채널의 보정을 줄이거나 접으세요. verdict가 `worse`였던
사례가 이번과 비슷하면 더 보수적으로, `better`였던 사례의 배분을 참고하세요. reflect가 지목한 실수를
되풀이하지 마세요.

어느 날을 왜 움직였는지 밝히는 **한국어 2~3문장**의 reasoning을 함께 쓰세요.
JSON 키와 숫자는 그대로 두고, reasoning 본문만 한국어로 작성하세요."""

calibration_system = """# 역할
당신은 공변량 attribution을 사용해 {target}의 기준 예측 Ŷ_base를 horizon 구간에서 보정하는 전문가입니다.
"forecast"에 보정된 {horizon}일치 일별 값을 출력하세요.
""" + _CALIB_RULES

# 보정 크기 상한 — --scale-pct 5 → ±5%, 20 → ±20%. 0(기본)이면 이 블록을 아예 붙이지 않는다.
calibration_scale_note = """

# 보정 크기 규칙 (반드시 지킬 것)
각 날의 조정은 기준 예측 Ŷ_base의 **±{pct}% 이내**를 기본 상한으로 삼으세요. 분명하고 강한 신호
(strong 라벨 + 뚜렷한 이벤트가 그날 걸림)가 있는 날에 한해서만 그 이상으로 움직일 수 있고,
그 외의 날은 ±{pct}%를 넘기지 마세요. 당신의 출력은 사후 보정 없이 그대로 사용되므로, 이 상한을
스스로 지켜야 합니다."""

calibration_user = """# 기준 예측 Ŷ_base (1일차 ... {horizon}일차)
{y_base}

# ATTRIBUTION (채널별 기여 라벨)
#   future-known 채널은 [1일차,...,{horizon}일차] 날짜별 라벨 리스트 — up/down인 그 날만 그 방향으로 움직이세요.
#   past-only 채널은 window 전체 라벨 하나 — 특정 날짜가 아니라 전체 수준/추세만.
{attribution}

# ATTRIBUTION 근거 (R_attr)
{r_attr}

# covariate knowledge (움직여야 하는 채널만; 크기와 상한)
{knowledge_active}

# HORIZON 구간의 공변량 (일별 — 조정을 올바른 날에 배치하는 데 사용)
{future_cov}

# 유사한 과거 사례 (attribution으로 검색; base->calib 및 그 reflection)
{retrieved}"""
