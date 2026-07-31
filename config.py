"""공통 설정 — 경로, 대상 채널, 타깃 설명."""

import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset")
MEMDIR = os.path.join(HERE, "memory")

# 어트리뷰션 대상 채널 (23)
#   숙박비 + 환율3 + 휴일4(주말+공휴일) + 한국 평일공휴일 + 콘서트 + 항공편4(국가별+전체) + 서울기상8 + 금요일
CHANNELS = [
    "hotel_price", "fx_usd", "fx_cny", "fx_jpy",
    "hol_cn", "hol_jp", "hol_us", "hol_kr", "hol_kr_wd", "concert_cap",
    "flight_cn", "flight_jp", "flight_us", "flight_total",
    "w_seoul_AVGTA", "w_seoul_SUMRN", "w_seoul_AVGWS", "w_seoul_AVGRHM",
    "w_seoul_AVGPS", "w_seoul_SUMGSR", "w_seoul_DDMES", "w_seoul_AVGTCA",
    "is_friday",
]

# 채널 서브셋 프리셋 — 잔차에 대한 효과크기(Cohen's d)로 고른 상위 채널만 남긴 조합.
#   all  : 23채널 전부
#   sig4 : 상위 4 (평일공휴일·금요일·항공총합·콘서트)
#   sig2 : d≥0.5 인 2 (평일공휴일·금요일) — 기본값
CHANNEL_SETS = {
    "all":  CHANNELS,
    "sig4": ["hol_kr_wd", "is_friday", "flight_total", "concert_cap"],
    "sig2": ["hol_kr_wd", "is_friday"],
}


def resolve_channels(spec):
    """'all'|'sig4'|'sig2' 프리셋 또는 쉼표구분 채널명 → 채널 리스트."""
    if spec in CHANNEL_SETS:
        return list(CHANNEL_SETS[spec])
    chs = [c.strip() for c in spec.split(",") if c.strip()]
    bad = [c for c in chs if c not in CHANNELS]
    if bad:
        raise SystemExit(f"[config] 알 수 없는 채널: {bad}\n사용 가능: {CHANNELS}")
    if not chs:
        raise SystemExit("[config] --channels 가 비어 있음")
    return chs


def _custom_tag(chs):
    """쉼표구분 채널 목록 → 조합마다 고유한 태그."""
    short = [c.replace("w_seoul_", "w").replace("flight_", "fl").replace("hol_", "h")
             for c in chs]
    name = "-".join(short)
    if len(name) <= 40:
        return name
    h = hashlib.sha1(",".join(chs).encode()).hexdigest()[:6]
    return f"ch{len(chs)}_{h}"


def run_tag(chan_spec, no_knowledge=False, no_experience=False):
    """실행 조건 → 산출물 파일명 태그 (arm끼리 덮어쓰지 않도록)."""
    t = chan_spec if chan_spec in CHANNEL_SETS else _custom_tag(resolve_channels(chan_spec))
    if no_knowledge:
        t += "_nok"
    if no_experience:
        t += "_noexp"
    return t


# 프롬프트에 넣는 타깃 설명
TARGET_DESC_KO = "일별 방한 외국인 입국자 수"
