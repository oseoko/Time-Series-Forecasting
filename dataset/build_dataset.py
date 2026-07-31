"""
데이터셋 재구축 — 타깃 1 + 공변량 23채널.

공변량(23): hotel_price / fx_usd,cny,jpy / hol_cn,jp,us,kr(주말+공휴일) / hol_kr_wd(평일 공휴일만) /
            concert_cap / flight_cn,jp,us,total / 서울기상 8(AVGTA,SUMRN,AVGWS,AVGRHM,AVGPS,SUMGSR,DDMES,AVGTCA) /
            is_friday
future-known(11): hol_* 4, hol_kr_wd, concert_cap, flight_* 4, is_friday  — 달력·스케줄이라 미래 값이 확정
past-only(12):    hotel_price, fx_* 3, w_seoul_* 8                        — 미래 값이 없음
출력: master_daily.parquet/csv, covariate_roles.json, channel_desc.json
      (parquet/csv = 파이프라인 입력, covariate_roles = 채널 역할, channel_desc = 프롬프트에 쓰는 채널 설명문)

원본 xlsx는 raw/ 에 둔다. parquet이 이미 있으면 이 스크립트를 돌릴 필요는 없다.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")                                       # 원본 xlsx 둘: 수요예측 원본 + 항공편수
# 파일명 한글이 NFD(분해형)로 저장될 수 있어 NFC 정규화 후 매칭해야 부분문자열이 맞는다.
import unicodedata
def _has(name, kw):
    return kw in unicodedata.normalize("NFC", os.path.basename(name))
_xlsx = glob.glob(os.path.join(RAW, "*.xlsx"))
if not _xlsx:
    raise SystemExit(f"원본 xlsx를 찾을 수 없습니다: {RAW}/*.xlsx\n"
                     "재구축이 목적이 아니라면 이 스크립트는 실행할 필요가 없습니다 "
                     "(master_daily.parquet이 이미 있습니다).")
XLS = pd.ExcelFile(next(f for f in _xlsx if not _has(f, "항공")))     # 수요예측 원본(타깃·환율·기상·휴일·콘서트)
FLIGHT_XLS = next((f for f in _xlsx if _has(f, "항공")), None)         # 항공사·공항별 항공편수 (별도 파일)
SEOUL = 108
WCOLS = ["AVGTA", "SUMRN", "AVGWS", "AVGRHM", "AVGPS", "SUMGSR", "DDMES", "AVGTCA"]
HOL_COUNTRIES = ["CN", "JP", "US", "KR"]
FX_MAP = {"원/미국달러(매매기준율)": "fx_usd", "원/위안(매매기준율)": "fx_cny",
          "원/일본엔(100엔)": "fx_jpy"}


def to_date(s, fmt="%Y%m%d"):
    return pd.to_datetime(s.astype("Int64").astype(str), format=fmt, errors="coerce")


def parse_capacity(text):
    if not isinstance(text, str):
        return np.nan
    nums = re.findall(r"[\d,]+", text.replace(" ", ""))
    vals = [int(n.replace(",", "")) for n in nums if n.replace(",", "").isdigit() and int(n.replace(",", "")) >= 100]
    return float(max(vals)) if vals else np.nan


# 1) target + index
tgt = pd.read_excel(XLS, "외국인 입국자 수")
tgt["date"] = to_date(tgt["DT"])
y = tgt.set_index("date")["FOREIGNER"].sort_index()
idx = pd.date_range(y.index.min(), y.index.max(), freq="D"); idx.name = "date"
df = pd.DataFrame(index=idx)
df["FOREIGNER"] = y.reindex(idx)
# 요약에서 누락 명시된 날짜 → 선형보간
for dt in ["2025-12-29", "2025-12-30", "2026-04-27"]:
    if pd.Timestamp(dt) in df.index:
        df.loc[pd.Timestamp(dt), "FOREIGNER"] = np.nan
df["FOREIGNER"] = df["FOREIGNER"].interpolate().ffill().bfill()

# 2) hotel_price
ht = pd.read_excel(XLS, "호텔 평균 가격"); ht["date"] = to_date(ht["check_in"])
df["hotel_price"] = ht.groupby("date")["y_mean"].mean().reindex(idx).ffill().bfill()

# 3) fx
fx = pd.read_excel(XLS, "환율"); fx["date"] = to_date(fx["TIME"])
fx["col"] = fx["ITEM_NAME1"].map(FX_MAP)
w = fx.pivot_table(index="date", columns="col", values="DATA_VALUE", aggfunc="last").reindex(idx).ffill().bfill()
for c in ["fx_usd", "fx_cny", "fx_jpy"]:
    df[c] = w[c]

# 4) 서울 기상 8
wx = pd.read_excel(XLS, "종관기상관측정보"); wx["date"] = to_date(wx["TM"])
s = wx[wx["STNID"] == SEOUL].set_index("date")[WCOLS].reindex(idx)
s["DDMES"] = s["DDMES"].fillna(0.0)                 # 적설: 없는 날 0
s = s.interpolate(limit_direction="both")
for c in WCOLS:
    df[f"w_seoul_{c}"] = s[c]

# 5) holidays (주말 OR 해당국 공휴일)
hol = pd.read_excel(XLS, "휴일"); hol["date"] = to_date(hol["date"])
is_weekend = idx.dayofweek >= 5
for c in HOL_COUNTRIES:
    days = set(hol.loc[hol["country"] == c, "date"])
    df[f"hol_{c.lower()}"] = (is_weekend | idx.isin(days)).astype(int)
# 평일에 걸린 한국 관공서 공휴일만 (주말 제외). hol_kr 은 '주말 OR 공휴일'이라 주말과 대부분
# 겹치므로, 주말을 뺀 이 채널이 있어야 공휴일 신호가 주말 신호와 분리된다.
# 노동절(근로자의 날)은 관공서 공휴일이 아니므로 제외.
kr = hol[hol["country"] == "KR"]
kr_days = set(kr.loc[~kr["name"].astype(str).str.contains("노동절"), "date"])
df["hol_kr_wd"] = (idx.isin(kr_days) & ~is_weekend).astype(int)
# 금요일 여부 — 주말(hol_*)에는 안 잡히는 요일이라 따로 둔다.
df["is_friday"] = (idx.dayofweek == 4).astype(int)

# 6) concert_cap
con = pd.read_excel(XLS, "콘서트 정보"); con["cap"] = con["CAPACITY"].apply(parse_capacity)
cap = pd.Series(0.0, index=idx)
for _, r in con.iterrows():
    rng = pd.date_range(r["START_DT"], r["END_DT"], freq="D"); rng = rng[rng.isin(idx)]
    if not np.isnan(r["cap"]):
        cap.loc[rng] += r["cap"]
df["concert_cap"] = cap

# 7) 항공편수 (별도 xlsx) — 출발 공항 코드로 국가를 판별해 cn/jp/us + 총합으로 집계.
#    한국 공항 출발분은 국내선이므로 제외.
FLIGHT_CN = {"PVG", "TAO", "PEK", "CAN", "DLC", "SZX", "SHE", "TSN", "YNT", "NKG", "HGH", "XIY",
             "CTU", "CKG", "WUH", "CGO", "XMN", "KMG", "CSX", "TNA", "HRB", "FOC", "NNG", "KWL",
             "HAK", "SYX", "URC", "LHW", "INC", "HFE", "TYN", "CGQ", "WNZ", "ZUH", "JJN", "LYG",
             "SJW", "HET", "YNJ", "WEH", "HKG", "MFM"}          # 홍콩·마카오 포함
FLIGHT_JP = {"KIX", "NRT", "FUK", "CTS", "NGO", "OKA", "HND", "KKJ", "HIJ", "KMJ", "KOJ", "TAK",
             "FSZ", "KMQ", "AOJ", "NGS", "OIT", "MYJ", "TOY", "ISG", "ITM", "SDJ", "KIJ", "AXT",
             "GAJ", "HKD", "OKJ", "UBJ", "MMJ", "KMI", "TKS", "IZO", "YGJ", "HSG", "OKD"}
FLIGHT_US = {"LAX", "SFO", "JFK", "SEA", "GUM", "ATL", "ORD", "DFW", "IAD", "EWR", "HNL", "DTW",
             "BOS", "IAH", "LAS", "DEN", "SPN"}
FLIGHT_KR = {"ICN", "GMP", "PUS", "CJU", "TAE", "KWJ", "RSU", "USN", "KUV", "HIN", "WJU",
             "YNY", "MWX", "KPO", "CHN"}                        # 국내선 → 제외

FLIGHT_COLS = []
if FLIGHT_XLS:
    fl = pd.read_excel(FLIGHT_XLS)
    fl["date"] = to_date(fl["flight_date"])
    fl = fl[~fl["AIRPORTCODE"].isin(FLIGHT_KR)]                 # 국내선 제외

    def _ctry(code):
        if code in FLIGHT_CN: return "cn"
        if code in FLIGHT_JP: return "jp"
        if code in FLIGHT_US: return "us"
        return "etc"
    fl["ctry"] = fl["AIRPORTCODE"].map(_ctry)
    for c in ["cn", "jp", "us"]:
        col = f"flight_{c}"
        df[col] = (fl[fl["ctry"] == c].groupby("date")["flight_count"].sum()
                   .reindex(idx).fillna(0.0))
        FLIGHT_COLS.append(col)
    # 전체 국제선 편수 (국내선은 이미 제외됨) — etc는 별도 채널로 두지 않음
    df["flight_total"] = fl.groupby("date")["flight_count"].sum().reindex(idx).ffill().bfill()
    FLIGHT_COLS.append("flight_total")
else:
    print("[warn] 항공편 xlsx 없음 → flight_* 생략")

# 역할 — horizon 구간의 값을 프롬프트에 넣을 수 있는 채널만 future-known.
# 휴일·콘서트·항공편·요일은 달력/스케줄이라 미래 값이 확정된다.
# 환율·호텔가·기상은 미래 값을 알 수 없으므로 past-only(lookback 구간의 수준/추세만 사용).
future_known = (["hol_cn", "hol_jp", "hol_us", "hol_kr", "hol_kr_wd", "concert_cap"]
                + FLIGHT_COLS + ["is_friday"])
past_only = ["hotel_price", "fx_usd", "fx_cny", "fx_jpy"] + [f"w_seoul_{c}" for c in WCOLS]
assert len(future_known) + len(past_only) == 19 + len(FLIGHT_COLS)

df.index.name = "date"
df.to_parquet(os.path.join(HERE, "master_daily.parquet"))
df.to_csv(os.path.join(HERE, "master_daily.csv"))
json.dump({"TARGET": "FOREIGNER", "FUTURE_KNOWN": ", ".join(future_known),
           "PAST_ONLY": ", ".join(past_only)},
          open(os.path.join(HERE, "covariate_roles.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
desc = {
    "hotel_price": "일별 평균 호텔 가격 (원).",
    "fx_usd": "원/달러 환율 (USD 1당 원).",
    "fx_cny": "원/위안 환율 (중국 위안 1당 원).",
    "fx_jpy": "원/엔 환율 (일본 엔 100당 원).",
    "hol_cn": "중국: 주말 또는 공휴일 여부 (1/0). 미래 값 알려짐.",
    "hol_jp": "일본: 주말 또는 공휴일 여부 (1/0). 미래 값 알려짐.",
    "hol_us": "미국: 주말 또는 공휴일 여부 (1/0). 미래 값 알려짐.",
    "hol_kr": "한국: 주말 또는 공휴일 여부 (1/0). 미래 값 알려짐.",
    "hol_kr_wd": "한국: 평일에 걸린 관공서 공휴일 여부, 주말·노동절 제외 (1/0). 미래 값 알려짐.",
    "concert_cap": "그날 예정된 콘서트 총 수용인원 (없으면 0). 미래 값 알려짐.",
    "flight_cn": "중국발(홍콩·마카오 포함) 일별 입국 항공편수. 미래 값 알려짐(스케줄).",
    "flight_jp": "일본발 일별 입국 항공편수. 미래 값 알려짐(스케줄).",
    "flight_us": "미국발(괌·사이판 포함) 일별 입국 항공편수. 미래 값 알려짐(스케줄).",
    "flight_total": "한국행 일별 총 입국 국제선 항공편수 (국내선 제외). 미래 값 알려짐(스케줄).",
    "w_seoul_AVGTA": "서울 일평균 기온 (℃). 미래 값은 알 수 없음(과거 관측치만 사용).",
    "w_seoul_SUMRN": "서울 일 강수량 합계 (mm). 미래 값은 알 수 없음(과거 관측치만 사용).",
    "w_seoul_AVGWS": "서울 일평균 풍속 (m/s). 미래 값은 알 수 없음(과거 관측치만 사용).",
    "w_seoul_AVGRHM": "서울 일평균 상대습도 (%). 미래 값은 알 수 없음(과거 관측치만 사용).",
    "w_seoul_AVGPS": "서울 일평균 해면기압 (hPa). 미래 값은 알 수 없음(과거 관측치만 사용).",
    "w_seoul_SUMGSR": "서울 일 합계 일사량 (MJ/m²). 미래 값은 알 수 없음(과거 관측치만 사용).",
    "w_seoul_DDMES": "서울 일 최심 적설 (cm). 미래 값은 알 수 없음(과거 관측치만 사용).",
    "w_seoul_AVGTCA": "서울 일평균 전운량 (0~10). 미래 값은 알 수 없음(과거 관측치만 사용).",
    "is_friday": "금요일 여부 (1/0). 미래 값 알려짐(달력). TSFM이 과거 숫자 패턴만으로는 못 잡는 금요일 수요 편향을 LLM의 요일 사전지식으로 보정하기 위한 변수.",
}
json.dump(desc, open(os.path.join(HERE, "channel_desc.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print(f"rows={len(df)}  {df.index.min().date()}~{df.index.max().date()}")
print(f"future_known({len(future_known)}): {future_known}")
print(f"past_only({len(past_only)}): {past_only}")
print("NaN:", int(df.isna().sum().sum()))
print(df[["FOREIGNER", "w_seoul_SUMGSR", "w_seoul_DDMES", "hol_cn", "concert_cap"]].describe().round(1).to_string())
