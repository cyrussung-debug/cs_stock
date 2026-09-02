# -*- coding: utf-8 -*-
# ============================================================
#  종목선정 필살기 스캐너 (Streamlit)
#  - 아이폰/아이패드/PC 브라우저에서 동작하는 반응형 웹앱
#  - 책 「종목선정」(김정수)의 매수/회피 패턴을 조건식으로 근사 구현
#
#  로컬 실행:  pip install -r requirements.txt  →  streamlit run app.py
#  클라우드 배포: README_배포가이드.md 참고 (Streamlit Community Cloud)
# ============================================================

import time
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 국내 데이터 라이브러리 (미설치 환경 대비 예외 처리)
try:
    import FinanceDataReader as fdr
    KR_DATA_AVAILABLE = True
except Exception:
    KR_DATA_AVAILABLE = False

st.set_page_config(
    page_title="종목선정 스캐너",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",   # 모바일에서는 접힌 상태로 시작
)

# ---------------- 모바일(아이폰/아이패드 Safari) 반응형 CSS ----------------
st.markdown("""
<style>
@media (max-width: 768px) {
    .block-container {padding: 0.8rem 0.6rem 2rem 0.6rem;}
    h1 {font-size: 1.4rem !important;}
    div[data-testid="stMetricValue"] {font-size: 1.05rem;}
    div[data-testid="stDataFrame"] {font-size: 12px;}
}
div[data-testid="stMetricValue"] {font-weight: 700;}
.stButton > button {height: 3em; font-size: 1.05rem; font-weight: 700;}
.small-note {color:#888; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. 데이터 수집 계층 (국내 / 해외 분리)
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_kr_universe(market: str) -> pd.DataFrame:
    """국내 전 종목 시세 스냅샷 (FinanceDataReader.StockListing)
    반환 컬럼: Code, Name, Close, ChagesRatio(등락률), Volume, Marcap"""
    if not KR_DATA_AVAILABLE:
        raise RuntimeError("finance-datareader 가 설치되어 있지 않습니다.")
    df = fdr.StockListing(market)           # 'KOSPI' 또는 'KOSDAQ'
    if df is None or df.empty:
        raise RuntimeError(f"{market} 종목 목록을 가져오지 못했습니다.")
    # 라이브러리 컬럼명 오타(ChagesRatio) 호환 처리
    if "ChagesRatio" in df.columns and "ChangesRatio" not in df.columns:
        df = df.rename(columns={"ChagesRatio": "ChangesRatio"})
    need = {"Code", "Name", "Close", "ChangesRatio", "Volume"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"예상 컬럼 누락: {missing}")
    return df[list(need | {"Marcap"} & set(df.columns))].copy()


def get_kr_top_gainers(market: str, top_n: int, min_volume: int, min_price: int) -> pd.DataFrame:
    """당일 등락률 상위 종목 (책의 1단계: 등락률 상위 + 거래량/가격 필터)"""
    df = get_kr_universe(market)
    df = df[(df["Volume"] >= min_volume) & (df["Close"] >= min_price)]
    df = df.sort_values("ChangesRatio", ascending=False).head(top_n)
    return df.rename(columns={"Code": "Ticker", "Name": "종목명",
                              "ChangesRatio": "등락률", "Close": "종가", "Volume": "거래량"})


@st.cache_data(ttl=3600, show_spinner=False)
def get_kr_ohlcv(ticker: str, lookback_days: int = 600) -> pd.DataFrame:
    """국내 종목 일봉 OHLCV"""
    end = datetime.today()
    start = end - timedelta(days=int(lookback_days * 1.6))
    df = fdr.DataReader(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        raise RuntimeError("시세 데이터 없음")
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(lookback_days)


@st.cache_data(ttl=3600, show_spinner=False)
def get_us_ohlcv(ticker: str, lookback_days: int = 600) -> pd.DataFrame:
    """해외(미국) 종목 일봉 OHLCV - yfinance"""
    df = yf.Ticker(ticker).history(period=f"{int(lookback_days * 1.6)}d", interval="1d")
    if df is None or df.empty:
        raise RuntimeError("시세 데이터 없음")
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(lookback_days)


@st.cache_data(ttl=1800, show_spinner=False)
def get_us_top_gainers(universe: tuple, min_volume: int, min_price: float) -> pd.DataFrame:
    """워치리스트 내 당일 등락률 상위 (미국은 무료 전체시장 스크리너가 없어 워치리스트 기준)"""
    if not universe:
        return pd.DataFrame()
    data = yf.download(list(universe), period="5d", interval="1d",
                       group_by="ticker", progress=False, threads=True)
    rows = []
    for t in universe:
        try:
            sub = (data[t] if len(universe) > 1 else data).dropna()
            if len(sub) < 2:
                continue
            prev_close, close = sub["Close"].iloc[-2], sub["Close"].iloc[-1]
            vol = sub["Volume"].iloc[-1]
            if vol >= min_volume and close >= min_price:
                rows.append({"Ticker": t, "종목명": t, "종가": close,
                             "등락률": (close - prev_close) / prev_close * 100, "거래량": vol})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("등락률", ascending=False) if rows else pd.DataFrame()


# ============================================================
# 2. 종목선정 알고리즘 엔진 (책의 원칙 → 조건식)
#    ※ 시각적 패턴은 정량 근사치이며, 최종 판단은 차트 확인 필수
# ============================================================

def compute_base_levels(df: pd.DataFrame) -> dict:
    """원바닥(장기 최저 지지) / 판바닥(최근 120일 지지) 근사"""
    base = float(df["Low"].min())
    recent = float(df["Low"].tail(120).min())
    return {"원바닥": base, "판바닥": max(base, recent)}


def classify_zone(price, base_price, period_high) -> str:
    r_base = price / base_price if base_price > 0 else np.nan
    r_high = price / period_high if period_high > 0 else np.nan
    if r_high >= 0.9 or r_base > 4:
        return "고점"
    if r_base <= 2:
        return "저점"
    return "중점"


def detect_jangdae_yangbong(df, gain_th=0.10, vol_mult=2.0, wick_th=0.3) -> bool:
    """장대양봉: 종가>시가, 몸통 +10%↑, 거래량 전일比 200%↑, 윗꼬리 짧음, 전고점 돌파"""
    if len(df) < 2:
        return False
    t, p = df.iloc[-1], df.iloc[-2]
    if t["Open"] <= 0 or p["Volume"] <= 0:
        return False
    body_gain = (t["Close"] - t["Open"]) / t["Open"]
    vol_ratio = t["Volume"] / p["Volume"]
    rng = t["High"] - t["Low"]
    upper_wick = (t["High"] - t["Close"]) / rng if rng > 0 else 1
    prior_high = df["High"].iloc[-21:-1].max() if len(df) > 21 else df["High"].iloc[:-1].max()
    return bool(t["Close"] > t["Open"] and body_gain >= gain_th and vol_ratio >= vol_mult
                and upper_wick < wick_th and t["Close"] > prior_high)


def detect_dolpa(df, box_days=120, vol_mult=1.8) -> bool:
    """돌파: 장기 박스권(저변동) 상단을 거래량 급증과 함께 상향 돌파"""
    if len(df) < box_days + 5:
        return False
    box = df.iloc[-(box_days + 1):-1]
    t = df.iloc[-1]
    box_vol = box["Close"].std() / box["Close"].mean()
    return bool(t["Close"] > box["Close"].max() and t["Volume"] >= box["Volume"].mean() * vol_mult
                and box_vol < 0.25)


def detect_nulimmok(df, lookback=20, ma_period=20) -> bool:
    """눌림목: 최근 장대양봉 이력 + 20일선 부근 얕은 조정 + 거래량 감소 + 저점 미붕괴"""
    if len(df) < lookback + ma_period:
        return False
    had_signal = any(detect_jangdae_yangbong(df.iloc[:i + 1]) for i in range(len(df) - lookback, len(df) - 1))
    ma = df["Close"].rolling(ma_period).mean()
    t = df.iloc[-1]
    near_ma = abs(t["Close"] - ma.iloc[-1]) / ma.iloc[-1] < 0.05
    vol_declining = df["Volume"].iloc[-5:].mean() < df["Volume"].iloc[-20:-5].mean()
    not_broken = t["Close"] > df["Low"].iloc[-lookback:].min() * 0.97
    return bool(had_signal and near_ma and vol_declining and not_broken)


def detect_goganori(df, tight_days=5, range_th=0.05) -> bool:
    """고가놀이: 60일 고점의 90%↑ 유지, 좁은 범위, 거래량 감소 횡보"""
    if len(df) < 60:
        return False
    r = df.iloc[-tight_days:]
    high60 = df["Close"].iloc[-60:].max()
    tight = (r["High"].max() - r["Low"].min()) / r["Close"].mean() < range_th
    near_high = r["Close"].mean() >= high60 * 0.9
    vol_declining = r["Volume"].mean() < df["Volume"].iloc[-30:-tight_days].mean()
    return bool(tight and near_high and vol_declining)


def detect_overextended(df, base_price, mult_th=4.0) -> bool:
    """앞/뒤폭탄 근사: 원바닥 대비 과도한 배수 상승(과열)"""
    return bool(base_price > 0 and df["Close"].iloc[-1] / base_price > mult_th)


def detect_downtrend(df, ma_s=20, ma_l=60) -> bool:
    """내리막폭포/계단/외봉 근사: 이평 역배열 + 저점 갱신 하락"""
    if len(df) < ma_l + 10:
        return False
    s, l = df["Close"].rolling(ma_s).mean(), df["Close"].rolling(ma_l).mean()
    down = s.iloc[-1] < l.iloc[-1] and s.iloc[-1] < s.iloc[-10]
    lows = df["Low"].iloc[-40:]
    return bool(down and lows.iloc[-10:].min() < lows.iloc[:-10].min())


def detect_choppy(df, window=40) -> bool:
    """톱니바퀴 근사: 변동성 크지만 추세 기울기 ≈ 0"""
    if len(df) < window:
        return False
    r = df["Close"].iloc[-window:]
    slope = np.polyfit(np.arange(len(r)), r.values, 1)[0]
    return bool(abs(slope) / r.mean() < 0.001 and r.pct_change().std() > 0.03)


def detect_repeated_resistance(df, window=90, band=0.03) -> bool:
    """다중턱/다중봉/다중꼬리/쌍봉/쌍꼬리 근사: 같은 저항대 반복 실패"""
    if len(df) < window:
        return False
    highs = df["High"].iloc[-window:]
    peak = highs.max()
    return bool((highs >= peak * (1 - band)).sum() >= 2 and df["Close"].iloc[-1] < peak * (1 - band))


def detect_top_distribution(df, window=20, band=0.05) -> bool:
    """고점횡보(분산) 근사: 전고점 부근 장기 정체 + 거래량 안 줄어듦"""
    if len(df) < window * 2:
        return False
    r = df.iloc[-window:]
    in_band = (r["Close"] >= df["Close"].max() * (1 - band)).sum()
    vol_hold = r["Volume"].mean() >= df["Volume"].iloc[-window * 2:-window].mean() * 0.9
    return bool(in_band >= window * 0.6 and vol_hold)


def analyze_ticker(ticker, name, df, is_light_cap=True, gain_th=0.10, vol_mult=2.0,
                    overheat_mult=4.0) -> dict | None:
    """Zone 분류 + 매수/회피 패턴 판정 + 이익실현·손절 기준가 산출"""
    df = df.dropna()
    if len(df) < 60:
        return None
    base = compute_base_levels(df)
    base_price = base["원바닥"]
    price = float(df["Close"].iloc[-1])
    zone = classify_zone(price, base_price, float(df["Close"].max()))

    buy, avoid = [], []
    if detect_jangdae_yangbong(df, gain_th, vol_mult):
        buy.append(f"{zone} 장대양봉")
    if detect_dolpa(df):
        buy.append(f"{zone} 돌파")
    if detect_nulimmok(df):
        buy.append(f"{zone} 눌림목")
    if detect_goganori(df):
        buy.append(f"{zone} 고가놀이")

    if detect_overextended(df, base_price, mult_th=overheat_mult):
        avoid.append(f"앞/뒤폭탄(과열, 원바닥×{overheat_mult:.1f} 초과)")
    if detect_downtrend(df):
        avoid.append("내리막(폭포/계단/외봉)")
    if detect_choppy(df):
        avoid.append("톱니바퀴")
    if detect_repeated_resistance(df):
        avoid.append("다중봉/쌍봉")
    if detect_top_distribution(df):
        avoid.append("고점횡보")

    verdict = "✅ 매수후보" if buy and not avoid else ("⛔ 회피" if avoid else "👀 관찰")
    target_pct = 0.10 if is_light_cap else 0.05
    stop_ref = max(base["판바닥"], float(df["Low"].iloc[-20:].min()))

    return {
        "판정": verdict,
        "Ticker": ticker,
        "종목명": name,
        "현재가": round(price, 2),
        "Zone": zone,
        "매수패턴": ", ".join(buy) if buy else "-",
        "회피패턴": ", ".join(avoid) if avoid else "-",
        "원바닥": round(base_price, 2),
        "원바닥배수": round(price / base_price, 2) if base_price else None,
        "이익실현목표가": round(price * (1 + target_pct), 2),
        "손절기준지지선": round(stop_ref, 2),
        "스캔시각": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def run_scan(market_type, params) -> pd.DataFrame:
    """스캔 전체 파이프라인 (진행률 표시 포함)"""
    if market_type == "KR":
        gainers = get_kr_top_gainers(params["kr_market"], params["top_n"],
                                     params["min_volume"], params["min_price"])
        fetch = get_kr_ohlcv
    else:
        gainers = get_us_top_gainers(tuple(params["universe"]), params["min_volume"], params["min_price"])
        gainers = gainers.head(params["top_n"]) if not gainers.empty else gainers
        fetch = get_us_ohlcv

    if gainers.empty:
        return pd.DataFrame()

    results, errors = [], 0
    bar = st.progress(0, text="분석 중...")
    rows = list(gainers.itertuples(index=False))
    for i, row in enumerate(rows, 1):
        bar.progress(i / len(rows), text=f"분석 중... {i}/{len(rows)}  {row.종목명}")
        try:
            df = fetch(str(row.Ticker), 600)
            res = analyze_ticker(str(row.Ticker), row.종목명, df,
                                 is_light_cap=(df["Close"].iloc[-1] <= params["light_cap_th"]),
                                 gain_th=params["gain_th"], vol_mult=params["vol_mult"],
                                 overheat_mult=params.get("overheat_mult", 4.0))
            if res:
                res["당일등락률(%)"] = round(float(row.등락률), 2)
                results.append(res)
        except Exception:
            errors += 1
    bar.empty()
    if errors:
        st.caption(f"※ {errors}개 종목은 데이터 조회 실패로 제외되었습니다.")
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    order = {"✅ 매수후보": 0, "👀 관찰": 1, "⛔ 회피": 2}
    out["_o"] = out["판정"].map(order)
    return out.sort_values(["_o", "당일등락률(%)"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


# ============================================================
# 3. UI
# ============================================================

st.title("📈 종목선정 필살기 스캐너")
st.markdown('<p class="small-note">장대양봉·돌파·눌림목·고가놀이 매수패턴 / 12가지 회피패턴 자동 판정 · '
            '본 프로그램은 서적 내용을 조건식으로 근사한 참고 도구이며 투자 조언이 아닙니다.</p>',
            unsafe_allow_html=True)

tab_scan, tab_detail, tab_money, tab_help = st.tabs(["🔍 스캔", "📊 종목 상세", "💰 자금관리", "📖 사용법"])

# ---------- 스캔 탭 ----------
with tab_scan:
    market_type = st.radio("시장", ["KR", "US"], horizontal=True,
                           format_func=lambda x: "🇰🇷 국내" if x == "KR" else "🇺🇸 미국")

    params = {}
    if market_type == "KR":
        c1, c2 = st.columns(2)
        params["kr_market"] = c1.selectbox("시장 구분", ["KOSPI", "KOSDAQ"])
        params["top_n"] = c2.slider("등락률 상위 N개 분석", 10, 100, 30, step=10,
                                    help="많을수록 정확하지만 오래 걸립니다(휴대폰: 30개 권장)")
        with st.expander("세부 필터 (책 기본값 적용됨)"):
            params["min_volume"] = st.number_input("최소 거래량(주)", value=100_000, step=10_000)
            params["min_price"] = st.number_input("최소 가격(원)", value=1_000, step=100)
            params["light_cap_th"] = st.number_input("소형주 기준가(원) — 이하 +10%, 초과 +5% 목표",
                                                     value=50_000, step=1_000)
            params["gain_th"] = st.slider("장대양봉 최소 상승폭(%)", 5, 20, 10) / 100
            params["vol_mult"] = st.slider("장대양봉 거래량 배수(전일比)", 1.5, 4.0, 2.0, 0.5)
            params["overheat_mult"] = st.slider(
                "과열(회피) 판단 배수 — 원바닥×N 초과면 회피", 2.0, 8.0, 4.0, 0.5,
                help="이미 많이 오른 종목을 걸러내는 기준입니다. 값을 높이면 이미 급등한 종목도 '매수후보'로 나올 수 있습니다.")
    else:
        default_univ = "AAPL,MSFT,NVDA,TSLA,AMD,META,AMZN,GOOGL,NFLX,AVGO,PLTR,COIN,RBLX,SMCI,ARM"
        univ_text = st.text_area("워치리스트 (티커, 쉼표 구분)", default_univ, height=80)
        params["universe"] = [x.strip().upper() for x in univ_text.split(",") if x.strip()]
        params["top_n"] = st.slider("등락률 상위 N개 분석", 5, 50, 15)
        with st.expander("세부 필터"):
            params["min_volume"] = st.number_input("최소 거래량(주)", value=500_000, step=50_000)
            params["min_price"] = st.number_input("최소 가격($)", value=3.0, step=1.0)
            params["light_cap_th"] = st.number_input("소형주 기준가($) — 이하 +10%, 초과 +5% 목표",
                                                     value=20.0, step=1.0)
            params["gain_th"] = st.slider("장대양봉 최소 상승폭(%)", 5, 20, 10) / 100
            params["vol_mult"] = st.slider("장대양봉 거래량 배수(전일比)", 1.5, 4.0, 2.0, 0.5)
            params["overheat_mult"] = st.slider(
                "과열(회피) 판단 배수 — 원바닥×N 초과면 회피", 2.0, 10.0, 4.0, 0.5,
                help="이미 많이 오른 대형 성장주가 많은 워치리스트라면 값을 높여보세요(예: 6~8).")

    if st.button("🚀 지금 스캔하기", use_container_width=True, type="primary"):
        t0 = time.time()
        try:
            with st.spinner("등락률 상위 종목 조회 중..."):
                result_df = run_scan(market_type, params)
            if result_df.empty:
                st.info("조건에 맞는 종목이 없거나 데이터를 가져오지 못했습니다. (휴장일이거나 필터가 과도할 수 있습니다)")
            else:
                st.session_state["scan_result"] = result_df
                st.session_state["scan_market"] = market_type
                st.success(f"{len(result_df)}개 종목 분석 완료 ({time.time() - t0:.0f}초)")
        except Exception as e:
            st.error(f"⚠️ 데이터 로드 실패: {e}\n\n네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요.")

    if "scan_result" in st.session_state:
        df_r = st.session_state["scan_result"]
        n_buy = int((df_r["판정"] == "✅ 매수후보").sum())
        n_watch = int((df_r["판정"] == "👀 관찰").sum())
        n_avoid = int((df_r["판정"] == "⛔ 회피").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("✅ 매수후보", n_buy)
        m2.metric("👀 관찰", n_watch)
        m3.metric("⛔ 회피", n_avoid)
        if n_buy == 0 and n_watch == 0 and n_avoid > 0:
            st.caption("💡 전부 '회피'로 나왔다면 워치리스트/종목들이 대부분 이미 많이 오른 상태일 수 있어요. "
                       "세부 필터의 **'과열(회피) 판단 배수'** 를 높이거나(예: 6~8), 워치리스트를 다양화해보세요.")

        show = st.multiselect("표시할 판정", ["✅ 매수후보", "👀 관찰", "⛔ 회피"],
                              default=["✅ 매수후보", "👀 관찰"])
        f = df_r[df_r["판정"].isin(show)]

        # 휴대폰: 카드형 요약 / PC: 전체 표
        mobile_view = st.toggle("📱 카드형 보기 (휴대폰 추천)", value=True)
        if mobile_view:
            if f.empty:
                st.info("표시할 종목이 없습니다.")
            for _, r in f.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{r['판정']}  {r['종목명']}** `({r['Ticker']})`  ·  {r['Zone']}")
                    a, b = st.columns(2)
                    a.markdown(f"현재가 **{r['현재가']:,}**  \n당일 {r['당일등락률(%)']:+.2f}%  \n원바닥 ×{r['원바닥배수']}")
                    b.markdown(f"목표가 **{r['이익실현목표가']:,}**  \n손절선 {r['손절기준지지선']:,}")
                    st.markdown(f"매수패턴: {r['매수패턴']}  \n회피패턴: {r['회피패턴']}")
        else:
            st.dataframe(f, use_container_width=True, hide_index=True)

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            f.to_excel(w, index=False, sheet_name="스캔결과")
        st.download_button("📥 엑셀로 저장", data=buf.getvalue(),
                           file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    else:
        if market_type == "KR":
            st.info("위에서 시장을 고른 뒤 **지금 스캔하기**를 누르세요.\n\n"
                    "※ '12:30 이후'는 프로그램의 제약이 아니라 **국내 장중 참고용 안내**입니다. "
                    "장 시작 직후엔 등락률 상위 종목이 자주 바뀌어서, 어느 정도 안정되는 12:30 이후 확인을 "
                    "권장한다는 뜻일 뿐 — 아무 때나 스캔해도 정상 작동합니다.")
        else:
            st.info("위에서 시장을 고른 뒤 **지금 스캔하기**를 누르세요. 미국 시장은 시간 제약 없이 언제든 스캔 가능합니다.")

# ---------- 종목 상세 탭 ----------
with tab_detail:
    d_market = st.radio("시장 ", ["KR", "US"], horizontal=True, key="d_market",
                        format_func=lambda x: "🇰🇷 국내" if x == "KR" else "🇺🇸 미국")
    d_ticker = st.text_input("종목코드 / 티커", "005930" if d_market == "KR" else "AAPL")
    if st.button("차트 보기", use_container_width=True):
        try:
            with st.spinner("차트 데이터 조회 중..."):
                df = get_kr_ohlcv(d_ticker, 600) if d_market == "KR" else get_us_ohlcv(d_ticker, 600)
            base = compute_base_levels(df)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                         low=df["Low"], close=df["Close"], name=d_ticker,
                                         increasing_line_color="#e53935", decreasing_line_color="#1e88e5"))
            fig.add_hline(y=base["원바닥"], line_dash="dash", line_color="blue", annotation_text="원바닥")
            fig.add_hline(y=base["원바닥"] * 2, line_dash="dot", line_color="gray", annotation_text="원바닥×2")
            fig.add_hline(y=base["판바닥"], line_dash="dot", line_color="orange", annotation_text="판바닥")
            fig.update_layout(height=420, xaxis_rangeslider_visible=False,
                              margin=dict(l=5, r=5, t=25, b=5), dragmode="pan")
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

            vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color="#999"))
            vol_fig.update_layout(height=160, margin=dict(l=5, r=5, t=5, b=5), title_text="거래량")
            st.plotly_chart(vol_fig, use_container_width=True)

            res = analyze_ticker(d_ticker, d_ticker, df)
            if res:
                c1, c2, c3 = st.columns(3)
                c1.metric("현재가", f"{res['현재가']:,}")
                c2.metric("Zone", res["Zone"])
                c3.metric("판정", res["판정"])
                st.markdown(f"**매수패턴:** {res['매수패턴']}  \n**회피패턴:** {res['회피패턴']}  \n"
                            f"**이익실현목표가:** {res['이익실현목표가']:,}  ·  **손절기준지지선:** {res['손절기준지지선']:,}")
        except Exception as e:
            st.error(f"⚠️ 차트 데이터를 가져오지 못했습니다: {e}")

# ---------- 자금관리 탭 ----------
with tab_money:
    st.subheader("분할매수 · 현금비중 계산기")
    total = st.number_input("총 투자 가능 금액", value=10_000_000, step=100_000)
    cash_pct = st.slider("최소 현금 비중 유지(%)", 0, 50, 20)
    splits = st.slider("분할 매수 횟수", 1, 5, 3)
    invest = total * (1 - cash_pct / 100)
    c1, c2, c3 = st.columns(3)
    c1.metric("투자 가능 총액", f"{invest:,.0f}")
    c2.metric("회당 분할 매수액", f"{invest / splits:,.0f}")
    c3.metric("상시 유지 현금", f"{total - invest:,.0f}")
    st.caption("책의 원칙: 현금 20% 이상 상시 보유 · 신용은 총 잔고의 30% 이내 · "
               "지수 -2%/-3% 급락 시 신용매수 중단 및 잔고 축소")

# ---------- 사용법 탭 ----------
with tab_help:
    st.markdown("""
**하루 사용 흐름 (책의 STEP 기준)**
1. **12:30 이후** 스캔 → 등락률 상위 종목 중 ✅ 매수후보 / 👀 관찰 확인
2. 후보 종목을 **종목 상세** 탭에서 차트로 직접 확인 (원바닥·판바닥 라인 참고)
3. **14:00** 재스캔 후 3~5개 → 최종 1~2개로 압축
4. **14:30 이후** 호가창 보고 저가 지정가 또는 종가 부근 매수
5. 이익실현: 소형주 +10% / 대형주 +5% · 손절: 지지선 붕괴 + 거래량 급증 시에만

**판정 기준**
- ✅ 매수후보: 매수패턴 1개 이상 + 회피패턴 없음
- 👀 관찰: 패턴 없음(추가 관찰)
- ⛔ 회피: 회피패턴 감지 (앞/뒤폭탄·내리막·톱니바퀴·다중봉·고점횡보)

**아이폰/아이패드 홈화면에 앱처럼 추가하기**
Safari에서 이 페이지 열기 → 공유(□↑) → **홈 화면에 추가**
    """)
