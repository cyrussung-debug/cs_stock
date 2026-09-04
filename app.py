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

try:
    from pykrx import stock as krx
    KRX_AVAILABLE = True
except Exception:
    KRX_AVAILABLE = False

st.set_page_config(
    page_title="종목선정 스캐너",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- 디자인: 카드/배지/타이포 커스텀 CSS (다크 테마 자체 적용) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.stApp { background-color: #0B1220; }
[data-testid="stSidebar"] { background-color: #10192B; }
.stApp, .stApp p, .stApp span, .stApp label, .stApp div { color: #E6EAF2; }
h1, h2, h3, h4 { color: #F1F5F9 !important; }

@media (max-width: 768px) {
    .block-container {padding: 0.8rem 0.6rem 2rem 0.6rem;}
    div[data-testid="stDataFrame"] {font-size: 12px;}
}
.block-container {max-width: 900px; margin: 0 auto;}

.app-hero {
    background: linear-gradient(135deg, #16321f 0%, #0B1220 70%);
    border: 1px solid #1f2c22;
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 18px;
}
.app-hero h1 {
    font-size: 1.55rem; font-weight: 800; margin: 0 0 4px 0;
    background: linear-gradient(90deg, #22C55E, #86EFAC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-hero p { color: #93A0B4; font-size: 0.85rem; margin: 0; line-height: 1.5; }

div[data-testid="stMetricValue"] {font-weight: 800; color: #E6EAF2;}
div[data-testid="stMetricLabel"] {color: #93A0B4;}
.stButton > button, .stLinkButton > a {
    border-radius: 12px !important; font-weight: 700 !important; height: 2.8em;
    background-color: #1B2537 !important; color: #E6EAF2 !important; border: 1px solid #2A3550 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #16A34A, #22C55E) !important; border: none !important; color: #06210F !important;
}

.stock-card {
    border: 1px solid #232D42; border-radius: 16px; padding: 16px 18px;
    margin-bottom: 12px; background: #10192B;
}
.stock-card.buy   { border-left: 4px solid #22C55E; }
.stock-card.watch { border-left: 4px solid #64748B; }
.stock-card.avoid { border-left: 4px solid #EF4444; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; margin-right: 6px;
}
.badge.buy   { background: rgba(34,197,94,0.15); color: #4ADE80; }
.badge.watch { background: rgba(100,116,139,0.2); color: #94A3B8; }
.badge.avoid { background: rgba(239,68,68,0.15); color: #F87171; }

.stock-title { font-size: 1.05rem; font-weight: 700; color: #E6EAF2; }
.stock-sub { color: #7C8AA5; font-size: 0.8rem; }
.metric-label { color: #7C8AA5; font-size: 0.78rem; }
.metric-value { color: #E6EAF2; font-size: 0.98rem; font-weight: 700; }
.small-note { color: #7C8AA5; font-size: 0.82rem; }

div[data-testid="stExpander"] { border: 1px solid #232D42; border-radius: 12px; background: #0E1626; }
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, textarea {
    background-color: #131C2E !important; color: #E6EAF2 !important; border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. 데이터 수집 계층 (국내 / 해외 분리)
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_kr_universe(market: str) -> pd.DataFrame:
    """국내 전 종목 시세 스냅샷. 반환 컬럼: Code, Name, Close, ChangesRatio, Volume, Marcap(원)"""
    if not KR_DATA_AVAILABLE:
        raise RuntimeError("finance-datareader 가 설치되어 있지 않습니다.")
    df = fdr.StockListing(market)
    if df is None or df.empty:
        raise RuntimeError(f"{market} 종목 목록을 가져오지 못했습니다.")
    if "ChagesRatio" in df.columns and "ChangesRatio" not in df.columns:
        df = df.rename(columns={"ChagesRatio": "ChangesRatio"})
    need = {"Code", "Name", "Close", "ChangesRatio", "Volume"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"예상 컬럼 누락: {missing}")
    cols = list(need | ({"Marcap"} & set(df.columns)))
    return df[cols].copy()


def get_kr_top_gainers(market: str, top_n: int, min_volume: int, min_price: int,
                        min_marketcap_eok: float = 0) -> pd.DataFrame:
    """당일 등락률 상위 종목 (거래량/가격/시가총액 필터 적용)"""
    df = get_kr_universe(market)
    df = df[(df["Volume"] >= min_volume) & (df["Close"] >= min_price)]
    if min_marketcap_eok > 0 and "Marcap" in df.columns:
        df = df[df["Marcap"] >= min_marketcap_eok * 1e8]
    df = df.sort_values("ChangesRatio", ascending=False).head(top_n)
    return df.rename(columns={"Code": "Ticker", "Name": "종목명",
                              "ChangesRatio": "등락률", "Close": "종가", "Volume": "거래량"})


@st.cache_data(ttl=3600, show_spinner=False)
def get_kr_ohlcv(ticker: str, lookback_days: int = 600, end_date: str | None = None) -> pd.DataFrame:
    """국내 종목 일봉 OHLCV. end_date(YYYY-MM-DD) 지정 시 그 시점까지의 과거 데이터(백테스트용)"""
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()
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
    """워치리스트 내 당일 등락률 상위 (100개씩 나눠 조회하여 안정성 확보)"""
    if not universe:
        return pd.DataFrame()
    universe = list(universe)
    chunk_size = 100
    rows = []
    for i in range(0, len(universe), chunk_size):
        chunk = universe[i:i + chunk_size]
        try:
            data = yf.download(chunk, period="5d", interval="1d",
                               group_by="ticker", progress=False, threads=True)
        except Exception:
            continue
        for t in chunk:
            try:
                sub = (data[t] if len(chunk) > 1 else data).dropna()
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_us_market_cap(ticker: str) -> float:
    """미국 종목 시가총액(달러). 조회 실패 시 0 반환"""
    try:
        fi = yf.Ticker(ticker).fast_info
        cap = fi.get("market_cap") or fi.get("marketCap")
        return float(cap) if cap else 0.0
    except Exception:
        return 0.0


# ============================================================
# 2. 종목선정 알고리즘 엔진 (책의 원칙 → 조건식)
#    ※ 시각적 패턴은 정량 근사치이며, 최종 판단은 차트 확인 필수
# ============================================================

def compute_base_levels(df: pd.DataFrame) -> dict:
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
    if len(df) < box_days + 5:
        return False
    box = df.iloc[-(box_days + 1):-1]
    t = df.iloc[-1]
    box_vol = box["Close"].std() / box["Close"].mean()
    return bool(t["Close"] > box["Close"].max() and t["Volume"] >= box["Volume"].mean() * vol_mult
                and box_vol < 0.25)


def detect_nulimmok(df, lookback=20, ma_period=20) -> bool:
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
    if len(df) < 60:
        return False
    r = df.iloc[-tight_days:]
    high60 = df["Close"].iloc[-60:].max()
    tight = (r["High"].max() - r["Low"].min()) / r["Close"].mean() < range_th
    near_high = r["Close"].mean() >= high60 * 0.9
    vol_declining = r["Volume"].mean() < df["Volume"].iloc[-30:-tight_days].mean()
    return bool(tight and near_high and vol_declining)


def detect_overextended(df, base_price, mult_th=4.0) -> bool:
    return bool(base_price > 0 and df["Close"].iloc[-1] / base_price > mult_th)


def detect_downtrend(df, ma_s=20, ma_l=60) -> bool:
    if len(df) < ma_l + 10:
        return False
    s, l = df["Close"].rolling(ma_s).mean(), df["Close"].rolling(ma_l).mean()
    down = s.iloc[-1] < l.iloc[-1] and s.iloc[-1] < s.iloc[-10]
    lows = df["Low"].iloc[-40:]
    return bool(down and lows.iloc[-10:].min() < lows.iloc[:-10].min())


def detect_choppy(df, window=40) -> bool:
    if len(df) < window:
        return False
    r = df["Close"].iloc[-window:]
    slope = np.polyfit(np.arange(len(r)), r.values, 1)[0]
    return bool(abs(slope) / r.mean() < 0.001 and r.pct_change().std() > 0.03)


def detect_repeated_resistance(df, window=90, band=0.03) -> bool:
    if len(df) < window:
        return False
    highs = df["High"].iloc[-window:]
    peak = highs.max()
    return bool((highs >= peak * (1 - band)).sum() >= 2 and df["Close"].iloc[-1] < peak * (1 - band))


def detect_top_distribution(df, window=20, band=0.05) -> bool:
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
        "target_pct": target_pct,
        "스캔시각": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def run_scan(market_type, params) -> pd.DataFrame:
    """스캔 전체 파이프라인 (진행률 표시 포함)"""
    if market_type == "KR":
        gainers = get_kr_top_gainers(params["kr_market"], params["top_n"],
                                     params["min_volume"], params["min_price"],
                                     params.get("min_marketcap_eok", 0))
        fetch = get_kr_ohlcv
    else:
        gainers = get_us_top_gainers(tuple(params["universe"]), params["min_volume"], params["min_price"])
        gainers = gainers.head(params["top_n"] * 3) if not gainers.empty else gainers  # 시총 필터로 줄어들 것 감안 여유분 확보
        fetch = get_us_ohlcv

    if gainers.empty:
        return pd.DataFrame()

    min_cap_usd = params.get("min_marketcap_usd_m", 0) * 1_000_000
    results, errors, cap_filtered = [], 0, 0
    bar = st.progress(0, text="분석 중...")
    rows = list(gainers.itertuples(index=False))
    kept = 0
    for i, row in enumerate(rows, 1):
        bar.progress(i / len(rows), text=f"분석 중... {i}/{len(rows)}  {row.종목명}")
        if kept >= params["top_n"]:
            break
        try:
            if market_type == "US" and min_cap_usd > 0:
                cap = get_us_market_cap(str(row.Ticker))
                if cap < min_cap_usd:
                    cap_filtered += 1
                    continue
            df = fetch(str(row.Ticker), 600)
            res = analyze_ticker(str(row.Ticker), row.종목명, df,
                                 is_light_cap=(df["Close"].iloc[-1] <= params["light_cap_th"]),
                                 gain_th=params["gain_th"], vol_mult=params["vol_mult"],
                                 overheat_mult=params.get("overheat_mult", 4.0))
            if res:
                res["당일등락률(%)"] = round(float(row.등락률), 2)
                results.append(res)
                kept += 1
        except Exception:
            errors += 1
    bar.empty()
    if errors:
        st.caption(f"※ {errors}개 종목은 데이터 조회 실패로 제외되었습니다.")
    if cap_filtered:
        st.caption(f"※ 시가총액 기준 미달로 {cap_filtered}개 종목이 제외되었습니다.")
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    order = {"✅ 매수후보": 0, "👀 관찰": 1, "⛔ 회피": 2}
    out["_o"] = out["판정"].map(order)
    return out.sort_values(["_o", "당일등락률(%)"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


# ============================================================
# 3. 차트 렌더링 (스캔 카드의 아코디언 / 종목상세 탭 공용)
# ============================================================

def render_chart_block(ticker: str, market: str, name: str | None = None, end_date: str | None = None):
    """캔들차트 + 거래량 + 원바닥/판바닥 라인 + 판정 요약을 그린다."""
    name = name or ticker
    try:
        df = (get_kr_ohlcv(ticker, 600, end_date) if market == "KR" else get_us_ohlcv(ticker, 600))
    except Exception as e:
        st.error(f"⚠️ 차트 데이터를 가져오지 못했습니다: {e}")
        return

    base = compute_base_levels(df)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name=ticker,
                                 increasing_line_color="#EF4444", decreasing_line_color="#3B82F6"))
    fig.add_hline(y=base["원바닥"], line_dash="dash", line_color="#60A5FA", annotation_text="원바닥")
    fig.add_hline(y=base["원바닥"] * 2, line_dash="dot", line_color="#94A3B8", annotation_text="원바닥×2")
    fig.add_hline(y=base["판바닥"], line_dash="dot", line_color="#FBBF24", annotation_text="판바닥")
    fig.update_layout(height=380, xaxis_rangeslider_visible=False,
                      margin=dict(l=5, r=5, t=25, b=5), dragmode="pan",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#E6EAF2")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True},
                    key=f"chart_{ticker}_{market}_{end_date or 'live'}")

    vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color="#3B4A63"))
    vol_fig.update_layout(height=130, margin=dict(l=5, r=5, t=5, b=5),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6EAF2")
    st.plotly_chart(vol_fig, use_container_width=True, key=f"vol_{ticker}_{market}_{end_date or 'live'}")

    res = analyze_ticker(ticker, name, df)
    if res:
        c1, c2, c3 = st.columns(3)
        c1.metric("현재가", f"{res['현재가']:,}")
        c2.metric("Zone", res["Zone"])
        c3.metric("판정", res["판정"])
        st.markdown(f"**매수패턴:** {res['매수패턴']}  \n**회피패턴:** {res['회피패턴']}  \n"
                    f"**이익실현목표가:** {res['이익실현목표가']:,}  ·  **손절기준지지선:** {res['손절기준지지선']:,}")

    ext_url = (f"https://finance.naver.com/item/main.naver?code={ticker}"
              if market == "KR" else f"https://finance.yahoo.com/quote/{ticker}")
    ext_label = "🔗 네이버금융에서 보기" if market == "KR" else "🔗 야후파이낸스에서 보기"
    st.link_button(ext_label, ext_url, use_container_width=True)


# ============================================================
# 4. 백테스트 엔진 (근사 / 국내 전용)
#    ※ 매일의 실제 "등락률 상위 200"을 재현하는 대신, 오늘 기준
#      유동성·시가총액 상위 종목군을 고정 후보군으로 삼아 과거
#      각 거래일의 매수신호 여부를 확인하는 근사 방식입니다.
#      (완전한 점-in-시간 재현은 아니며, 대략적 성과 감을 보기 위한 참고용입니다)
# ============================================================

def get_backtest_universe(market: str, top_n: int) -> list:
    """시가총액 상위 종목 중심의 백테스트 후보군 구성"""
    df = get_kr_universe(market)
    if "Marcap" in df.columns:
        df = df.sort_values("Marcap", ascending=False)
    else:
        df = df.sort_values("Volume", ascending=False)
    return df.head(top_n)[["Code", "Name"]].values.tolist()


def simulate_trade(df: pd.DataFrame, signal_idx: int, target_pct: float, max_hold_days: int = 20):
    """신호 발생 다음날 시가 매수 → 목표가/손절선 중 먼저 닿는 쪽으로 청산 (근사 시뮬레이션)"""
    if signal_idx + 1 >= len(df):
        return None
    entry = df["Open"].iloc[signal_idx + 1]
    if entry <= 0 or np.isnan(entry):
        return None
    hist_up_to_signal = df.iloc[:signal_idx + 1]
    base = compute_base_levels(hist_up_to_signal)
    stop_ref = max(base["판바닥"], float(hist_up_to_signal["Low"].iloc[-20:].min()))
    target_price = entry * (1 + target_pct)

    end = min(signal_idx + 1 + max_hold_days, len(df))
    for j in range(signal_idx + 1, end):
        day = df.iloc[j]
        if day["Low"] <= stop_ref:
            return {"결과": "손절", "수익률": (stop_ref - entry) / entry * 100, "보유일": j - signal_idx}
        if day["High"] >= target_price:
            return {"결과": "익절", "수익률": target_pct * 100, "보유일": j - signal_idx}
    last = df.iloc[end - 1]
    return {"결과": "기간만료청산", "수익률": (last["Close"] - entry) / entry * 100, "보유일": end - 1 - signal_idx}


def run_backtest(market: str, universe_n: int, months: int, gain_th: float, vol_mult: float,
                 overheat_mult: float, max_hold_days: int) -> pd.DataFrame:
    tickers = get_backtest_universe(market, universe_n)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=months * 31 + 650)  # 지표 계산용 여유 버퍼 포함

    records = []
    bar = st.progress(0, text="종목별 과거 데이터 조회 중...")
    for i, (code, name) in enumerate(tickers, 1):
        bar.progress(i / len(tickers), text=f"백테스트 진행 중... {i}/{len(tickers)}  {name}")
        try:
            df = fdr.DataReader(code, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            continue
        if len(df) < 200:
            continue

        backtest_start_idx = max(200, len(df) - months * 21)  # 최근 months*21거래일만 신호 탐색
        for idx in range(backtest_start_idx, len(df) - 1):
            window = df.iloc[:idx + 1]
            base = compute_base_levels(window)
            zone = classify_zone(float(window["Close"].iloc[-1]), base["원바닥"], float(window["Close"].max()))
            if not detect_jangdae_yangbong(window, gain_th, vol_mult):
                continue
            if detect_overextended(window, base["원바닥"], overheat_mult) or detect_downtrend(window) or \
               detect_choppy(window) or detect_repeated_resistance(window) or detect_top_distribution(window):
                continue
            price = float(window["Close"].iloc[-1])
            is_light = price <= 50_000
            target_pct = 0.10 if is_light else 0.05
            trade = simulate_trade(df, idx, target_pct, max_hold_days)
            if trade:
                trade.update({"종목명": name, "코드": code, "신호일": df.index[idx].strftime("%Y-%m-%d"), "Zone": zone})
                records.append(trade)
    bar.empty()
    return pd.DataFrame(records)


# ============================================================
# 5. UI
# ============================================================

st.markdown("""
<div class="app-hero">
  <h1>📈 종목선정 필살기 스캐너</h1>
  <p>장대양봉·돌파·눌림목·고가놀이 매수패턴 / 12가지 회피패턴 자동 판정 · 본 프로그램은 서적 내용을 조건식으로
  근사한 참고 도구이며 투자 조언이 아닙니다.</p>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_detail, tab_backtest, tab_money, tab_help = st.tabs(
    ["🔍 스캔", "📊 종목상세", "📈 백테스트", "💰 자금관리", "📖 사용법"])

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
        params["min_marketcap_eok"] = st.number_input(
            "최소 시가총액 필터 (억원, 0=제한없음)", min_value=0, value=0, step=100,
            help="예: 1,000 입력 시 시가총액 1,000억원 이상 종목만 분석합니다.")
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
        default_univ = ("GOOG,META,AAPL,BABA,MSFT,FAS,FAZ,TNA,TZA,BTCS,TTI,TSLA,DJT,NVDA,UBER,PLTR,AMWL,AMC,NKLA,ZIM,"
                        "ZM,DBX,TAUG,LCID,RIVN,FMCC,FNMA,IONQ,UA,SPY,HOOD,SNAP,CHPT,GLD,PFE,CPNG,CTRM,AXP,TDOC,ERJ,"
                        "OUST,BA,NFLX,AMZN,CME,GRPN,F,BAC,C,EBAY,DIS,XOM,BKNG,EXPE,ADBE,V,ICE,KO,TWLO,NUS,TRIP,PSX,"
                        "USO,BRK-A,PYPL,SBUX,TQQQ,SRPT,SND,HLF,RIOT,PM,ACN,LYFT,CHGG,ASML,MU,HLT,AMD,TLT,COST,DAL,"
                        "INO,RCL,MAR,AAL,NASDX,JBLU,UAL,WMT,COTY,LULU,GILD,GIL,ULTA,MDT,TRV,JNJ,CCL,CAT,CMG,TOPS,"
                        "VAL,MRNA,AOR,LTPZ,VT,REGN,MGM,CZR,CLX,CHTR,ORCL,HD,ENB,HON,INTC,JD,QQQ,UONE,LMT,NKE,T,MCD,"
                        "JPM,KODK,HAS,EB,CRM,INTU,LEGN,RVMD,TSM,SNOW,CNXS,FSLY,TRMB,BEKE,MGA,ARKQ,RBLX,COIN,AZN,"
                        "ARKK,XLI,XLF,XLB,SPOT,SONY,PINS,RDW,DIA,VO,GDDY,PLUG,BLNK,LAC,ALB,FCX,QCOM,SOXL,WRBY,SDGR,"
                        "RPRX,FND,U,SLDP,MRVL,DOCU,CVX,UTSL,TECS,HCP,PDBC,PXGYF,BATT,LIT,FDX,UPS,SCHW,SPHD,LVMHF,"
                        "HESAY,LVMUY,HESAF,PDRDF,PRNDY,O,MRK,GME,UPST,IBM,AVGO,CVS,DIDIY,SQ,ABNB,VERU,GOEV,PDD,ARDX,"
                        "HAFC,QYLD,VEON,ABR,NIO,D,XYLD,JEPI,RWLK,GCT,TER,FXI,LBPH,HTZ,CAR,ABBV,UVV,MO,CL,NBR,SAVE,"
                        "NVAX,DUOL,WBTN,VFS,STLA,HYG,SOF,ECL,TMO,SMCI,LLY,ISRG,SGOV,NVO,MSTR,CAPV,COF,MRNY,ARM,BKH,"
                        "WRD,SOUN,NBIS,SERV,TSLL,GEV,STZ,NMAX,CRSP,BEAM,CRSH,PONY,XYZ,R,APUS,MP,OSIS,KTOS,AISP,SHOP,"
                        "NOC,OXY,IOT,JOBY,VEEV,TOST,PG,MSTU,MSTX,SOXX,SUPL,PNC,SMR,PEW,DE,FIG,BLSH,HP,LRCX,XRPC,EIX,"
                        "BB,SPCX,SKHY")
        univ_text = st.text_area("워치리스트 (티커, 쉼표 구분) — 286개 등록됨", default_univ, height=140)
        st.caption("⚠️ 여기서 고친 목록은 페이지를 완전히 새로고침/재접속하면 이 기본값으로 되돌아갑니다.")
        params["universe"] = [x.strip().upper().lstrip("$").replace("/", "-")
                              for x in univ_text.split(",") if x.strip()]
        params["top_n"] = st.slider("등락률 상위 N개 분석", 5, 60, 20)
        params["min_marketcap_usd_m"] = st.number_input(
            "최소 시가총액 필터 (백만$, 0=제한없음)", min_value=0, value=0, step=100,
            help="예: 1,000 입력 시 시가총액 10억달러 이상 종목만 분석합니다. (시총 조회가 추가로 필요해 조금 더 걸릴 수 있어요)")
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
                st.session_state.pop("scan_result", None)
                st.session_state.pop("scan_market", None)
                st.info("조건에 맞는 종목이 없거나 데이터를 가져오지 못했습니다. (휴장일이거나 필터가 과도할 수 있습니다)")
            else:
                st.session_state["scan_result"] = result_df
                st.session_state["scan_market"] = market_type
                st.success(f"{len(result_df)}개 종목 분석 완료 ({time.time() - t0:.0f}초)")
        except Exception as e:
            st.session_state.pop("scan_result", None)
            st.session_state.pop("scan_market", None)
            st.error(f"⚠️ 데이터 로드 실패: {e}\n\n네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요.")

    if "scan_result" in st.session_state and st.session_state.get("scan_market") == market_type:
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

        mobile_view = st.toggle("📱 카드형 보기 (휴대폰 추천)", value=True)
        if mobile_view:
            if f.empty:
                st.info("표시할 종목이 없습니다.")
            for _, r in f.iterrows():
                css_cls = "buy" if r["판정"].startswith("✅") else ("avoid" if r["판정"].startswith("⛔") else "watch")
                st.markdown(f"""
                <div class="stock-card {css_cls}">
                  <span class="badge {css_cls}">{r['판정']}</span>
                  <span class="stock-title">{r['종목명']}</span>
                  <span class="stock-sub"> ({r['Ticker']}) · {r['Zone']}</span>
                </div>
                """, unsafe_allow_html=True)
                a, b = st.columns(2)
                a.markdown(f"<span class='metric-label'>현재가</span><br><span class='metric-value'>{r['현재가']:,}</span>"
                          f"<br><span class='metric-label'>당일 {r['당일등락률(%)']:+.2f}% · 원바닥 ×{r['원바닥배수']}</span>",
                          unsafe_allow_html=True)
                b.markdown(f"<span class='metric-label'>목표가</span><br><span class='metric-value'>{r['이익실현목표가']:,}</span>"
                          f"<br><span class='metric-label'>손절선 {r['손절기준지지선']:,}</span>",
                          unsafe_allow_html=True)
                st.markdown(f"<span class='small-note'>매수패턴: {r['매수패턴']} · 회피패턴: {r['회피패턴']}</span>",
                           unsafe_allow_html=True)

                ext_url = (f"https://finance.naver.com/item/main.naver?code={r['Ticker']}"
                          if market_type == "KR" else f"https://finance.yahoo.com/quote/{r['Ticker']}")
                ext_label = "🔗 네이버금융" if market_type == "KR" else "🔗 야후파이낸스"
                st.link_button(ext_label, ext_url, use_container_width=True)

                with st.expander("📊 차트 바로 보기 (누르면 즉시 펼쳐집니다)"):
                    render_chart_block(str(r["Ticker"]), market_type, r["종목명"])
                st.write("")
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

# ---------- 종목 상세 탭 (임의 티커 직접 조회) ----------
with tab_detail:
    d_market = st.radio("시장 ", ["KR", "US"], horizontal=True, key="d_market",
                        format_func=lambda x: "🇰🇷 국내" if x == "KR" else "🇺🇸 미국")
    default_ticker = "005930" if d_market == "KR" else "AAPL"
    d_ticker = st.text_input("종목코드 / 티커", default_ticker)
    if st.button("차트 보기", use_container_width=True):
        with st.spinner("차트 데이터 조회 중..."):
            render_chart_block(d_ticker, d_market)

# ---------- 백테스트 탭 ----------
with tab_backtest:
    st.markdown("#### 📈 과거 신호 백테스트 (국내, 근사치)")
    st.caption("⚠️ 매일의 실제 '등락률 상위 200종목'을 그대로 재현하는 게 아니라, **오늘 기준 시가총액 상위 종목군**을 "
               "고정 후보로 놓고 과거 각 거래일에 매수신호(장대양봉 등)가 떴는지 확인하는 **근사 백테스트**입니다. "
               "실제 그날의 등락률 상위와는 다를 수 있으니 참고용으로만 봐주세요. 종목 수·기간이 클수록 오래 걸립니다"
               "(예: 100종목×6개월 ≈ 수 분).")

    bt_market = st.selectbox("시장", ["KOSPI", "KOSDAQ"], key="bt_market")
    bc1, bc2, bc3 = st.columns(3)
    bt_universe_n = bc1.slider("후보 종목 수 (시총 상위)", 20, 300, 80, step=10)
    bt_months = bc2.slider("백테스트 기간(개월)", 1, 12, 3)
    bt_hold_days = bc3.slider("최대 보유일", 5, 40, 20)

    with st.expander("세부 조건 (스캔 탭 기본값과 동일)"):
        bt_gain_th = st.slider("장대양봉 최소 상승폭(%) ", 5, 20, 10, key="bt_gain") / 100
        bt_vol_mult = st.slider("거래량 배수(전일比) ", 1.5, 4.0, 2.0, 0.5, key="bt_vol")
        bt_overheat = st.slider("과열 판단 배수 ", 2.0, 8.0, 4.0, 0.5, key="bt_heat")

    if st.button("🧪 백테스트 실행", use_container_width=True, type="primary"):
        if not KR_DATA_AVAILABLE:
            st.error("국내 데이터 라이브러리가 없어 백테스트를 실행할 수 없습니다.")
        else:
            t0 = time.time()
            with st.spinner("백테스트 실행 중... (종목 수가 많으면 수 분 걸릴 수 있어요)"):
                bt_df = run_backtest(bt_market, bt_universe_n, bt_months, bt_gain_th,
                                     bt_vol_mult, bt_overheat, bt_hold_days)
            if bt_df.empty:
                st.info("해당 조건에서 발생한 매수 신호가 없습니다. 기간/종목 수를 늘려보세요.")
            else:
                st.success(f"신호 {len(bt_df)}건 시뮬레이션 완료 ({time.time() - t0:.0f}초)")
                win_rate = (bt_df["결과"] == "익절").mean() * 100
                avg_ret = bt_df["수익률"].mean()
                avg_hold = bt_df["보유일"].mean()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("표본 수", f"{len(bt_df)}건")
                m2.metric("승률(익절 비율)", f"{win_rate:.1f}%")
                m3.metric("평균 수익률", f"{avg_ret:+.2f}%")
                m4.metric("평균 보유일", f"{avg_hold:.1f}일")
                st.caption("승률 = 목표가(익절선)에 먼저 닿은 비율. 손절/기간만료청산 건은 승률 분모엔 포함되고 분자엔 제외됩니다.")

                st.dataframe(bt_df.sort_values("신호일", ascending=False),
                            use_container_width=True, hide_index=True)

                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                    bt_df.to_excel(w, index=False, sheet_name="백테스트결과")
                st.download_button("📥 백테스트 결과 엑셀로 저장", data=buf.getvalue(),
                                   file_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

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
2. 궁금한 종목 카드에서 **📊 차트 바로 보기**를 눌러 그 자리에서 차트 확인 (원바닥·판바닥 라인 참고)
3. **14:00** 재스캔 후 3~5개 → 최종 1~2개로 압축
4. **14:30 이후** 호가창 보고 저가 지정가 또는 종가 부근 매수
5. 이익실현: 소형주 +10% / 대형주 +5% · 손절: 지지선 붕괴 + 거래량 급증 시에만

**판정 기준**
- ✅ 매수후보: 매수패턴 1개 이상 + 회피패턴 없음
- 👀 관찰: 패턴 없음(추가 관찰)
- ⛔ 회피: 회피패턴 감지 (앞/뒤폭탄·내리막·톱니바퀴·다중봉·고점횡보)

**새로 추가된 기능**
- 💰 **시가총액 필터**: 스캔 탭에서 최소 시가총액을 직접 입력해 소형주를 걸러낼 수 있습니다.
- 📊 **즉시 차트보기**: 카드의 '차트 바로 보기'를 누르면 탭 이동 없이 그 자리에서 바로 펼쳐집니다.
- 📈 **백테스트**: 과거 데이터로 이 전략의 대략적인 승률·수익률을 미리 확인할 수 있습니다 (근사치, 참고용).

**아이폰/아이패드 홈화면에 앱처럼 추가하기**
Safari에서 이 페이지 열기 → 공유(□↑) → **홈 화면에 추가**
    """)
