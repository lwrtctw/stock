import os

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader

PERIOD_DAYS = {
    "2wk": 14,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
}
PERIOD_LABELS = {
    "2wk": "2 週",
    "1mo": "1 個月",
    "3mo": "3 個月",
    "6mo": "6 個月",
    "1y": "1 年",
    "2y": "2 年",
}

# FinMind 回傳英文欄位，對應為三大法人
INST_COL_MAP = {
    "外資": ["Foreign_Investor", "Foreign_Dealer_Self", "外資", "外陸資"],
    "投信": ["Investment_Trust", "投信"],
    "自營商": ["Dealer_self", "Dealer_Hedging", "自營商", "自營"],
}


def find_ma_crosses(df):
    """偵測 MA20 與 MA60 的黃金交叉 / 死亡交叉。"""
    crosses = []
    diff = df["MA20"] - df["MA60"]
    for i in range(1, len(df)):
        if pd.isna(diff.iloc[i]) or pd.isna(diff.iloc[i - 1]):
            continue
        if diff.iloc[i - 1] <= 0 < diff.iloc[i]:
            crosses.append(("golden", df.index[i], df["Close"].iloc[i]))
        elif diff.iloc[i - 1] >= 0 > diff.iloc[i]:
            crosses.append(("death", df.index[i], df["Close"].iloc[i]))
    return crosses


def sum_net_buy_lots(series, days):
    """累計買賣超，換算為張數。"""
    return series.tail(days).sum() / 1000


def is_taiwan_stock(stock_code):
    """判斷是否為台股（FinMind 法人資料僅支援台股）。"""
    code = stock_code.upper()
    return code.endswith(".TW") or code.endswith(".TWO") or stock_code.isdigit()


def parse_ticker_input(raw_input):
    """將使用者輸入轉為 yfinance 代碼與 FinMind stock_id。台股可直接輸入數字。"""
    code = raw_input.strip().upper()
    if not code:
        return "", ""

    if code.isdigit():
        return f"{code}.TW", code

    if code.endswith(".TW") or code.endswith(".TWO"):
        return code, code.split(".")[0]

    return code, code.split(".")[0]


def normalize_chip_df(df_chip):
    """將 FinMind 法人資料統一為「外資 / 投信 / 自營商」欄位。"""
    if df_chip is None or df_chip.empty:
        return None

    normalized = pd.DataFrame(index=df_chip.index)
    for cn_name, candidates in INST_COL_MAP.items():
        matched = [c for c in candidates if c in df_chip.columns]
        if matched:
            normalized[cn_name] = df_chip[matched].sum(axis=1)

    return normalized if not normalized.empty else None


def get_finmind_token():
    """從 Streamlit secrets 或環境變數讀取 FinMind Token。"""
    try:
        return st.secrets.get("FINMIND_TOKEN") or os.environ.get("FINMIND_TOKEN")
    except Exception:
        return os.environ.get("FINMIND_TOKEN")


@st.cache_data(ttl=86400)
def get_taiwan_stock_name_map():
    """快取台股代碼與中文名稱對照表。"""
    try:
        api = DataLoader()
        token = get_finmind_token()
        if token:
            api.login_by_token(api_token=token)
        df = api.taiwan_stock_info()
        if df is None or df.empty:
            return {}
        return dict(zip(df["stock_id"].astype(str), df["stock_name"]))
    except Exception:
        return {}


def get_stock_display_name(stock_code, stock_num):
    """取得標的顯示名稱（台股優先使用中文名）。"""
    if is_taiwan_stock(stock_code):
        name_map = get_taiwan_stock_name_map()
        cn_name = name_map.get(str(stock_num))
        if cn_name:
            return f"{stock_num} {cn_name}"
    try:
        info = yf.Ticker(stock_code).info
        en_name = info.get("shortName") or info.get("longName")
        if en_name:
            return f"{stock_num} {en_name}"
    except Exception:
        pass
    return stock_num


# 1. 網頁基本設定
st.set_page_config(page_title="投資決策視覺化工具", layout="wide")

# 2. 側邊欄：使用者輸入參數
st.sidebar.header("📊 參數設定")
ticker_input = st.sidebar.text_input(
    "輸入股票代碼 (台股直接輸入數字，如 2330；美股如 AAPL)",
    value="2330",
)
period = st.sidebar.selectbox(
    "查詢區間",
    options=list(PERIOD_DAYS.keys()),
    index=3,
    format_func=lambda x: PERIOD_LABELS[x],
)

ticker, stock_id = parse_ticker_input(ticker_input)
stock_display_name = get_stock_display_name(ticker, stock_id)

st.title(f"📈 {stock_display_name}" if stock_id else "📈 股票趨勢與量能決策儀表板")
st.markdown("結合 K 線、移動平均線與成交量能，加速你的波段進出場決策。")


# 3. 核心資料抓取與處理函數
@st.cache_data(ttl=3600)
def load_all_data(stock_code, stock_num, data_period):
    start_date = (
        pd.Timestamp.now() - pd.Timedelta(days=PERIOD_DAYS[data_period])
    ).strftime("%Y-%m-%d")
    df_price = yf.download(stock_code, start=start_date)
    if df_price.empty:
        return None, None, "empty", None

    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)

    df_price["MA20"] = df_price["Close"].rolling(window=20).mean()
    df_price["MA60"] = df_price["Close"].rolling(window=60).mean()
    df_price["Volume_Shares"] = df_price["Volume"] / 1000

    chip_status = "not_taiwan"
    chip_error = None
    df_chip = None

    if is_taiwan_stock(stock_code):
        try:
            api = DataLoader()
            token = get_finmind_token()
            if token:
                api.login_by_token(api_token=token)

            raw_chip = api.taiwan_stock_institutional_investors(
                stock_id=stock_num,
                start_date=start_date,
            )

            if raw_chip is not None and not raw_chip.empty:
                raw_chip["net_buy"] = raw_chip["buy"] - raw_chip["sell"]
                raw_chip["date"] = pd.to_datetime(raw_chip["date"])
                pivoted = raw_chip.pivot_table(
                    index="date", columns="name", values="net_buy", aggfunc="sum"
                ).fillna(0)
                df_chip = normalize_chip_df(pivoted)
                chip_status = "ok" if df_chip is not None else "empty"
            else:
                chip_status = "empty"
        except Exception as exc:
            chip_status = "error"
            chip_error = str(exc)
    else:
        chip_status = "not_taiwan"

    return df_price, df_chip, chip_status, chip_error


df_p, df_c, chip_status, chip_error = load_all_data(ticker, stock_id, period)

# 4. 畫面渲染與圖表繪製
if df_p is not None:
    latest_data = df_p.iloc[-1]
    latest_date = df_p.index[-1].strftime("%Y-%m-%d")
    crosses = find_ma_crosses(df_p)

    # ---- 均線交叉訊號（側邊欄）----
    st.sidebar.subheader("📡 均線交叉訊號")
    if pd.notna(latest_data["MA20"]) and pd.notna(latest_data["MA60"]):
        if latest_data["MA20"] > latest_data["MA60"]:
            st.sidebar.success("🟢 多頭排列：MA20 在 MA60 上方")
        else:
            st.sidebar.warning("🔴 空頭排列：MA20 在 MA60 下方")

        if crosses:
            last_type, last_date, _ = crosses[-1]
            days_ago = (df_p.index[-1] - last_date).days
            if last_type == "golden" and days_ago <= 20:
                st.sidebar.success(
                    f"✨ 近期黃金交叉：{last_date.strftime('%Y-%m-%d')}（{days_ago} 天前）"
                )
            elif last_type == "death" and days_ago <= 20:
                st.sidebar.error(
                    f"⚠️ 近期死亡交叉：{last_date.strftime('%Y-%m-%d')}（{days_ago} 天前）"
                )
    else:
        st.sidebar.info("ℹ️ 資料不足，尚無法計算均線交叉")

    # ---- 資訊看板 (Metrics) ----
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"最新收盤價 ({latest_date})", value=f"${latest_data['Close']:.2f}")
    with col2:
        st.metric(label="20MA (月線)", value=f"${latest_data['MA20']:.2f}")
    with col3:
        st.metric(label="60MA (季線)", value=f"${latest_data['MA60']:.2f}")

    # ---- 法人籌碼摘要 ----
    has_institutional_data = False
    foreign_col = sitc_col = dealer_col = None

    if df_c is not None and len(df_c) > 0:
        df_c = df_c.reindex(df_p.index).fillna(0)
        foreign_col = "外資" if "外資" in df_c.columns else None
        sitc_col = "投信" if "投信" in df_c.columns else None
        dealer_col = "自營商" if "自營商" in df_c.columns else None

        if foreign_col or sitc_col or dealer_col:
            has_institutional_data = True
            st.sidebar.success("🟢 法人籌碼數據：串接成功")

            st.subheader("🏦 法人籌碼摘要（張）")
            chip_cols = st.columns(3)

            if foreign_col:
                f5 = sum_net_buy_lots(df_c[foreign_col], 5)
                f20 = sum_net_buy_lots(df_c[foreign_col], 20)
                with chip_cols[0]:
                    st.metric("外資近 5 日", f"{f5:+,.0f}", delta="偏多" if f5 > 0 else "偏空")
                    st.caption(f"近 20 日：{f20:+,.0f} 張")

            if sitc_col:
                s5 = sum_net_buy_lots(df_c[sitc_col], 5)
                s20 = sum_net_buy_lots(df_c[sitc_col], 20)
                with chip_cols[1]:
                    st.metric("投信近 5 日", f"{s5:+,.0f}", delta="偏多" if s5 > 0 else "偏空")
                    st.caption(f"近 20 日：{s20:+,.0f} 張")

            inst_cols = [c for c in [foreign_col, sitc_col, dealer_col] if c]
            if inst_cols:
                total_series = df_c[inst_cols].sum(axis=1)
                t5 = sum_net_buy_lots(total_series, 5)
                t20 = sum_net_buy_lots(total_series, 20)
                with chip_cols[2]:
                    st.metric("三大法人近 5 日", f"{t5:+,.0f}", delta="偏多" if t5 > 0 else "偏空")
                    st.caption(f"近 20 日：{t20:+,.0f} 張")

    # 5. 動態子圖表建立
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
    )

    # ---- 上圖：K 線與均線 ----
    fig.add_trace(
        go.Candlestick(
            x=df_p.index,
            open=df_p["Open"],
            high=df_p["High"],
            low=df_p["Low"],
            close=df_p["Close"],
            name="K 線",
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_p.index,
            y=df_p["MA20"],
            name="20MA (月線)",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_p.index,
            y=df_p["MA60"],
            name="60MA (季線)",
            line=dict(color="#2ca02c", width=1.5, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # 交叉點標記
    golden = [(d, p) for t, d, p in crosses if t == "golden"]
    death = [(d, p) for t, d, p in crosses if t == "death"]
    if golden:
        fig.add_trace(
            go.Scatter(
                x=[d for d, _ in golden],
                y=[p for _, p in golden],
                mode="markers",
                name="黃金交叉",
                marker=dict(symbol="triangle-up", size=12, color="#ffd700", line=dict(width=1, color="#333")),
            ),
            row=1,
            col=1,
        )
    if death:
        fig.add_trace(
            go.Scatter(
                x=[d for d, _ in death],
                y=[p for _, p in death],
                mode="markers",
                name="死亡交叉",
                marker=dict(symbol="triangle-down", size=12, color="#333", line=dict(width=1, color="#d62728")),
            ),
            row=1,
            col=1,
        )

    # ---- 下圖：籌碼判定與量能繪製 ----
    if has_institutional_data:
        if foreign_col:
            foreign_data = pd.Series(df_c[foreign_col]) / 1000
            f_colors = ["#2ca02c" if val >= 0 else "#d62728" for val in foreign_data]
            fig.add_trace(
                go.Bar(
                    x=df_c.index,
                    y=foreign_data,
                    name="外資買賣超 (張)",
                    marker_color=f_colors,
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )
        if sitc_col:
            sitc_data = pd.Series(df_c[sitc_col]) / 1000
            fig.add_trace(
                go.Scatter(
                    x=df_c.index,
                    y=sitc_data,
                    name="投信買賣超 (張)",
                    line=dict(color="#9467bd", width=1.8),
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="法人買賣超 (張)", row=2, col=1)

    if not has_institutional_data:
        if chip_status == "not_taiwan":
            st.sidebar.info("ℹ️ 非台股代碼，下圖顯示「市場總成交量」")
        elif chip_status == "error":
            st.sidebar.warning(f"⚠️ 法人資料取得失敗：{chip_error or '未知錯誤'}")
            st.sidebar.caption("可在 `.streamlit/secrets.toml` 設定 FINMIND_TOKEN 提高 API 額度")
            st.sidebar.info("ℹ️ 已自動切換為「市場總成交量」")
        elif chip_status == "empty":
            st.sidebar.info("ℹ️ 查無法人籌碼資料，顯示「市場總成交量」")
        else:
            st.sidebar.info("ℹ️ 法人資料不可用，顯示「市場總成交量」")
        volume_colors = []
        for i in range(len(df_p)):
            if i == 0 or df_p["Close"].iloc[i] >= df_p["Close"].iloc[i - 1]:
                volume_colors.append("#2ca02c")
            else:
                volume_colors.append("#d62728")

        fig.add_trace(
            go.Bar(
                x=df_p.index,
                y=df_p["Volume_Shares"],
                name="成交量 (張)",
                marker_color=volume_colors,
                opacity=0.6,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="成交量 (張)", row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"{stock_display_name} 走勢與籌碼分析",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        height=650,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)

    st.plotly_chart(fig, width="stretch")

    with st.expander("查看歷史數據原始表格"):
        st.dataframe(df_p.tail(10))
else:
    st.error(f"找不到代碼「{ticker_input}」的數據，請確認代碼是否正確。")
