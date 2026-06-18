import os

import streamlit as st
import streamlit.components.v1 as components
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


def prepare_trading_axis(df_index):
    """將交易日轉為連續 X 軸，去除週末與假日空白。"""
    labels = [d.strftime("%Y-%m-%d") for d in df_index]
    x_vals = list(range(len(labels)))
    date_to_x = {d: i for i, d in enumerate(df_index)}
    step = max(1, len(x_vals) // 8)
    return x_vals, labels, date_to_x, x_vals[::step], labels[::step]


def is_taiwan_stock(stock_code):
    """判斷是否為台股（FinMind 法人資料僅支援台股）。"""
    code = stock_code.upper()
    return code.endswith(".TW") or code.endswith(".TWO") or stock_code.isdigit()


def contains_chinese(text):
    return any("\u4e00" <= c <= "\u9fff" for c in text)


@st.cache_data(ttl=86400)
def get_taiwan_stock_catalog():
    """快取台股代碼、中文名稱與市場別。"""
    try:
        api = DataLoader()
        token = get_finmind_token()
        if token:
            api.login_by_token(api_token=token)
        df = api.taiwan_stock_info()
        if df is None or df.empty:
            return []

        df = df.drop_duplicates(subset=["stock_id", "type"])
        catalog = []
        for _, row in df.iterrows():
            market = row["type"]
            if market == "tpex":
                suffix = ".TWO"
            elif market == "twse":
                suffix = ".TW"
            else:
                continue
            catalog.append(
                {
                    "stock_id": str(row["stock_id"]),
                    "name": row["stock_name"],
                    "suffix": suffix,
                }
            )
        return catalog
    except Exception:
        return []


def get_taiwan_stock_name_map():
    """台股代碼 → 中文名稱對照表。"""
    catalog = get_taiwan_stock_catalog()
    return {item["stock_id"]: item["name"] for item in catalog}


def search_stocks_by_name(query):
    """依中文名稱搜尋台股，支援完整或部分比對。"""
    catalog = get_taiwan_stock_catalog()
    q = query.strip()
    if not q:
        return []

    exact = [s for s in catalog if s["name"] == q]
    if exact:
        return exact

    partial = [s for s in catalog if q in s["name"]]
    partial.sort(key=lambda x: (not x["name"].startswith(q), len(x["name"]), x["stock_id"]))

    seen = set()
    results = []
    for item in partial:
        if item["stock_id"] not in seen:
            seen.add(item["stock_id"])
            results.append(item)
    return results[:30]


def resolve_stock_match(match):
    """將搜尋結果轉為 yfinance 代碼與 stock_id。"""
    return f"{match['stock_id']}{match['suffix']}", match["stock_id"]


def parse_ticker_input(raw_input):
    """將使用者輸入轉為 yfinance 代碼與 FinMind stock_id。支援數字、代碼或中文名。"""
    raw = raw_input.strip()
    if not raw:
        return "", "", []

    if contains_chinese(raw):
        matches = search_stocks_by_name(raw)
        if len(matches) == 1:
            ticker, stock_id = resolve_stock_match(matches[0])
            return ticker, stock_id, matches
        return "", "", matches

    code = raw.upper()
    if code.isdigit():
        catalog = get_taiwan_stock_catalog()
        match = next((s for s in catalog if s["stock_id"] == code), None)
        suffix = match["suffix"] if match else ".TW"
        return f"{code}{suffix}", code, []

    if code.endswith(".TW") or code.endswith(".TWO"):
        return code, code.split(".")[0], []

    return code, code.split(".")[0], []


def clean_price_df(df_price):
    """清理 yfinance 資料：扁平化欄位並移除收盤價缺失的列。"""
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price = df_price.copy()
        df_price.columns = df_price.columns.get_level_values(0)
    return df_price.dropna(subset=["Close"])


def create_finmind_api():
    api = DataLoader()
    token = get_finmind_token()
    if token:
        api.login_by_token(api_token=token)
    return api


def load_taiwan_price(api, stock_num, start_date):
    """台股股價優先使用 FinMind（yfinance 台股資料常延遲或缺漏）。"""
    df = api.taiwan_stock_daily(stock_id=stock_num, start_date=start_date)
    if df is None or df.empty:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df_price = pd.DataFrame(
        {
            "Open": df["open"],
            "High": df["max"],
            "Low": df["min"],
            "Close": df["close"],
            "Volume": df["Trading_Volume"],
        }
    )
    return df_price.dropna(subset=["Close"])


def load_yfinance_price(stock_code, start_date):
    df_price = yf.download(stock_code, start=start_date)
    if df_price.empty:
        return None
    return clean_price_df(df_price)


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


def get_stock_display_name(stock_code, stock_num):
    """取得標的顯示名稱（台股優先使用中文名）。"""
    if is_taiwan_stock(stock_code):
        cn_name = get_taiwan_stock_name_map().get(str(stock_num))
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


def evaluate_entry_signal(df_p, latest_data, crosses, df_c, has_institutional_data):
    """綜合均線、交叉訊號與法人籌碼，產生進場燈號。"""
    checks = []

    if pd.notna(latest_data.get("MA20")) and pd.notna(latest_data.get("MA60")):
        ok = latest_data["MA20"] > latest_data["MA60"]
        checks.append(("均線排列", ok, "MA20 在 MA60 上方（多頭）" if ok else "MA20 在 MA60 下方（空頭）"))

    if pd.notna(latest_data.get("MA20")):
        ok = latest_data["Close"] > latest_data["MA20"]
        checks.append(("站上月線", ok, "收盤價高於 20MA" if ok else "收盤價低於 20MA"))

    if pd.notna(latest_data.get("MA60")):
        ok = latest_data["Close"] > latest_data["MA60"]
        checks.append(("站上季線", ok, "收盤價高於 60MA" if ok else "收盤價低於 60MA"))

    if crosses:
        last_type, last_date, _ = crosses[-1]
        days_ago = (df_p.index[-1] - last_date).days
        if days_ago <= 20:
            if last_type == "golden":
                checks.append(("交叉訊號", True, f"近 {days_ago} 天出現黃金交叉"))
            else:
                checks.append(("交叉訊號", False, f"近 {days_ago} 天出現死亡交叉"))

    if has_institutional_data and df_c is not None:
        if "外資" in df_c.columns:
            f5 = sum_net_buy_lots(df_c["外資"], 5)
            checks.append(("外資近 5 日", f5 > 0, f"累計 {f5:+,.0f} 張"))
        if "投信" in df_c.columns:
            s5 = sum_net_buy_lots(df_c["投信"], 5)
            checks.append(("投信近 5 日", s5 > 0, f"累計 {s5:+,.0f} 張"))
        inst_cols = [c for c in ["外資", "投信", "自營商"] if c in df_c.columns]
        if inst_cols:
            t5 = sum_net_buy_lots(df_c[inst_cols].sum(axis=1), 5)
            checks.append(("三大法人近 5 日", t5 > 0, f"累計 {t5:+,.0f} 張"))

    if not checks:
        return "yellow", "🟡 進場燈號：資料不足", "查詢區間過短，請改用 3 個月以上再判斷", checks

    bullish = sum(1 for _, ok, _ in checks if ok)
    ratio = bullish / len(checks)
    has_death = any(not ok and "死亡交叉" in detail for _, ok, detail in checks)

    if ratio >= 0.7 and not has_death:
        return "green", "🟢 進場燈號：偏多", "多項指標同步偏多，可列入進場觀察清單", checks
    if ratio <= 0.35 or has_death:
        return "red", "🔴 進場燈號：偏空", "趨勢偏弱或出現死亡交叉，建議暫不進場", checks
    return "yellow", "🟡 進場燈號：觀望", "多空指標分歧，建議等待更明確訊號", checks


def render_entry_light(light, title, message, checks, show_details=True):
    """渲染進場燈號區塊。"""
    colors = {
        "green": ("#d4edda", "#28a745"),
        "yellow": ("#fff3cd", "#ffc107"),
        "red": ("#f8d7da", "#dc3545"),
    }
    bg, border = colors[light]
    st.markdown(
        f"""
        <div style="padding:1rem 1.25rem;border-radius:8px;background:{bg};
        border-left:6px solid {border};margin-bottom:1rem;color:#111;">
        <h3 style="margin:0 0 0.25rem 0;color:#111;">{title}</h3>
        <p style="margin:0;font-size:1rem;color:#111;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_details:
        with st.expander("查看燈號判斷細項"):
            for name, ok, detail in checks:
                icon = "🟢" if ok else "🔴"
                st.write(f"{icon} **{name}**：{detail}")
            st.caption("※ 燈號僅供技術面參考，不構成投資建議。")


def inject_responsive_css():
    """注入手機版 RWD 樣式。"""
    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
            .block-container { padding-top: 0.75rem; padding-left: 0.75rem; padding-right: 0.75rem; }
            section[data-testid="stSidebar"] { display: none !important; }
            button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
            div[data-testid="stPlotlyChart"] { min-height: 380px !important; }
            div[data-testid="stPlotlyChart"] iframe { height: 380px !important; }
            h1 { font-size: 1.35rem !important; }
            div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
            div[data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_mobile_detect():
    """小螢幕自動啟用手機精簡模式（透過 query param）。"""
    components.html(
        """
        <script>
        (function () {
            const win = window.parent || window;
            const params = new URLSearchParams(win.location.search);
            const isMobile = win.innerWidth <= 768;
            if (isMobile && params.get("mobile") !== "1") {
                params.set("mobile", "1");
                win.location.search = params.toString();
            } else if (!isMobile && params.get("mobile") === "1") {
                params.delete("mobile");
                win.location.search = params.toString();
            }
        })();
        </script>
        """,
        height=0,
    )


# 1. 網頁基本設定
st.set_page_config(page_title="投資決策視覺化工具", layout="wide")
inject_responsive_css()
inject_mobile_detect()

compact_mode = st.query_params.get("mobile", "0") == "1"

# 2. 主畫面查詢設定（手機無需開側邊欄）
with st.container():
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        ticker_input = st.text_input(
            "輸入股票代碼或中文名 (如 2330、台積電；美股如 AAPL)",
            value="2330",
        )
    with search_col2:
        period = st.selectbox(
            "查詢區間",
            options=list(PERIOD_DAYS.keys()),
            index=3,
            format_func=lambda x: PERIOD_LABELS[x],
        )

    if compact_mode:
        show_full = st.toggle(
            "顯示完整資訊",
            value=False,
            help="顯示法人籌碼摘要、燈號細項與歷史表格",
        )
        compact_mode = not show_full
    else:
        compact_mode = st.toggle("手機精簡模式", value=False, help="精簡版面，適合小螢幕")

ticker, stock_id, name_matches = parse_ticker_input(ticker_input)

if len(name_matches) > 1:
    selected = st.selectbox(
        "找到多筆符合，請選擇：",
        options=name_matches,
        format_func=lambda m: f"{m['stock_id']} {m['name']}",
    )
    ticker, stock_id = resolve_stock_match(selected)
elif ticker_input.strip() and not ticker and not name_matches:
    st.warning(f"找不到「{ticker_input.strip()}」對應的台股標的")

stock_display_name = get_stock_display_name(ticker, stock_id)
chart_height = 400 if compact_mode else 650

st.title(stock_display_name if stock_id else "股票趨勢與量能決策儀表板")
if not compact_mode:
    st.markdown("結合 K 線、移動平均線與成交量能，加速你的波段進出場決策。")


# 3. 核心資料抓取與處理函數
@st.cache_data(ttl=3600)
def load_all_data(stock_code, stock_num, data_period):
    start_date = (
        pd.Timestamp.now() - pd.Timedelta(days=PERIOD_DAYS[data_period])
    ).strftime("%Y-%m-%d")

    df_price = None
    chip_status = "not_taiwan"
    chip_error = None
    df_chip = None

    if is_taiwan_stock(stock_code):
        try:
            api = create_finmind_api()
            df_price = load_taiwan_price(api, stock_num, start_date)

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
            chip_error = str(exc)
            chip_status = "error"
            df_price = None

    if df_price is None or df_price.empty:
        df_price = load_yfinance_price(stock_code, start_date)

    if df_price is None or df_price.empty:
        return None, None, chip_status, chip_error

    df_price["MA20"] = df_price["Close"].rolling(window=20).mean()
    df_price["MA60"] = df_price["Close"].rolling(window=60).mean()
    df_price["Volume_Shares"] = df_price["Volume"] / 1000

    return df_price, df_chip, chip_status, chip_error


if ticker:
    df_p, df_c, chip_status, chip_error = load_all_data(ticker, stock_id, period)
else:
    df_p, df_c, chip_status, chip_error = None, None, "empty", None

# 4. 畫面渲染與圖表繪製
if df_p is not None:
    latest_data = df_p.iloc[-1]
    latest_date = df_p.index[-1].strftime("%Y-%m-%d")
    crosses = find_ma_crosses(df_p)

    # ---- 法人資料預處理 ----
    has_institutional_data = False
    foreign_col = sitc_col = dealer_col = None
    if df_c is not None and len(df_c) > 0:
        df_c = df_c.reindex(df_p.index).fillna(0)
        foreign_col = "外資" if "外資" in df_c.columns else None
        sitc_col = "投信" if "投信" in df_c.columns else None
        dealer_col = "自營商" if "自營商" in df_c.columns else None
        has_institutional_data = bool(foreign_col or sitc_col or dealer_col)

    # ---- 進場燈號 ----
    light, light_title, light_msg, light_checks = evaluate_entry_signal(
        df_p, latest_data, crosses, df_c, has_institutional_data
    )
    render_entry_light(
        light, light_title, light_msg, light_checks, show_details=not compact_mode
    )

    if not compact_mode:
        st.sidebar.subheader("進場燈號")
        if light == "green":
            st.sidebar.success(light_title)
        elif light == "red":
            st.sidebar.error(light_title)
        else:
            st.sidebar.warning(light_title)

        st.sidebar.subheader("均線交叉訊號")
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
    if has_institutional_data and not compact_mode:
        st.sidebar.success("🟢 法人籌碼數據：串接成功")

        st.subheader("法人籌碼摘要（張）")
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

    # 5. 動態子圖表建立（X 軸僅顯示實際交易日，無週末／假日空白）
    x_vals, x_labels, date_to_x, tick_vals, tick_text = prepare_trading_axis(df_p.index)

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
            x=x_vals,
            open=df_p["Open"],
            high=df_p["High"],
            low=df_p["Low"],
            close=df_p["Close"],
            name="K 線",
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
            customdata=x_labels,
            hovertemplate="日期: %{customdata}<br>"
            "開: %{open}<br>高: %{high}<br>低: %{low}<br>收: %{close}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_p["MA20"],
            name="20MA (月線)",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            customdata=x_labels,
            hovertemplate="日期: %{customdata}<br>20MA: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_p["MA60"],
            name="60MA (季線)",
            line=dict(color="#2ca02c", width=1.5, dash="dot"),
            customdata=x_labels,
            hovertemplate="日期: %{customdata}<br>60MA: %{y:.2f}<extra></extra>",
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
                x=[date_to_x[d] for d, _ in golden if d in date_to_x],
                y=[p for d, p in golden if d in date_to_x],
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
                x=[date_to_x[d] for d, _ in death if d in date_to_x],
                y=[p for d, p in death if d in date_to_x],
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
                    x=x_vals,
                    y=foreign_data,
                    name="外資買賣超 (張)",
                    marker_color=f_colors,
                    opacity=0.7,
                    customdata=x_labels,
                    hovertemplate="日期: %{customdata}<br>外資: %{y:,.0f} 張<extra></extra>",
                ),
                row=2,
                col=1,
            )
        if sitc_col:
            sitc_data = pd.Series(df_c[sitc_col]) / 1000
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=sitc_data,
                    name="投信買賣超 (張)",
                    line=dict(color="#9467bd", width=1.8),
                    customdata=x_labels,
                    hovertemplate="日期: %{customdata}<br>投信: %{y:,.0f} 張<extra></extra>",
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="法人買賣超 (張)", row=2, col=1)

    if not has_institutional_data and not compact_mode:
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

    if not has_institutional_data:
        volume_colors = []
        for i in range(len(df_p)):
            if i == 0 or df_p["Close"].iloc[i] >= df_p["Close"].iloc[i - 1]:
                volume_colors.append("#2ca02c")
            else:
                volume_colors.append("#d62728")

        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=df_p["Volume_Shares"],
                name="成交量 (張)",
                marker_color=volume_colors,
                opacity=0.6,
                customdata=x_labels,
                hovertemplate="日期: %{customdata}<br>成交量: %{y:,.0f} 張<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="成交量 (張)", row=2, col=1)

    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"{stock_display_name} 走勢與籌碼分析",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        height=chart_height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            font=dict(size=10 if compact_mode else 12),
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)

    st.plotly_chart(fig, width="stretch")

    if not compact_mode:
        with st.expander("查看歷史數據原始表格"):
            st.dataframe(df_p.tail(10))
else:
    st.error(f"找不到代碼「{ticker_input}」的數據，請確認代碼是否正確。")
