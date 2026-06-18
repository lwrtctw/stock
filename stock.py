import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# 1. 網頁標題與設定
st.set_page_config(page_title="投資決策視覺化工具", layout="wide")
st.title("📈 股票趨勢快速決策儀表板")
st.markdown("輸入股票代碼，快速查看移動平均線（MA）走勢與黃金/死亡交叉，輔助進出場決策。")

# 2. 側邊欄：使用者輸入參數
st.sidebar.header("📊 參數設定")
# 支援美股 (如 AAPL) 或台股 (如 2330.TW)
ticker = st.sidebar.text_input("輸入股票代碼 (台股請加 .TW)", value="2330.TW")
period = st.sidebar.selectbox("查詢區間", options=["3mo", "6mo", "1y", "2y"], index=2)

# 3. 資料抓取與處理
@st.cache_data(ttl=3600)  # 快取機制：一小時內重複輸入不重新下載，加速載入
def load_data(stock_code, data_period):
    # 抓取歷史K線
    df = yf.download(stock_code, period=data_period)
    if df.empty:
        return None
    
    # 清理多重索引 (yfinance 有時會產生)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # 計算技術指標 (均線)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    return df

df = load_data(ticker, period)

# 4. 畫面呈現
if df is not None:
    # 取得最新一天的數據做摘要
    latest_data = df.iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    
    # 資訊看板 (Metrics)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"最新收盤價 ({latest_date})", value=f"${latest_data['Close']:.2f}")
    with col2:
        st.metric(label="20MA (月線)", value=f"${latest_data['MA20']:.2f}")
    with col3:
        st.metric(label="60MA (季線)", value=f"${latest_data['MA60']:.2f}")
        
    # 決策輔助小提示
    st.subheader("💡 決策參考")
    if latest_data['MA20'] > latest_data['MA60']:
        st.success("🟢 目前月線在季線之上（黃金交叉/多頭排列），暗示中期趨勢偏多。")
    else:
        st.error("🔴 目前月線在季線之下（死亡交叉/空頭排列），建議謹慎佈局。")

    # 5. 繪製互動式圖表 (Plotly)
    st.subheader("📊 歷史走勢與技術指標")
    fig = go.Figure()
    
    # 股價線
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盤價', line=dict(color='#1f77b4', width=2)))
    # 20 MA
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20MA (月線)', line=dict(color='#ff7f0e', width=1.5, dash='dash')))
    # 60 MA
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='60MA (季線)', line=dict(color='#2ca02c', width=1.5, dash='dot')))
    
    # 圖表樣式優化
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="日期",
        yaxis_title="價格"
    )
    
    # 在 Streamlit 中顯示圖表
    st.plotly_chart(fig, use_container_width=True)
    
    # 可選：顯示原始數據表格
    with st.expander("查看原始數據"):
        st.dataframe(df.tail(10))
else:
    st.error(st.error(f"找不到代碼「{ticker}」的數據，請確認代碼是否正確（例如台股台積電為 2330.TW，美股蘋果為 AAPL）。"))