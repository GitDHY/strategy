import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
import requests
import io

import datetime

# Set page config must be the first streamlit command
st.set_page_config(layout="wide", page_title="Stock Strategy Analyzer")

# --- Helper Functions for Indicators ---

# Removed cache for debugging connection issues
def fetch_fred_data(series_id):
    """
    Robust fetch for FRED data using requests with User-Agent.
    Priority:
    1. Local file (fred_{series_id}.csv)
    2. Network fetch (fred.stlouisfed.org)
    """
    # 1. Check local file override
    local_file = os.path.join(os.path.dirname(__file__), f"fred_{series_id}.csv")
    if os.path.exists(local_file):
        try:
            df = pd.read_csv(local_file, parse_dates=['observation_date'], index_col='observation_date')
            df.columns = [series_id]
            # st.toast(f"Using local file for {series_id}", icon="📂") # Optional toast
            return df
        except Exception as e:
            st.warning(f"Found local file {local_file} but failed to read: {e}")

    # 2. Network Fetch
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    import time
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Added verify=False to bypass SSL errors (common in some networks)
            # Suppress warnings for verify=False
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Increased timeout to 30s
            response = requests.get(url, headers=headers, timeout=30, verify=False)
            response.raise_for_status()
            content = response.content.decode('utf-8')
            df = pd.read_csv(io.StringIO(content), parse_dates=['observation_date'], index_col='observation_date')
            df.columns = [series_id]
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1) # Wait 1s before retry
                continue
            
            # Show friendly warning instead of error
            st.warning(f"⚠️ 无法连接 FRED 数据源 ({series_id})。网络连接被切断。\n\n**解决方法**：请展开页面顶部的 **“📂 手动导入宏观数据”** 面板，上传该数据文件即可恢复正常。")
            print(f"Error fetching FRED data ({series_id}): {e}")
            # Return empty DataFrame on failure
            return pd.DataFrame()

@st.cache_data
def get_fred_indpro():
    """
    Fetches Industrial Production Index (INDPRO) from FRED as a proxy for Economic Cycle.
    """
    df = fetch_fred_data("INDPRO")
    if not df.empty:
        # Calculate YoY Growth
        df['INDPRO_YoY'] = df['INDPRO'].pct_change(12) * 100
        return df
    else:
        return pd.DataFrame()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calculate_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower # Upper, Mid, Lower

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_kdj(df, period=9):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    
    # Calculate RSV
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    
    # Calculate K, D, J (using alpha=1/3 for standard smoothing)
    # Initialize with 50 if needed, but ewm handles start well enough
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci

@st.cache_data
def get_vix_data(start, end):
    """
    Fetches VIX data from Yahoo Finance.
    """
    try:
        # Added auto_adjust=False to maintain consistent behavior
        df = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)
        return df[['Close']].rename(columns={'Close': 'VIX'})
    except Exception:
        return pd.DataFrame()

# --- Portfolio Manager ---
PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolios.json")

def load_portfolios():
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_portfolio(name, tickers, weights):
    data = load_portfolios()
    data[name] = {"tickers": tickers, "weights": weights}
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

def delete_portfolio(name):
    data = load_portfolios()
    if name in data:
        del data[name]
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(data, f, indent=4)

# --- Shared Logic for Backtest & State Machine ---

def get_target_percentages(s, gold_bear=False, value_regime=False):
    """
    Returns target asset allocation based on macro state.
    Shared by State Machine Diagnosis and Backtest.
    """
    targets = {}
    
    if s == "INFLATION_SHOCK":
        # Rate Spike: Kill Duration, Cash is King
        targets = {
            'IWY': 0.00, 'WTMF': 0.40, 'LVHI': 0.10,
            'G3B.SI': 0.10, 'MBH.SI': 0.00, 'GSD.SI': 0.25,
            'SRT.SI': 0.00, 'AJBU.SI': 0.00
        }
    elif s == "DEFLATION_RECESSION":
        # Recession: Long Bonds, Gold
        targets = {
            'IWY': 0.10, 'WTMF': 0.15, 'LVHI': 0.05,
            'G3B.SI': 0.05, 'MBH.SI': 0.35, 'GSD.SI': 0.25,
            'SRT.SI': 0.00, 'AJBU.SI': 0.00
        }
    elif s == "EXTREME_ACCUMULATION":
        # Buy Dip
        targets = {
            'IWY': 0.70, 'WTMF': 0.00, 'LVHI': 0.00,
            'G3B.SI': 0.10, 'MBH.SI': 0.05, 'GSD.SI': 0.05,
            'SRT.SI': 0.06, 'AJBU.SI': 0.04
        }
    elif s == "CAUTIOUS_TREND":
        # Bear Trend: Defensive
        growth_w = 0.15
        value_w = 0.25
        if value_regime:
            growth_w = 0.10
            value_w = 0.30
        
        targets = {
            'IWY': growth_w, 'WTMF': 0.10, 'LVHI': value_w,
            'G3B.SI': 0.20, 'MBH.SI': 0.15, 'GSD.SI': 0.10,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }
    elif s == "CAUTIOUS_VOL":
        # High Vol: Hedge
        targets = {
            'IWY': 0.40, 'WTMF': 0.20, 'LVHI': 0.10,
            'G3B.SI': 0.10, 'MBH.SI': 0.10, 'GSD.SI': 0.05,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }
    else: # NEUTRAL
        # Style Rotation
        growth_w = 0.50
        value_w = 0.10
        if value_regime:
            growth_w = 0.40
            value_w = 0.20
            
        targets = {
            'IWY': growth_w, 'WTMF': 0.10, 'LVHI': value_w,
            'G3B.SI': 0.10, 'MBH.SI': 0.10, 'GSD.SI': 0.05,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }
        
    # Gold Trend Filter: If Gold is Bearish, cut allocation by half, move to WTMF (Cash proxy)
    if gold_bear and targets.get('GSD.SI', 0) > 0:
        cut_amount = targets['GSD.SI'] * 0.5
        targets['GSD.SI'] -= cut_amount
        targets['WTMF'] += cut_amount
    
    return targets

def run_dynamic_backtest(df_states, start_date, end_date, initial_capital=10000.0):
    """
    Simulates the strategy over historical states.
    df_states: DataFrame with 'State', 'Gold_Bear', 'Value_Regime' columns, indexed by Date.
    """
    # 1. Define Asset Universe
    assets = ['IWY', 'WTMF', 'LVHI', 'G3B.SI', 'MBH.SI', 'GSD.SI', 'SRT.SI', 'AJBU.SI', 'TLT', 'SPY']
    
    # 2. Fetch Price Data
    fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=7)
    
    try:
        price_data = yf.download(assets, start=fetch_start, end=end_date, progress=False, auto_adjust=False)['Adj Close']
    except:
        # Fallback if Adj Close issue
        try:
             price_data = yf.download(assets, start=fetch_start, end=end_date, progress=False, auto_adjust=False)['Close']
        except Exception as e:
            return None, f"Data fetch failed: {e}"

    if price_data.empty:
         return None, "No price data fetched."

    # Fill missing
    price_data = price_data.fillna(method='ffill').fillna(method='bfill')
    
    # Filter to requested range
    price_data = price_data[(price_data.index >= pd.to_datetime(start_date)) & (price_data.index <= pd.to_datetime(end_date))]
    if price_data.empty:
         return None, "Price data empty after filtering."

    # Align states with prices
    common_idx = price_data.index.intersection(df_states.index)
    price_data = price_data.loc[common_idx]
    df_states = df_states.loc[common_idx]
    
    if len(price_data) < 10:
        return None, "Insufficient data points for backtest."

    # 3. Strategy Simulation (Daily Rebalancing Approximation)
    portfolio_values = []
    current_val = initial_capital
    
    # We iterate daily. To speed up, we could vectorise, but logic is complex.
    # Logic: Daily return = Sum(Weight_i * Return_i)
    # This assumes we rebalance to target weights DAILY.
    
    returns_df = price_data.pct_change().fillna(0)
    
    for date, row in df_states.iterrows():
        # Get Targets for today (based on today's state)
        # Note: In reality, we trade tomorrow based on today's close state?
        # Or trade today at close? Assuming trade at close.
        s = row['State']
        gb = row['Gold_Bear']
        vr = row['Value_Regime']
        
        targets = get_target_percentages(s, gold_bear=gb, value_regime=vr)
        
        # Calculate Portfolio Return for this day
        # If we rebalanced yesterday to these weights?
        # Simpler: Assume we hold these weights TODAY.
        # So return is sum(w * r).
        
        daily_ret = 0.0
        if date in returns_df.index:
            rets = returns_df.loc[date]
            for t, w in targets.items():
                if t in rets:
                    daily_ret += w * rets[t]
        
        current_val = current_val * (1 + daily_ret)
        portfolio_values.append(current_val)
        
    s_strategy = pd.Series(portfolio_values, index=df_states.index, name="Strategy")
    
    # 4. Benchmarks
    # IWY
    s_iwy = pd.Series(dtype=float)
    if 'IWY' in price_data.columns:
        iwy_prices = price_data['IWY']
        s_iwy = (iwy_prices / iwy_prices.iloc[0]) * initial_capital
        s_iwy.name = "IWY (Growth)"

    # 60/40
    s_6040 = pd.Series(dtype=float)
    if 'SPY' in price_data.columns and 'TLT' in price_data.columns:
        spy = price_data['SPY'] / price_data['SPY'].iloc[0]
        tlt = price_data['TLT'] / price_data['TLT'].iloc[0]
        s_6040 = (0.6 * spy + 0.4 * tlt) * initial_capital
        s_6040.name = "60/40 (SPY/TLT)"
        
    # Neutral Config (Buy & Hold / Fixed Weight)
    default_targets = get_target_percentages("NEUTRAL", False, False)
    neutral_vals = []
    curr_n = initial_capital
    
    for date in df_states.index:
        daily_ret = 0.0
        if date in returns_df.index:
            rets = returns_df.loc[date]
            for t, w in default_targets.items():
                if t in rets:
                    daily_ret += w * rets[t]
        curr_n = curr_n * (1 + daily_ret)
        neutral_vals.append(curr_n)
        
    s_neutral = pd.Series(neutral_vals, index=df_states.index, name="Neutral Config")
    
    return pd.DataFrame({
        "Dynamic Strategy": s_strategy,
        "IWY (Benchmark)": s_iwy,
        "60/40 (Balanced)": s_6040,
        "Neutral (Fixed)": s_neutral
    }), None


# --- Page 1: Single Stock Analysis ---

def render_single_stock_analysis():
    st.header("📈 单只股票策略分析 (Single Stock Analysis)")

    # Sidebar Inputs
    st.sidebar.header("配置 (Single Stock)")
    market = st.sidebar.radio("市场", ["US (美股)", "SG (新加坡)"])
    ticker_input = st.sidebar.text_input("股票代码 (e.g., AAPL, D05)", value="AAPL")
    start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("结束日期", pd.to_datetime("today"))

    st.sidebar.subheader("Strategy 4 · Smart DCA")
    strat4_mode = st.sidebar.radio(
        "Risk Profile (风险偏好)",
        ["Standard (Balanced)", "Conservative (Low Drawdown)", "Aggressive (Growth)"],
        index=0,
        help="Conservative: 更快止损，上涨时分批止盈，不接飞刀。\nAggressive: 允许左侧抄底，不止盈，追求复利。"
    )

    with st.sidebar.expander("⚙️ Advanced Parameters (高级设置)", expanded=False):
        st.markdown("**买入增强条件 (Accumulate Boosters)**")
        p_vix_high = st.number_input("VIX High Threshold (Level 1)", value=30, step=1, help="VIX 高于此值加 1 单位")
        p_vix_extreme = st.number_input("VIX Extreme Threshold (Level 2)", value=40, step=1, help="VIX 高于此值加 2 单位")
        
        p_dd_1 = st.slider("Drawdown Level 1 (%)", -50, -10, -20, step=5, help="回撤超过此幅度加 1 单位")
        p_dd_2 = st.slider("Drawdown Level 2 (%)", -60, -15, -30, step=5)
        
        p_val_1 = st.slider("Value Factor 1 (% of MA200)", 0.7, 1.0, 0.9, 0.01, help="价格低于 0.9 * MA200 加 1 单位")
        p_val_2 = st.slider("Value Factor 2 (% of MA200)", 0.6, 0.95, 0.85, 0.01)
        
        st.markdown("---")
        st.markdown("**卖出策略优化 (Exit Strategy)**")
        p_weeks_below_ma10 = st.number_input("Weeks Below Month MA10", value=2, min_value=1, max_value=5, help="[Legacy] 连续多少周低于月线 MA10 触发趋势减弱信号")
        
        col_exit_1, col_exit_2 = st.columns(2)
        with col_exit_1:
            p_exit_ma_period = st.selectbox("Trend Guard MA (趋势防守)", [20, 60, 120, 200], index=1, help="价格跌破此均线增加卖出权重。建议: 60(季线)或120(半年线)")
        with col_exit_2:
            p_atr_mult = st.number_input("Trailing Stop ATR (移动止盈)", value=3.0, step=0.5, min_value=1.5, help="从持仓高点回撤 N倍ATR 即清仓。设为 0 禁用。")
            
        p_max_units = st.number_input("Max Units Cap", value=5, min_value=1, max_value=10, help="单次最大买入单位限制")

    st.sidebar.info(
        "**Long-Term Weekly ETF Strategy (Smart State)**\n\n"
        "**1. Accumulate State (Regular Buying):**\n"
        "- Base: 1 unit/week\n"
        f"- **Boosters**: VIX≥{p_vix_high}/{p_vix_extreme}, DD≤{p_dd_1}%/{p_dd_2}%, Value<{p_val_1}/{p_val_2}*MA200\n"
        f"- **Cap**: Max {p_max_units} units. **Bear Filter**: Halved if Price < MA200 & MA200↓\n\n"
        "**2. Exit State (Clearance):**\n"
        f"- **Trailing Stop**: Drop > {p_atr_mult}*ATR from High\n"
        f"- **Trend Break**: Price < MA{p_exit_ma_period}\n"
        "- Trigger: ≥2 Sell Signals (Mixed Weights)\n\n"
        "**3. Re-entry State (Recovery):**\n"
        "- Trigger: Price > MA200 & MA200↑ & 6M Mom > 0\n"
        "- Action: **Tentative Buy** (0.5 units/week)."
    )

    # Process Ticker
    ticker = ticker_input.strip().upper()
    if market == "SG (新加坡)" and not ticker.endswith(".SI"):
        ticker += ".SI"

    if st.sidebar.button("Analyze Stock"):
        with st.spinner(f"Fetching data for {ticker}..."):
            # 1. Fetch Data
            fetch_start = pd.to_datetime(start_date) - pd.DateOffset(months=18)
            try:
                df_fred = get_fred_indpro()
                df_vix = get_vix_data(fetch_start, end_date)
                # Added auto_adjust=False to maintain consistent behavior
                df_daily = yf.download(ticker, start=fetch_start, end=end_date, progress=False, auto_adjust=False)
                
                if df_daily.empty:
                    st.error("No data found. Please check the ticker symbol.")
                    return
                
                if isinstance(df_daily.columns, pd.MultiIndex):
                     df_daily.columns = df_daily.columns.get_level_values(0)

                # Get Stock Info Name
                try:
                    stock_info = yf.Ticker(ticker).info
                    stock_name = stock_info.get('longName', ticker)
                except:
                    stock_name = ticker

                st.subheader(f"{stock_name} ({ticker})")

                # 2. Prepare Data (Daily)
                df_daily['MA20'] = df_daily['Close'].rolling(window=20).mean()
                df_daily['MA60'] = df_daily['Close'].rolling(window=60).mean()
                df_daily['MA120'] = df_daily['Close'].rolling(window=120).mean()
                df_daily['MA200'] = df_daily['Close'].rolling(window=200).mean()
                
                rolling_max = df_daily['Close'].rolling(window=252, min_periods=1).max()
                df_daily['Drawdown'] = (df_daily['Close'] / rolling_max - 1) * 100
                
                df_daily['MACD'], df_daily['MACD_Signal'], df_daily['MACD_Hist'] = calculate_macd(df_daily['Close'])
                df_daily['RSI'] = calculate_rsi(df_daily['Close'])
                df_daily['BOLL_Upper'], df_daily['BOLL_Mid'], df_daily['BOLL_Lower'] = calculate_bollinger(df_daily['Close'])
                df_daily['ATR'] = calculate_atr(df_daily)
                df_daily['K'], df_daily['D'], df_daily['J'] = calculate_kdj(df_daily)
                df_daily['CCI'] = calculate_cci(df_daily)

                # 3. Prepare Data (Monthly)
                df_monthly = df_daily.resample('M').agg({'Close': 'last'})
                df_monthly['MA10_Month'] = df_monthly['Close'].rolling(window=10).mean()
                df_monthly['Mom_12M'] = df_monthly['Close'].diff(12)
                df_monthly['Mom_6M'] = df_monthly['Close'].diff(6)

                # 4. Merge Data
                df_monthly_resampled = df_monthly[['MA10_Month', 'Mom_12M', 'Mom_6M']].reindex(df_daily.index, method='ffill')
                df_monthly_resampled = df_monthly_resampled.rename(columns={'MA10_Month': 'Month_MA10', 'Mom_12M': 'Month_Mom_12M', 'Mom_6M': 'Month_Mom_6M'})
                
                df_fred_resampled = pd.DataFrame()
                if not df_fred.empty:
                    df_fred_resampled = df_fred.reindex(df_daily.index, method='ffill')
                    
                df_vix_resampled = pd.DataFrame()
                if not df_vix.empty:
                    df_vix_resampled = df_vix.reindex(df_daily.index, method='ffill')

                dfs_to_concat = [df_daily, df_monthly_resampled]
                if not df_fred_resampled.empty:
                     dfs_to_concat.append(df_fred_resampled)
                if not df_vix_resampled.empty:
                     dfs_to_concat.append(df_vix_resampled)

                df = pd.concat(dfs_to_concat, axis=1)
                
                df['Volatility_Ann'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
                
                df = df[df.index >= pd.to_datetime(start_date)]
                if df.empty:
                    st.warning("所选时间范围内没有足够的数据，请调整日期后重试。")
                    return
                
                # 5. Apply Strategy 4
                trades = []
                dca_positions = []
                strat4_buys = []
                strat4_sells = []

                df['Week'] = df.index.to_period('W')
                week_ends = df.groupby('Week').apply(lambda x: x.index[-1])
                week_end_dates = set(week_ends.values)
                df = df.drop(columns=['Week'])

                avg_costs = [np.nan] * len(df)
                weeks_below_ma10 = 0
                first_ma200 = df.iloc[0]['MA200'] if 'MA200' in df.columns else np.nan
                prev_week_ma200 = first_ma200 if not np.isnan(first_ma200) else 0
                strategy_state = "ACCUMULATE"
                trade_high_price = 0.0

                for i in range(1, len(df)):
                    curr = df.iloc[i]
                    date = df.index[i]
                    
                    if dca_positions:
                        if curr['Close'] > trade_high_price:
                            trade_high_price = curr['Close']

                    if date in week_end_dates:
                        ma200_current = curr.get('MA200', np.nan)
                        ma200_slope = 0
                        if not np.isnan(ma200_current) and prev_week_ma200 > 0:
                            ma200_slope = ma200_current - prev_week_ma200

                        ma10_m = curr.get('Month_MA10', np.nan)
                        if not np.isnan(ma10_m) and curr['Close'] < ma10_m:
                            weeks_below_ma10 += 1
                        else:
                            weeks_below_ma10 = 0

                        mom6 = curr.get('Month_Mom_6M', 0)
                        mom12 = curr.get('Month_Mom_12M', 0)

                        sell_conditions = 0
                        current_sell_reasons = []

                        atr_val = curr.get('ATR', 0)
                        if dca_positions and p_atr_mult > 0 and atr_val > 0:
                            if (trade_high_price - curr['Close']) > (p_atr_mult * atr_val):
                                sell_conditions += 2
                                current_sell_reasons.append(f"Trailing Stop (-{p_atr_mult}ATR)")

                        exit_ma_val = curr.get(f'MA{p_exit_ma_period}', np.nan)
                        if not np.isnan(exit_ma_val) and curr['Close'] < exit_ma_val:
                            sell_conditions += 1
                            current_sell_reasons.append(f"Broken MA{p_exit_ma_period}")

                        if weeks_below_ma10 >= p_weeks_below_ma10:
                            sell_conditions += 1
                            current_sell_reasons.append("Weak Trend (M-MA10)")
                        
                        if mom6 < 0 and mom12 < 0:
                            sell_conditions += 1
                            current_sell_reasons.append("Neg Momentum")

                        indpro_vals = []
                        for m in range(6):
                            idx = max(0, i - (m * 21))
                            val = df.iloc[idx].get('INDPRO_YoY', np.nan)
                            indpro_vals.append(val)
                        is_macro_weak = False
                        if all(v < 0 for v in indpro_vals[:3]) and (indpro_vals[0] - indpro_vals[2] < 0):
                            is_macro_weak = True
                        if all(v < 0 for v in indpro_vals[:6]):
                            is_macro_weak = True
                        
                        if is_macro_weak:
                            sell_conditions += 1
                            current_sell_reasons.append("Macro Stress")

                        potential_buy_units = 1
                        vix = curr.get('VIX', 0)
                        if vix >= p_vix_extreme:
                            potential_buy_units += 2
                        elif vix >= p_vix_high:
                            potential_buy_units += 1

                        dd = curr.get('Drawdown', 0)
                        if dd <= p_dd_1:
                            potential_buy_units += 1
                        if dd <= p_dd_2:
                            potential_buy_units += 1
                        
                        if dd <= (p_dd_2 - 10): 
                            potential_buy_units += 1

                        if not np.isnan(ma200_current):
                            if curr['Close'] < ma200_current * p_val_1:
                                potential_buy_units += 1
                            if curr['Close'] < ma200_current * p_val_2:
                                potential_buy_units += 1

                        potential_buy_units = min(potential_buy_units, p_max_units)
                        if not np.isnan(ma200_current) and curr['Close'] < ma200_current and ma200_slope < 0:
                            potential_buy_units = int(np.floor(potential_buy_units * 0.5))

                        if strat4_mode == "Aggressive (Growth)":
                            if sell_conditions > 0 and mom12 > 0:
                                sell_conditions = 0
                                current_sell_reasons = [] 
                                if is_macro_weak:
                                    sell_conditions = 2
                                    current_sell_reasons = ["Macro Stress (Force)"]

                        action = "HOLD"
                        actual_buy_units = 0
                        
                        sell_threshold = 2
                        if strat4_mode == "Conservative (Low Drawdown)":
                            sell_threshold = 1

                        if strategy_state != "EXIT" and sell_conditions >= sell_threshold:
                            strategy_state = "EXIT"
                            action = "SELL_ALL"
                        elif strategy_state == "EXIT":
                            reentry_ok = (
                                not np.isnan(ma200_current)
                                and curr['Close'] > ma200_current
                                and ma200_slope > 0
                                and mom6 > 0
                                and sell_conditions == 0
                            )
                            if reentry_ok:
                                strategy_state = "REENTRY"
                            else:
                                action = "HOLD"

                        if strategy_state == "REENTRY" and action != "SELL_ALL":
                            accum_ok = (
                                mom12 > 0
                                and not np.isnan(ma10_m)
                                and curr['Close'] > ma10_m
                            )
                            if accum_ok:
                                strategy_state = "ACCUMULATE"
                            else:
                                action = "BUY"
                                actual_buy_units = 0.5

                        if strategy_state == "ACCUMULATE" and action != "SELL_ALL":
                            if potential_buy_units > 0:
                                action = "BUY"
                                actual_buy_units = potential_buy_units
                            else:
                                action = "HOLD"

                        if strat4_mode == "Conservative (Low Drawdown)":
                            is_deep_value = False
                            if not np.isnan(ma200_current):
                                is_deep_value = curr['Close'] < ma200_current * 0.95
                            is_high_vix = curr.get('VIX', 0) > 25
                            if action == "BUY" and not (is_deep_value or is_high_vix or strategy_state == "REENTRY"):
                                action = "HOLD"
                            
                        elif strat4_mode == "Aggressive (Growth)":
                            if action == "BUY":
                                actual_buy_units = min(actual_buy_units + 1, 10)

                        if action == "BUY" and actual_buy_units > 0:
                            if not dca_positions:
                                trade_high_price = curr['Close']
                            
                            dca_positions.append({
                                'Date': date,
                                'Price': curr['Close'],
                                'Units': actual_buy_units,
                                'Reason': f'Units {actual_buy_units} ({strategy_state})'
                            })
                            strat4_buys.append({
                                'Date': date,
                                'Price': curr['Close'],
                                'Units': actual_buy_units,
                                'Reason': f"State: {strategy_state}<br>Units: {actual_buy_units}<br>VIX: {curr.get('VIX',0):.1f}<br>DD: {curr.get('Drawdown',0):.1f}%"
                            })
                        elif action == "SELL_ALL" and dca_positions:
                            sell_price = curr['Close']
                            
                            reason_short = ", ".join(current_sell_reasons) if current_sell_reasons else "Manual/Stop"
                            reason_html = "<br>".join(current_sell_reasons) if current_sell_reasons else "Manual/Stop"
                            
                            strat4_sells.append({
                                'Date': date,
                                'Price': sell_price,
                                'Reason': f"Clearance (State: EXIT)<br>Signals: {sell_conditions}<br>Causes: {reason_html}"
                            })
                            total_cost = sum([p['Price'] * p.get('Units', 1) for p in dca_positions])
                            total_units = sum([p.get('Units', 1) for p in dca_positions])
                            avg_buy_price = total_cost / total_units if total_units > 0 else curr['Close']
                            first_buy_date = dca_positions[0]['Date']
                            pct_change = (sell_price - avg_buy_price) / avg_buy_price * 100
                            trades.append({
                                "Buy Date": first_buy_date,
                                "Buy Price": avg_buy_price,
                                "Sell Date": date,
                                "Sell Price": sell_price,
                                "Return (%)": pct_change,
                                "Status": "Closed",
                                "Reason": f"{reason_short} (Sigs: {sell_conditions})",
                                "DCA Count": len(dca_positions)
                            })
                            dca_positions = []
                            trade_high_price = 0.0

                        if not np.isnan(ma200_current):
                            prev_week_ma200 = ma200_current

                    if dca_positions:
                        total_cost = sum([p['Price'] * p.get('Units', 1) for p in dca_positions])
                        total_units = sum([p.get('Units', 1) for p in dca_positions])
                        if total_units > 0:
                            avg_costs[i] = total_cost / total_units

                df['Avg_Cost'] = avg_costs
                
                # --- Post-Processing: Close Open Trades for Stats ---
                if dca_positions:
                    current_price = df.iloc[-1]['Close']
                    prices = [p['Price'] for p in dca_positions]
                    avg_price = sum(prices) / len(prices)
                    first_buy_date = dca_positions[0]['Date']
                    floating_return = (current_price - avg_price) / avg_price * 100
                    trades.append({
                        "Buy Date": first_buy_date, "Buy Price": avg_price, "Sell Date": pd.NaT,
                        "Sell Price": current_price, "Return (%)": floating_return, "Status": "Open",
                        "Reason": "Holding", "DCA Count": len(dca_positions)
                    })

                df_trades = pd.DataFrame(trades)
                
                # --- Calculate Summary Metrics ---
                if not df_trades.empty:
                    if 'Status' not in df_trades.columns: df_trades['Status'] = 'Closed'
                    else: df_trades['Status'] = df_trades['Status'].fillna('Closed')
                    
                    if not df.empty:
                        first_close = df.iloc[0]['Close']
                        last_close = df.iloc[-1]['Close']
                        buy_hold_return = (last_close - first_close) / first_close * 100
                    else:
                        buy_hold_return = 0.0

                    cumulative_return = ((1 + df_trades['Return (%)'] / 100).prod() - 1) * 100
                    days = (df.index[-1] - df.index[0]).days
                    if days > 365:
                        ann_return = ((1 + cumulative_return / 100) ** (365 / days) - 1) * 100
                    else:
                        ann_return = cumulative_return

                    avg_return = df_trades['Return (%)'].mean()
                    win_rate = (df_trades['Return (%)'] > 0).mean() * 100
                else:
                    cumulative_return = 0.0
                    ann_return = 0.0
                    buy_hold_return = 0.0
                    avg_return = 0.0
                    win_rate = 0.0

                # 6. Display Results (New UI)
                
                # A. Top KPI Dashboard
                st.markdown("### 📊 Strategy Performance")
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                kpi1.metric("Strategy Return", f"{cumulative_return:.2f}%", help="Compound return of all trades")
                kpi2.metric("Annualized", f"{ann_return:.2f}%", help="CAGR")
                kpi3.metric("Buy & Hold", f"{buy_hold_return:.2f}%", delta=f"{cumulative_return - buy_hold_return:.2f}%")
                kpi4.metric("Avg Trade", f"{avg_return:.2f}%")
                kpi5.metric("Win Rate", f"{win_rate:.1f}%")

                # B. Market Environment Dashboard
                st.markdown("### 🌍 Market Environment")
                with st.container():
                    col_env1, col_env2, col_env3 = st.columns(3)
                    
                    # VIX Status
                    curr_vix = df.iloc[-1].get('VIX', np.nan)
                    vix_status = "Unknown"
                    vix_color = "off"
                    if not np.isnan(curr_vix):
                        if curr_vix < 20: vix_status = "Calm (Low Fear)"; vix_color = "normal"
                        elif curr_vix < 30: vix_status = "Elevated"; vix_color = "off"
                        else: vix_status = "High Fear"; vix_color = "inverse"
                    
                    col_env1.metric("VIX Level", f"{curr_vix:.2f}", vix_status)

                    # Macro Status (INDPRO)
                    indpro_last = df.iloc[-1].get('INDPRO_YoY', np.nan)
                    indpro_status = "Neutral"
                    if not np.isnan(indpro_last):
                        if indpro_last > 0: indpro_status = "Expansion"
                        else: indpro_status = "Contraction (Weak Macro)"
                    
                    col_env2.metric("Macro Cycle (INDPRO)", f"{indpro_last:.2f}%" if not np.isnan(indpro_last) else "N/A", indpro_status)
                    
                    # Trend Status
                    curr_price = df.iloc[-1]['Close']
                    curr_ma200 = df.iloc[-1].get('MA200', np.nan)
                    trend_status = "Neutral"
                    if not np.isnan(curr_ma200):
                        dist = (curr_price / curr_ma200 - 1) * 100
                        if dist > 0: trend_status = f"Bullish (+{dist:.1f}% vs MA200)"
                        else: trend_status = f"Bearish ({dist:.1f}% vs MA200)"
                    col_env3.metric("Long Term Trend", trend_status)

                # C. Charts
                st.markdown("### 📈 Technical Analysis")
                tab1, tab2, tab3 = st.tabs(["Price & Strategy", "Momentum (RSI/MACD)", "Oscillators (KDJ/CCI)"])

                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price'
                    ))

                    if 'MA20' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='Daily MA20'))

                    if 'Avg_Cost' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['Avg_Cost'], line=dict(color='purple', width=2, dash='dot'), name='Avg Cost'))

                    if 'MA200' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='gray', width=1), name='MA200 (Base)'))

                    if f'MA{p_exit_ma_period}' in df.columns and p_exit_ma_period not in [20, 200]: 
                        fig.add_trace(go.Scatter(x=df.index, y=df[f'MA{p_exit_ma_period}'], line=dict(color='blue', width=1, dash='dot'), name=f'Trend Guard (MA{p_exit_ma_period})'))

                    if 'Month_MA10' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['Month_MA10'], line=dict(color='orange', width=1, dash='dash'), name='Month MA10 (Exit)'))

                    if strat4_buys:
                        b_df = pd.DataFrame(strat4_buys)
                        fig.add_trace(go.Scatter(
                            x=b_df['Date'], y=b_df['Price'] * 0.98, mode='markers',
                            marker=dict(symbol='triangle-up', size=b_df['Units'] * 4 + 6, color='green', opacity=0.8),
                            text=b_df['Reason'], hovertemplate='<b>Buy</b><br>Date: %{x}<br>Price: %{y:.2f}<br>%{text}<extra></extra>', name='Buy (Scaled)'
                        ))

                    if strat4_sells:
                        s_df = pd.DataFrame(strat4_sells)
                        fig.add_trace(go.Scatter(
                            x=s_df['Date'], y=s_df['Price'] * 1.02, mode='markers',
                            marker=dict(symbol='x', size=12, color='red', line=dict(width=2)),
                            text=s_df['Reason'], hovertemplate='<b>SELL ALL</b><br>Date: %{x}<br>Price: %{y:.2f}<br>%{text}<extra></extra>', name='Clearance'
                        ))

                    fig.update_layout(
                        title=dict(text=f"{stock_name} Strategy Execution", font=dict(size=24)),
                        yaxis_title="Price", xaxis_title="Date", height=600, xaxis_rangeslider_visible=False,
                        template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.5])
                    # RSI
                    fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
                    fig2.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                    fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                    
                    # MACD
                    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
                    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
                    fig2.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Hist'), row=2, col=1)
                    
                    fig2.update_layout(height=500, title="RSI & MACD", hovermode="x unified", template="plotly_white")
                    st.plotly_chart(fig2, use_container_width=True)

                with tab3:
                    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.5])
                    # KDJ
                    if 'K' in df.columns:
                        fig3.add_trace(go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='orange')), row=1, col=1)
                        fig3.add_trace(go.Scatter(x=df.index, y=df['D'], name='D', line=dict(color='blue')), row=1, col=1)
                        fig3.add_trace(go.Scatter(x=df.index, y=df['J'], name='J', line=dict(color='purple')), row=1, col=1)
                    
                    # CCI
                    if 'CCI' in df.columns:
                        fig3.add_trace(go.Scatter(x=df.index, y=df['CCI'], name='CCI', line=dict(color='brown')), row=2, col=1)
                        fig3.add_hline(y=100, line_dash="dash", line_color="red", row=2, col=1)
                        fig3.add_hline(y=-100, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig3.update_layout(height=500, title="KDJ & CCI", hovermode="x unified", template="plotly_white")
                    st.plotly_chart(fig3, use_container_width=True)
                
                # D. Transaction History
                st.markdown("### 📝 Transaction Log")
                
                if not df_trades.empty:
                    df_trades['Buy Date'] = pd.to_datetime(df_trades['Buy Date']).dt.date
                    
                    # Allow Download
                    csv = df_trades.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Trades as CSV",
                        csv,
                        "trades.csv",
                        "text/csv",
                        key='download-csv'
                    )

                    st.dataframe(
                        df_trades, use_container_width=True,
                        column_config={
                            "Buy Date": st.column_config.DateColumn("Buy Date", format="YYYY-MM-DD"),
                            "Sell Date": st.column_config.DateColumn("Sell Date", format="YYYY-MM-DD"),
                            "Buy Price": st.column_config.NumberColumn("Buy Price", format="$%.2f"),
                            "Sell Price": st.column_config.NumberColumn("Sell Price", format="$%.2f"),
                            "Return (%)": st.column_config.NumberColumn("Return (%)", format="%.2f%%", help="Trade Return"),
                            "DCA Count": st.column_config.NumberColumn("DCA Orders"),
                            "Status": st.column_config.TextColumn("Status"),
                        }, hide_index=True
                    )
                else:
                    st.info("No trades executed in this period.")


            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Page 3: State Machine Check ---

@st.cache_data
def get_historical_macro_data(start_date, end_date):
    """
    Fetches and calculates macro states for a given date range.
    Includes buffer to ensure valid data at start_date.
    """
    # Add buffer for rolling windows (MA200 needs 200 days, Correlation 60 days, etc.)
    # We add 365 days to be safe.
    buffer_days = 365
    fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
    fetch_end = pd.to_datetime(end_date)

    # 1. Fetch Market Data
    tickers = ['IWY', 'TLT', '^TNX', '^VIX', 'GLD', 'IWD']
    try:
        df_all = yf.download(tickers, start=fetch_start, end=fetch_end, progress=False, auto_adjust=False)
        
        # Handle MultiIndex
        if isinstance(df_all.columns, pd.MultiIndex):
            if 'Adj Close' in df_all.columns.get_level_values(0):
                data = df_all['Adj Close']
            elif 'Close' in df_all.columns.get_level_values(0):
                data = df_all['Close']
            else:
                data = df_all
        else:
            data = df_all
            
        # Basic check
        if data.empty:
             return pd.DataFrame(), "Market data fetch failed or incomplete."
             
    except Exception as e:
        return pd.DataFrame(), f"Error fetching market data: {str(e)}"

    # 2. Fetch FRED Data (UNRATE & T10Y2Y)
    try:
        # Use robust fetch function
        unrate = fetch_fred_data("UNRATE")
        yc = fetch_fred_data("T10Y2Y") # Yield Curve
        
        if unrate.empty:
            raise ValueError("Fetched empty data for UNRATE")
            
        unrate.columns = ['UNRATE']
        unrate = unrate[unrate.index >= fetch_start]
        unrate_daily = unrate.reindex(data.index, method='ffill')
        
        if not yc.empty:
            yc.columns = ['T10Y2Y']
            yc = yc[yc.index >= fetch_start]
            yc_daily = yc.reindex(data.index, method='ffill')
        else:
            yc_daily = pd.DataFrame(0.0, index=data.index, columns=['T10Y2Y'])

    except Exception as e:
        # Fallback
        return pd.DataFrame(), f"Error fetching FRED data: {str(e)}"

    # 3. Calculate Indicators
    try:
        # --- FIX: Calculate Sahm Rule on MONTHLY data first ---
        # Sahm Rule: (3m avg) - (min of prev 12m of 3m avg)
        u_monthly = unrate['UNRATE']
        u_3m_avg = u_monthly.rolling(window=3).mean()
        # Strict Sahm: Min of *previous* 12 months (shift 1 to exclude current)
        u_12m_low = u_3m_avg.rolling(window=12).min().shift(1)
        sahm_monthly = u_3m_avg - u_12m_low
        
        # Reindex Sahm Rule to Daily (ffill)
        sahm_series = sahm_monthly.reindex(data.index, method='ffill')
        
        # Rate Shock: TNX 21-day ROC
        # Handle missing columns gracefully
        tnx_col = '^TNX' if '^TNX' in data.columns else data.columns[0] # Fallback if missing
        tnx_roc = (data[tnx_col] - data[tnx_col].shift(21)) / data[tnx_col].shift(21)
        
        # Correlation: IWY vs TLT 60-day
        if 'IWY' in data.columns and 'TLT' in data.columns:
            corr = data['IWY'].rolling(60).corr(data['TLT'])
            iwy_series = data['IWY']
        else:
            corr = pd.Series(0, index=data.index)
            iwy_series = data.iloc[:, 0]
        
        # Trend: IWY vs MA200
        iwy_ma200 = iwy_series.rolling(200).mean()
        trend_bear = iwy_series < iwy_ma200
        
        # Gold Trend (GLD vs MA200)
        gold_trend_bear = pd.Series(False, index=data.index)
        if 'GLD' in data.columns:
            gld_ma200 = data['GLD'].rolling(200).mean()
            gold_trend_bear = data['GLD'] < gld_ma200
            
        # Style Trend (Growth vs Value)
        style_value_regime = pd.Series(False, index=data.index)
        if 'IWY' in data.columns and 'IWD' in data.columns:
            pair_ratio = data['IWY'] / data['IWD']
            pair_ma200 = pair_ratio.rolling(200).mean()
            style_value_regime = pair_ratio < pair_ma200 # If Ratio < MA200, Value is winning

        # Assemble DataFrame
        df_hist = pd.DataFrame({
            'IWY': iwy_series,
            'Sahm': sahm_series,
            'RateShock': tnx_roc,
            'Corr': corr,
            'VIX': data.get('^VIX', pd.Series(0, index=data.index)),
            'Trend_Bear': trend_bear,
            'YieldCurve': yc_daily['T10Y2Y'],
            'Gold_Bear': gold_trend_bear,
            'Value_Regime': style_value_regime
        }).dropna()
        
        # 4. Determine States
        def determine_row_state(row):
            is_rec = row['Sahm'] >= 0.50
            is_shock = row['RateShock'] > 0.20
            is_c_broken = row['Corr'] > 0.3
            is_f = row['VIX'] > 32
            is_down = row['Trend_Bear']
            is_vol_elevated = row['VIX'] > 20
            
            # Yield Curve logic: Inverted is bad, but Un-inverting is recessionary.
            # Simplified: Use it as a reinforcement? 
            # For backtest state, we stick to core definitions but maybe refine "DEFLATION_RECESSION"
            # if Yield Curve is un-inverting (T10Y2Y > 0 after being < 0). 
            # Ideally needs state history. For now keeping simple.
            
            if is_shock or (is_rec and is_c_broken):
                return "INFLATION_SHOCK"
            elif is_rec or (is_down and row['VIX'] > 35):
                return "DEFLATION_RECESSION"
            elif is_f and not is_shock and not is_rec:
                return "EXTREME_ACCUMULATION"
            elif is_down and not is_vol_elevated:
                return "CAUTIOUS_TREND" # Bearish but low vol
            elif is_vol_elevated: 
                 return "CAUTIOUS_VOL"
            else:
                return "NEUTRAL"

        df_hist['State'] = df_hist.apply(determine_row_state, axis=1)
        
        # Filter Output to requested range (removing buffer)
        df_final = df_hist.loc[(df_hist.index >= pd.to_datetime(start_date)) & (df_hist.index <= pd.to_datetime(end_date))]
        
        return df_final, None

    except Exception as e:
        return pd.DataFrame(), f"Error in calculation: {str(e)}"

def render_state_machine_check():
    st.header("🛡️ 宏观状态机与资产配置 (Macro State & Allocation)")
    st.caption("基于宏观因子 (利率、失业率、波动率、相关性) 的全自动资产配置生成器。")

    # --- Manual Data Import (Fallback) ---
    with st.expander("📂 手动导入宏观数据 (网络受限时使用)", expanded=False):
        st.info("如果网络受限导致 FRED 数据 (UNRATE, T10Y2Y) 获取失败，请手动下载并上传 CSV 文件。系统将自动优先读取本地文件。")
        
        col_u1, col_u2 = st.columns(2)
        import time

        with col_u1:
            st.markdown("**1. 失业率 (UNRATE)**")
            
            # Check local file
            unrate_path = os.path.join(os.path.dirname(__file__), "fred_UNRATE.csv")
            if os.path.exists(unrate_path):
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(unrate_path)).strftime('%Y-%m-%d %H:%M')
                st.success(f"✅ 已检测到本地数据 (更新于 {file_time})")
            else:
                st.warning("⚠️ 未检测到本地文件")

            st.markdown("[📥 点击下载 UNRATE.csv](https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE)")
            uploaded_file = st.file_uploader("更新 UNRATE.csv", type=['csv'], key="uploader_unrate")
            
            if uploaded_file is not None:
                # Deduplication logic to avoid infinite reruns
                file_id = f"{uploaded_file.name}-{uploaded_file.size}"
                if st.session_state.get("processed_unrate_id") != file_id:
                    try:
                        df_test = pd.read_csv(uploaded_file)
                        if 'observation_date' in df_test.columns:
                            uploaded_file.seek(0)
                            with open(unrate_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            st.session_state["processed_unrate_id"] = file_id
                            st.success("✅ UNRATE 已保存! 正在刷新...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("CSV 格式错误: 缺少 'observation_date' 列")
                    except Exception as e:
                        st.error(f"文件处理错误: {e}")

        with col_u2:
            st.markdown("**2. 收益率曲线 (T10Y2Y)**")
            
            # Check local file
            yc_path = os.path.join(os.path.dirname(__file__), "fred_T10Y2Y.csv")
            if os.path.exists(yc_path):
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(yc_path)).strftime('%Y-%m-%d %H:%M')
                st.success(f"✅ 已检测到本地数据 (更新于 {file_time})")
            else:
                st.warning("⚠️ 未检测到本地文件")

            st.markdown("[📥 点击下载 T10Y2Y.csv](https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y)")
            uploaded_yc = st.file_uploader("更新 T10Y2Y.csv", type=['csv'], key="uploader_t10y2y")
            
            if uploaded_yc is not None:
                # Deduplication logic to avoid infinite reruns
                file_id = f"{uploaded_yc.name}-{uploaded_yc.size}"
                if st.session_state.get("processed_t10y2y_id") != file_id:
                    try:
                        df_test = pd.read_csv(uploaded_yc)
                        if 'observation_date' in df_test.columns:
                            uploaded_yc.seek(0)
                            with open(yc_path, "wb") as f:
                                f.write(uploaded_yc.getbuffer())
                            
                            st.session_state["processed_t10y2y_id"] = file_id
                            st.cache_data.clear()
                            st.success("✅ T10Y2Y 已保存! 正在刷新...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("CSV 格式错误: 缺少 'observation_date' 列")
                    except Exception as e:
                        st.error(f"文件处理错误: {e}")

    # --- State Reference Guide (Resident Folded Display) ---
    with st.expander("📖 宏观状态定义与策略对照表 (State Machine Reference)", expanded=False):
        st.info("💡 提示：本模型实时扫描市场，一旦触发以下任一状态，系统将自动建议对应的防御或进攻仓位。")
        
        c_ref1, c_ref2, c_ref3, c_ref4, c_ref5 = st.columns(5)
        
        with c_ref1:
            st.error("🔴 滞胀/加息冲击\n(INFLATION SHOCK)")
            st.markdown("""
            **🛡️ 触发条件**:  
            1. **利率冲击**: TNX 21日涨幅 > 20%  
            2. **或** (衰退 + 股债相关性失效)
            
            **📉 策略**: **现金为王 (Anti-Duration)**  
            - **清仓** 成长股 & REITs (0%)  
            - **清仓** 长债 (0%)  
            - **重仓** 危机Alpha (WTMF) & 现金  
            - **配置** 黄金 (适量)
            """)
            
        with c_ref2:
            st.info("🔵 衰退/崩盘\n(DEFLATION RECESSION)")
            st.markdown("""
            **🛡️ 触发条件**:  
            1. **衰退**: 萨姆规则 (Sahm ≥ 0.5)  
            2. **或** (趋势向下 + VIX > 35)
            
            **📉 策略**: **全面防御 (Long Duration)**  
            - **减持** 成长股 (至 10%)  
            - **重仓** 国债 (MBH/TLT) 锁定收益  
            - **重仓** 黄金 (避险)  
            - **低配** REITs (规避信用风险)
            """)
            
        with c_ref3:
            st.warning("🚀 极度贪婪/抄底\n(EXTREME ACCUMULATION)")
            st.markdown("""
            **🛡️ 触发条件**:  
            1. **恐慌**: VIX > 32  
            2. 且 **非** 利率冲击  
            3. 且 **非** 衰退
            
            **📈 策略**: **重仓进攻 (Leverage)**  
            - **增持** 成长股 (至 70%)  
            - **卖出** 防御资产  
            - **利用** 恐慌买入
            """)
            
        with c_ref4:
            st.warning("⚠️ 谨慎 (CAUTIOUS)")
            st.markdown("""
            **A. 趋势破位 (Trend)**:  
            Price < MA200 且 VIX < 20  
            *策略*: **防御 (Defensive)** - 重仓红利/现金
            
            **B. 高波震荡 (Vol)**:  
            Price > MA200 且 VIX > 20  
            *策略*: **对冲 (Hedge)** - 保留成长+危机Alpha
            """)
            
        with c_ref5:
            st.success("🟢 常态/牛市\n(NEUTRAL)")
            st.markdown("""
            **🛡️ 触发条件**:  
            - 无上述风险信号  
            - VIX < 20 且 趋势向上
            
            **📈 策略**: **标准配置 (Growth)**  
            - **成长** IWY (50%)  
            - **红利** LVHI (10%)  
            - **蓝筹** G3B (10%)  
            - **债券** MBH (10%)
            """)

    # --- Import from Portfolio Backtest ---
    saved_portfolios = load_portfolios()
    if saved_portfolios:
        with st.expander("📥 从已保存的投资组合导入 (Import from Saved)", expanded=False):
            st.info("提示：这将把选定组合的配置比例转换为对应市值的持仓。")
            c_imp1, c_imp2, c_imp3 = st.columns([2, 1, 1])
            with c_imp1:
                sel_port_name = st.selectbox("选择组合", list(saved_portfolios.keys()), key="sm_imp_name")
            with c_imp2:
                imp_total_cap = st.number_input("设定总本金 (Total Value)", value=10000.0, step=1000.0, key="sm_imp_cap")
            with c_imp3:
                if st.button("应用到下方持仓", type="secondary"):
                    # Logic to populate session state
                    if sel_port_name in saved_portfolios:
                        p_data = saved_portfolios[sel_port_name]
                        weights = p_data.get("weights", {})
                        
                        # 1. Reset known fields
                        known_tickers = ['IWY', 'WTMF', 'LVHI', 'G3B.SI', 'MBH.SI', 'GSD.SI', 'SRT.SI', 'AJBU.SI']
                        for t in known_tickers:
                            st.session_state[f"hold_{t}"] = 0.0
                        st.session_state["hold_OTHERS"] = 0.0
                            
                        # 2. Populate from portfolio
                        other_val = 0.0
                        for t, w in weights.items():
                            val = imp_total_cap * (w / 100.0)
                            if t in known_tickers:
                                st.session_state[f"hold_{t}"] = val
                            else:
                                other_val += val
                        
                        st.session_state["hold_OTHERS"] = other_val
                            
                        st.toast(f"已导入组合: {sel_port_name} (含其他资产 ${other_val:,.0f})", icon="✅")
                        st.rerun()

    # --- 1. Input Section (Current Holdings) ---
    with st.expander("💼 输入当前持仓 (Current Portfolio Holdings)", expanded=True):
        st.markdown("请输入您当前账户中各标的的**市值 (Value)** (单位：美元/新元均可，统一即可)。")
        
        col_in1, col_in2, col_in3, col_in4 = st.columns(4)
        
        # Initialize session state keys if they don't exist
        defaults = {
            "hold_IWY": 0.0, "hold_WTMF": 0.0, "hold_LVHI": 0.0,
            "hold_G3B.SI": 0.0, "hold_MBH.SI": 0.0, "hold_GSD.SI": 0.0,
            "hold_SRT.SI": 0.0, "hold_AJBU.SI": 0.0, "hold_OTHERS": 0.0
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # We use number_input keys to auto-update session state
        # Note: 'value' argument is removed to rely solely on session_state[key]
        with col_in1:
            val_iwy = st.number_input("IWY (美股成长)", step=100.0, key="hold_IWY")
            val_wtmf = st.number_input("WTMF (危机Alpha)", step=100.0, key="hold_WTMF")
        with col_in2:
            val_lvhi = st.number_input("LVHI (美股红利)", step=100.0, key="hold_LVHI")
            val_g3b = st.number_input("G3B.SI (新加坡蓝筹)", step=100.0, key="hold_G3B.SI")
        with col_in3:
            val_mbh = st.number_input("MBH.SI (新元债券)", step=100.0, key="hold_MBH.SI")
            val_gsd = st.number_input("GSD.SI (黄金)", step=100.0, key="hold_GSD.SI")
        with col_in4:
            val_srt = st.number_input("SRT.SI (REITs)", step=100.0, key="hold_SRT.SI")
            val_ajbu = st.number_input("AJBU.SI (REITs)", step=100.0, key="hold_AJBU.SI")
            val_others = st.number_input("其他资产 (Others)", step=100.0, key="hold_OTHERS", help="Imported assets not in strategy list (Action: Sell)")
            
        current_holdings = {
            'IWY': val_iwy, 'WTMF': val_wtmf, 'LVHI': val_lvhi,
            'G3B.SI': val_g3b, 'MBH.SI': val_mbh, 'GSD.SI': val_gsd,
            'SRT.SI': val_srt, 'AJBU.SI': val_ajbu, 'OTHERS': val_others
        }
        
        total_value = sum(current_holdings.values())
        st.caption(f"💰 当前账户总市值: **{total_value:,.2f}**")

    # --- 2. Diagnosis Button ---
    if st.button("🚀 开始诊断与配置生成 (Run Analysis)", type="primary", use_container_width=True):
        # Result Containers
        status_container = st.container()
        result_container = st.container()

        # Data placeholders
        data = pd.DataFrame()
        unrate_curr = pd.Series()
        fetch_errors = []
        df_hist = pd.DataFrame()
        
        with status_container:
            with st.status("正在进行全市场扫描与宏观诊断...", expanded=True) as status:
                
                # --- A. Fetch Data ---
                st.write("📡 步骤 1/3: 获取实时市场行情 (Yahoo Finance)...")
                end = datetime.datetime.now()
                # Increase lookback to 3 years to ensure enough history for Sahm Rule (12m + 3m rolling)
                start = end - datetime.timedelta(days=1095)
                
                try:
                    tickers = ['IWY', 'TLT', '^TNX', '^VIX', 'GLD', 'IWD']
                    df_all = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
                    
                    if df_all.empty:
                        raise ValueError("Yahoo Finance 返回空数据")
                    
                    # Handle MultiIndex columns safely
                    if isinstance(df_all.columns, pd.MultiIndex):
                        if 'Adj Close' in df_all.columns.get_level_values(0):
                            data = df_all['Adj Close']
                        elif 'Close' in df_all.columns.get_level_values(0):
                             data = df_all['Close']
                        else:
                             data = df_all
                    else:
                        data = df_all
                    
                    if data.empty:
                         raise ValueError("解析后的行情数据为空")
                         
                    st.write("✅ 市场行情数据获取成功")
                except Exception as e:
                    fetch_errors.append(f"市场数据错误: {str(e)}")
                    st.error(f"❌ 市场数据获取失败: {e}")
                    status.update(label="诊断中断：数据获取失败", state="error")
                    st.stop()

                st.write("📊 步骤 2/3: 同步美联储宏观数据 (FRED)...")
                is_fred_live = True
                try:
                    # Use robust fetch function
                    unrate = fetch_fred_data("UNRATE")
                    yc_data = fetch_fred_data("T10Y2Y") # Yield Curve
                    
                    if unrate.empty:
                        raise ValueError("Fetched empty data for UNRATE")
                        
                    unrate.columns = ['UNRATE']
                    unrate = unrate[(unrate.index >= start) & (unrate.index <= end)]
                    unrate_curr = unrate['UNRATE']
                    
                    if not yc_data.empty:
                        yc_data.columns = ['T10Y2Y']
                        yc_data = yc_data[(yc_data.index >= start) & (yc_data.index <= end)]
                        yc_curr = yc_data['T10Y2Y']
                    else:
                        yc_curr = pd.Series()

                    if unrate_curr.empty:
                        raise ValueError("FRED 数据范围为空")
                    st.write("✅ 宏观数据 (UNRATE/Yield) 同步成功")
                except Exception as e:
                    is_fred_live = False
                    st.write(f"⚠️ FRED 连接超时/失败 ({str(e)})，启用应急兜底数据 (常态假设)...")
                    dates = pd.date_range(start=start, end=end, freq='M')
                    unrate_curr = pd.Series([4.0]*len(dates), index=dates)
                    yc_curr = pd.Series([0.5]*len(dates), index=dates)

                # --- B. Process Signals ---
                st.write("⚙️ 步骤 3/3: 正在计算核心因子 (萨姆规则、利率冲击、波动率)...")
                
                try:
                    # 1. Sahm Rule
                    unrate_clean = unrate_curr.dropna()
                    u_3m = unrate_clean.rolling(3).mean().dropna()
                    
                    if len(u_3m) >= 14:
                        current_sahm_avg = u_3m.iloc[-1]
                        prev_12m_min = u_3m.iloc[-13:-1].min()
                        sahm_val = current_sahm_avg - prev_12m_min
                    else:
                        sahm_val = 0.0
                    
                    if pd.isna(sahm_val): sahm_val = 0.0
                    is_recession = sahm_val >= 0.50

                    # 2. Rate Shock
                    tnx = data['^TNX'].dropna()
                    if len(tnx) >= 22:
                        tnx_roc = (tnx.iloc[-1] - tnx.iloc[-21]) / tnx.iloc[-21]
                    else:
                        tnx_roc = 0.0
                    is_rate_shock = tnx_roc > 0.20

                    # 3. Correlation
                    corr_series = data['IWY'].rolling(60).corr(data['TLT']).dropna()
                    if not corr_series.empty:
                        corr = corr_series.iloc[-1]
                    else:
                        corr = 0.0
                    is_corr_broken = corr > 0.3

                    # 4. Fear (Smoothed)
                    vix_series = data['^VIX'].dropna()
                    if not vix_series.empty:
                        vix_ma = vix_series.rolling(5).mean()
                        vix = vix_ma.iloc[-1] if not vix_ma.empty else vix_series.iloc[-1]
                    else:
                        vix = 0.0
                    is_fear = vix > 32
                    is_elevated_vix = vix > 20

                    # 5. Trend (Price vs MA200)
                    iwy_series = data['IWY'].dropna()
                    is_downtrend = False
                    if len(iwy_series) >= 200:
                        iwy_price = iwy_series.iloc[-1]
                        iwy_ma200 = iwy_series.rolling(200).mean().iloc[-1]
                        is_downtrend = iwy_price < iwy_ma200
                        
                    # 6. Yield Curve (Recession Warning)
                    yc_val = 0.0
                    yc_status = "Normal"
                    yc_un_inverting = False
                    if not yc_curr.empty:
                        yc_clean = yc_curr.dropna()
                        if not yc_clean.empty:
                            yc_val = yc_clean.iloc[-1]
                            # Check for un-inversion (was negative recently, now rising)
                            # Simple check: Current > -0.1 AND Min of last 6 months < -0.3
                            recent_min = yc_clean.iloc[-126:].min() if len(yc_clean) > 126 else -1.0
                            if yc_val < 0:
                                yc_status = "Inverted (Warning)"
                            elif yc_val < 0.2 and recent_min < -0.2:
                                yc_status = "Un-inverting (Danger)"
                                yc_un_inverting = True
                            else:
                                yc_status = "Normal"

                    # 7. Gold Trend (Optimization)
                    is_gold_bear = False
                    if 'GLD' in data.columns:
                        gld_s = data['GLD'].dropna()
                        if len(gld_s) > 200:
                            gld_price = gld_s.iloc[-1]
                            gld_ma200 = gld_s.rolling(200).mean().iloc[-1]
                            is_gold_bear = gld_price < gld_ma200

                    # 8. Style Trend (Growth vs Value)
                    is_value_regime = False
                    if 'IWY' in data.columns and 'IWD' in data.columns:
                        iwy_s = data['IWY'].dropna()
                        iwd_s = data['IWD'].dropna()
                        # Align
                        common_idx = iwy_s.index.intersection(iwd_s.index)
                        if len(common_idx) > 200:
                            ratio = iwy_s.loc[common_idx] / iwd_s.loc[common_idx]
                            ratio_ma200 = ratio.rolling(200).mean()
                            if ratio.iloc[-1] < ratio_ma200.iloc[-1]:
                                is_value_regime = True
                    
                    st.write("⏳ 正在回溯历史状态序列 (用于绘图)...")
                    # --- Historical State Backtest ---
                    # 1. Align Data
                    # Reindex UNRATE to daily (ffill)
                    unrate_daily = unrate_curr.reindex(data.index, method='ffill')
                    
                    # 2. Vectorized Calculations
                    # Sahm: (3m avg) - (min of prev 12m of 3m avg)
                    u_3m_hist = unrate_daily.rolling(3).mean()
                    # FIX: Strict lag (shift 1) to exclude current month from min
                    u_3m_min_hist = u_3m_hist.rolling(12).min().shift(1)
                    sahm_series_hist = u_3m_hist - u_3m_min_hist
                    
                    # Rate Shock: 21-day ROC
                    tnx_hist = data['^TNX']
                    tnx_roc_hist = (tnx_hist - tnx_hist.shift(21)) / tnx_hist.shift(21)
                    
                    # Correlation: 60-day rolling
                    corr_hist = data['IWY'].rolling(60).corr(data['TLT'])
                    
                    # VIX
                    vix_hist = data['^VIX']
                    
                    # Trend
                    iwy_hist = data['IWY']
                    iwy_ma200_hist = iwy_hist.rolling(200).mean()
                    
                    # 3. Combine into DataFrame
                    df_hist = pd.DataFrame({
                        'IWY': iwy_hist,
                        'Sahm': sahm_series_hist,
                        'RateShock': tnx_roc_hist,
                        'Corr': corr_hist,
                        'VIX': vix_hist,
                        'Trend_Bear': iwy_hist < iwy_ma200_hist
                    }).dropna()
                    
                    # 4. Determine State for each row
                    def determine_row_state(row):
                        is_rec = row['Sahm'] >= 0.50
                        is_shock = row['RateShock'] > 0.20
                        is_c_broken = row['Corr'] > 0.3
                        is_f = row['VIX'] > 32
                        is_down = row['Trend_Bear']
                        is_vol_elevated = row['VIX'] > 20
                        
                        if is_shock or (is_rec and is_c_broken):
                            return "INFLATION_SHOCK"
                        elif is_rec or (is_down and row['VIX'] > 35):
                            return "DEFLATION_RECESSION"
                        elif is_f and not is_shock and not is_rec:
                            return "EXTREME_ACCUMULATION"
                        elif is_down and not is_vol_elevated:
                            return "CAUTIOUS_TREND"
                        elif is_vol_elevated:
                             return "CAUTIOUS_VOL"
                        else:
                            return "NEUTRAL"

                    df_hist['State'] = df_hist.apply(determine_row_state, axis=1)

                    st.write("✅ 因子计算完成")
                    status.update(label="诊断完成", state="complete", expanded=False)
                    
                except Exception as e:
                    st.error(f"❌ 计算过程出错: {e}")
                    status.update(label="诊断失败", state="error")
                    st.stop()

        # --- C. Determine & Render State ---
        with result_container:
            # Logic Determination
            state = "NEUTRAL"
            state_display = "🟢 常态 / 牛市 (Neutral)"
            state_desc = "市场运行平稳，维持标准增长配置。"
            state_bg_color = "#e6f4ea" # Light Green
            state_border_color = "#1e8e3e"
            state_icon = "🟢"

            # Cautious triggers
            is_cautious_trend = is_downtrend and (vix <= 20)
            is_cautious_vol = (vix > 20)

            # Optimization: Yield Curve Steepening (Un-inverting) is a Recession Risk
            is_recession_risk = is_recession or (yc_un_inverting)

            if is_rate_shock or (is_recession_risk and is_corr_broken):
                state = "INFLATION_SHOCK"
                state_display = "🔴 滞胀 / 加息冲击 (Inflation Shock)"
                state_desc = "⚠️ 警报：利率飙升或出现股债双杀。现金与抗通胀资产为王。"
                state_bg_color = "#fce8e6" # Light Red
                state_border_color = "#d93025"
                state_icon = "🔴"
            elif is_recession_risk or (is_downtrend and vix > 35):
                state = "DEFLATION_RECESSION"
                state_display = "🔵 衰退 / 崩盘 (Deflation/Crash)"
                state_desc = "⚠️ 警报：经济衰退或流动性危机确认。全面防御，持有国债与美元。"
                state_bg_color = "#e8f0fe" # Light Blue
                state_border_color = "#1a73e8"
                state_icon = "🔵"
            elif is_fear and not is_rate_shock and not is_recession_risk:
                state = "EXTREME_ACCUMULATION"
                state_display = "🚀 极度贪婪 / 抄底 (Accumulation)"
                state_desc = "🔔 机会：恐慌过度但基本面未崩坏。建议重仓抄底成长股。"
                state_bg_color = "rgba(142, 36, 170, 0.2)" # Purple
                state_border_color = "#8e24aa"
                state_icon = "🚀"
            elif is_cautious_trend:
                state = "CAUTIOUS_TREND"
                state_display = "⚠️ 谨慎 / 趋势破位 (Bear Trend)"
                state_desc = "📉 提示：长期趋势转空但恐慌未起。阴跌风险大，建议重仓红利与现金。"
                state_bg_color = "#fff3e0" # Light Orange
                state_border_color = "#f57c00"
                state_icon = "📉"
            elif is_cautious_vol:
                state = "CAUTIOUS_VOL"
                state_display = "⚡ 谨慎 / 高波震荡 (High Volatility)"
                state_desc = "🌊 提示：趋势尚可但波动加剧。可能是牛市回调，建议保留成长并增加对冲。"
                state_bg_color = "#fff8e1" # Lighter Orange
                state_border_color = "#ffb74d"
                state_icon = "⚡"

            # 1. State Header (Custom styled box)
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {state_bg_color}; border-left: 6px solid {state_border_color}; margin-bottom: 20px;">
                <h2 style="margin:0; color: #202124;">{state_icon} {state_display}</h2>
                <p style="margin-top:10px; font-size: 16px; color: #5f6368;">{state_desc}</p>
            </div>
            """, unsafe_allow_html=True)

            if not is_fred_live:
                st.warning("⚠️ 注意：宏观数据 (UNRATE/Yield) 使用的是默认安全值，可能会掩盖真实的衰退信号。请稍后重试或手动确认 FRED 数据源。", icon="⚠️")

            # 1.5. Factor Dashboard (New)
            st.markdown("### 📊 核心因子看板 (Key Factors)")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            
            with f_col1:
                st.metric(
                    "利率冲击 (TNX ROC)", 
                    f"{tnx_roc:+.1%}", 
                    f"{'⚠️ 触发 (>20%)' if is_rate_shock else '✅ 安全'}",
                    delta_color="inverse" if is_rate_shock else "normal"
                )
            with f_col2:
                # Combine Sahm & Yield Curve
                sub_label = "Sahm Risk" if is_recession else ("Yield Curve Danger" if yc_un_inverting else "Safe")
                st.metric(
                    "衰退信号 (Recession)", 
                    f"{sahm_val:.2f}", 
                    f"{'⚠️ ' + sub_label if is_recession or yc_un_inverting else '✅ 安全'}",
                    delta_color="inverse" if is_recession or yc_un_inverting else "normal",
                    help=f"Sahm Rule: {sahm_val:.2f} (>=0.50 Risk)\nYield Curve: {yc_status}"
                )
            with f_col3:
                 st.metric(
                    "股债相关性 (Corr)", 
                    f"{corr:.2f}", 
                    f"{'⚠️ 失效 (>0.30)' if is_corr_broken else '✅ 正常'}",
                    delta_color="inverse" if is_corr_broken else "normal"
                )
            with f_col4:
                 st.metric(
                    "恐慌指数 (VIX)", 
                    f"{vix:.1f}", 
                    f"{'⚠️ 恐慌 (>32)' if is_fear else '✅ 正常'}",
                    delta_color="inverse" if is_fear else "normal"
                )
            
            st.markdown("---")
            
            # Sub-dashboard for Optimizations
            st.markdown("#### 🎯 策略优化指标 (Optimization Signals)")
            opt_c1, opt_c2, opt_c3 = st.columns(3)
            with opt_c1:
                st.metric("美债收益率曲线 (10Y-2Y)", f"{yc_val:.2f}%", yc_status, delta_color="off" if "Normal" in yc_status else "inverse")
            with opt_c2:
                st.metric("黄金趋势 (Gold Trend)", "Bearish (Weak)" if is_gold_bear else "Bullish (Strong)", "Avoid Gold" if is_gold_bear else "Hold Gold", delta_color="inverse" if is_gold_bear else "normal")
            with opt_c3:
                st.metric("风格轮动 (Style)", "Value Regime" if is_value_regime else "Growth Regime", "Tilt Value" if is_value_regime else "Tilt Growth", delta_color="off")


            # 2. Logic Breakdown (Detailed Table)
            st.subheader("🔍 状态判定逻辑详解 (Logic Breakdown)")
            
            logic_data = [
                {
                    "Factor": "利率冲击 (Rate Shock)",
                    "Indicator": "TNX (10Y Yield) ROC",
                    "Current Value": f"{tnx_roc:+.1%}",
                    "Threshold": "> +20%",
                    "Status": "🚨 触发 (Triggered)" if is_rate_shock else "✅ 安全 (Safe)",
                    "Description": "美债收益率是否在短期内剧烈飙升，导致资产估值重估。"
                },
                {
                    "Factor": "衰退信号 (Recession)",
                    "Indicator": "Sahm Rule | Yield Curve",
                    "Current Value": f"{sahm_val:.2f} | {yc_val:.2f}%",
                    "Threshold": "Sahm≥0.5 | Un-invert",
                    "Status": "🚨 触发 (Triggered)" if is_recession_risk else "✅ 安全 (Safe)",
                    "Description": "基于萨姆规则或收益率曲线陡峭化（解倒挂），判断衰退风险。"
                },
                {
                    "Factor": "股债相关性 (Correlation)",
                    "Indicator": "Corr(IWY, TLT) 60d",
                    "Current Value": f"{corr:.2f}",
                    "Threshold": "> 0.30",
                    "Status": "🚨 失效 (Broken)" if is_corr_broken else "✅ 负相关 (Normal)",
                    "Description": "股债是否同涨同跌。若失效，则传统对冲策略无效。"
                },
                {
                    "Factor": "市场恐慌 (Fear)",
                    "Indicator": "VIX Index",
                    "Current Value": f"{vix:.1f}",
                    "Threshold": "> 32.0",
                    "Status": "🚨 恐慌 (Panic)" if is_fear else "✅ 正常 (Normal)",
                    "Description": "市场波动率指数，反映投资者恐慌程度。"
                },
                {
                    "Factor": "趋势形态 (Trend)",
                    "Indicator": "IWY Price vs MA200",
                    "Current Value": "Bearish" if is_downtrend else "Bullish",
                    "Threshold": "Price < MA200",
                    "Status": "📉 下行 (Downtrend)" if is_downtrend else "📈 上行 (Uptrend)",
                    "Description": "长期趋势是否已经破坏。"
                }
            ]
            
            st.dataframe(
                pd.DataFrame(logic_data),
                column_config={
                    "Factor": "宏观因子",
                    "Indicator": "观测指标",
                    "Current Value": "当前数值",
                    "Threshold": "触发阈值",
                    "Status": "状态判定",
                    "Description": "说明"
                },
                use_container_width=True,
                hide_index=True
            )

            # 3. Allocation Calculation
            # Optimization Modifiers
            # If Gold is Bearish (Price < MA200), reduce Gold, add to Cash/Bonds (WTMF/MBH)
            # If Value Regime, Tilt Growth -> Dividend
            
            targets = get_target_percentages(state, gold_bear=is_gold_bear, value_regime=is_value_regime)
            
            # Table Data Prep
            rebal_data = []
            if total_value == 0:
                st.warning("⚠️ 请在上方位输入持仓市值，以获取调仓建议。")
                
            # Use Union of Strategy Targets and Current Holdings (to catch OTHERS)
            all_tickers = set(targets.keys()).union(current_holdings.keys())
            
            for tkr in all_tickers:
                tgt_pct = targets.get(tkr, 0.0) # 0% for non-strategy assets (OTHERS)
                curr_val = current_holdings.get(tkr, 0)
                curr_pct = curr_val / total_value if total_value > 0 else 0
                
                diff_pct = tgt_pct - curr_pct
                diff_val = diff_pct * total_value if total_value > 0 else 0
                
                # Action Logic
                action = "✅ 持有 (Hold)"
                
                # Force Sell for Non-Strategy Assets
                if tkr not in targets:
                     if curr_val > 0:
                         action = f"🔴 清仓 (Sell All) {curr_val:,.0f}"
                else:
                    # Crisis Mode: Strict
                    if state in ["INFLATION_SHOCK", "DEFLATION_RECESSION"]:
                         if abs(diff_pct) > 0.01:
                            if diff_val > 0: action = f"🟢 买入 {abs(diff_val):,.0f}"
                            else: action = f"🔴 卖出 {abs(diff_val):,.0f}"
                    else:
                        # Normal Mode: Buffer
                        if diff_pct > 0.05:
                            action = f"🟢 买入 {abs(diff_val):,.0f}"
                        elif diff_pct < -0.03:
                            action = f"🔴 卖出 {abs(diff_val):,.0f}"
                
                rebal_data.append({
                    "Ticker": tkr,
                    "Target %": f"{tgt_pct:.0%}",
                    "Current %": f"{curr_pct:.1%}",
                    "Current Val": f"{curr_val:,.0f}",
                    "Diff Val": diff_val, # for sorting/color
                    "Action": action
                })
            
            df_rebal = pd.DataFrame(rebal_data)
            
            # Display Table
            st.markdown("### 📋 调仓建议 (Rebalancing Plan)")
            st.dataframe(
                df_rebal,
                column_config={
                    "Ticker": "标的",
                    "Target %": "目标仓位",
                    "Current %": "当前仓位",
                    "Current Val": "当前市值",
                    "Action": "建议操作 (市值)"
                },
                use_container_width=True,
                hide_index=True
            )
            


            # 5. Charts (Visual Context)
            with st.expander("📉 查看详细图表 (Visual Context)", expanded=False):
                # Simple Plotly charts for signals
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("IWY Price & MA200", "VIX", "TNX Yield"))
                
                # IWY
                fig.add_trace(go.Scatter(x=data.index, y=data['IWY'], name='IWY'), row=1, col=1)
                iwy_ma = data['IWY'].rolling(200).mean()
                fig.add_trace(go.Scatter(x=data.index, y=iwy_ma, name='MA200'), row=1, col=1)
                
                # VIX
                fig.add_trace(go.Scatter(x=data.index, y=data['^VIX'], name='VIX', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=32, line_dash='dash', line_color='red', annotation_text='Panic (32)', row=2, col=1)
                
                # TNX
                fig.add_trace(go.Scatter(x=data.index, y=data['^TNX'], name='TNX', line=dict(color='orange')), row=3, col=1)
                
                fig.update_layout(height=800, template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

    # --- New Independent Historical Backtest Section ---
    st.markdown("---")
    st.markdown("### 🕰️ 历史状态回溯与策略仿真 (Historical State & Strategy Backtest)")
    st.caption("基于历史宏观状态，模拟策略的动态资产配置表现，并与基准进行对比。")
    
    col_hist_1, col_hist_2, col_hist_3 = st.columns([2, 1, 1])
    with col_hist_1:
        # Default to last 5 years
        default_start = datetime.date.today() - datetime.timedelta(days=365*5)
        default_end = datetime.date.today()
        
        hist_dates = st.date_input(
            "选择回测时间范围",
            value=(default_start, default_end),
            max_value=datetime.date.today()
        )
    with col_hist_2:
        init_cap_hist = st.number_input("初始资金 ($)", value=10000, step=1000)
    with col_hist_3:
        st.write("") # Spacer
        st.write("") 
        run_hist_btn = st.button("🚀 运行策略回测", type="primary")

    if run_hist_btn:
        if isinstance(hist_dates, tuple) and len(hist_dates) == 2:
            h_start, h_end = hist_dates
            
            with st.spinner(f"正在获取历史数据并进行模拟 ({h_start} 至 {h_end})..."):
                # 1. Get Macro States
                df_states, err_msg = get_historical_macro_data(h_start, h_end)
                
                if not df_states.empty:
                    # 2. Run Backtest
                    df_res, bt_err = run_dynamic_backtest(df_states, h_start, h_end, init_cap_hist)
                    
                    if df_res is not None:
                        st.success("回测完成!")
                        
                        # --- Metrics Calculation ---
                        metrics_list = []
                        for col in df_res.columns:
                            s = df_res[col]
                            if s.empty: continue
                            
                            tot_ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
                            days = (s.index[-1] - s.index[0]).days
                            cagr = ((s.iloc[-1] / s.iloc[0]) ** (365 / days) - 1) * 100 if days > 0 else 0
                            
                            rolling_max = s.cummax()
                            dd = (s / rolling_max - 1) * 100
                            max_dd = dd.min()
                            
                            daily_ret = s.pct_change().dropna()
                            vol = daily_ret.std() * np.sqrt(252) * 100
                            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
                            
                            metrics_list.append({
                                "Strategy": col,
                                "Total Return": tot_ret,
                                "CAGR": cagr,
                                "Max Drawdown": max_dd,
                                "Volatility": vol,
                                "Sharpe": sharpe
                            })
                            
                        st.dataframe(
                            pd.DataFrame(metrics_list).set_index("Strategy"),
                            use_container_width=True,
                            column_config={
                                "Total Return": st.column_config.NumberColumn(format="%.2f%%"),
                                "CAGR": st.column_config.NumberColumn(format="%.2f%%"),
                                "Max Drawdown": st.column_config.NumberColumn(format="%.2f%%"),
                                "Volatility": st.column_config.NumberColumn(format="%.2f%%"),
                                "Sharpe": st.column_config.NumberColumn(format="%.2f")
                            }
                        )
                        
                        # --- Charts ---
                        tab_b1, tab_b2 = st.tabs(["📈 净值曲线 (Equity Curve)", "📉 回撤 (Drawdown)"])
                        
                        with tab_b1:
                            fig_bt = go.Figure()
                            colors = {'Dynamic Strategy': '#2962FF', 'IWY (Benchmark)': '#FF6D00', '60/40 (Balanced)': '#00C853', 'Neutral (Fixed)': '#AA00FF'}
                            
                            for col in df_res.columns:
                                width = 3 if col == "Dynamic Strategy" else 1.5
                                fig_bt.add_trace(go.Scatter(
                                    x=df_res.index, y=df_res[col],
                                    name=col,
                                    line=dict(width=width, color=colors.get(col, 'gray'))
                                ))
                                
                            fig_bt.update_layout(
                                title="Strategy vs Benchmarks Performance",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                height=600,
                                template="plotly_white",
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig_bt, use_container_width=True)
                            
                        with tab_b2:
                            fig_dd = go.Figure()
                            for col in df_res.columns:
                                s = df_res[col]
                                dd = (s / s.cummax() - 1) * 100
                                fig_dd.add_trace(go.Scatter(
                                    x=dd.index, y=dd,
                                    name=col,
                                    fill='tozeroy' if col == "Dynamic Strategy" else None
                                ))
                            fig_dd.update_layout(title="Drawdown (%)", template="plotly_white", height=500)
                            st.plotly_chart(fig_dd, use_container_width=True)
                            
                    else:
                        st.error(f"Backtest Failed: {bt_err}")
                else:
                    if err_msg:
                        st.error(err_msg)
                    else:
                        st.warning("无有效宏观数据。")
        else:
            st.info("请选择完整的日期范围。")

# --- Page 2: Portfolio Backtest ---

def render_portfolio_backtest():
    st.header("📊 投资组合回测 (Portfolio Backtest)")
    st.caption("Design, test, and optimize your investment strategy.")
    
    # Init session state
    if 'port_selected_popular' not in st.session_state:
        st.session_state['port_selected_popular'] = ["SPY", "TLT"]
    if 'port_custom_tickers' not in st.session_state:
        st.session_state['port_custom_tickers'] = ""
    
    popular_etfs = {
        "US": ["SPY", "QQQ", "VOO", "VTI", "TLT", "GLD", "XLK", "XLF", "VNQ", "IWM"],
        "SG": ["ES3.SI", "G3B.SI", "S27.SI", "A35.SI", "O9P.SI", "CLR.SI"]
    }
    all_popular = popular_etfs["US"] + popular_etfs["SG"]

    # --- Sidebar: Global Settings & Portfolio Management ---
    st.sidebar.header("⚙️ Configuration")
    
    with st.sidebar.expander("📅 Time & Capital", expanded=True):
        start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("today"))
        initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000, format="%d")

    # Portfolio Load/Save Management
    saved_portfolios = load_portfolios()
    
    with st.sidebar.expander("📂 Portfolio Manager", expanded=False):
        selected_saved = st.selectbox("Select Saved Portfolio", ["-- New / Unselected --"] + list(saved_portfolios.keys()))
        
        col_load, col_del = st.columns(2)
        if col_load.button("Load", use_container_width=True):
            if selected_saved != "-- New / Unselected --":
                p_data = saved_portfolios[selected_saved]
                saved_tickers = p_data.get("tickers", [])
                saved_weights = p_data.get("weights", {})
                
                # Update Tickers
                new_popular = [t for t in saved_tickers if t in all_popular]
                new_custom = [t for t in saved_tickers if t not in all_popular]
                
                st.session_state['port_selected_popular'] = new_popular
                st.session_state['port_custom_tickers'] = ", ".join(new_custom)
                
                # Update Weights
                for t, w in saved_weights.items():
                    st.session_state[f"w_{t}"] = w
                    
                st.toast(f"Loaded: {selected_saved}", icon="✅")
                st.rerun()

        if col_del.button("Delete", use_container_width=True):
            if selected_saved != "-- New / Unselected --":
                delete_portfolio(selected_saved)
                st.toast(f"Deleted {selected_saved}", icon="🗑️")
                st.rerun()

    # --- Main Area: Composition & Analysis ---
    
    # 1. Portfolio Composition Area (Card Style)
    with st.container():
        st.subheader("🛠️ Build Your Portfolio")
        
        col_comp_1, col_comp_2 = st.columns([1, 1])
        
        with col_comp_1:
            st.markdown("**1. Select Assets**")
            selected_popular = st.multiselect("Popular ETFs", all_popular, key="port_selected_popular")
            custom_tickers = st.text_input("Custom Tickers (comma separated)", placeholder="e.g. MSFT, D05.SI", key="port_custom_tickers")
            
            # Merge tickers
            tickers = list(selected_popular)
            if custom_tickers:
                custom_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
                for t in custom_list:
                    if t not in tickers:
                        tickers.append(t)
        
        with col_comp_2:
            st.markdown("**2. Allocation (%)**")
            if not tickers:
                st.info("Please select assets first.")
                weights = {}
            else:
                # --- Fix Data Editor State Sync ---
                # Initialize state variables if not present
                if 'last_tickers' not in st.session_state:
                    st.session_state['last_tickers'] = []
                if 'internal_weights_df' not in st.session_state:
                    st.session_state['internal_weights_df'] = pd.DataFrame(columns=["Ticker", "Weight"])

                # Check if tickers have changed (Add/Remove assets)
                current_tickers = tickers
                if current_tickers != st.session_state['last_tickers']:
                    # Re-initialize weights logic
                    default_w = 100.0 / len(current_tickers)
                    new_data = []
                    
                    # Try to preserve existing weights from the internal DF or previous individual keys
                    prev_weights = {}
                    if not st.session_state['internal_weights_df'].empty:
                         prev_weights = dict(zip(st.session_state['internal_weights_df']['Ticker'], st.session_state['internal_weights_df']['Weight']))
                    
                    # Fallback to check individual keys if DF is empty (migration)
                    for t in current_tickers:
                        if t in prev_weights:
                            w = prev_weights[t]
                        elif f"w_{t}" in st.session_state:
                             w = st.session_state[f"w_{t}"]
                        else:
                            w = default_w
                        new_data.append({"Ticker": t, "Weight": w})
                    
                    st.session_state['internal_weights_df'] = pd.DataFrame(new_data)
                    st.session_state['last_tickers'] = current_tickers

                # Render Data Editor using the persistent DataFrame
                edited_df = st.data_editor(
                    st.session_state['internal_weights_df'],
                    column_config={
                        "Ticker": st.column_config.TextColumn("Asset", disabled=True),
                        "Weight": st.column_config.NumberColumn("Weight (%)", min_value=0, max_value=100, step=1, format="%.1f")
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="weight_editor"
                )
                
                # Update the persistent DataFrame immediately with edited values
                st.session_state['internal_weights_df'] = edited_df
                
                # Extract weights for calculation
                weights = dict(zip(edited_df['Ticker'], edited_df['Weight']))
                
                # Sync back to individual keys (optional, but keeps compatibility)
                for t, w in weights.items():
                    st.session_state[f"w_{t}"] = w

        # Validation & Actions
        if tickers:
            # --- Benchmark Selection (Moved Up) ---
            st.markdown("---")
            col_bench_1, col_bench_2 = st.columns([3, 1])
            with col_bench_1:
                default_benchmarks = {
                    "Benchmark: S&P 500 (SPY)": {"tickers": ["SPY"], "weights": {"SPY": 100}},
                    "Benchmark: Nasdaq 100 (QQQ)": {"tickers": ["QQQ"], "weights": {"QQQ": 100}},
                    "Benchmark: 60/40 Balanced": {"tickers": ["SPY", "TLT"], "weights": {"SPY": 60, "TLT": 40}}
                }
                available_comparisons = list(default_benchmarks.keys()) + list(saved_portfolios.keys())
                selected_comparisons = st.multiselect(
                    "⚔️ Benchmark / Compare Against (Optional):", 
                    available_comparisons,
                    placeholder="Select benchmarks to compare performance..."
                )
            
            st.divider()

            total_weight = sum(weights.values())
            
            # Action Bar
            col_act_1, col_act_2, col_act_3 = st.columns([2, 1, 1])
            with col_act_1:
                 if abs(total_weight - 100) > 0.1:
                    st.warning(f"Total Allocation: {total_weight:.1f}% (Will be normalized to 100%)", icon="⚠️")
                 else:
                    st.success(f"Total Allocation: {total_weight:.1f}%", icon="✅")
            
            with col_act_2:
                # Save Logic
                with st.popover("💾 Save Portfolio"):
                    save_name = st.text_input("Name", placeholder="My Portfolio")
                    if st.button("Confirm Save", type="primary"):
                        if save_name:
                            save_portfolio(save_name, tickers, weights)
                            st.toast(f"Saved '{save_name}'!", icon="💾")
                        else:
                            st.error("Name required.")

            with col_act_3:
                run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
        else:
            run_backtest = False

    st.markdown("---")

    # 2. Backtest Analysis Area
    if run_backtest and tickers:
        
        with st.spinner("Crunching numbers..."):
            try:
                # 1. Collect Tickers
                all_tickers_set = set(tickers)
                comparison_specs = {} 
                
                for comp_name in selected_comparisons:
                    # Check if it's a default benchmark
                    if comp_name in default_benchmarks:
                        comp_data = default_benchmarks[comp_name]
                    else:
                        comp_data = saved_portfolios.get(comp_name, {})
                        
                    c_tickers = comp_data.get("tickers", [])
                    c_weights_raw = comp_data.get("weights", {})
                    
                    if c_tickers:
                        all_tickers_set.update(c_tickers)
                        c_total_w = sum(c_weights_raw.values())
                        if c_total_w > 0:
                            c_weights_norm = {k: v / c_total_w for k, v in c_weights_raw.items()}
                            comparison_specs[comp_name] = {"tickers": c_tickers, "weights": c_weights_norm}

                download_tickers = list(all_tickers_set)

                # 2. Fetch Data
                # Added auto_adjust=False to maintain consistent behavior
                data_raw = yf.download(download_tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
                
                if data_raw.empty:
                    st.error("No data found. Check tickers or internet connection.")
                    return
                
                if isinstance(data_raw, pd.Series):
                    data_raw = data_raw.to_frame(name=download_tickers[0])
                elif isinstance(data_raw, pd.DataFrame) and len(download_tickers) == 1:
                    data_raw.columns = download_tickers
                
                data = data_raw.dropna(axis=1, how='all')
                available_tickers = set(data.columns.tolist())
                
                if not available_tickers:
                    st.error("No data for selected assets.")
                    return
                
                data = data.fillna(method='ffill').fillna(method='bfill')
                normalized_prices = data / data.iloc[0]

                # --- Calculation Helper ---
                def calculate_portfolio_performance(p_tickers, p_weights_norm, p_name):
                    valid_p_tickers = [t for t in p_tickers if t in available_tickers]
                    if not valid_p_tickers: return None
                    
                    valid_w_sum = sum([p_weights_norm.get(t, 0) for t in valid_p_tickers])
                    if valid_w_sum == 0: return None
                    
                    p_weights_final = {t: p_weights_norm.get(t, 0) / valid_w_sum for t in valid_p_tickers}
                    
                    # Calc Value
                    val_series = pd.Series(0, index=data.index)
                    for t in valid_p_tickers:
                        w = p_weights_final[t]
                        val_series += normalized_prices[t] * (initial_capital * w)
                    
                    val_series.name = p_name
                    
                    # Metrics
                    tot_ret = (val_series.iloc[-1] / val_series.iloc[0] - 1) * 100
                    days = (val_series.index[-1] - val_series.index[0]).days
                    cagr = ((val_series.iloc[-1] / val_series.iloc[0]) ** (365 / days) - 1) * 100 if days > 0 else 0
                    
                    rolling_max = val_series.cummax()
                    dd = (val_series / rolling_max - 1) * 100
                    max_dd = dd.min()
                    
                    daily_ret = val_series.pct_change().dropna()
                    vol = daily_ret.std() * np.sqrt(252) * 100
                    
                    rf_daily = 0.03 / 252
                    excess = daily_ret - rf_daily
                    sharpe = (excess.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
                    
                    # Sortino Ratio
                    downside_returns = daily_ret[daily_ret < 0]
                    downside_std = downside_returns.std() * np.sqrt(252)
                    sortino = (excess.mean() * 252 * 100) / (downside_std * 100) if downside_std > 0 else 0
                    
                    # Calmar Ratio
                    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

                    # Max Drawdown Duration (Longest Recovery Time)
                    # We can define this as the max duration between a peak and the recovery to that peak
                    is_in_drawdown = dd < 0
                    current_duration = 0
                    max_duration_days = 0
                    
                    for is_dd in is_in_drawdown:
                        if is_dd:
                            current_duration += 1
                        else:
                            # Ended a drawdown period (or was never in one)
                            # Assuming daily data, approximated by count of trading days
                            # More precise would be difference in dates, but this is a good approximation for 'trading days'
                            if current_duration > max_duration_days:
                                max_duration_days = current_duration
                            current_duration = 0
                    # Check if the last period was the longest
                    if current_duration > max_duration_days:
                        max_duration_days = current_duration
                    
                    return {
                        "name": p_name,
                        "series": val_series,
                        "drawdown": dd,
                        "metrics": {
                            "Final Balance": val_series.iloc[-1],
                            "Total Return (%)": tot_ret,
                            "CAGR (%)": cagr,
                            "Max Drawdown (%)": max_dd,
                            "Max DD Duration (Days)": max_duration_days,
                            "Volatility (%)": vol,
                            "Sharpe Ratio": sharpe,
                            "Sortino Ratio": sortino,
                            "Calmar Ratio": calmar
                        }
                    }

                # 3. Calculate "Current" Portfolio
                current_perf = calculate_portfolio_performance(tickers, weights, "Current Portfolio")
                if not current_perf:
                    st.error("Invalid current portfolio data.")
                    return
                
                results = [current_perf]
                
                # 4. Calculate Comparison Portfolios
                for c_name, c_spec in comparison_specs.items():
                    c_perf = calculate_portfolio_performance(c_spec["tickers"], c_spec["weights"], c_name)
                    if c_perf:
                        results.append(c_perf)
                    else:
                        st.warning(f"Skipping '{c_name}': insufficient data.")
                
                # --- Display Results ---
                st.subheader("📈 Backtest Results")
                
                # A. Summary Metrics (Top Row - KPI Cards)
                curr_metrics = results[0]["metrics"]
                
                cols_kpi = st.columns(4)
                cols_kpi[0].metric("Total Return", f"{curr_metrics['Total Return (%)']:.2f}%", help="Cumulative return over period")
                cols_kpi[1].metric("CAGR", f"{curr_metrics['CAGR (%)']:.2f}%", help="Compound Annual Growth Rate")
                cols_kpi[2].metric("Max Drawdown", f"{curr_metrics['Max Drawdown (%)']:.2f}%", help="Deepest peak-to-valley decline")
                cols_kpi[3].metric("Sharpe Ratio", f"{curr_metrics['Sharpe Ratio']:.2f}", help="Risk-adjusted return")

                # B. Interactive Charts & Details
                tab_chart, tab_dd, tab_monthly, tab_stats, tab_corr = st.tabs(["💰 Value Growth", "📉 Drawdowns", "📅 Monthly Returns", "📋 Detailed Stats", "🔥 Correlation"])
                
                with tab_chart:
                    fig = go.Figure()
                    # Add Current (Thicker line)
                    curr_s = results[0]["series"]
                    fig.add_trace(go.Scatter(x=curr_s.index, y=curr_s, name=results[0]["name"], line=dict(width=3, color='#2962FF')))
                    
                    # Add Comparisons
                    colors = ['#FF6D00', '#00C853', '#AA00FF', '#FFD600', '#D50000', '#3E2723']
                    for i, res in enumerate(results[1:]):
                        col = colors[i % len(colors)]
                        fig.add_trace(go.Scatter(
                            x=res["series"].index, 
                            y=res["series"], 
                            name=res["name"], 
                            line=dict(width=2, color=col, dash='dot')
                        ))
                    
                    fig.update_layout(
                        title="Portfolio Value Comparison",
                        xaxis_title="Date",
                        yaxis_title="Value ($)",
                        height=550,
                        template="plotly_white",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab_dd:
                    fig_dd = go.Figure()
                    # Current
                    fig_dd.add_trace(go.Scatter(x=results[0]["drawdown"].index, y=results[0]["drawdown"], name=results[0]["name"], line=dict(width=2, color='#2962FF'), fill='tozeroy'))
                    
                    # Comparisons
                    for i, res in enumerate(results[1:]):
                        col = colors[i % len(colors)]
                        fig_dd.add_trace(go.Scatter(x=res["drawdown"].index, y=res["drawdown"], name=res["name"], line=dict(width=1, color=col)))
                        
                    fig_dd.update_layout(title="Portfolio Drawdown (%)", yaxis_title="Drawdown %", template="plotly_white", height=500, hovermode="x unified")
                    st.plotly_chart(fig_dd, use_container_width=True)

                with tab_monthly:
                    st.markdown("#### 📅 Monthly Returns Heatmap")
                    
                    # Select portfolio to visualize
                    port_names = [r["name"] for r in results]
                    selected_heatmap_port = st.selectbox("Select Portfolio:", port_names, key="heatmap_port_select")
                    
                    # Find selected result
                    sel_res = next((r for r in results if r["name"] == selected_heatmap_port), results[0])
                    
                    # Calculate Monthly Returns
                    daily_s = sel_res["series"]
                    monthly_s = daily_s.resample('M').last().pct_change() * 100
                    
                    if not monthly_s.empty:
                        # Prepare Pivot Table
                        monthly_df = monthly_s.to_frame(name='Return')
                        monthly_df['Year'] = monthly_df.index.year
                        monthly_df['Month'] = monthly_df.index.month_name().str[:3] # Jan, Feb...
                        
                        # Pivot: Index=Year, Columns=Month
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        pivot_ret = monthly_df.pivot_table(index='Year', columns='Month', values='Return')
                        pivot_ret = pivot_ret.reindex(columns=month_order)
                        
                        # Add Year Total
                        year_ret = daily_s.resample('Y').last().pct_change() * 100
                        year_ret.index = year_ret.index.year
                        pivot_ret['YTD'] = year_ret
                        
                        # Heatmap using Plotly
                        fig_hm = go.Figure(data=go.Heatmap(
                            z=pivot_ret.values,
                            x=pivot_ret.columns,
                            y=pivot_ret.index,
                            colorscale='RdBu',
                            zmid=0,
                            text=np.round(pivot_ret.values, 1),
                            texttemplate="%{text}%",
                            showscale=True
                        ))
                        fig_hm.update_layout(
                            title=f"{selected_heatmap_port} - Monthly Returns (%)",
                            height=max(400, len(pivot_ret)*30 + 100),
                            yaxis=dict(autorange="reversed", type='category')
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)
                    else:
                        st.info("Not enough data for monthly analysis.")

                with tab_stats:
                    metrics_data = []
                    for res in results:
                        m = res["metrics"]
                        row = {"Portfolio": res["name"]}
                        row.update(m)
                        metrics_data.append(row)
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(
                        metrics_df,
                        use_container_width=True,
                        column_config={
                            "Final Balance": st.column_config.NumberColumn(format="$%.2f"),
                            "Total Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                            "CAGR (%)": st.column_config.NumberColumn(format="%.2f%%"),
                            "Max Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
                            "Max DD Duration (Days)": st.column_config.NumberColumn(help="Longest time to recover from a drawdown (in trading days)"),
                            "Volatility (%)": st.column_config.NumberColumn(format="%.2f%%"),
                            "Sharpe Ratio": st.column_config.NumberColumn(format="%.2f"),
                            "Sortino Ratio": st.column_config.NumberColumn(format="%.2f"),
                            "Calmar Ratio": st.column_config.NumberColumn(format="%.2f"),
                        },
                        hide_index=True
                    )

                with tab_corr:
                    if len(tickers) > 1:
                        # Extract data for current portfolio tickers only
                        valid_curr_tickers = [t for t in tickers if t in available_tickers]
                        if len(valid_curr_tickers) > 1:
                            curr_data = data[valid_curr_tickers]
                            corr = curr_data.pct_change().corr()
                            fig_corr = go.Figure(data=go.Heatmap(
                                z=corr.values,
                                x=corr.columns,
                                y=corr.index,
                                colorscale='RdBu',
                                zmin=-1, zmax=1,
                                text=np.round(corr.values, 2),
                                texttemplate="%{text}",
                                showscale=True
                            ))
                            fig_corr.update_layout(height=600, title="Asset Correlation Matrix")
                            st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Correlation matrix requires at least 2 assets in the portfolio.")

                # --- Download Section ---
                st.markdown("### 📥 Export Data")
                
                # Prepare Daily Data CSV
                df_export = pd.DataFrame(index=data.index)
                for res in results:
                    df_export[f"{res['name']} Value"] = res["series"]
                    df_export[f"{res['name']} Drawdown"] = res["drawdown"]
                
                csv_data = df_export.to_csv().encode('utf-8')
                
                st.download_button(
                    label="Download Daily Backtest Data (CSV)",
                    data=csv_data,
                    file_name="backtest_daily_data.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Analysis Error: {e}")


# --- Main App Navigation ---

st.sidebar.title("App Navigation")
page = st.sidebar.radio("选择功能", ["单只股票策略分析", "投资组合回测", "状态机检查"])

if page == "单只股票策略分析":
    render_single_stock_analysis()
elif page == "投资组合回测":
    render_portfolio_backtest()
else:
    render_state_machine_check()
