import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os

# Set page config must be the first streamlit command
st.set_page_config(layout="wide", page_title="Stock Strategy Analyzer")

# --- Helper Functions for Indicators ---

@st.cache_data
def get_fred_indpro():
    """
    Fetches Industrial Production Index (INDPRO) from FRED as a proxy for Economic Cycle.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDPRO"
    try:
        df = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date')
        df.columns = ['INDPRO']
        # Calculate YoY Growth
        df['INDPRO_YoY'] = df['INDPRO'].pct_change(12) * 100
        return df
    except Exception as e:
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
page = st.sidebar.radio("选择功能", ["单只股票策略分析", "投资组合回测"])

if page == "单只股票策略分析":
    render_single_stock_analysis()
else:
    render_portfolio_backtest()
