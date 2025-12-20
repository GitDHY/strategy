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
    1. Local file (fred_{series_id}.csv or {series_id}.csv) in app dir or CWD
    2. Network fetch (fred.stlouisfed.org)
    """
    # 1. Check local file override
    # Define search candidates
    candidates = [
        os.path.join(os.path.dirname(__file__), f"fred_{series_id}.csv"),
        os.path.join(os.getcwd(), f"fred_{series_id}.csv"),
        os.path.join(os.path.dirname(__file__), f"{series_id}.csv"),
        os.path.join(os.getcwd(), f"{series_id}.csv"),
    ]
    # Remove duplicates
    candidates = list(dict.fromkeys(candidates))
    
    for local_file in candidates:
        if os.path.exists(local_file):
            try:
                df = pd.read_csv(local_file, parse_dates=['observation_date'], index_col='observation_date')
                df.columns = [series_id]
                # st.toast(f"Using local file for {series_id}", icon="📂") # Optional toast
                return df
            except Exception as e:
                st.warning(f"Found local file {local_file} but failed to read: {e}")
                # Continue to try network or other files? 
                # If we found a file but failed to read, it's likely the intended file. 
                # But let's fallback to network just in case.

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
            st.warning(f"⚠️ 无法连接 FRED 数据源 ({series_id})。网络连接被切断。\n\n**尝试寻找的本地路径**:\n{candidates}\n\n**解决方法**：请展开页面顶部的 **“📂 手动导入宏观数据”** 面板，上传该数据文件即可恢复正常。")
            print(f"Error fetching FRED data ({series_id}): {e}")
            # Return empty DataFrame on failure
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

def calculate_equity_curve_metrics(series, risk_free_rate=0.03):
    """
    Calculates comprehensive performance metrics for an equity curve.
    series: pd.Series of portfolio values or prices, indexed by datetime.
    risk_free_rate: Annualized risk-free rate (decimal).
    """
    if series.empty or len(series) < 2:
        return {}
    
    # 1. Basic Returns
    total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
    days = (series.index[-1] - series.index[0]).days
    if days > 0:
        cagr = ((series.iloc[-1] / series.iloc[0]) ** (365 / days) - 1) * 100
    else:
        cagr = 0.0

    # 2. Drawdown
    rolling_max = series.cummax()
    drawdown = (series / rolling_max - 1) * 100
    max_dd = drawdown.min()
    
    # 3. Daily Returns Analysis
    daily_ret = series.pct_change().fillna(0)
    
    # 4. Volatility (Annualized)
    vol = daily_ret.std() * np.sqrt(252) * 100
    
    # 5. Risk-Adjusted Returns
    rf_daily = risk_free_rate / 252
    excess_ret = daily_ret - rf_daily
    
    if daily_ret.std() > 0:
        sharpe = (excess_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    # Sortino (Downside Deviation)
    downside_ret = daily_ret[daily_ret < 0]
    if len(downside_ret) > 0:
        downside_std = downside_ret.std() * np.sqrt(252)
        if downside_std > 0:
            sortino = (excess_ret.mean() * 252) / downside_std # Annualized excess return / Annualized downside deviation
        else:
            sortino = 0.0
    else:
        sortino = 0.0 # No downside
        
    # Calmar
    if abs(max_dd) > 0:
        calmar = cagr / abs(max_dd)
    else:
        calmar = 0.0
        
    # 6. Trade/Win Analysis (Daily Basis)
    # Win Rate: % of days with positive return
    winning_days = daily_ret[daily_ret > 0].count()
    losing_days = daily_ret[daily_ret < 0].count()
    total_trading_days = winning_days + losing_days
    
    win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0.0
    
    # Profit/Loss Ratio: Avg Win / Avg Loss
    avg_win = daily_ret[daily_ret > 0].mean() if winning_days > 0 else 0
    avg_loss = abs(daily_ret[daily_ret < 0].mean()) if losing_days > 0 else 0
    
    pl_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    
    return {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Max Drawdown (%)": max_dd,
        "Volatility (%)": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Win Rate (Daily %)": win_rate,
        "Profit/Loss Ratio": pl_ratio
    }

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




# --- Page 3: State Machine Check ---

# --- Constants & Config ---
MACRO_STATES = {
    "INFLATION_SHOCK": {
        "display": "🔴 滞胀 / 加息冲击 (Inflation Shock)",
        "desc": "⚠️ **严重警报**：利率飙升或出现股债双杀。传统资产失效。**现金为王**，清仓长久期资产。",
        "bg_color": "#fce8e6", "border_color": "#d93025", "icon": "🔴"
    },
    "DEFLATION_RECESSION": {
        "display": "🔵 衰退 / 崩盘 (Deflation/Crash)",
        "desc": "⚠️ **严重警报**：经济衰退确认或流动性危机。**全面防御**，锁定国债收益，配置黄金避险。",
        "bg_color": "#e8f0fe", "border_color": "#1a73e8", "icon": "🔵"
    },
    "EXTREME_ACCUMULATION": {
        "display": "🚀 极度贪婪 / 抄底 (Accumulation)",
        "desc": "🔔 **机会提示**：市场极度恐慌但基本面未崩坏。建议**重仓抄底**成长股，利用别人的恐慌获利。",
        "bg_color": "rgba(142, 36, 170, 0.2)", "border_color": "#8e24aa", "icon": "🚀"
    },
    "CAUTIOUS_TREND": {
        "display": "⚠️ 谨慎 / 趋势破位 (Bear Trend)",
        "desc": "📉 **风险提示**：长期趋势转空但恐慌未起（阴跌）。建议转为**防御配置**，重仓红利与现金。",
        "bg_color": "#fff3e0", "border_color": "#f57c00", "icon": "📉"
    },
    "CAUTIOUS_VOL": {
        "display": "⚡ 谨慎 / 高波震荡 (High Volatility)",
        "desc": "🌊 **风险提示**：趋势尚可但波动加剧。建议保留成长仓位，但增加**危机Alpha (WTMF)** 进行对冲。",
        "bg_color": "#fff8e1", "border_color": "#ffb74d", "icon": "⚡"
    },
    "NEUTRAL": {
        "display": "🟢 常态 / 牛市 (Neutral)",
        "desc": "✅ 市场运行平稳，波动率低且趋势向上。建议维持**标准增长配置**，享受复利增长。",
        "bg_color": "#e6f4ea", "border_color": "#1e8e3e", "icon": "🟢"
    }
}

ASSET_NAMES = {
    'IWY': '美股成长 (Russell Top 200 Growth)',
    'WTMF': '危机Alpha (Managed Futures)',
    'LVHI': '美股红利 (High Div Low Vol)',
    'G3B.SI': '新加坡蓝筹 (STI ETF)',
    'MBH.SI': '新元债券 (Govt Bond)',
    'GSD.SI': '黄金 (Gold)',
    'SRT.SI': 'S-REITs (Supermarket)',
    'AJBU.SI': 'Keppel DC REIT',
    'TLT': '美债 (20Y Treasury)',
    'SPY': '标普500 (S&P 500)',
    'OTHERS': '其他/待清理资产 (Others)'
}

def determine_macro_state(row):
    """
    Determines macro state based on a row of indicators.
    Expected row keys: Sahm, RateShock, Corr, VIX, Trend_Bear
    """
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

@st.cache_data
def get_historical_macro_data(start_date, end_date):
    """
    Fetches and calculates macro states for a given date range.
    Includes buffer to ensure valid data at start_date.
    """
    buffer_days = 365 * 2 # Increase buffer for Sahm Rule (12m min)
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
            
        if data.empty:
             return pd.DataFrame(), "Market data fetch failed or incomplete."
             
    except Exception as e:
        return pd.DataFrame(), f"Error fetching market data: {str(e)}"

    # 2. Fetch FRED Data (UNRATE & T10Y2Y)
    try:
        unrate = fetch_fred_data("UNRATE")
        yc = fetch_fred_data("T10Y2Y") 
        
        if unrate.empty:
            raise ValueError("Fetched empty data for UNRATE")
            
        unrate.columns = ['UNRATE']
        unrate = unrate[unrate.index >= fetch_start]
        # Reindex
        unrate_daily = unrate.reindex(data.index, method='ffill')
        
        if not yc.empty:
            yc.columns = ['T10Y2Y']
            yc = yc[yc.index >= fetch_start]
            yc_daily = yc.reindex(data.index, method='ffill')
        else:
            yc_daily = pd.DataFrame(0.0, index=data.index, columns=['T10Y2Y'])

    except Exception as e:
        return pd.DataFrame(), f"Error fetching FRED data: {str(e)}"

    # 3. Calculate Indicators
    try:
        # Sahm Rule
        u_monthly = unrate['UNRATE']
        u_3m_avg = u_monthly.rolling(window=3).mean()
        u_12m_low = u_3m_avg.rolling(window=12).min().shift(1)
        sahm_monthly = u_3m_avg - u_12m_low
        sahm_series = sahm_monthly.reindex(data.index, method='ffill')
        
        # Rate Shock
        tnx_col = '^TNX' if '^TNX' in data.columns else data.columns[0]
        tnx_roc = (data[tnx_col] - data[tnx_col].shift(21)) / data[tnx_col].shift(21)
        
        # Correlation
        if 'IWY' in data.columns and 'TLT' in data.columns:
            corr = data['IWY'].rolling(60).corr(data['TLT'])
            iwy_series = data['IWY']
        else:
            corr = pd.Series(0, index=data.index)
            iwy_series = data.iloc[:, 0]
        
        # Trend
        iwy_ma200 = iwy_series.rolling(200).mean()
        trend_bear = iwy_series < iwy_ma200
        
        # Gold Trend
        gold_trend_bear = pd.Series(False, index=data.index)
        if 'GLD' in data.columns:
            gld_ma200 = data['GLD'].rolling(200).mean()
            gold_trend_bear = data['GLD'] < gld_ma200
            
        # Style Trend
        style_value_regime = pd.Series(False, index=data.index)
        if 'IWY' in data.columns and 'IWD' in data.columns:
            pair_ratio = data['IWY'] / data['IWD']
            pair_ma200 = pair_ratio.rolling(200).mean()
            style_value_regime = pair_ratio < pair_ma200 

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
        df_hist['State'] = df_hist.apply(determine_macro_state, axis=1)
        
        # Filter Output
        df_final = df_hist.loc[(df_hist.index >= pd.to_datetime(start_date)) & (df_hist.index <= pd.to_datetime(end_date))]
        return df_final, None

    except Exception as e:
        return pd.DataFrame(), f"Error in calculation: {str(e)}"

# --- UI Components ---

def render_manual_data_import():
    """Renders the manual data import expander."""
    with st.expander("📂 手动导入宏观数据 (网络受限时使用)", expanded=False):
        st.info("如果网络受限导致 FRED 数据 (UNRATE, T10Y2Y) 获取失败，请手动下载并上传 CSV 文件。")
        col_u1, col_u2 = st.columns(2)
        import time

        # UNRATE Import
        with col_u1:
            st.markdown("**1. 失业率 (UNRATE)**")
            unrate_path = os.path.join(os.path.dirname(__file__), "fred_UNRATE.csv")
            if os.path.exists(unrate_path):
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(unrate_path)).strftime('%Y-%m-%d %H:%M')
                st.success(f"✅ 已检测到本地数据 ({file_time})")
            else:
                st.warning("⚠️ 未检测到本地文件")

            st.markdown("[📥 下载 UNRATE.csv](https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE)")
            uploaded_file = st.file_uploader("上传 UNRATE.csv", type=['csv'], key="uploader_unrate")
            
            if uploaded_file is not None:
                file_id = f"{uploaded_file.name}-{uploaded_file.size}"
                if st.session_state.get("processed_unrate_id") != file_id:
                    try:
                        df_test = pd.read_csv(uploaded_file)
                        if 'observation_date' in df_test.columns:
                            uploaded_file.seek(0)
                            with open(unrate_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            st.session_state["processed_unrate_id"] = file_id
                            st.success("✅ UNRATE 已保存! 刷新中...")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("格式错误: 缺少 'observation_date'")
                    except Exception as e:
                        st.error(f"错误: {e}")

        # T10Y2Y Import
        with col_u2:
            st.markdown("**2. 收益率曲线 (T10Y2Y)**")
            yc_path = os.path.join(os.path.dirname(__file__), "fred_T10Y2Y.csv")
            if os.path.exists(yc_path):
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(yc_path)).strftime('%Y-%m-%d %H:%M')
                st.success(f"✅ 已检测到本地数据 ({file_time})")
            else:
                st.warning("⚠️ 未检测到本地文件")

            st.markdown("[📥 下载 T10Y2Y.csv](https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y)")
            uploaded_yc = st.file_uploader("上传 T10Y2Y.csv", type=['csv'], key="uploader_t10y2y")
            
            if uploaded_yc is not None:
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
                            st.success("✅ T10Y2Y 已保存! 刷新中...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("格式错误: 缺少 'observation_date'")
                    except Exception as e:
                        st.error(f"错误: {e}")

def render_reference_guide():
    """Renders the state reference guide."""
    with st.expander("📖 新手指南：市场状态与应对策略 (Beginner's Guide)", expanded=False):
        st.info("💡 **系统逻辑**：自动分析宏观数据，判断当前“经济季节”，并建议“穿什么衣服”（资产配置）。")
        cols = st.columns(5)
        # Using shared constants
        states_order = ["INFLATION_SHOCK", "DEFLATION_RECESSION", "EXTREME_ACCUMULATION", "CAUTIOUS_TREND", "NEUTRAL"]
        
        for i, s_key in enumerate(states_order):
            s = MACRO_STATES[s_key]
            with cols[i]:
                # Simple color coding for headers
                header_colors = {"🔴": "red", "🔵": "blue", "🚀": "violet", "⚠️": "orange", "⚡": "orange", "🟢": "green"}
                color = header_colors.get(s['icon'], "gray")
                st.markdown(f":{color}[**{s['display']}**]")
                st.markdown(s['desc'])

def render_portfolio_import():
    """Renders the import from saved portfolios section."""
    saved_portfolios = load_portfolios()
    if saved_portfolios:
        with st.expander("📥 从已保存的投资组合导入 (Import)", expanded=False):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                sel_name = st.selectbox("选择组合", list(saved_portfolios.keys()), key="sm_imp_name")
            with c2:
                imp_cap = st.number_input("总本金", value=10000.0, step=1000.0, key="sm_imp_cap")
            with c3:
                if st.button("应用到持仓", type="secondary"):
                    if sel_name in saved_portfolios:
                        p = saved_portfolios[sel_name]
                        weights = p.get("weights", {})
                        
                        # Reset known
                        known = ['IWY', 'WTMF', 'LVHI', 'G3B.SI', 'MBH.SI', 'GSD.SI', 'SRT.SI', 'AJBU.SI']
                        for t in known: st.session_state[f"hold_{t}"] = 0.0
                        st.session_state["hold_OTHERS"] = 0.0
                        
                        # Populate
                        other_val = 0.0
                        for t, w in weights.items():
                            val = imp_cap * (w / 100.0)
                            if t in known:
                                st.session_state[f"hold_{t}"] = val
                            else:
                                other_val += val
                        st.session_state["hold_OTHERS"] = other_val
                        st.toast(f"已导入: {sel_name}", icon="✅")
                        st.rerun()

def render_holdings_input():
    """Renders the holdings input section and returns the total value."""
    with st.expander("💼 输入当前持仓 (Current Portfolio)", expanded=True):
        st.markdown("请输入当前账户各标的的**市值 (Value)**。")
        cols = st.columns(4)
        
        inputs = [
            ("IWY (美股成长)", "hold_IWY"), ("WTMF (危机Alpha)", "hold_WTMF"),
            ("LVHI (美股红利)", "hold_LVHI"), ("G3B.SI (新加坡蓝筹)", "hold_G3B.SI"),
            ("MBH.SI (新元债券)", "hold_MBH.SI"), ("GSD.SI (黄金)", "hold_GSD.SI"),
            ("SRT.SI (超市REITs)", "hold_SRT.SI"), ("AJBU.SI (数据中心)", "hold_AJBU.SI"),
            ("其他资产 (Others)", "hold_OTHERS")
        ]
        
        # Init state
        for _, key in inputs:
            if key not in st.session_state: st.session_state[key] = 0.0
            
        for i, (label, key) in enumerate(inputs):
            with cols[i % 4]:
                st.number_input(label, step=100.0, key=key)
        
        current_holdings = {k.replace("hold_", ""): st.session_state[k] for _, k in inputs}
        total_value = sum(current_holdings.values())
        st.caption(f"💰 当前账户总市值: **{total_value:,.2f}**")
        return current_holdings, total_value

def render_status_card(state):
    """Renders the main status card."""
    s_conf = MACRO_STATES.get(state, MACRO_STATES["NEUTRAL"])
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {s_conf['bg_color']}; border-left: 6px solid {s_conf['border_color']}; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h2 style="margin:0; color: #202124; font-size: 28px;">{s_conf['icon']} {s_conf['display']}</h2>
        <p style="margin-top:10px; font-size: 16px; color: #3c4043; font-weight: 500;">{s_conf['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

def render_factor_dashboard(metrics):
    """Renders the metrics dashboard."""
    st.markdown("### 📊 核心宏观因子 (Macro Factors)")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        is_trig = metrics['rate_shock']
        st.metric("利率冲击 (TNX ROC)", f"{metrics['tnx_roc']:+.1%}", 
                  "⚠️ 触发" if is_trig else "✅ 安全", 
                  delta_color="inverse" if is_trig else "normal")
    with c2:
        is_trig = metrics['recession']
        val = metrics['sahm']
        st.metric("衰退信号 (Sahm)", f"{val:.2f}", 
                  "⚠️ 触发" if is_trig else "✅ 安全",
                  delta_color="inverse" if is_trig else "normal")
    with c3:
        is_trig = metrics['corr_broken']
        st.metric("股债相关性 (Corr)", f"{metrics['corr']:.2f}", 
                  "⚠️ 失效" if is_trig else "✅ 正常",
                  delta_color="inverse" if is_trig else "normal")
    with c4:
        is_trig = metrics['fear']
        st.metric("恐慌指数 (VIX)", f"{metrics['vix']:.1f}", 
                  "⚠️ 恐慌" if is_trig else "✅ 正常",
                  delta_color="inverse" if is_trig else "normal")

    st.markdown("#### 🎯 战术微调 (Tactical Modifiers)")
    c1, c2, c3 = st.columns(3)
    with c1:
        yc = metrics['yield_curve']
        status = "⚠️ 倒挂/解倒挂" if (yc < 0 or metrics['yc_un_invert']) else "✅ 正常"
        st.metric("收益率曲线 (10Y-2Y)", f"{yc:.2f}%", status, delta_color="off" if yc > 0 else "inverse")
    with c2:
        gb = metrics['gold_bear']
        st.metric("黄金趋势", "Bearish (Weak)" if gb else "Bullish (Strong)", "Avoid Gold" if gb else "Hold", delta_color="inverse" if gb else "normal")
    with c3:
        vr = metrics['value_regime']
        st.metric("风格轮动", "Value Regime" if vr else "Growth Regime", "Tilt Value" if vr else "Tilt Growth", delta_color="off")

def render_rebalancing_table(state, current_holdings, total_value, is_gold_bear, is_value_regime):
    """Renders the rebalancing table."""
    targets = get_target_percentages(state, gold_bear=is_gold_bear, value_regime=is_value_regime)
    
    # Add Current Holdings not in targets
    all_tickers = set(targets.keys()).union(current_holdings.keys())
    
    data = []
    if total_value == 0:
        st.warning("⚠️ 请输入持仓市值以获取建议。")
        return

    for tkr in all_tickers:
        tgt_pct = targets.get(tkr, 0.0)
        curr_val = current_holdings.get(tkr, 0.0)
        curr_pct = curr_val / total_value if total_value > 0 else 0
        
        diff_val = (tgt_pct - curr_pct) * total_value
        
        # Action Text
        action = "✅ 持有"
        if tkr not in targets:
            if curr_val > 1: action = f"🔴 清仓 (-{curr_val:,.0f})"
        else:
            if abs(diff_val) > total_value * 0.01: # 1% threshold
                if diff_val > 0: action = f"🟢 买入 (+{diff_val:,.0f})"
                else: action = f"🔴 卖出 ({diff_val:,.0f})"
        
        data.append({
            "代码": tkr,
            "名称": ASSET_NAMES.get(tkr, tkr),
            "目标仓位": tgt_pct,
            "当前仓位": curr_pct,
            "当前市值": curr_val,
            "建议操作": action,
            "diff": diff_val # For sort
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        # Sort: Sells first, then Buys
        df['sort_key'] = df['diff'].apply(lambda x: 0 if x < 0 else (1 if x > 0 else 2))
        df = df.sort_values('sort_key')
        
        st.dataframe(
            df,
            column_config={
                "目标仓位": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                "当前仓位": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                "当前市值": st.column_config.NumberColumn(format="$%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )

def render_historical_backtest_section():
    """Renders the independent historical backtest section."""
    st.markdown("---")
    st.markdown("### 🕰️ 历史状态回溯与策略仿真")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        dates = st.date_input("回测时间", [datetime.date.today()-datetime.timedelta(days=365*5), datetime.date.today()])
    with c2:
        cap = st.number_input("初始资金", value=10000)
    with c3:
        st.write(""); st.write("")
        run = st.button("🚀 运行回测", type="primary")
        
    if run and isinstance(dates, tuple) and len(dates)==2:
        with st.spinner("回测中..."):
            df_states, err = get_historical_macro_data(dates[0], dates[1])
            if not df_states.empty:
                res, err = run_dynamic_backtest(df_states, dates[0], dates[1], cap)
                if res is not None:
                    # Metrics & Charts (Simplified for brevity as logic exists in run_dynamic_backtest return)
                    st.success("回测完成")
                    
                    # 1. Curve
                    fig = go.Figure()
                    for c in res.columns:
                        fig.add_trace(go.Scatter(x=res.index, y=res[c], name=c))
                    fig.update_layout(title="净值曲线", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Drawdown
                    fig_dd = go.Figure()
                    for c in res.columns:
                        dd = (res[c]/res[c].cummax()-1)*100
                        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, name=c, fill='tozeroy' if 'Dynamic' in c else None))
                    fig_dd.update_layout(title="最大回撤", template="plotly_white")
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                    # 3. Metrics Table
                    metrics_list = []
                    for col in res.columns:
                        m = calculate_equity_curve_metrics(res[col])
                        row = {"Strategy": col}
                        row.update(m)
                        metrics_list.append(row)
                    
                    st.markdown("#### 📊 详细性能指标 (Performance Metrics)")
                    st.dataframe(
                        pd.DataFrame(metrics_list), 
                        use_container_width=True,
                        column_config={
                            "Strategy": st.column_config.TextColumn("策略名称", width="medium"),
                            "Total Return (%)": st.column_config.NumberColumn("总收益率", format="%.2f%%"),
                            "CAGR (%)": st.column_config.NumberColumn("年化收益 (CAGR)", format="%.2f%%"),
                            "Max Drawdown (%)": st.column_config.NumberColumn("最大回撤", format="%.2f%%"),
                            "Volatility (%)": st.column_config.NumberColumn("波动率 (年化)", format="%.2f%%"),
                            "Sharpe Ratio": st.column_config.NumberColumn("夏普比率 (Sharpe)", format="%.2f"),
                            "Sortino Ratio": st.column_config.NumberColumn("索提诺比率 (Sortino)", format="%.2f"),
                            "Calmar Ratio": st.column_config.NumberColumn("卡玛比率 (Calmar)", format="%.2f"),
                            "Win Rate (Daily %)": st.column_config.NumberColumn("胜率 (日度)", format="%.1f%%"),
                            "Profit/Loss Ratio": st.column_config.NumberColumn("盈亏比 (P/L)", format="%.2f"),
                        },
                        hide_index=True
                    )
                    
            else:
                st.error(f"无法获取数据: {err}")

def render_state_machine_check():
    st.header("🛡️ 宏观状态机与资产配置 (Macro State & Allocation)")
    st.caption("全自动资产配置生成器 (Auto-Allocator)")
    
    render_manual_data_import()
    render_reference_guide()
    render_portfolio_import()
    current_holdings, total_value = render_holdings_input()
    
    if st.button("🚀 开始诊断 (Run Analysis)", type="primary", use_container_width=True):
        with st.status("正在进行宏观扫描...", expanded=True) as status:
            st.write("📡 获取数据...")
            # Use get_historical_macro_data for "Now" by asking for recent window
            end = datetime.date.today()
            start = end - datetime.timedelta(days=365*3)
            
            # Re-use the robust fetcher
            df_hist, err = get_historical_macro_data(start, end)
            
            if df_hist.empty:
                status.update(label="诊断失败", state="error")
                st.error(err)
            else:
                st.write("✅ 数据获取与计算完成")
                status.update(label="诊断完成", state="complete", expanded=False)
                
                # Extract latest state
                last_row = df_hist.iloc[-1]
                state = last_row['State']
                
                # Logic helpers for dashboard
                # Calculate Yield Curve Un-inversion signal (Steepening from deep inversion)
                yc_series = df_hist['YieldCurve']
                yc_un_invert = False
                if len(yc_series) > 126:
                    recent_min = yc_series.iloc[-126:].min()
                    current_yc = yc_series.iloc[-1]
                    # Logic: Was deeply inverted (<-0.2) recently, now rising but still low (<0.2)
                    yc_un_invert = (current_yc < 0.2) and (recent_min < -0.2)

                metrics = {
                    'tnx_roc': last_row['RateShock'],
                    'rate_shock': last_row['RateShock'] > 0.20,
                    'sahm': last_row['Sahm'],
                    'recession': last_row['Sahm'] >= 0.50,
                    'corr': last_row['Corr'],
                    'corr_broken': last_row['Corr'] > 0.30,
                    'vix': last_row['VIX'],
                    'fear': last_row['VIX'] > 32,
                    'yield_curve': last_row['YieldCurve'],
                    'yc_un_invert': yc_un_invert,
                    'gold_bear': last_row['Gold_Bear'],
                    'value_regime': last_row['Value_Regime']
                }
                
                # Render Results
                render_status_card(state)
                render_factor_dashboard(metrics)
                
                st.markdown("---")
                render_rebalancing_table(state, current_holdings, total_value, metrics['gold_bear'], metrics['value_regime'])
                
    render_historical_backtest_section()


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
page = st.sidebar.radio("选择功能", ["状态机检查", "投资组合回测"])

if page == "状态机检查":
    render_state_machine_check()
elif page == "投资组合回测":
    render_portfolio_backtest()
