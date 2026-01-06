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

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time

# Set page config must be the first streamlit command
st.set_page_config(layout="wide", page_title="Stock Strategy Analyzer v1.4")

# --- Helper Functions for Indicators ---

def get_adjustment_reasons(s, gold_bear=False, value_regime=False, asset_trends=None, vix=None, yield_curve=None):
    """
    Returns a list of strings explaining why the allocation differs from the base static model.
    """
    if asset_trends is None: asset_trends = {}
    reasons = []
    
    # 1. Style Regime
    if s in ["NEUTRAL", "CAUTIOUS_TREND"] and value_regime:
        reasons.append("🧱 风格轮动: 价值占优 (Value Regime) -> 增加红利，减少成长")
        
    # 2. Dynamic Risk Control
    if s == "NEUTRAL":
        if vix is not None:
            if vix < 13.0:
                reasons.append("🚀 极度平稳 (VIX < 13): 激进模式 -> 清空WTMF/减债，加仓成长")
            elif vix > 20.0:
                reasons.append("🌬️ 早期预警 (VIX > 20): 避险模式 -> 减仓成长 20%，增加 WTMF")
    
    if s in ["DEFLATION_RECESSION", "CAUTIOUS_TREND"]:
        if yield_curve is not None and yield_curve < -0.30:
            reasons.append("⚠️ 深度倒挂 (Yield Curve < -0.3%): 债券陷阱 -> 大幅削减 MBH，转入 WTMF")

    # 3. Trend Filters
    if s != "EXTREME_ACCUMULATION":
        # Global Filter
        assets_to_check = ['G3B.SI', 'LVHI', 'MBH.SI', 'GSD.SI', 'SRT.SI', 'AJBU.SI']
        bear_assets = [t for t in assets_to_check if asset_trends.get(t, False)]
        if bear_assets:
            reasons.append(f"📉 趋势熔断: {', '.join(bear_assets)} 破位 -> 清仓")
            
        # Core IWY Filter
        if asset_trends.get('IWY', False):
            cut = "80%" if (vix and vix > 25) else "50%"
            reasons.append(f"🛡️ 核心熔断: IWY 破位 -> 削减 {cut} 仓位")
            
    # 4. Gold
    if gold_bear:
        reasons.append("🐻 黄金熊市: Gold < MA200 -> 清仓 GSD.SI")
        
    return reasons

# Removed cache for debugging connection issues
def fetch_fred_data(series_id, max_attempts: int = 2, timeout_sec: int = 10):
    """
    Robust fetch for FRED data with Auto-Update & Caching logic.
    Priority:
    1. Fresh Local File (modified today): Use directly.
    2. Network Fetch: Download and save to local (fred_{series_id}.csv), then use.
    3. Stale Local File: Fallback if network fails.
    
    改进：
    - 更详细的错误信息（状态码、异常原因 + 返回体预览）。
    - 增加 http 备份 URL，兼容部分 TLS 拦截/证书问题的网络。
    - 增加 Accept 头，避免被判为机器人流量。
    - 当日文件支持多路径/多命名 (fred_{id}.csv 或 {id}.csv)，避免手动下载后未被识别。
    - 缩短 UI 等待时间：默认 2 次尝试，每次超时 10 秒，避免前端卡顿。
    """
    base_dir = os.path.dirname(__file__)
    file_name = f"fred_{series_id}.csv"
    alt_name = f"{series_id}.csv"
    candidates = [
        os.path.join(base_dir, file_name),
        os.path.join(os.getcwd(), file_name),
        os.path.join(base_dir, alt_name),
        os.path.join(os.getcwd(), alt_name),
        os.path.join(base_dir, "data", file_name),
        os.path.join(base_dir, "data", alt_name),
    ]
    candidates = list(dict.fromkeys(candidates))
    target_path = candidates[0]
    
    # 1) 当日本地缓存（识别手动下载的两种命名）
    for path in candidates:
        if os.path.exists(path):
            try:
                mtime = datetime.date.fromtimestamp(os.path.getmtime(path))
                if mtime == datetime.date.today():
                    df = pd.read_csv(path, parse_dates=['observation_date'], index_col='observation_date')
                    df.columns = [series_id]
                    return df
            except Exception as e:
                print(f"Error reading fresh local file {path}: {e}")
    
    # 2) 网络下载（含 https -> http 备份）
    urls = [
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
        f"http://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    last_err = None
    for attempt in range(max_attempts):
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=timeout_sec, verify=False, allow_redirects=True)
                status = resp.status_code
                preview = resp.text[:200] if resp is not None else ""
                if status != 200:
                    raise RuntimeError(f"HTTP {status}, preview: {preview}")
                content = resp.content.decode('utf-8', errors='ignore')
                lower_head = content[:200].lower()
                if "<html" in lower_head or "<!doctype" in lower_head:
                    raise RuntimeError(f"HTML page returned, preview: {content[:200]}")
                if 'observation_date' not in content:
                    raise RuntimeError(f"Missing observation_date, preview: {content[:200]}")
                if len(content) < 50:
                    raise RuntimeError(f"Empty/short content (len={len(content)}), preview: {content[:200]}")
                try:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception as e:
                    print(f"Failed to write cache file: {e}")
                df = pd.read_csv(io.StringIO(content), parse_dates=['observation_date'], index_col='observation_date')
                df.columns = [series_id]
                return df
            except Exception as e:
                last_err = f"{url} -> {e}"
                continue
        time.sleep(1)
    
    if last_err:
        print(f"Error fetching FRED data ({series_id}): {last_err}")
        safe_warn(f"⚠️ 自动下载 FRED 数据失败 ({series_id})。错误: {last_err}\n\n**解决方法**：1) 检查网络/代理，2) 可手动下载并放入程序目录 (fred_{series_id}.csv 或 {series_id}.csv)。")

    # 3) 兜底使用本地旧文件
    for local_file in candidates:
        if os.path.exists(local_file):
            try:
                df = pd.read_csv(local_file, parse_dates=['observation_date'], index_col='observation_date')
                df.columns = [series_id]
                file_date = datetime.date.fromtimestamp(os.path.getmtime(local_file))
                safe_warn(f"⚠️ 无法连接 FRED 数据源 ({series_id})。已使用本地历史数据 (日期: {file_date})。\n\n**解决方法**：请检查网络，或手动更新数据。")
                return df
            except Exception:
                continue

    safe_warn(f"⚠️ 无法连接 FRED 数据源 ({series_id}) 且无本地备份。\n\n**解决方法**：请展开页面顶部的 **‘📂 手动导入宏观数据’** 面板，上传该数据文件。")
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

# --- Alert & Automation Config ---
ALERT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "alert_config.json")
DEFAULT_ALERT_CONFIG = {
    "enabled": False,
    "email_to": "",
    "email_from": "",
    "email_pwd": "",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "frequency": "Manual",  # Manual, Daily, Weekly
    "trigger_time": "09:30",  # Singapore Time (UTC+8)
    "last_run": "",
}

def load_alert_config():
    if not os.path.exists(ALERT_CONFIG_FILE):
        return DEFAULT_ALERT_CONFIG.copy()
    try:
        with open(ALERT_CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("alert_config is not a dict")
    except Exception as e:
        print(f"[AlertConfig] load failed, using defaults: {e}")
        return DEFAULT_ALERT_CONFIG.copy()
    merged = DEFAULT_ALERT_CONFIG.copy()
    merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged

def save_alert_config(config):
    with open(ALERT_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


def safe_warn(msg: str):
    try:
        if threading.current_thread().name == "MainThread":
            st.warning(msg)
        else:
            print(msg)
    except Exception as e:
        print(f"[warn] {msg} (streamlit warn failed: {e})")


# --- Idempotent Daily Lock to Prevent Duplicate Sends ---
LOCK_DIR = os.path.join(os.path.dirname(__file__), ".locks")


def _ensure_lock_dir():
    try:
        os.makedirs(LOCK_DIR, exist_ok=True)
    except Exception as e:
        print(f"[Lock] Failed to ensure lock dir: {e}")


def acquire_daily_lock(date_str: str, ttl_minutes: int = 120) -> bool:
    """Create a dated lock file to avoid duplicate daily sends.
    Returns True if lock acquired; False if an unexpired lock already exists."""
    _ensure_lock_dir()
    lock_path = os.path.join(LOCK_DIR, f"alert_{date_str}.lock")
    now_ts = time.time()
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r") as f:
                ts = float(f.read().strip() or "0")
            if now_ts - ts < ttl_minutes * 60:
                return False
        except Exception:
            # If reading fails, overwrite to be safe
            pass
    try:
        with open(lock_path, "w") as f:
            f.write(str(now_ts))
    except Exception as e:
        print(f"[Lock] Failed to write lock file: {e}")
    return True


def release_daily_lock(date_str: str):
    """Optional: remove the lock file for the given date."""
    lock_path = os.path.join(LOCK_DIR, f"alert_{date_str}.lock")
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception as e:
        print(f"[Lock] Failed to release lock: {e}")

def analyze_market_state_logic():
    """
    Core logic to fetch data and determine current market state.
    Returns: (success, result_dict_or_error_msg)
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365*3)
    
    # Re-use the robust fetcher
    df_hist, err = get_historical_macro_data(start, end)
    
    if df_hist.empty:
        return False, err
    
    # Extract latest state
    last_row = df_hist.iloc[-1]
    state = last_row['State']
    
    # --- Fetch Portfolio Asset Trends (Dual Momentum) ---
    asset_trends = {}
    try:
        check_assets = ['G3B.SI', 'LVHI', 'SRT.SI', 'AJBU.SI', 'IWY', 'MBH.SI', 'GSD.SI']
        trend_start = datetime.date.today() - datetime.timedelta(days=400)
        # Fetch latest data (add timeout to avoid blocking UI)
        data_raw = yf.download(check_assets, start=trend_start, progress=False, auto_adjust=False, timeout=12)
        
        df_assets = pd.DataFrame()
        if not data_raw.empty:
            # Handle MultiIndex or Flat
            if isinstance(data_raw.columns, pd.MultiIndex):
                if 'Adj Close' in data_raw.columns.get_level_values(0):
                    df_assets = data_raw['Adj Close']
                elif 'Close' in data_raw.columns.get_level_values(0):
                    df_assets = data_raw['Close']
                else:
                    df_assets = data_raw
            elif 'Adj Close' in data_raw.columns:
                 df_assets = data_raw['Adj Close']
            elif 'Close' in data_raw.columns:
                 df_assets = data_raw['Close']
            else:
                 df_assets = data_raw

        if not df_assets.empty:
            df_assets = df_assets.ffill()
            ma200 = df_assets.rolling(200).mean()
            
            latest_prices = df_assets.iloc[-1]
            latest_ma = ma200.iloc[-1]
            
            for t in check_assets:
                if t in df_assets.columns:
                    # Bearish if Price < MA200
                    try:
                        p = latest_prices[t]
                        m = latest_ma[t]
                        if pd.notna(p) and pd.notna(m):
                            asset_trends[t] = bool(p < m)
                        else:
                            asset_trends[t] = False
                    except:
                        asset_trends[t] = False
    except Exception as e:
        print(f"Error fetching asset trends: {e}")

    # Logic helpers
    yc_series = df_hist['YieldCurve']
    yc_un_invert = False
    if len(yc_series) > 126:
        recent_min = yc_series.iloc[-126:].min()
        current_yc = yc_series.iloc[-1]
        yc_un_invert = (current_yc < 0.2) and (recent_min < -0.2)

    metrics = {
        'date': last_row.name.strftime('%Y-%m-%d'),
        'state': state,
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
        'value_regime': last_row['Value_Regime'],
        'asset_trends': asset_trends
    }
    
    return True, metrics

def send_strategy_email(metrics, config):
    """发送策略分析邮件，返回 (success, message)。"""
    email_to = str(config.get("email_to", "")).strip()
    email_from = str(config.get("email_from", "")).strip()
    email_pwd = config.get("email_pwd", "")
    smtp_server = str(config.get("smtp_server", "smtp.gmail.com")).strip() or "smtp.gmail.com"
    try:
        smtp_port = int(config.get("smtp_port", 587))
    except Exception:
        smtp_port = 587

    if not email_to or not email_from or not email_pwd:
        return False, "邮箱配置不完整"

    state = metrics['state']
    s_conf = MACRO_STATES.get(state, MACRO_STATES["NEUTRAL"])
    sent_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    report_date = metrics.get('date', sent_at.split(' ')[0])
    
    # Calculate Targets
    targets = get_target_percentages(
        state, 
        gold_bear=metrics['gold_bear'], 
        value_regime=metrics['value_regime'], 
        asset_trends=metrics.get('asset_trends', {}),
        vix=metrics.get('vix'),
        yield_curve=metrics.get('yield_curve')
    )
    
    # Build Target Table
    target_rows = ""
    for t, w in targets.items():
        if w > 0:
            target_rows += f"<tr><td>{ASSET_NAMES.get(t, t)}</td><td style='color:#555'>{t}</td><td><b>{w*100:.1f}%</b></td></tr>"

    # Get Adjustment Reasons
    adjustments = get_adjustment_reasons(
        state, 
        gold_bear=metrics['gold_bear'], 
        value_regime=metrics['value_regime'], 
        asset_trends=metrics.get('asset_trends', {}),
        vix=metrics.get('vix'),
        yield_curve=metrics.get('yield_curve')
    )

    adj_html = ""
    if adjustments:
        adj_list = "".join([f"<li>{r}</li>" for r in adjustments])
        adj_html = f"""
        <div style="background:#fff6f2;border:1px solid #ffd7c2;border-radius:10px;padding:14px 16px;margin:12px 0;">
            <div style="font-weight:600;color:#d93025;margin-bottom:6px;">🔧 动态风控触发</div>
            <ul style="line-height:1.6;margin:0;color:#b23c17;">{adj_list}</ul>
        </div>
        """
    else:
        adj_html = """
        <div style="background:#f6ffed;border:1px solid #b7eb8f;border-radius:10px;padding:14px 16px;margin:12px 0;">
            <div style="font-weight:600;color:#237804;">✅ 当前未触发额外风控</div>
        </div>
        """

    # Quick summary pills
    yc_val = metrics.get('yield_curve', 0)
    summary_points = [
        f"数据截至 {report_date}",
        f"状态: {s_conf['display']}",
        f"VIX {metrics['vix']:.1f} ({'⚠️ 高波动' if metrics['fear'] else '✅ 正常'})",
        f"10Y-2Y {yc_val:.2f}% ({'⚠️ 倒挂/解倒挂' if (yc_val < 0 or metrics.get('yc_un_invert', False)) else '✅ 正常'})",
        f"Sahm {metrics['sahm']:.2f} ({'⚠️ 衰退信号' if metrics['recession'] else '✅ 未触发'})"
    ]
    summary_html = "".join([f"<span style='display:inline-block;background:#f0f4ff;color:#1a73e8;padding:6px 10px;border-radius:20px;margin:4px 4px 0 0;font-size:13px;'>{p}</span>" for p in summary_points])

    html_content = f"""
    <html>
    <body style="font-family: 'Helvetica Neue', Arial, sans-serif; color: #1f2937; background:#f7f8fa;">
        <div style="max-width: 680px; margin: 24px auto; background:#fff; border:1px solid #e5e7eb; border-radius:14px; overflow:hidden; box-shadow:0 10px 30px rgba(0,0,0,0.05);">
            <div style="padding:22px 24px; background: linear-gradient(135deg, {s_conf['border_color']} 0%, #1f1f1f 100%); color:#fff;">
                <div style="font-size:13px; opacity:0.85;">数据截至 {report_date}</div>
                <div style="font-size:12px; opacity:0.75;">发送时间 {sent_at}</div>
                <h2 style="margin:6px 0 4px 0; font-weight:700; letter-spacing:0.3px;">{s_conf['icon']} 宏观策略快报</h2>
                <div style="opacity:0.9; line-height:1.5; font-size:14px;">{s_conf['desc']}</div>
            </div>

            <div style="padding:22px 24px;">
                <div style="margin-bottom:12px;">{summary_html}</div>

                <h3 style="margin:18px 0 10px 0; font-size:16px;">📈 核心指标 (Key Metrics)</h3>
                <table style="width:100%; border-collapse:separate; border-spacing:0 8px; font-size:14px;">
                    <tr style="background:#f9fafb;"><td style="padding:10px 12px; border-radius:10px 0 0 10px;">利率冲击</td><td style="padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#d93025' if metrics['rate_shock'] else '#15803d'};">{metrics['tnx_roc']:.1%} ({'⚠️ 触发' if metrics['rate_shock'] else '✅ 安全'})</td></tr>
                    <tr style="background:#f9fafb;"><td style="padding:10px 12px; border-radius:10px 0 0 10px;">Sahm Rule</td><td style="padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#d93025' if metrics['recession'] else '#15803d'};">{metrics['sahm']:.2f} ({'⚠️ 触发' if metrics['recession'] else '✅ 安全'})</td></tr>
                    <tr style="background:#f9fafb;"><td style="padding:10px 12px; border-radius:10px 0 0 10px;">VIX</td><td style="padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#ea580c' if metrics['fear'] else '#15803d'};">{metrics['vix']:.1f} ({'⚠️ 恐慌' if metrics['fear'] else '✅ 正常'})</td></tr>
                    <tr style="background:#f9fafb;"><td style="padding:10px 12px; border-radius:10px 0 0 10px;">股债相关性</td><td style="padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#d93025' if metrics['corr_broken'] else '#15803d'};">{metrics['corr']:.2f} ({'⚠️ 失效' if metrics['corr_broken'] else '✅ 正常'})</td></tr>
                    <tr style="background:#f9fafb;"><td style="padding:10px 12px; border-radius:10px 0 0 10px;">收益率曲线 (10Y-2Y)</td><td style="padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#d93025' if (yc_val < 0 or metrics.get('yc_un_invert', False)) else '#15803d'};">{yc_val:.2f}%</td></tr>
                </table>

                <h3 style="margin:20px 0 10px 0; font-size:16px;">🎯 战术概览 (Tactical)</h3>
                <ul style="line-height:1.6; margin-top:6px; padding-left:18px; color:#374151;">
                    <li><b>黄金趋势:</b> {'🐻 回避' if metrics['gold_bear'] else '🐂 持有/增配'}</li>
                    <li><b>风格轮动:</b> {'🧱 Value 价值占优' if metrics['value_regime'] else '🚀 Growth 成长占优'}</li>
                </ul>

                {adj_html}

                <h3 style="margin:20px 0 10px 0; font-size:16px;">📊 建议配置 (Target Allocation)</h3>
                <table border="0" cellpadding="10" cellspacing="0" style="width: 100%; border-collapse: collapse; margin-top: 8px; font-size:14px;">
                    <tr style="background-color: #f3f4f6; text-align: left;">
                        <th style="border-bottom: 2px solid #e5e7eb;">资产名称</th>
                        <th style="border-bottom: 2px solid #e5e7eb;">代码</th>
                        <th style="border-bottom: 2px solid #e5e7eb;">目标仓位</th>
                    </tr>
                    {target_rows}
                </table>

                <p style="font-size: 12px; color: #6b7280; margin-top: 26px; text-align: center; border-top: 1px solid #e5e7eb; padding-top: 10px;">
                    此邮件由 Stock Strategy Analyzer 自动生成，供参考，不构成投资建议。
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    msg = MIMEMultipart()
    msg['From'] = email_from
    msg['To'] = email_to
    msg['Subject'] = f"[{state}] 宏观策略状态更新 - {sent_at} (数据截至 {report_date})"
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        timeout = 20
        use_ssl = int(smtp_port) == 465
        if use_ssl:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=timeout)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=timeout)
            try:
                server.starttls()
            except Exception as e:
                server.quit()
                return False, f"TLS 握手失败: {e}"
        server.login(email_from, email_pwd)
        server.send_message(msg)
        server.quit()
        return True, "邮件发送成功"
    except Exception as e:
        return False, f"邮件发送失败: {str(e)}"

# --- Background Scheduler (Lightweight) ---

@st.cache_resource
def start_scheduler_service():
    """
    Starts the background scheduler in a singleton thread.
    Uses @st.cache_resource to ensure only one thread runs per server process,
    preventing duplicate emails when multiple tabs are open.
    """
    def run_scheduler_check():
        """Checks if alert needs to be sent. Runs in background thread."""
        while True:
            try:
                cfg = load_alert_config() or {}
                enabled = bool(cfg.get("enabled", False))
                freq = str(cfg.get("frequency", "Manual") or "Manual")
                if enabled and freq != "Manual":
                    sg_tz = datetime.timezone(datetime.timedelta(hours=8))
                    now = datetime.datetime.now(sg_tz)
                    trigger_hm = str(cfg.get("trigger_time", "09:30") or "09:30")
                    last_run_str = str(cfg.get("last_run", "") or "")
                    
                    should_run = False
                    today_str = now.strftime('%Y-%m-%d')
                    
                    # Simple check: Is it past trigger time AND haven't run today?
                    try:
                        trigger_dt = datetime.datetime.strptime(f"{today_str} {trigger_hm}", "%Y-%m-%d %H:%M").replace(tzinfo=sg_tz)
                    except Exception:
                        # Fallback if time parse fails
                        trigger_dt = datetime.datetime.strptime(f"{today_str} 09:30", "%Y-%m-%d %H:%M").replace(tzinfo=sg_tz)
                    
                    if now >= trigger_dt:
                        # Check frequency
                        if freq == "Daily":
                            if last_run_str != today_str:
                                should_run = True
                        elif freq == "Weekly":
                            # Assume Monday is trigger day (weekday=0)
                            if now.weekday() == 0 and last_run_str != today_str:
                                should_run = True
                    
                    if should_run:
                        # Idempotent guard: prevent duplicate sends across threads/processes
                        if not acquire_daily_lock(today_str, ttl_minutes=120):
                            print(f"[Scheduler] Skip duplicate send for {today_str} (lock exists).")
                        else:
                            print(f"[Scheduler] Triggering auto-analysis at {now}")
                            success, res = analyze_market_state_logic()
                            if success:
                                email_ok, msg = send_strategy_email(res, cfg)
                                if email_ok:
                                    print(f"[Scheduler] Email sent: {msg}")
                                    cfg = load_alert_config()
                                    cfg["last_run"] = today_str
                                    save_alert_config(cfg)
                                else:
                                    print(f"[Scheduler] Email failed: {msg}")
                            else:
                                print(f"[Scheduler] Analysis failed: {res}")
            except Exception as e:
                print(f"[Scheduler] Loop error: {e}")
            
            time.sleep(60) # Check every minute

    # Create and start the thread
    t = threading.Thread(target=run_scheduler_check, daemon=True)
    t.start()
    print("[System] Global background scheduler service started.")
    return t

# Start scheduler (Singleton)
start_scheduler_service()

# --- Shared Logic for Backtest & State Machine ---

def get_target_percentages(s, gold_bear=False, value_regime=False, asset_trends=None, vix=None, yield_curve=None):
    """
    Returns target asset allocation based on macro state.
    Shared by State Machine Diagnosis and Backtest.
    asset_trends: dict {ticker: is_bearish_bool} - optional override for asset specific trends
    """
    if asset_trends is None:
        asset_trends = {}

    targets = {}
    
    # --- 1. Base Allocation (基于宏观状态的原始配置) ---
    if s == "INFLATION_SHOCK":
        # Rate Spike: Kill Duration, Cash is King, Trend Following (WTMF)
        # Optimized: Increased WTMF to capture trend, Removed G3B (Equity exposure)
        targets = {
            'IWY': 0.00, 'WTMF': 0.50, 'LVHI': 0.15,
            'G3B.SI': 0.00, 'MBH.SI': 0.00, 'GSD.SI': 0.25,
            'SRT.SI': 0.00, 'AJBU.SI': 0.10
        }
    elif s == "DEFLATION_RECESSION":
        # Recession: Long Bonds, Gold
        targets = {
            'IWY': 0.05, 'WTMF': 0.20, 'LVHI': 0.05,
            'G3B.SI': 0.00, 'MBH.SI': 0.40, 'GSD.SI': 0.25,
            'SRT.SI': 0.00, 'AJBU.SI': 0.05
        }
    elif s == "EXTREME_ACCUMULATION":
        # Buy Dip (左侧交易，不进行动量过滤)
        targets = {
            'IWY': 0.75, 'WTMF': 0.00, 'LVHI': 0.00,
            'G3B.SI': 0.10, 'MBH.SI': 0.05, 'GSD.SI': 0.05,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }
    elif s == "CAUTIOUS_TREND":
        # Bear Trend: Defensive, but use WTMF for downside protection
        # Optimized: Increased WTMF, Reduced localized equity (G3B)
        growth_w = 0.10
        value_w = 0.20
        if value_regime:
            growth_w = 0.05
            value_w = 0.25
        
        targets = {
            'IWY': growth_w, 'WTMF': 0.25, 'LVHI': value_w,
            'G3B.SI': 0.10, 'MBH.SI': 0.15, 'GSD.SI': 0.10,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }
    elif s == "CAUTIOUS_VOL":
        # High Vol: Hedge
        # Optimized: Reduced Equity, Increased Crisis Alpha
        targets = {
            'IWY': 0.30, 'WTMF': 0.30, 'LVHI': 0.10,
            'G3B.SI': 0.10, 'MBH.SI': 0.10, 'GSD.SI': 0.05,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }
    else: # NEUTRAL
        # Style Rotation
        # Optimized: Slightly higher growth base, less drag from diversifiers
        growth_w = 0.55
        value_w = 0.10
        if value_regime:
            growth_w = 0.45
            value_w = 0.20
            
        targets = {
            'IWY': growth_w, 'WTMF': 0.10, 'LVHI': value_w,
            'G3B.SI': 0.05, 'MBH.SI': 0.10, 'GSD.SI': 0.05,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }

    # === 🚀 新增：动态风控层 (Dynamic Risk Control) ===
    
    # 1. 牛市增强与预警 (Aggressive Growth in Calm Waters)
    # 只有在"常态"下才进行激进微调
    if s == "NEUTRAL":
        if vix is not None:
            # 极度平稳期 (VIX < 13)：大胆加仓，减少保险
            if vix < 13.0:
                # 从 WTMF (保险) 挪到 IWY (成长)
                wtmf_amt = targets.get('WTMF', 0)
                targets['WTMF'] = 0.0
                targets['IWY'] += wtmf_amt
                
                # 若还不够激进，可适当减少低效债券 (MBH)
                mbh_amt = targets.get('MBH.SI', 0) * 0.5
                targets['MBH.SI'] -= mbh_amt
                targets['IWY'] += mbh_amt
                
            # 早期动荡预警 (VIX > 20)：虽然没到恐慌(32)，但先跑为敬
            elif vix > 20.0:
                cut_amt = 0.20 # 减仓 20%
                # 从 IWY (成长) 挪到 WTMF (保险)
                available_growth = targets.get('IWY', 0)
                move_amt = min(available_growth, cut_amt)
                targets['IWY'] -= move_amt
                targets['WTMF'] += move_amt

    # 2. 债券陷阱规避 (Avoid Bond Trap)
    # 如果处于衰退或震荡期，且收益率曲线深度倒挂，长债可能不仅不避险，反而下跌
    if s in ["DEFLATION_RECESSION", "CAUTIOUS_TREND"]:
        if yield_curve is not None and yield_curve < -0.30:
             # 削减长债/新元债，转为抗跌的 WTMF 或现金
             if targets.get('MBH.SI', 0) > 0:
                 move_amt = targets['MBH.SI'] * 0.7 # 大幅削减
                 targets['MBH.SI'] -= move_amt
                 targets['WTMF'] += move_amt

    # --- 2. Global Dynamic Trend Filter (全局动态趋势过滤) ---
    # 逻辑：除了"极度贪婪/抄底"模式外，任何资产如果处于熊市趋势（价格 < MA200），都应该被削减。
    # 目的：避免在宏观误判或流动性危机时死守下跌资产。
    
    if s != "EXTREME_ACCUMULATION":
        # 需要检查趋势的资产列表 (IWY 单独处理，这里检查配角)
        assets_to_check = ['G3B.SI', 'LVHI', 'MBH.SI', 'GSD.SI', 'SRT.SI', 'AJBU.SI']
        
        for asset in assets_to_check:
            # 如果该资产在当前配置中有仓位，且处于熊市趋势
            if targets.get(asset, 0) > 0 and asset_trends.get(asset, False):
                weight_to_move = targets[asset]
                targets[asset] = 0.0 # 清仓该弱势资产
                
                # --- 资金去向逻辑 ---
                if s == "NEUTRAL":
                    # 牛市逻辑：如果配角弱，资金去主角 (IWY)。
                    # 但前提是主角 (IWY) 自己必须强。
                    if not asset_trends.get('IWY', False): # IWY is NOT bearish
                         targets['IWY'] += weight_to_move
                    else:
                         # 如果连 IWY 都弱，那就去避险 (WTMF)
                         targets['WTMF'] += weight_to_move
                else:
                    # 熊市/震荡逻辑 (Cautious/Recession/Shock)：
                    # 风险厌恶。如果防御资产都跌（例如债熊），资金必须去现金/危机Alpha (WTMF)。
                    targets['WTMF'] += weight_to_move

    # --- 3. IWY (Core) Safety Valve (核心资产熔断) ---
    # 如果处于非抄底模式，且核心资产 IWY 破位，必须大幅降低风险
    if s != "EXTREME_ACCUMULATION" and targets.get('IWY', 0) > 0:
        if asset_trends.get('IWY', False): # IWY is Bearish
            # 如果 VIX 高，说明是恐慌性下跌，砍得更狠
            severity = 0.5
            if vix is not None and vix > 25:
                severity = 0.8
                
            cut_amount = targets['IWY'] * severity
            targets['IWY'] -= cut_amount
            targets['WTMF'] += cut_amount

    # --- 4. Gold Trend Filter (Legacy) ---
    # 保留原有的黄金独立判断，作为最后一道防线
    if gold_bear and targets.get('GSD.SI', 0) > 0:
        cut_amount = targets['GSD.SI'] # 直接清仓
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

    # 2. Drawdown & Duration (Updated)
    rolling_max = series.cummax()
    drawdown = (series / rolling_max - 1) * 100
    max_dd = drawdown.min()

    # Calculate Max Drawdown Duration (Days Underwater)
    # Logic: Find peaks, fill dates forward, subtract current date from last peak date
    is_peak = series == rolling_max
    peak_dates = pd.Series(series.index, index=series.index).where(is_peak).ffill()
    dd_days = series.index - peak_dates
    max_dd_days = dd_days.max().days if not dd_days.empty else 0
    
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
            sortino = (excess_ret.mean() * 252) / downside_std 
        else:
            sortino = 0.0
    else:
        sortino = 0.0 
        
    # Calmar
    if abs(max_dd) > 0:
        calmar = cagr / abs(max_dd)
    else:
        calmar = 0.0
        
    # 6. Trade/Win Analysis
    winning_days = daily_ret[daily_ret > 0].count()
    losing_days = daily_ret[daily_ret < 0].count()
    total_trading_days = winning_days + losing_days
    
    win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0.0
    
    avg_win = daily_ret[daily_ret > 0].mean() if winning_days > 0 else 0
    avg_loss = abs(daily_ret[daily_ret < 0].mean()) if losing_days > 0 else 0
    
    pl_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    # 7. Annual Returns (New)
    annual_rets = {}
    yearly_vals = series.groupby(series.index.year).last()
    previous_val = series.iloc[0]
    
    for year in yearly_vals.index:
        current_val = yearly_vals.loc[year]
        # Return for the year = (End Value / Start Value) - 1
        ret = (current_val / previous_val) - 1
        annual_rets[f"{year} (%)"] = ret * 100
        previous_val = current_val

    # Construct Final Result
    results = {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Max Drawdown (%)": max_dd,
        "Max DD Days": max_dd_days, # 新增字段
        "Volatility (%)": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Win Rate (Daily %)": win_rate,
        "Profit/Loss Ratio": pl_ratio
    }
    
    # Merge Annual Returns into results
    results.update(annual_rets)

    return results

def run_dynamic_backtest(df_states, start_date, end_date, initial_capital=10000.0, ma_window=200, use_proxies=False):
    """
    Simulates the strategy over historical states.
    df_states: DataFrame with 'State', 'Gold_Bear', 'Value_Regime' columns, indexed by Date.
    """
    # 1. Define Asset Universe
    # If using proxies (for long-term history > 20 years), we map ETFs to Indices
    if use_proxies:
        # Proxy Mapping:
        # IWY -> ^GSPC (S&P 500) as generic equity
        # WTMF -> Cash (Simulated) or similar? Hard to proxy. We'll use Gold as partial proxy or just Cash?
        # Let's map WTMF to Gold for Crisis Alpha in history? Or just Cash.
        # Let's use ^GSPC for Equity, TLT for Bonds (needs check), GLD for Gold.
        # Note: Yahoo data for GLD starts 2004. TLT 2002. 
        # For meaningful 1990s backtest, we need Indices.
        # But yfinance index data for 'Total Return' is hard. ^GSPC is price only (no div).
        assets = ['^GSPC', '^NDX', 'TLT', 'GLD', 'VUSTX', 'GC=F'] # Minimal set
    else:
        assets = ['IWY', 'WTMF', 'LVHI', 'G3B.SI', 'MBH.SI', 'GSD.SI', 'SRT.SI', 'AJBU.SI', 'TLT', 'SPY']
    
    # 2. Fetch Price Data
    fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=365)
    
    try:
        price_data = yf.download(assets, start=fetch_start, end=end_date, progress=False, auto_adjust=False)['Adj Close']
    except:
        # Fallback if Adj Close issue
        try:
             price_data = yf.download(assets, start=fetch_start, end=end_date, progress=False, auto_adjust=False)['Close']
        except Exception as e:
            return None, None, f"Data fetch failed: {e}"

    if price_data.empty:
         return None, None, "No price data fetched."

    # Fill missing
    price_data = price_data.ffill().bfill()
    
    # Calculate Asset Trends for Backtest (Dual Momentum)
    # Use dynamic MA window
    ma_all = price_data.rolling(ma_window).mean()
    trend_bear_all = price_data < ma_all
    
    # Filter to requested range
    mask = (price_data.index >= pd.to_datetime(start_date)) & (price_data.index <= pd.to_datetime(end_date))
    price_data = price_data[mask]
    trend_bear_all = trend_bear_all[mask]
    
    if price_data.empty:
         return None, None, "Price data empty after filtering."

    # Align states with prices
    common_idx = price_data.index.intersection(df_states.index)
    price_data = price_data.loc[common_idx]
    trend_bear_all = trend_bear_all.loc[common_idx]
    df_states = df_states.loc[common_idx]
    
    if len(price_data) < 10:
        return None, None, "Insufficient data points for backtest."

    # 3. Strategy Simulation (Daily Rebalancing Approximation)
    portfolio_values = []
    current_val = initial_capital
    
    # Track allocation history
    history_records = []
    
    # Turnover tracking
    prev_targets = {}
    prev_rets = None
    
    # We iterate daily. To speed up, we could vectorise, but logic is complex.
    # Logic: Daily return = Sum(Weight_i * Return_i)
    # This assumes we rebalance to target weights DAILY.
    
    returns_df = price_data.pct_change().fillna(0)
    
    # Proxy Mapper Function
    def map_target_to_asset(target_ticker, current_date=None):
        if not use_proxies:
            return target_ticker
        
        # Simple Proxy Logic
        if target_ticker in ['IWY', 'G3B.SI', 'LVHI', 'SRT.SI', 'AJBU.SI', 'SPY']:
            if '^NDX' in price_data.columns and target_ticker == 'IWY': return '^NDX'
            return '^GSPC' if '^GSPC' in price_data.columns else target_ticker
        if target_ticker in ['TLT', 'MBH.SI']:
            if 'VUSTX' in price_data.columns: return 'VUSTX'
            return 'TLT' 
        if target_ticker in ['GSD.SI']:
            if current_date and current_date < pd.Timestamp('2004-11-18') and 'GC=F' in price_data.columns:
                return 'GC=F'
            return 'GLD'
        if target_ticker in ['WTMF']:
            return 'CASH' # Simulate Cash for managed futures in proxy mode
        return target_ticker

    for date, row in df_states.iterrows():
        # Get Targets for today (based on today's state)
        # Note: In reality, we trade tomorrow based on today's close state?
        # Or trade today at close? Assuming trade at close.
        s = row['State']
        gb = row['Gold_Bear']
        vr = row['Value_Regime']
        
        # Get trends for this date
        daily_trends = {}
        # We need to map the trends check to proxies too? 
        # The strategy logic checks specific tickers. We should probably simulate trends on the Proxy.
        # But get_target_percentages expects original tickers. 
        # We will let get_target_percentages run as is, but we feed it "Proxy Trends" masquerading as Asset Trends.
        
        if use_proxies:
            # Map proxy trends back to original tickers for the logic to consume
            # E.g. If ^GSPC is Bearish, then IWY is Bearish
            proxy_trend_bear = False
            if '^GSPC' in trend_bear_all.columns:
                proxy_trend_bear = trend_bear_all.loc[date]['^GSPC']
            
            # Apply to all equity
            for t in ['IWY', 'G3B.SI', 'LVHI', 'SRT.SI', 'AJBU.SI']:
                daily_trends[t] = proxy_trend_bear
                
            # Gold
            gold_proxy = 'GLD'
            if date < pd.Timestamp('2004-11-18') and 'GC=F' in trend_bear_all.columns:
                 gold_proxy = 'GC=F'
            
            if gold_proxy in trend_bear_all.columns:
                daily_trends['GSD.SI'] = trend_bear_all.loc[date][gold_proxy]

            # Bond Proxy Trend (MBH.SI -> VUSTX/TLT)
            bond_proxy = 'TLT'
            if 'VUSTX' in trend_bear_all.columns:
                bond_proxy = 'VUSTX'
            
            if bond_proxy in trend_bear_all.columns:
                daily_trends['MBH.SI'] = trend_bear_all.loc[date][bond_proxy]
        else:
            if date in trend_bear_all.index:
                daily_trends = trend_bear_all.loc[date].to_dict()
        
        vix_val = row.get('VIX')
        yc_val = row.get('YieldCurve')
        targets = get_target_percentages(s, gold_bear=gb, value_regime=vr, asset_trends=daily_trends, vix=vix_val, yield_curve=yc_val)
        
        # --- Map Targets to Available Assets (Proxy Translation) ---
        final_weights = {}
        for t, w in targets.items():
            mapped_asset = map_target_to_asset(t, date)
            if mapped_asset == 'CASH':
                # Cash means 0 return, we just don't invest it
                pass 
            elif mapped_asset in price_data.columns:
                final_weights[mapped_asset] = final_weights.get(mapped_asset, 0.0) + w
            else:
                # If mapped asset missing (e.g. TLT before 2002), hold Cash
                pass
        
        # --- Calculate Turnover (Trading Volume) ---
        # Compare current 'targets' with 'prev_targets' adjusted for drift
        daily_turnover = 0.0
        
        if not prev_targets:
            # First day: turnover is the sum of all positions (building portfolio)
            daily_turnover = sum(final_weights.values())
        else:
            # Calculate "Drifted Weights" from previous day
            # Formula: W_drifted_i = W_prev_i * (1 + r_i) / (1 + R_port)
            # R_port = Sum(W_prev_i * r_i) + W_cash * 0
            
            # 1. Calculate value of each component after drift
            drifted_values = {}
            total_drifted_val = 0.0
            
            # Assets
            for t, w in prev_targets.items():
                r = 0.0
                if prev_rets is not None and t in prev_rets:
                    r = prev_rets[t]
                
                val = w * (1 + r)
                drifted_values[t] = val
                total_drifted_val += val
                
            # Cash (implied)
            prev_cash_w = max(0.0, 1.0 - sum(prev_targets.values()))
            drifted_cash_val = prev_cash_w * 1.0 # Cash return 0
            total_drifted_val += drifted_cash_val
            
            # 2. Normalize to get Drifted Weights
            if total_drifted_val > 0:
                drifted_weights = {t: v / total_drifted_val for t, v in drifted_values.items()}
                drifted_cash_w = drifted_cash_val / total_drifted_val
            else:
                drifted_weights = prev_targets
                drifted_cash_w = prev_cash_w

            # 3. Compare with New Targets
            diff_sum = 0.0
            all_assets = set(final_weights.keys()) | set(drifted_weights.keys())
            
            for t in all_assets:
                w_tgt = final_weights.get(t, 0.0)
                w_drift = drifted_weights.get(t, 0.0)
                diff_sum += abs(w_tgt - w_drift)
            
            # Don't forget Cash difference
            curr_cash_w = max(0.0, 1.0 - sum(final_weights.values()))
            diff_sum += abs(curr_cash_w - drifted_cash_w)
            
            daily_turnover = diff_sum / 2.0 # One-sided turnover
            
        # Record history
        rec = targets.copy() # Record original logic targets for debug
        rec['Date'] = date
        rec['State'] = s
        rec['Turnover'] = daily_turnover
        history_records.append(rec)
        
        # Calculate Portfolio Return for this day
        daily_ret = 0.0
        current_rets = pd.Series(dtype=float)
        
        if date in returns_df.index:
            current_rets = returns_df.loc[date]
            for t, w in final_weights.items():
                if t in current_rets:
                    daily_ret += w * current_rets[t]
        
        current_val = current_val * (1 + daily_ret)
        portfolio_values.append(current_val)
        
        # Prepare for next iteration
        prev_targets = final_weights
        prev_rets = current_rets

        
    s_strategy = pd.Series(portfolio_values, index=df_states.index, name="Strategy")
    
    # Create History DataFrame
    df_history = pd.DataFrame(history_records)
    if not df_history.empty:
        df_history = df_history.set_index('Date')
    
    # 4. Benchmarks
    # SPY
    s_spy = pd.Series(dtype=float)
    bench_ticker = 'SPY'
    if use_proxies and '^GSPC' in price_data.columns:
        bench_ticker = '^GSPC'
        
    if bench_ticker in price_data.columns:
        spy_prices = price_data[bench_ticker]
        s_spy = (spy_prices / spy_prices.iloc[0]) * initial_capital
        s_spy.name = f"{bench_ticker} (Benchmark)"

    # IWY
    s_iwy = pd.Series(dtype=float)
    growth_ticker = 'IWY'
    if use_proxies and '^NDX' in price_data.columns:
        growth_ticker = '^NDX'
    elif use_proxies and '^GSPC' in price_data.columns:
        growth_ticker = '^GSPC'
        
    if growth_ticker in price_data.columns:
        iwy_prices = price_data[growth_ticker]
        s_iwy = (iwy_prices / iwy_prices.iloc[0]) * initial_capital
        s_iwy.name = f"{growth_ticker} (Growth)"

    # 60/40
    s_6040 = pd.Series(dtype=float)
    bond_ticker = 'TLT'
    if use_proxies and 'VUSTX' in price_data.columns:
        bond_ticker = 'VUSTX'
    # If TLT missing in proxy mode, maybe we can't do 60/40 easily without a bond proxy
    
    if bench_ticker in price_data.columns and bond_ticker in price_data.columns:
        spy = price_data[bench_ticker] / price_data[bench_ticker].iloc[0]
        tlt = price_data[bond_ticker] / price_data[bond_ticker].iloc[0]
        s_6040 = (0.6 * spy + 0.4 * tlt) * initial_capital
        s_6040.name = "60/40 (Balanced)"
        
    # Neutral Config (Buy & Hold / Fixed Weight)
    # Note: Neutral config logic relies on original ETFs. 
    # In proxy mode, we need to map default targets too.
    default_targets = get_target_percentages("NEUTRAL", False, False)
    neutral_vals = []
    curr_n = initial_capital
    
    for date in df_states.index:
        daily_ret = 0.0
        if date in returns_df.index:
            rets = returns_df.loc[date]
            for t, w in default_targets.items():
                mapped_t = map_target_to_asset(t, date)
                if mapped_t in rets and mapped_t != 'CASH':
                    daily_ret += w * rets[mapped_t]
        curr_n = curr_n * (1 + daily_ret)
        neutral_vals.append(curr_n)
        
    s_neutral = pd.Series(neutral_vals, index=df_states.index, name="Neutral Config")
    
    return pd.DataFrame({
        "Dynamic Strategy": s_strategy,
        "SPY (Benchmark)": s_spy,
        "Growth (Benchmark)": s_iwy,
        "60/40 (Balanced)": s_6040,
        "Neutral (Fixed)": s_neutral
    }), df_history, None




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

def determine_macro_state(row, params=None):
    """
    Determines macro state based on a row of indicators.
    Expected row keys: Sahm, RateShock, Corr, VIX, Trend_Bear
    """
    if params is None:
        params = {
            'sahm_threshold': 0.50,
            'rate_shock_threshold': 0.20,
            'corr_threshold': 0.30,
            'vix_panic': 32,
            'vix_recession': 35,
            'vix_elevated': 20
        }
        
    is_rec = row['Sahm'] >= params['sahm_threshold']
    is_shock = row['RateShock'] > params['rate_shock_threshold']
    is_c_broken = row['Corr'] > params['corr_threshold']
    is_f = row['VIX'] > params['vix_panic']
    is_down = row['Trend_Bear']
    is_vol_elevated = row['VIX'] > params['vix_elevated']
    
    if is_shock or (is_rec and is_c_broken):
        return "INFLATION_SHOCK"
    elif is_rec or (is_down and row['VIX'] > params['vix_recession']):
        return "DEFLATION_RECESSION"
    elif is_f and not is_shock and not is_rec:
        return "EXTREME_ACCUMULATION"
    elif is_down:
        # Optimized: If Trend is Down, prioritize Trend signal (Defensive) over Volatility signal.
        return "CAUTIOUS_TREND"
    elif is_vol_elevated:
        return "CAUTIOUS_VOL"
    else:
        return "NEUTRAL"

@st.cache_data
def get_historical_macro_data(start_date, end_date, ma_window=200, params=None, use_proxies=False):
    """
    Fetches and calculates macro states for a given date range.
    Includes buffer to ensure valid data at start_date.
    use_proxies: If True, prioritizes Indices (^GSPC, VUSTX) over ETFs for longer history.
    """
    if params is None:
        params = {
            'sahm_threshold': 0.50,
            'rate_shock_threshold': 0.20,
            'corr_threshold': 0.30,
            'vix_panic': 32,
            'vix_recession': 35,
            'vix_elevated': 20
        }

    buffer_days = 365 * 2 # Increase buffer for Sahm Rule (12m min)
    fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
    fetch_end = pd.to_datetime(end_date)

    # 1. Fetch Market Data
    # Added ^GSPC (S&P 500) for longer history check if IWY is missing
    # Added VUSTX (Vanguard Long-Term Treasury) for longer bond history (since 1986)
    tickers = ['IWY', 'TLT', '^TNX', '^VIX', 'GLD', 'IWD', '^GSPC', 'VUSTX']
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
        unrate_daily = unrate.reindex(data.index).ffill()
        
        if not yc.empty:
            yc.columns = ['T10Y2Y']
            yc = yc[yc.index >= fetch_start]
            yc_daily = yc.reindex(data.index).ffill()
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
        sahm_series = sahm_monthly.reindex(data.index).ffill()
        
        # Rate Shock
        tnx_col = '^TNX' if '^TNX' in data.columns else data.columns[0]
        tnx_roc = (data[tnx_col] - data[tnx_col].shift(21)) / data[tnx_col].shift(21)
        
        # Correlation & Series Selection
        # If use_proxies is True, we FORCE the use of Indices (^GSPC, VUSTX) to ensure we get data back to 1990s.
        # Otherwise, we prefer the actual ETFs (IWY, TLT).
        
        prefer_etfs = ('IWY' in data.columns) and ('TLT' in data.columns) and (not use_proxies)

        if prefer_etfs:
            corr = data['IWY'].rolling(60).corr(data['TLT'])
            iwy_series = data['IWY']
        elif '^GSPC' in data.columns:
            # Fallback or Proxy mode: Use S&P 500
            iwy_series = data['^GSPC']
            
            # For correlation, prefer VUSTX if using proxies or if TLT is missing/short
            bond_series = None
            if use_proxies and 'VUSTX' in data.columns:
                bond_series = data['VUSTX']
            elif 'TLT' in data.columns:
                # Check if TLT has enough history? 
                # For simplicity, if not forcing proxies, try TLT first, fallback to VUSTX
                tlt_series = data['TLT']
                if 'VUSTX' in data.columns:
                    bond_series = data['VUSTX']
                else:
                    bond_series = data['TLT']
            elif 'VUSTX' in data.columns:
                bond_series = data['VUSTX']
            
            if bond_series is not None:
                corr = data['^GSPC'].rolling(60).corr(bond_series)
            else:
                corr = pd.Series(0, index=data.index)
        else:
            corr = pd.Series(0, index=data.index)
            iwy_series = data.iloc[:, 0] if not data.empty else pd.Series(dtype=float)
        
        # Trend
        iwy_ma = iwy_series.rolling(ma_window).mean()
        trend_bear = iwy_series < iwy_ma
        
        # Gold Trend
        gold_trend_bear = pd.Series(False, index=data.index)
        if 'GLD' in data.columns:
            gld_ma = data['GLD'].rolling(ma_window).mean()
            gold_trend_bear = data['GLD'] < gld_ma
            
        # Style Trend
        style_value_regime = pd.Series(False, index=data.index)
        # Only use IWY/IWD if NOT using proxies (since IWD history is short)
        if not use_proxies and 'IWY' in data.columns and 'IWD' in data.columns:
            pair_ratio = data['IWY'] / data['IWD']
            pair_ma = pair_ratio.rolling(ma_window).mean()
            style_value_regime = pair_ratio < pair_ma 
        else:
            # Fallback for style if ETFs missing
            style_value_regime = pd.Series(False, index=data.index)

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
        # Pass params to the state determinator
        df_hist['State'] = df_hist.apply(lambda row: determine_macro_state(row, params), axis=1)
        
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
        st.info(
            """💡 **判定流程（越上面优先级越高）**
1) 拉取数据：股指/长债 (^TNX, IWY/TLT)、VIX、失业率 UNRATE、收益率曲线 T10Y2Y。
2) 计算指标：Sahm≥0.50 判衰退；21日利率涨幅 >20% 判利率冲击；股债相关性>0.30 判相关性失效；VIX>32 判恐慌；价格<MA200 判趋势破位。
3) 状态判定优先级：Inflation Shock → Deflation/Recession → Extreme Accumulation → Cautious Trend → Cautious Vol → Neutral。
4) 输出对应的资产配置建议（见下表）。"""
        )
        
        guide_cards = [
            {
                "key": "INFLATION_SHOCK",
                "trigger": "利率21日涨幅>20% 或 股债相关性>0.30 且波动上升",
                "action": "现金为王，削减股票/长久期债，提升危机Alpha (WTMF)。",
            },
            {
                "key": "DEFLATION_RECESSION",
                "trigger": "Sahm≥0.50 或 趋势破位且VIX>35（衰退/流动性风险）",
                "action": "全面防御：长债+黄金为主，股票权重大幅下调。",
            },
            {
                "key": "EXTREME_ACCUMULATION",
                "trigger": "VIX>32 恐慌但未触发利率/衰退条件",
                "action": "左侧抄底：加大成长股权重，保留一定防御。",
            },
            {
                "key": "CAUTIOUS_TREND",
                "trigger": "价格跌破MA200（阴跌趋势），但未触发恐慌/衰退",
                "action": "防御配置：提高红利/价值与现金，降低成长敞口。",
            },
            {
                "key": "CAUTIOUS_VOL",
                "trigger": "趋势尚可但VIX>20（高波震荡）",
                "action": "保留核心成长，但用 WTMF/防御资产对冲波动。",
            },
            {
                "key": "NEUTRAL",
                "trigger": "未触发以上任一警报",
                "action": "标准增长配置，跟随趋势持有。",
            },
        ]
        
        # 3 columns per row
        cols = st.columns(3)
        for idx, card in enumerate(guide_cards):
            s = MACRO_STATES[card["key"]]
            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div style="padding: 12px; border-radius: 8px; background-color: {s['bg_color']}; border-left: 4px solid {s['border_color']}; margin-bottom: 12px; min-height: 190px;">
                        <div style="font-weight: 700; font-size: 15px; margin-bottom: 6px;">{s['icon']} {s['display']}</div>
                        <div style="font-size: 13px; color: #3c4043; line-height: 1.5; margin-bottom: 6px;">{s['desc']}</div>
                        <div style="font-size: 12px; color: #111827; line-height: 1.6;">
                            <b>触发条件：</b>{card['trigger']}<br/>
                            <b>应对策略：</b>{card['action']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        st.markdown(
            """
            **单个资产的动态处理规则（结合趋势/波动自动调整）：**
            - **IWY（成长）**：核心进攻仓。Neutral/Accumulation 重仓；Cautious/Inflation/Recession 阶梯削减；若价格跌破 MA200，则标记为熊市并降低权重。
            - **WTMF（危机 Alpha）**：对冲资产。高波震荡 (Cautious Vol) 和 通胀冲击 时显著加仓；Neutral 维持小权重；若回到平稳则逐步降回基准。
            - **LVHI（红利/价值）**：防御权益。Cautious Trend/Vol 提升；Inflation Shock 下仍保留小比例；Neutral 适中配置。
            - **G3B.SI（本地蓝筹）**：与成长同向但更防御。趋势破位或高波时下调；通胀/衰退场景大幅削减。
            - **MBH.SI / TLT（长债）**：衰退防御主力。Deflation/Recession 大幅加仓；Inflation Shock 降至极低；Normal/Vol 维持中等。
            - **GSD.SI（黄金）**：系统性风险与通胀对冲。Inflation Shock/Deflation 提升；平稳期降低至防御底仓。
            - **SRT.SI / AJBU.SI（REITs/数据中心）**：在 Cautious/Inflation/Recession 阶段减少，Neutral 维持小权重，Accumulation 不额外加码。
            - **OTHERS**：默认低配或清理；仅在 Neutral/Accumulation 且基本面良好时酌情持有。
            """
        )


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

    # Show Active Adjustments
    adjustments = get_adjustment_reasons(
        metrics['state'], 
        gold_bear=metrics['gold_bear'], 
        value_regime=metrics['value_regime'], 
        asset_trends=metrics.get('asset_trends', {}),
        vix=metrics.get('vix'),
        yield_curve=metrics.get('yield_curve')
    )
    
    if adjustments:
        with st.expander("🔧 动态风控触发 (Active Strategy Adjustments)", expanded=True):
            for adj in adjustments:
                st.markdown(f"- {adj}")

def render_rebalancing_table(state, current_holdings, total_value, is_gold_bear, is_value_regime, asset_trends=None, vix=None, yield_curve=None):
    """Renders the rebalancing table."""
    if asset_trends is None: asset_trends = {}
    targets = get_target_percentages(state, gold_bear=is_gold_bear, value_regime=is_value_regime, asset_trends=asset_trends, vix=vix, yield_curve=yield_curve)
    
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
            "目标仓位": tgt_pct * 100,
            "当前仓位": curr_pct * 100,
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
                "目标仓位": st.column_config.NumberColumn(format="%.1f%%"),
                "当前仓位": st.column_config.NumberColumn(format="%.1f%%"),
                "当前市值": st.column_config.NumberColumn(format="$%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )

def render_historical_backtest_section():
    """Renders the independent historical backtest section."""
    st.markdown("---")
    st.markdown("### 🕰️ 历史状态回溯与策略仿真")
    
    # --- Advanced Settings (Sensitivity & Proxies) ---
    with st.expander("⚙️ 高级回测设置 (参数敏感性与样本外测试)", expanded=False):
        # Initialize Session State for Widgets to avoid warnings
        if "bt_use_proxies" not in st.session_state: st.session_state["bt_use_proxies"] = False
        if "bt_ma_window" not in st.session_state: st.session_state["bt_ma_window"] = 200
        if "bt_p_sahm" not in st.session_state: st.session_state["bt_p_sahm"] = 0.50
        if "bt_p_vix_panic" not in st.session_state: st.session_state["bt_p_vix_panic"] = 32
        if "bt_p_vix_rec" not in st.session_state: st.session_state["bt_p_vix_rec"] = 35

        # Reset Button
        if st.button("🔄 恢复默认设置"):
            st.session_state["bt_use_proxies"] = False
            st.session_state["bt_ma_window"] = 200
            st.session_state["bt_p_sahm"] = 0.50
            st.session_state["bt_p_vix_panic"] = 32
            st.session_state["bt_p_vix_rec"] = 35
            st.rerun()

        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            st.markdown("**1. 样本外测试 (Out-of-Sample)**")
            use_proxies = st.checkbox("启用代理资产 (Use Proxies)", help="使用 S&P500, VUSTX(1986+), GC=F 等替代 ETF 以支持更长历史回测 (1990+)。", key="bt_use_proxies")
            ma_window = st.number_input("动量窗口 (MA Window)", step=10, help="默认 200 日均线。尝试 150 或 250 测试敏感性。", key="bt_ma_window")
            
        with c_adv2:
            st.markdown("**2. 阈值敏感性 (Sensitivity)**")
            p_sahm = st.number_input("Sahm Rule", step=0.01, format="%.2f", key="bt_p_sahm")
            p_vix_panic = st.number_input("VIX Panic", step=1, key="bt_p_vix_panic")
            p_vix_rec = st.number_input("VIX Recession", step=1, key="bt_p_vix_rec")
    
    # Construct params dict
    custom_params = {
        'sahm_threshold': p_sahm,
        'rate_shock_threshold': 0.20,
        'corr_threshold': 0.30,
        'vix_panic': int(p_vix_panic),
        'vix_recession': int(p_vix_rec),
        'vix_elevated': 20
    }

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        # Default start date logic
        def_start = datetime.date(1990, 1, 1) if use_proxies else datetime.date.today()-datetime.timedelta(days=365*5)
        if def_start > datetime.date.today(): def_start = datetime.date.today() - datetime.timedelta(days=365)
        
        dates = st.date_input("回测时间", [def_start, datetime.date.today()])
    with c2:
        cap = st.number_input("初始资金", value=10000)
    with c3:
        st.write(""); st.write("")
        run = st.button("🚀 运行回测", type="primary")
        
    if run and isinstance(dates, tuple) and len(dates)==2:
        with st.spinner("回测中..."):
            df_states, err = get_historical_macro_data(dates[0], dates[1], ma_window=int(ma_window), params=custom_params, use_proxies=use_proxies)
            if not df_states.empty:
                res, df_history, err = run_dynamic_backtest(df_states, dates[0], dates[1], cap, ma_window=int(ma_window), use_proxies=use_proxies)
                if res is not None:
                    # Metrics & Charts (Simplified for brevity as logic exists in run_dynamic_backtest return)
                    st.success("回测完成")
                    
                    # 1. Curve
                    fig = go.Figure()
                    for c in res.columns:
                        fig.add_trace(go.Scatter(x=res.index, y=res[c], name=c))
                    
                    # Add Background Colors for States
                    shapes_curve = []
                    annotations_curve = []
                    
                    if df_history is not None and not df_history.empty:
                        # Create a copy to avoid affecting downstream logic
                        df_viz = df_history.copy()
                        df_viz['state_grp'] = (df_viz['State'] != df_viz['State'].shift()).cumsum()
                        
                        # Group by state segments
                        state_segments = df_viz.groupby(['state_grp', 'State'])['State'].agg(
                            ['first', lambda x: x.index[0], lambda x: x.index[-1]]
                        ).reset_index()
                        state_segments.columns = ['grp', 'State', 'State_Name', 'Start', 'End']
                        
                        for _, seg in state_segments.iterrows():
                            s_conf = MACRO_STATES.get(seg['State'], MACRO_STATES["NEUTRAL"])
                            color = s_conf['bg_color']
                            
                            # Add shape
                            shapes_curve.append(dict(
                                type="rect",
                                xref="x", yref="paper",
                                x0=seg['Start'], x1=seg['End'],
                                y0=0, y1=1,
                                fillcolor=color,
                                opacity=0.3,
                                layer="below",
                                line_width=0,
                            ))
                            
                            # Add icon label if segment is long enough
                            if (seg['End'] - seg['Start']).days > 15:
                                annotations_curve.append(dict(
                                    x=seg['Start'] + (seg['End'] - seg['Start'])/2,
                                    y=1.05,
                                    xref="x", yref="paper",
                                    text=s_conf['icon'],
                                    showarrow=False,
                                    font=dict(size=14)
                                ))

                    fig.update_layout(
                        title="净值曲线 (Net Value Curve)", 
                        template="plotly_white",
                        shapes=shapes_curve,
                        annotations=annotations_curve,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- NEW: State & Allocation Visualization ---
                    st.markdown("### 🏗️ 仓位历史与状态分布 (Allocation & Regimes)")
                    
                    if df_history is not None and not df_history.empty:
                        # Stacked Area Chart
                        fig_alloc = go.Figure()
                        
                        # Identify asset columns (float types)
                        asset_cols = df_history.select_dtypes(include=[np.number]).columns
                        
                        for asset in asset_cols:
                            fig_alloc.add_trace(go.Scatter(
                                x=df_history.index, 
                                y=df_history[asset],
                                mode='lines',
                                name=ASSET_NAMES.get(asset, asset),
                                stackgroup='one',
                                groupnorm='percent', # Normalize to 0-100
                                hoverinfo='x+y+name'
                            ))
                        
                        # Add Background Colors for States
                        # 1. Simplify states to segments
                        df_history['state_grp'] = (df_history['State'] != df_history['State'].shift()).cumsum()
                        state_segments = df_history.groupby(['state_grp', 'State'])['State'].agg(['first', lambda x: x.index[0], lambda x: x.index[-1]]).reset_index()
                        state_segments.columns = ['grp', 'State', 'State_Name', 'Start', 'End']
                        
                        shapes = []
                        annotations = []
                        
                        for _, seg in state_segments.iterrows():
                            s_conf = MACRO_STATES.get(seg['State'], MACRO_STATES["NEUTRAL"])
                            color = s_conf['bg_color']
                            
                            # Add shape
                            shapes.append(dict(
                                type="rect",
                                xref="x", yref="paper",
                                x0=seg['Start'], x1=seg['End'],
                                y0=0, y1=1,
                                fillcolor=color,
                                opacity=0.3,
                                layer="below",
                                line_width=0,
                            ))
                            
                            # Add label if segment is long enough (e.g. > 10 days)
                            if (seg['End'] - seg['Start']).days > 15:
                                annotations.append(dict(
                                    x=seg['Start'] + (seg['End'] - seg['Start'])/2,
                                    y=1.05,
                                    xref="x", yref="paper",
                                    text=s_conf['icon'],
                                    showarrow=False,
                                    font=dict(size=14)
                                ))

                        fig_alloc.update_layout(
                            title="历史持仓分布与市场状态 (Portfolio Allocation & Market Regimes)",
                            template="plotly_white",
                            yaxis=dict(title="Allocation %", range=[0, 100]),
                            shapes=shapes,
                            annotations=annotations,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_alloc, use_container_width=True)
                        
                        # Legend for states (Optional text below)
                        st.caption("背景颜色代表市场状态: " + " | ".join([f"{v['icon']} {k}" for k, v in MACRO_STATES.items()]))

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
                    df_metrics = pd.DataFrame(metrics_list)
                    
                    # Basic Configs
                    col_config = {
                        "Strategy": st.column_config.TextColumn("策略名称", width="medium"),
                        "Total Return (%)": st.column_config.NumberColumn("总收益率", format="%.2f%%"),
                        "CAGR (%)": st.column_config.NumberColumn("年化收益 (CAGR)", format="%.2f%%"),
                        "Max Drawdown (%)": st.column_config.NumberColumn("最大回撤", format="%.2f%%"),
                        "Max DD Days": st.column_config.NumberColumn("回撤修复 (天)", format="%d"),
                        "Volatility (%)": st.column_config.NumberColumn("波动率", format="%.2f%%"),
                        "Sharpe Ratio": st.column_config.NumberColumn("夏普比率", format="%.2f"),
                        "Sortino Ratio": st.column_config.NumberColumn("索提诺", format="%.2f"),
                        "Calmar Ratio": st.column_config.NumberColumn("卡玛", format="%.2f"),
                        "Win Rate (Daily %)": st.column_config.NumberColumn("胜率", format="%.1f%%"),
                        "Profit/Loss Ratio": st.column_config.NumberColumn("盈亏比", format="%.2f"),
                    }
                    
                    # Add dynamic configs for Years
                    for c in df_metrics.columns:
                        if " (%)" in c and c not in col_config:
                            col_config[c] = st.column_config.NumberColumn(c, format="%.2f%%")
                            
                    st.dataframe(
                        df_metrics, 
                        use_container_width=True,
                        column_config=col_config,
                        hide_index=True
                    )
                    
                    # --- 4. Trading Costs & Frequency Analysis ---
                    if df_history is not None and 'Turnover' in df_history.columns:
                        st.markdown("---")
                        st.markdown("#### 💸 交易成本与频率 (Trading Costs & Frequency)")
                        
                        # Calculate Stats
                        total_days = len(df_history)
                        years = total_days / 252.0 if total_days > 0 else 0
                        
                        # Total One-sided Turnover (sum of daily portions)
                        # We skip the first day (initial allocation) for "churn" metrics, 
                        # but keeping it shows total volume. Usually exclude day 1 for "Strategy Turnover".
                        
                        if total_days > 1:
                            turnover_series = df_history['Turnover'].iloc[1:] # Exclude initial setup
                            total_turnover = turnover_series.sum()
                            avg_daily_turnover = turnover_series.mean()
                            annual_turnover = avg_daily_turnover * 252
                            
                            # Est Cost (bps)
                            cost_bps = 10 # 0.10% per side
                            total_cost_est = total_turnover * (cost_bps / 10000)
                            annual_cost_est = annual_turnover * (cost_bps / 10000)
                            
                            # Avg Holding Period (Days)
                            # Formula: 1 / Daily Turnover (approx)
                            avg_hold_days = 1 / avg_daily_turnover if avg_daily_turnover > 0 else 0
                        else:
                            annual_turnover = 0
                            annual_cost_est = 0
                            avg_hold_days = 0

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("年化换手率 (Annual Turnover)", f"{annual_turnover:.1%}", help="平均每年调整仓位的总比例 (单边)")
                        with c2:
                            st.metric("平均持仓周期 (Avg Hold)", f"{avg_hold_days:.1f} 天", help="平均每笔资金持有的天数")
                        with c3:
                            st.metric("预估年化成本 (Est. Cost)", f"{annual_cost_est:.2%}", help=f"基于单边 {cost_bps}bps ({cost_bps/100}%) 手续费估算的年化拖累")
                        with c4:
                            # Trading Frequency (Days with > 1% turnover)
                            active_days = df_history[df_history['Turnover'] > 0.01].count()['Turnover']
                            freq_pct = active_days / total_days if total_days > 0 else 0
                            st.metric("活跃交易频率", f"{freq_pct:.1%}", help="日换手率超过 1% 的天数比例")

                        # Chart: Rolling Turnover
                        # st.bar_chart(df_history['Turnover']) # Simple bar
                        
                        fig_to = go.Figure()
                        fig_to.add_trace(go.Bar(x=df_history.index, y=df_history['Turnover'], name='Daily Turnover'))
                        fig_to.update_layout(
                            title="每日换手率 (Daily Turnover)", 
                            yaxis=dict(title="Turnover %", tickformat=".1%"),
                            template="plotly_white",
                            height=300
                        )
                        st.plotly_chart(fig_to, use_container_width=True)
                    
            else:
                msg = err if err else "该时间段内无有效数据 (可能是因为数据源不足，请尝试勾选 'Use Proxies' 或缩短时间范围)"
                st.error(f"无法获取数据: {msg}")

def render_alert_config_ui():
    """Renders the configuration UI for auto-alerts."""
    with st.expander("🔔 自动提醒设置 (Auto-Alert Configuration)", expanded=False):
        st.caption("设置定时自动分析市场状态，并将策略建议发送到您的邮箱。需保持后台脚本运行或网页开启。")
        
        config = load_alert_config()
        
        # Current snapshot for status cards
        email_to_saved = str(config.get("email_to", "")).strip()
        email_from_saved = str(config.get("email_from", "")).strip()
        email_ready = bool(email_to_saved and email_from_saved and config.get("email_pwd"))
        enabled_saved = bool(config.get("enabled", False))
        freq_saved = str(config.get("frequency", "Manual"))
        time_str_saved = str(config.get("trigger_time", "09:30"))
        try:
            trigger_time_saved = datetime.datetime.strptime(time_str_saved, "%H:%M").time()
        except Exception:
            trigger_time_saved = datetime.time(9, 30)

        def _next_run_preview(freq: str, trig_time: datetime.time):
            if freq not in ["Daily", "Weekly"]:
                return "手动触发"
            sg_tz = datetime.timezone(datetime.timedelta(hours=8))
            now = datetime.datetime.now(sg_tz)
            today_trigger = datetime.datetime.combine(now.date(), trig_time, tzinfo=sg_tz)
            if freq == "Daily":
                nxt = today_trigger if now < today_trigger else today_trigger + datetime.timedelta(days=1)
            else:  # Weekly (Monday)
                days_ahead = (0 - now.weekday()) % 7
                if days_ahead == 0 and now >= today_trigger:
                    days_ahead = 7
                nxt = today_trigger + datetime.timedelta(days=days_ahead)
            return nxt.strftime("%Y-%m-%d %H:%M")

        next_run_preview = _next_run_preview(freq_saved, trigger_time_saved)
        status_color = "🟢 已启用" if (enabled_saved and email_ready) else ("🟡 待补全" if enabled_saved else "⚪ 未启用")

        c_status, c_next, c_last = st.columns(3)
        with c_status:
            st.metric("当前状态", status_color, help="需要同时开启开关并填写邮件信息。")
        with c_next:
            st.metric("下次触发", next_run_preview)
        with c_last:
            st.metric("上次运行", config.get("last_run", "Never"))

        st.divider()

        with st.form("alert_config_form"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📧 邮件配置 (Email)")
                email_to = st.text_input("接收邮箱 (To)", value=email_to_saved, placeholder="you@example.com")
                email_from = st.text_input("发送邮箱 (From)", value=email_from_saved, placeholder="sender@gmail.com")
                email_pwd = st.text_input("授权码/密码 (App Password)", value=str(config.get("email_pwd", "")), type="password", help="Gmail/Outlook 请使用应用专用密码，避免使用真实登录密码。")
                
                c1a, c1b = st.columns(2)
                with c1a:
                    smtp_server = st.text_input("SMTP 服务器", value=str(config.get("smtp_server", "smtp.gmail.com")))
                with c1b:
                    val_port = config.get("smtp_port", 587)
                    try: val_port = int(val_port)
                    except: val_port = 587
                    smtp_port = st.number_input("SMTP 端口", value=val_port)
            
            with c2:
                st.subheader("⏰ 触发规则 (Trigger)")
                enabled = st.checkbox("启用自动提醒 (Enable)", value=enabled_saved)
                
                curr_freq = freq_saved if freq_saved in ["Manual", "Daily", "Weekly"] else "Manual"
                frequency = st.selectbox("触发频率", ["Manual", "Daily", "Weekly"], index=["Manual", "Daily", "Weekly"].index(curr_freq))
                
                time_str = time_str_saved
                try:
                    time_obj = datetime.datetime.strptime(time_str, "%H:%M").time()
                except Exception:
                    time_obj = datetime.time(9, 30)
                trigger_time = st.time_input("触发时间 (Local Time)", value=time_obj)
                
                st.info("仅在应用运行时触发；默认新加坡时间 09:30，请根据本地/服务器时区自行调整。")

            if st.form_submit_button("💾 保存配置"):
                email_ready_form = bool(email_to.strip() and email_from.strip() and email_pwd)
                if enabled and not email_ready_form:
                    st.error("启用自动提醒需要填写收件人、发件人和授权码。")
                else:
                    new_config = {
                        "enabled": enabled,
                        "email_to": email_to.strip(),
                        "email_from": email_from.strip(),
                        "email_pwd": email_pwd,
                        "smtp_server": smtp_server.strip() or "smtp.gmail.com",
                        "smtp_port": smtp_port,
                        "frequency": frequency,
                        "trigger_time": trigger_time.strftime("%H:%M"),
                        "last_run": config.get("last_run", "")
                    }
                    save_alert_config(new_config)
                    st.success("配置已保存!")
                    st.rerun()

        # Test Button
        if st.button("📨 立即发送测试邮件 (Send Test Email)", type="secondary"):
            with st.spinner("正在分析并发送..."):
                cfg = load_alert_config()
                if cfg.get("enabled") and (not cfg.get("email_to") or not cfg.get("email_from") or not cfg.get("email_pwd")):
                    st.error("请先补全邮箱配置后再测试发送。")
                else:
                    success, res = analyze_market_state_logic()
                    if success:
                        ok, msg = send_strategy_email(res, cfg)
                        if ok:
                            st.success(f"✅ 发送成功! 请检查邮箱: {cfg['email_to']}")
                        else:
                            st.error(f"❌ 发送失败: {msg}")
                    else:
                        st.error(f"❌ 分析失败: {res}")

def render_state_machine_check():
    st.header("🛡️ 宏观状态机与资产配置 (Macro State & Allocation)")
    st.caption("全自动资产配置生成器 (Auto-Allocator)")
    
    # 1. Alert Config
    render_alert_config_ui()
    
    # 2. Imports & Inputs
    render_manual_data_import()
    render_reference_guide()
    render_portfolio_import()
    current_holdings, total_value = render_holdings_input()
    
    # 3. Manual Analysis
    if st.button("🚀 开始诊断 (Run Analysis)", type="primary", use_container_width=True):
        with st.status("正在进行宏观扫描...", expanded=True) as status:
            st.write("📡 获取数据并计算指标...")
            
            # Use the new shared logic
            success, metrics = analyze_market_state_logic()
            
            if not success:
                status.update(label="诊断失败", state="error")
                st.error(metrics) # metrics is error msg here
            else:
                st.write("✅ 数据获取与计算完成")
                status.update(label="诊断完成", state="complete", expanded=False)
                
                # Render Results
                render_status_card(metrics['state'])
                render_factor_dashboard(metrics)
                
                st.markdown("---")
                render_rebalancing_table(
                    metrics['state'], 
                    current_holdings, 
                    total_value, 
                    metrics['gold_bear'], 
                    metrics['value_regime'], 
                    metrics.get('asset_trends', {}),
                    vix=metrics.get('vix'),
                    yield_curve=metrics.get('yield_curve')
                )
                
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
                
                data = data.ffill().bfill()
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
                    # Logic: Find peaks, fill dates forward, subtract current date from last peak date
                    # Uses Calendar Days
                    is_peak = val_series == rolling_max
                    peak_dates = pd.Series(val_series.index, index=val_series.index).where(is_peak).ffill()
                    dd_days = val_series.index - peak_dates
                    max_duration_days = dd_days.max().days if not dd_days.empty else 0
                    
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
