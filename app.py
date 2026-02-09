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
st.set_page_config(layout="wide", page_title="Stock Strategy Analyzer v1.5")

# --- Helper Functions for Indicators ---

# === 原有阈值 (v2.0 IWY核心+波动控制优化) ===
VIX_BOOST_LO = 13.0                 # 降低: 14→13，低波动时更激进配置IWY
VIX_CUT_HI = 22.0                   # 降低: 23→22，更早启动波动控制
VIX_PANIC = 30.0                    # 提高: 28→30，真正恐慌才大幅减仓
YIELD_CURVE_CUTOFF = -0.35          # 保持

# === 优化参数 (v2.0 IWY核心+波动控制) ===
# 1. 波动率目标机制 - 目标波动率控制，IWY主导但整体波动受控
TARGET_VOL = 0.15                   # 目标年化波动率15%
VOL_LOOKBACK = 20                   # 保持
VOL_SCALAR_MAX = 1.6                # 降低: 1.8→1.6，防止过度集中
VOL_SCALAR_MIN = 0.5                # 提高: 0.4→0.5，保持基础仓位

# 2. 动态止损机制 - v1.9 加速恢复优化
DRAWDOWN_STOP_LOSS = -0.12          # 保持: -12%触发止损
DRAWDOWN_REDUCE_RATIO = 0.35        # 降低: 0.4→0.35，止损减仓更温和
DRAWDOWN_RECOVERY_THRESHOLD = -0.04 # 收紧: -0.06→-0.04，更早开始恢复

# 3. VIX响应平滑化参数 - 提高响应阈值
VIX_SMOOTH_START = 18.0             # 提高: 15→18，减少低VIX区间的拖累
VIX_SMOOTH_END = 32.0               # 提高: 30→32
VIX_MAX_REDUCTION = 0.35            # 降低: 0.40→0.35，减少最大减仓

# 4. 信号确认延迟 - 对抄底信号更激进
SIGNAL_CONFIRM_DAYS = 2             # 保持（将对EXTREME_ACCUMULATION特殊处理）

# 5. 再平衡容忍带 - 略微放宽减少交易成本
REBALANCE_THRESHOLD = 0.06          # 提高: 0.05→0.06

# 6. 状态转换平滑 - v1.9 更快过渡
STATE_TRANSITION_DAYS = 1           # 降低: 2→1，更快响应状态恢复

# === 新增优化参数（低过拟合风险）v2.0 ===
# 7. 动量强度分层配置 - 更宽的牛市定义，IWY优先
MOMENTUM_STRONG_THRESHOLD = 1.02    # 降低: 1.03→1.02，更宽的牛市定义
MOMENTUM_WEAK_THRESHOLD = 0.92      # 降低: 0.93→0.92，更窄的熊市定义
MOMENTUM_NEUTRAL_REDUCTION = 0.05   # 降低: 0.08→0.05，中性区减仓更少

# 8. Sahm Rule 预警增强 - 收窄预警区间
SAHM_EARLY_WARNING_LO = 0.35        # 提高: 0.30→0.35，减少误报
SAHM_EARLY_WARNING_HI = 0.50        # 保持
SAHM_REDUCTION_RATE = 0.40          # 降低: 0.50→0.40

# 9. 收益率曲线解倒挂延保护 - v1.9 缩短保护期
YC_UNINVERT_PROTECTION_MONTHS = 6   # 降低: 9→6，更快恢复
YC_UNINVERT_REDUCTION = 0.10        # 降低: 0.15→0.10，减少减仓

# 10. VIX均值回归加仓 - v1.9 更激进回归加仓
VIX_MEAN_REVERSION_PEAK = 22.0      # 降低: 23→22，更早触发加仓
VIX_MEAN_REVERSION_RATIO = 0.70     # 降低: 0.75→0.70，更早确认VIX回落
VIX_MEAN_REVERSION_BOOST = 0.15     # 提高: 0.12→0.15，加仓更多

# 11. 相关性动态再配置 - 放宽触发条件
CORR_MID_THRESHOLD = 0.18           # 提高: 0.15→0.18
CORR_HIGH_THRESHOLD = 0.35          # 提高: 0.30→0.35
CORR_MAX_REALLOC = 0.12             # 降低: 0.15→0.12

# === 新增优化参数 v1.7 收益最大化 ===
# 12. 现金缓冲机制 - 完全禁用（用户要求）
CASH_BUFFER_BASE = 0.0              # 禁用: 0.02→0
CASH_BUFFER_VIX_THRESHOLD = 999.0   # 禁用: 永远不触发
CASH_BUFFER_MAX = 0.0               # 禁用: 0.10→0
CASH_BUFFER_VIX_SCALE = 0.0         # 禁用: 0.012→0

# 13. CAUTIOUS_VOL VIX分层 - v1.7 更激进的权益配置，动态IWY/WTMF轮换
CAUTIOUS_VOL_VIX_TIERS = {
    # VIX区间: (lo, hi, IWY权重, WTMF权重) - 更激进轮换
    'tier1': (20, 25, 0.40, 0.20),   # 20-25: IWY↑40%, WTMF↓20% (低波时更多成长)
    'tier2': (25, 30, 0.30, 0.30),   # 25-30: 均衡
    'tier3': (30, 40, 0.20, 0.40),   # 30-40: VIX高时WTMF对冲
    'tier4': (40, 999, 0.10, 0.50),  # 40+:   极端波动，WTMF主导
}

# 14. 双均线趋势确认 - 降低减仓幅度
TREND_MA_SHORT = 50                 # 保持
TREND_MA_LONG = 200                 # 保持
WEAK_BEAR_REDUCTION = 0.20          # 降低: 0.30→0.20
STRONG_BEAR_REDUCTION = 0.55        # 降低: 0.70→0.55

# 15. 止损分阶段恢复 - v1.9 加速恢复
STOP_LOSS_RECOVERY_STAGES = [
    # (回撤阈值, 恢复仓位比例) - 更激进恢复
    (-0.12, 0.60),   # -12%: 60%仓位 (原55%)
    (-0.08, 0.80),   # -8%:  80%仓位 (原75%)
    (-0.04, 0.95),   # -4%:  95%仓位 (原-5%, 90%)
    (-0.015, 1.00),  # -1.5%: 完全恢复 (原-2%)
]

# 16. 跨资产动量 - 降低减仓力度
MARKET_BREADTH_LOW = 0.25           # 降低: 0.30→0.25
MARKET_BREADTH_MID = 0.45           # 降低: 0.50→0.45
BREADTH_LOW_REDUCTION = 0.10        # 降低: 0.15→0.10
BREADTH_MID_REDUCTION = 0.03        # 降低: 0.05→0.03

# === 新增 v1.6 收益增强参数 ===
# 17. 趋势顺势加仓（新增）
TREND_BOOST_THRESHOLD = 1.08        # Price > MA * 1.08 时启动顺势加仓
TREND_BOOST_AMOUNT = 0.08           # 从WTMF/MBH转移8%到IWY
TREND_BOOST_VIX_MAX = 18.0          # 只在VIX<18时启用

# 18. 抄底加速确认（新增）
EXTREME_CONFIRM_DAYS = 1            # EXTREME_ACCUMULATION只需1天确认（更快抄底）

# 19. 牛市WTMF最小化（v1.7 完全取消）
NEUTRAL_MIN_WTMF = 0.0              # NEUTRAL状态WTMF=0%（原5%），全部转IWY
NEUTRAL_WTMF_BOOST_TO_IWY = True    # 启用WTMF→IWY转换

# === v1.7 新增: 成长/红利/WTMF动态轮换参数 ===
# 20. VIX驱动的成长↔红利轮换
VIX_GROWTH_TO_VALUE_START = 22.0    # VIX>22时开始从IWY转向LVHI
VIX_GROWTH_TO_VALUE_FULL = 35.0     # VIX≥35时达到最大转换比例
GROWTH_TO_VALUE_MAX_SHIFT = 0.20    # 最大从IWY转移20%到LVHI

# 21. 趋势强度驱动的成长↔WTMF轮换
TREND_STRONG_BULL = 1.10            # Price > MA*1.10: 强牛市
TREND_MILD_BULL = 1.03              # Price > MA*1.03: 温和牛市
TREND_MILD_BEAR = 0.97              # Price < MA*0.97: 温和熊市
TREND_STRONG_BEAR_LINE = 0.90       # Price < MA*0.90: 强熊市
# 转换幅度
BULL_IWY_BOOST = 0.10               # 强牛时从WTMF转10%到IWY
BEAR_WTMF_BOOST = 0.15              # 强熊时从IWY转15%到WTMF

# 22. 红利相对强弱
VALUE_OUTPERFORM_THRESHOLD = 0.03   # LVHI相对IWY跑赢3%时增配红利
VALUE_UNDERPERFORM_THRESHOLD = -0.05 # LVHI相对IWY跑输5%时减配红利
VALUE_ROTATION_AMOUNT = 0.08        # 轮换幅度8%

# === v2.0 新增: 波动率控制因子 ===
# 23. IWY波动率自适应 - 根据IWY自身波动调整仓位
IWY_VOL_HIGH = 0.25                 # IWY日波动率年化>25%时减仓
IWY_VOL_LOW = 0.15                  # IWY日波动率年化<15%时加仓
IWY_VOL_ADJUST_MAX = 0.12           # 最大调整12%

# 24. LVHI波动缓冲系数 - 红利作为波动缓冲器
LVHI_VOL_BUFFER_RATIO = 0.30        # IWY减仓时，30%转LVHI（红利抗跌）
LVHI_MIN_ALLOCATION = 0.10          # LVHI最低配置10%（波动锚定）

# 25. WTMF危机Alpha配置
WTMF_BASE_ALLOCATION = 0.05         # 基础WTMF配置5%（危机保险）
WTMF_VOL_TRIGGER = 20.0             # VIX>20时开始增配WTMF
WTMF_MAX_ALLOCATION = 0.25          # WTMF最大配置25%

# 26. 动态再平衡触发 - 基于波动率而非固定阈值
REBAL_VOL_MULTIPLIER = 2.0          # 再平衡阈值 = 基础阈值 × (当前波动/目标波动)
REBAL_MIN_THRESHOLD = 0.04          # 最小再平衡阈值4%
REBAL_MAX_THRESHOLD = 0.10          # 最大再平衡阈值10%

# 27. 风险平价系数 - IWY为主但控制边际风险贡献
IWY_MAX_RISK_CONTRIBUTION = 0.75    # IWY最大风险贡献75%
RISK_PARITY_ENABLED = True          # 启用风险平价调整

# === v2.1 新增: 状态机收益增强优化 ===
# 28. EXTREME_ACCUMULATION 进入阈值优化 - 更早抄底
VIX_EXTREME_THRESHOLD = 30.0        # 降低: 32→30，更早触发抄底（历史VIX>30即为恐慌）
VIX_EXTREME_EXIT = 25.0             # 抄底退出阈值（VIX回落到25以下退出抄底模式）

# 29. 动量加速因子 - 牛市中更激进
MOMENTUM_ACCELERATION_ENABLED = True
MOMENTUM_ACCEL_THRESHOLD = 1.05     # 价格>MA*1.05时触发加速
MOMENTUM_ACCEL_BOOST = 0.08         # 加速时额外加仓8%到IWY

# 30. 状态滞后惩罚 - 减少频繁切换
STATE_HYSTERESIS_DAYS = 3           # 状态切换需持续3天（防止日内噪音）
STATE_HYSTERESIS_EXCEPTION = ["EXTREME_ACCUMULATION"]  # 抄底不受限制

# 31. VIX均值回归加速 - 恐慌后更快加仓
VIX_REVERSION_ACCEL_THRESHOLD = 0.65  # VIX从峰值回落35%以上时加速
VIX_REVERSION_ACCEL_BOOST = 0.08      # 额外加仓8%

# 32. 趋势确认强化 - 减少假突破
TREND_CONFIRM_DAYS = 2              # 趋势信号需持续2天
TREND_CONFIRM_EXCEPTION_VIX = 25.0  # VIX>25时不需确认（危机中快速反应）

# 33. NEUTRAL状态IWY上限 - 牛市中放开限制
NEUTRAL_IWY_CAP_NORMAL = 0.75       # 正常情况IWY上限75%
NEUTRAL_IWY_CAP_BULL = 0.82         # 强牛市(VIX<15且动量>5%)IWY上限82%

# === v3.0 新增优化参数 ===
# 34. 时序动量 (Time-Series Momentum)
TSMOM_LOOKBACK_FAST = 21            # 1个月动量
TSMOM_LOOKBACK_SLOW = 126           # 6个月动量
TSMOM_COMBO_WEIGHT = 0.6            # 快慢动量组合权重 (60%快 + 40%慢)
TSMOM_THRESHOLD = 0.02              # 动量信号阈值 (>2%才算有效)
TSMOM_BOOST_AMOUNT = 0.10           # 强动量时额外加仓10%
TSMOM_ENABLED = True                # 启用时序动量

# 35. 尾部风险控制
TAIL_RISK_VIX_SPIKE = 8.0           # VIX单日涨幅超过8点触发
TAIL_RISK_REDUCTION = 0.25          # 尾部风险触发时减仓25%
TAIL_RISK_RECOVERY_DAYS = 5         # 恢复期5天
TAIL_RISK_ENABLED = True            # 启用尾部风险控制

# 36. 智能再平衡
REBAL_COST_BPS = 10                 # 交易成本10bps
REBAL_EXPECTED_ALPHA = 0.005        # 预期日超额收益0.5%
REBAL_MIN_EDGE = 2.0                # 预期收益/成本比>2才触发
SMART_REBAL_ENABLED = True          # 启用智能再平衡
SMART_REBAL_HARD_THRESHOLD = 0.15   # 单边换手>15%时强制再平衡

# 37. 概率化状态转换
STATE_CONFIDENCE_THRESHOLD = 0.70   # 状态置信度阈值
STATE_SMOOTHING_ALPHA = 0.3         # 指数平滑系数
STATE_LOOKBACK_DAYS = 5             # 状态概率计算回看天数
PROB_STATE_ENABLED = True           # 启用概率化状态

# 38. 风险平价混合
RISK_PARITY_TARGET_VOL = 0.12       # 组合目标波动率12%
RISK_PARITY_BLEND_RATIO = 0.20      # 风险平价混合比例20%
RISK_PARITY_MIN_WEIGHT = 0.05       # 风险平价最低权重5%
RISK_PARITY_BLEND_ENABLED = True    # 启用风险平价混合

# 39. 自适应VIX阈值
ADAPTIVE_VIX_ENABLED = True         # 启用自适应VIX阈值
VIX_PANIC_PERCENTILE = 0.90         # 使用历史90%分位数作为恐慌阈值
VIX_CALM_PERCENTILE = 0.20          # 使用历史20%分位数作为平静阈值
VIX_ADAPTIVE_LOOKBACK = 252         # 自适应VIX回看天数（1年）

# === 资产类别映射 (用于风险暴露分析和邮件生成) ===
ASSET_CATEGORIES = {
    'IWY': {'category': '权益', 'sub': '美股成长', 'risk_level': 'high'},
    'LVHI': {'category': '权益', 'sub': '美股红利', 'risk_level': 'medium'},
    'G3B.SI': {'category': '权益', 'sub': '新加坡蓝筹', 'risk_level': 'medium'},
    'SRT.SI': {'category': '另类', 'sub': 'REITs', 'risk_level': 'medium'},
    'AJBU.SI': {'category': '另类', 'sub': 'REITs', 'risk_level': 'medium'},
    'MBH.SI': {'category': '固收', 'sub': '新元债券', 'risk_level': 'low'},
    'GSD.SI': {'category': '商品', 'sub': '黄金', 'risk_level': 'medium'},
    'WTMF': {'category': '对冲', 'sub': '危机Alpha', 'risk_level': 'low'},
    'OTHERS': {'category': '其他', 'sub': '其他资产', 'risk_level': 'unknown'},
}

# === 资产名称映射 (用于邮件和UI显示) ===
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

SCHEDULER_LOCK = os.path.join(os.path.dirname(__file__), "data", "scheduler.lock")
STATE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "data", "state_history.json")
PORTFOLIO_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "data", "portfolio_history.json")
os.makedirs(os.path.dirname(SCHEDULER_LOCK), exist_ok=True)
os.makedirs(os.path.dirname(STATE_HISTORY_FILE), exist_ok=True)

def normalize_yf_prices(df_raw):
    """
    从 yfinance 返回的 DataFrame 中提取调整后收盘价
    兼容 yfinance 各版本的不同返回格式
    """
    if df_raw is None or len(df_raw) == 0:
        return pd.DataFrame()
    
    # yfinance 新版本可能返回 MultiIndex (Price, Ticker) 或 (Ticker, Price)
    if isinstance(df_raw.columns, pd.MultiIndex):
        level_values_0 = df_raw.columns.get_level_values(0).unique().tolist()
        level_values_1 = df_raw.columns.get_level_values(1).unique().tolist() if df_raw.columns.nlevels > 1 else []
        
        # 检查 'Adj Close' 或 'Close' 在哪个 level
        price_types = ['Adj Close', 'Close', 'Price']
        
        for price_type in price_types:
            if price_type in level_values_0:
                return df_raw[price_type]
            elif price_type in level_values_1:
                # 交换 level 后提取
                df_swapped = df_raw.swaplevel(axis=1)
                return df_swapped[price_type]
        
        # 如果都没找到，尝试直接返回第一个 level
        try:
            return df_raw[level_values_0[0]]
        except Exception:
            pass
        
        return df_raw
    
    # 单层列名
    if 'Adj Close' in df_raw.columns:
        return df_raw['Adj Close']
    if 'Close' in df_raw.columns:
        return df_raw['Close']
    if 'Price' in df_raw.columns:
        return df_raw['Price']
    return df_raw


def ensure_fred_cached(series_ids=("UNRATE", "T10Y2Y")):
    """Eager-download FRED CSVs into local cache before analysis/backtest/email."""
    for sid in series_ids:
        try:
            _ = fetch_fred_data(sid)
        except Exception as e:
            log_event("WARN", "fred_prefetch_failed", {"series": sid, "err": str(e)})

def evaluate_risk_triggers(s, gold_bear=False, value_regime=False, asset_trends=None, vix=None, yield_curve=None, sahm=None, corr=None, yc_recently_inverted=False, dual_ma_signals=None, breadth_score=None):
    if asset_trends is None:
        asset_trends = {}
    reasons = []

    # 1. Style Regime
    if s in ["NEUTRAL", "CAUTIOUS_TREND"] and value_regime:
        reasons.append("🧱 风格轮动: 价值占优 (Value Regime) -> 增加红利，减少成长")

    # 2. Dynamic Risk Control
    if s == "NEUTRAL" and vix is not None:
        if vix < VIX_BOOST_LO:
            reasons.append(f"🚀 极度平稳 (VIX < {VIX_BOOST_LO}): 激进模式 -> 清空WTMF/减债，加仓成长")
        elif vix > VIX_CUT_HI:
            reasons.append(f"🌬️ 早期预警 (VIX > {VIX_CUT_HI}): 避险模式 -> 减仓成长 20%，增加 WTMF")
    
    # v1.5: CAUTIOUS_VOL VIX分层
    if s == "CAUTIOUS_VOL" and vix is not None:
        if vix >= 30:
            reasons.append(f"🔴 高波动分层 (VIX={vix:.1f}≥30): IWY降至10%，WTMF升至40%")
        elif vix >= 25:
            reasons.append(f"🟠 中波动分层 (VIX={vix:.1f}≥25): IWY降至20%，WTMF升至35%")

    if s in ["DEFLATION_RECESSION", "CAUTIOUS_TREND"] and yield_curve is not None:
        if yield_curve < YIELD_CURVE_CUTOFF:
            reasons.append(f"⚠️ 深度倒挂 (Yield Curve < {YIELD_CURVE_CUTOFF}%): 债券陷阱 -> 大幅削减 MBH，转入 WTMF")

    # 3. Trend Filters (v1.5: 支持双均线)
    if s != "EXTREME_ACCUMULATION":
        if dual_ma_signals:
            strong_bear = [t for t, sig in dual_ma_signals.items() if sig == "STRONG_BEAR"]
            weak_bear = [t for t, sig in dual_ma_signals.items() if sig == "WEAK_BEAR"]
            if strong_bear:
                reasons.append(f"📉 强熊市信号: {', '.join(strong_bear)} (价格<MA200且MA50<MA200) -> 减仓70%")
            if weak_bear:
                reasons.append(f"📊 弱熊市信号: {', '.join(weak_bear)} (可能回调) -> 减仓30%")
        else:
            assets_to_check = ['G3B.SI', 'LVHI', 'MBH.SI', 'GSD.SI', 'SRT.SI', 'AJBU.SI']
            bear_assets = [t for t in assets_to_check if asset_trends.get(t, False)]
            if bear_assets:
                reasons.append(f"📉 趋势熔断: {', '.join(bear_assets)} 破位 -> 清仓")

        if asset_trends.get('IWY', False):
            cut = "80%" if (vix and vix > VIX_PANIC) else "50%"
            reasons.append(f"🛡️ 核心熔断: IWY 破位 -> 削减 {cut} 仓位")
    
    # 4. Sahm Rule 预警
    if sahm is not None and SAHM_EARLY_WARNING_LO <= sahm < SAHM_EARLY_WARNING_HI:
        reduction_pct = int((sahm - SAHM_EARLY_WARNING_LO) / (SAHM_EARLY_WARNING_HI - SAHM_EARLY_WARNING_LO) * SAHM_REDUCTION_RATE * 100)
        reasons.append(f"📉 Sahm预警 ({sahm:.2f}): 衰退风险上升 -> IWY预防性减仓 {reduction_pct}%")
    
    # 5. 收益率曲线解倒挂保护
    if yc_recently_inverted and yield_curve is not None and yield_curve > 0:
        reasons.append(f"📈 解倒挂保护: 曲线转正但近期曾倒挂 -> 维持防御配置 {int(YC_UNINVERT_REDUCTION*100)}%")
    
    # 6. 相关性调整 (v1.5: 渐进响应)
    if corr is not None and corr > CORR_MID_THRESHOLD:
        if corr > CORR_HIGH_THRESHOLD:
            reasons.append(f"🔗 相关性失效 (Corr={corr:.2f}): 股债同涨同跌 -> MBH渐进转移至WTMF/黄金 (最大{int(CORR_MAX_REALLOC*100)}%)")
        else:
            reasons.append(f"🔗 相关性升高 (Corr={corr:.2f}): 开始减少MBH配置")
    
    # v1.5: 市场广度
    if breadth_score is not None and breadth_score < MARKET_BREADTH_MID:
        if breadth_score < MARKET_BREADTH_LOW:
            reasons.append(f"📊 市场广度差 ({breadth_score*100:.0f}%<{MARKET_BREADTH_LOW*100:.0f}%): 多数资产下跌 -> 权益减仓{int(BREADTH_LOW_REDUCTION*100)}%")
        else:
            reasons.append(f"📊 市场广度一般 ({breadth_score*100:.0f}%): 权益小幅减仓{int(BREADTH_MID_REDUCTION*100)}%")
    
    # v1.5: 现金缓冲
    if vix is not None and vix > CASH_BUFFER_VIX_THRESHOLD and s != "EXTREME_ACCUMULATION":
        extra_cash = min((vix - CASH_BUFFER_VIX_THRESHOLD) / 5 * CASH_BUFFER_VIX_SCALE, CASH_BUFFER_MAX - CASH_BUFFER_BASE)
        total_cash = CASH_BUFFER_BASE + extra_cash
        reasons.append(f"💵 现金缓冲 (VIX={vix:.1f}): 保留{total_cash*100:.1f}%现金")

    # 7. Gold
    if gold_bear:
        reasons.append("🐻 黄金熊市: Gold < MA200 -> 清仓 GSD.SI")

    return reasons

def get_adjustment_reasons(s, gold_bear=False, value_regime=False, asset_trends=None, vix=None, yield_curve=None, sahm=None, corr=None, yc_recently_inverted=False, dual_ma_signals=None, breadth_score=None):
    """
    Returns a list of strings explaining why the allocation differs from the base static model.
    """
    return evaluate_risk_triggers(
        s,
        gold_bear=gold_bear,
        value_regime=value_regime,
        asset_trends=asset_trends,
        vix=vix,
        yield_curve=yield_curve,
        sahm=sahm,
        corr=corr,
        yc_recently_inverted=yc_recently_inverted,
        dual_ma_signals=dual_ma_signals,
        breadth_score=breadth_score,
    )

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
    cache_dir = os.path.join(base_dir, "data")
    os.makedirs(cache_dir, exist_ok=True)
    candidates = [
        os.path.join(base_dir, file_name),
        os.path.join(os.getcwd(), file_name),
        os.path.join(base_dir, alt_name),
        os.path.join(os.getcwd(), alt_name),
        os.path.join(cache_dir, file_name),
        os.path.join(cache_dir, alt_name),
    ]
    candidates = list(dict.fromkeys(candidates))
    target_path = candidates[0]
    lastgood_path = os.path.join(cache_dir, f"fred_{series_id}_lastgood.csv")
    
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
    
    # 2) 网络下载（含 https -> http 备份 + 退避重试）
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
        backoff = max(1, (attempt + 1))
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
                    with open(lastgood_path, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception as e:
                    print(f"Failed to write cache file: {e}")
                df = pd.read_csv(io.StringIO(content), parse_dates=['observation_date'], index_col='observation_date')
                df.columns = [series_id]
                return df
            except Exception as e:
                last_err = f"{url} -> {e}"
                continue
        time.sleep(backoff)
    
    if last_err:
        print(f"Error fetching FRED data ({series_id}): {last_err}")
        safe_warn(f"⚠️ 自动下载 FRED 数据失败 ({series_id})。错误: {last_err}\n\n**解决方法**：1) 检查网络/代理，2) 可手动下载并放入程序目录 (fred_{series_id}.csv 或 {series_id}.csv)。")

    # 3) 兜底使用本地旧文件（含 lastgood）
    fallback_candidates = list(candidates)
    if lastgood_path not in fallback_candidates:
        fallback_candidates.append(lastgood_path)

    for local_file in fallback_candidates:
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
    # New: real-time risk alerts
    "state_change_alert": False,
    "vix_alert_enabled": False,
    "vix_alert_threshold": 35,
    "channels": {
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "wechat_webhook": ""
    }
}

def load_alert_config():
    if os.path.exists(ALERT_CONFIG_FILE):
        try:
            with open(ALERT_CONFIG_FILE, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            log_event("ERROR", "[AlertConfig] load failed, using defaults", {"err": str(e)})
            cfg = DEFAULT_ALERT_CONFIG.copy()
    else:
        cfg = DEFAULT_ALERT_CONFIG.copy()

    merged, issues, warns = validate_alert_config(cfg)
    for w in warns:
        safe_warn(f"⚠️ 配置提醒: {w}")
        log_event("WARN", w)
    for i in issues:
        safe_warn(f"⚠️ {i}")
        log_event("ERROR", i)
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


def log_event(level: str, message: str, extra=None):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payload = {"ts": ts, "level": level.upper(), "msg": message}
    if extra:
        payload["extra"] = extra
    try:
        print(json.dumps(payload, ensure_ascii=False))
    except Exception:
        print(f"[{level}] {message} | extra={extra}")


def load_state_history():
    try:
        if os.path.exists(STATE_HISTORY_FILE):
            with open(STATE_HISTORY_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        log_event("ERROR", "state_history_load_failed", {"err": str(e)})
    return []


def save_state_history(history):
    try:
        with open(STATE_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        log_event("ERROR", "state_history_save_failed", {"err": str(e)})


def record_state_history(state, metrics):
    history = load_state_history()
    date_str = metrics.get('date') or datetime.date.today().isoformat()
    fetch_ts = metrics.get('fetch_ts') or datetime.datetime.now().isoformat(timespec='seconds')
    entry = {"date": date_str, "state": state, "ts": fetch_ts}

    if history and history[-1].get("date") == date_str:
        history[-1] = entry
    else:
        history.append(entry)
    save_state_history(history)
    return history


def get_state_change_info(history, current_state, current_date):
    if not current_date:
        return None
    streak_start = current_date
    prev_state = None
    prev_date = None
    for item in reversed(history):
        try:
            d = datetime.date.fromisoformat(item.get("date")) if item.get("date") else None
        except Exception:
            continue
        if item.get("state") == current_state:
            streak_start = d
        else:
            prev_state = item.get("state")
            prev_date = d
            break
    days_in_state = (current_date - streak_start).days + 1 if streak_start else None
    changed_on = streak_start
    return {
        "prev_state": prev_state,
        "prev_date": prev_date,
        "changed_on": changed_on,
        "days_in_state": days_in_state,
    }


# === 持仓历史追踪与回撤计算 ===
def load_portfolio_history():
    """加载持仓历史记录"""
    try:
        if os.path.exists(PORTFOLIO_HISTORY_FILE):
            with open(PORTFOLIO_HISTORY_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        log_event("ERROR", "portfolio_history_load_failed", {"err": str(e)})
    return {"records": [], "peak_value": 0, "cost_basis": 0}


def save_portfolio_history(history):
    """保存持仓历史记录"""
    try:
        with open(PORTFOLIO_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        log_event("ERROR", "portfolio_history_save_failed", {"err": str(e)})


def record_portfolio_snapshot(total_value, holdings_dict, state=None):
    """
    记录当前持仓快照，用于计算回撤
    total_value: 当前总市值
    holdings_dict: {ticker: value} 当前持仓
    """
    history = load_portfolio_history()
    date_str = datetime.date.today().isoformat()
    
    record = {
        "date": date_str,
        "ts": datetime.datetime.now().isoformat(timespec='seconds'),
        "total_value": total_value,
        "holdings": holdings_dict,
        "state": state
    }
    
    records: list[dict] = history.get("records", [])
    # 同一天只保留最新记录
    if records and records[-1].get("date") == date_str:
        records[-1] = record
    else:
        records.append(record)
    
    # 只保留最近90天数据
    if len(records) > 90:
        records = records[-90:]
    
    # 更新历史最高净值
    peak_value = history.get("peak_value", 0)
    if total_value > peak_value:
        peak_value = total_value
    
    history["records"] = records
    history["peak_value"] = peak_value
    
    save_portfolio_history(history)
    return history


def calculate_portfolio_drawdown(current_value, history=None):
    """
    计算当前组合回撤
    返回: (drawdown_pct, peak_value, days_since_peak, in_stop_loss_zone, recovery_ratio)
    """
    if history is None:
        history = load_portfolio_history()
    
    peak_value = history.get("peak_value", 0)
    if peak_value <= 0:
        return 0, current_value, 0, False, 1.0
    
    drawdown_pct = (current_value - peak_value) / peak_value
    
    # 计算距离峰值的天数
    records = history.get("records", [])
    days_since_peak = 0
    for rec in reversed(records):
        if rec.get("total_value", 0) >= peak_value * 0.999:  # 允许0.1%误差
            break
        days_since_peak += 1
    
    # 判断是否在止损区间
    in_stop_loss_zone = drawdown_pct < DRAWDOWN_STOP_LOSS
    
    # 计算恢复比例 (分阶段恢复)
    recovery_ratio = 1.0
    if in_stop_loss_zone:
        for threshold, ratio in STOP_LOSS_RECOVERY_STAGES:
            if drawdown_pct < threshold:
                recovery_ratio = ratio
                break
    
    return drawdown_pct, peak_value, days_since_peak, in_stop_loss_zone, recovery_ratio


def get_stop_loss_status(current_value, history=None):
    """
    获取止损状态的详细信息
    返回: dict with status details
    """
    drawdown_pct, peak_value, days_since_peak, in_stop_loss, recovery_ratio = calculate_portfolio_drawdown(current_value, history)
    
    status = {
        "current_value": current_value,
        "peak_value": peak_value,
        "drawdown_pct": drawdown_pct,
        "days_since_peak": days_since_peak,
        "in_stop_loss": in_stop_loss,
        "recovery_ratio": recovery_ratio,
        "should_reduce": in_stop_loss,
        "reduction_pct": (1 - recovery_ratio) * 100 if in_stop_loss else 0,
    }
    
    # 判断恢复阶段
    if in_stop_loss:
        status["stage"] = "止损中"
        status["stage_color"] = "#f5222d"
        status["advice"] = f"风险资产建议减仓至 {recovery_ratio*100:.0f}%"
    elif drawdown_pct < DRAWDOWN_RECOVERY_THRESHOLD:
        status["stage"] = "恢复中"
        status["stage_color"] = "#faad14"
        status["advice"] = f"回撤{drawdown_pct*100:.1f}%，接近止损线，保持警惕"
    else:
        status["stage"] = "正常"
        status["stage_color"] = "#52c41a"
        status["advice"] = "持仓健康，无需止损调整"
    
    return status


def reset_portfolio_peak(new_peak_value=None):
    """
    重置历史最高净值（用于注入新资金或手动调整）
    """
    history = load_portfolio_history()
    if new_peak_value is not None:
        history["peak_value"] = new_peak_value
    else:
        # 使用最近记录的最高值
        records = history.get("records", [])
        if records:
            history["peak_value"] = max(r.get("total_value", 0) for r in records)
    save_portfolio_history(history)
    return history


def validate_alert_config(cfg: dict):
    merged = DEFAULT_ALERT_CONFIG.copy()
    issues = []
    warns = []
    if not isinstance(cfg, dict):
        issues.append("配置文件格式错误，已恢复默认配置")
        return merged, issues, warns

    # Required keys & types
    merged.update({k: cfg.get(k, v) for k, v in merged.items()})

    # Email fields
    for key in ["email_to", "email_from"]:
        merged[key] = str(merged.get(key, "") or "").strip()
    merged["email_pwd"] = merged.get("email_pwd", "") or ""
    env_pwd = os.environ.get("ALERT_EMAIL_PWD") or os.environ.get("SMTP_PASSWORD")
    if not merged["email_pwd"] and env_pwd:
        merged["email_pwd"] = env_pwd
        warns.append("已使用环境变量中的邮箱密码/授权码")

    # SMTP port
    try:
        merged["smtp_port"] = int(merged.get("smtp_port", 587))
    except Exception:
        merged["smtp_port"] = 587
        warns.append("SMTP 端口无效，已回退 587")

    # Frequency
    freq = str(merged.get("frequency", "Manual") or "Manual")
    if freq not in ["Manual", "Daily", "Weekly"]:
        warns.append("frequency 非法，已回退 Manual")
        merged["frequency"] = "Manual"
    else:
        merged["frequency"] = freq

    # Trigger time
    trig = str(merged.get("trigger_time", "09:30") or "09:30")
    try:
        datetime.datetime.strptime(trig, "%H:%M")
        merged["trigger_time"] = trig
    except Exception:
        merged["trigger_time"] = "09:30"
        warns.append("触发时间格式无效，已回退 09:30")

    # Enabled flag
    merged["enabled"] = bool(merged.get("enabled", False))

    # Realtime alert controls
    merged["state_change_alert"] = bool(merged.get("state_change_alert", False))
    merged["vix_alert_enabled"] = bool(merged.get("vix_alert_enabled", False))
    try:
        merged["vix_alert_threshold"] = float(merged.get("vix_alert_threshold", 35))
    except Exception:
        merged["vix_alert_threshold"] = 35
        warns.append("VIX 阈值无效，已回退 35")

    # Channels placeholder (Telegram / WeCom)
    channels = merged.get("channels", {}) or {}
    if not isinstance(channels, dict):
        channels = {}
    merged["channels"] = {
        "telegram_bot_token": channels.get("telegram_bot_token", ""),
        "telegram_chat_id": channels.get("telegram_chat_id", ""),
        "wechat_webhook": channels.get("wechat_webhook", ""),
    }

    return merged, issues, warns


def check_data_health(df_hist: pd.DataFrame, freshness_limit_days: int = 5):
    warnings = []
    latest_date = None
    freshness_days = None
    if df_hist is None or df_hist.empty:
        warnings.append("数据为空，无法评估新鲜度")
        return warnings, latest_date, freshness_days

    latest_date = df_hist.index[-1].date()
    freshness_days = (datetime.date.today() - latest_date).days
    if freshness_days > freshness_limit_days:
        warnings.append(f"数据已滞后 {freshness_days} 天，请检查数据源或手动上传。")

    required_cols = ["State", "Sahm", "RateShock", "Corr", "VIX", "Trend_Bear", "YieldCurve"]
    missing = [c for c in required_cols if c not in df_hist.columns]
    if missing:
        warnings.append(f"缺少必要字段: {', '.join(missing)}")
    else:
        na_cols = [c for c in required_cols if df_hist[c].isna().any()]
        if na_cols:
            warnings.append(f"存在空值字段: {', '.join(na_cols)}，建议刷新或补齐数据。")

    return warnings, latest_date, freshness_days


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


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_scheduler_lock(ttl_hours: int = 6) -> bool:
    now_ts = time.time()
    if os.path.exists(SCHEDULER_LOCK):
        try:
            with open(SCHEDULER_LOCK, "r") as f:
                content = f.read().strip().split(",")
            pid = int(content[0]) if content and content[0].isdigit() else None
            ts = float(content[1]) if len(content) > 1 else 0
            if pid and is_pid_running(pid):
                # If lock is fresh and pid alive, refuse
                if now_ts - ts < ttl_hours * 3600:
                    return False
        except Exception:
            pass
    try:
        with open(SCHEDULER_LOCK, "w") as f:
            f.write(f"{os.getpid()},{now_ts}")
        return True
    except Exception as e:
        print(f"[Lock] Failed to write scheduler lock: {e}")
        return False

def analyze_market_state_logic():
    """
    Core logic to fetch data and determine current market state.
    Returns: (success, result_dict_or_error_msg)
    """
    ensure_fred_cached()
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365*3)
    
    # Re-use the robust fetcher
    df_hist, err = get_historical_macro_data(start, end)
    
    if df_hist.empty:
        return False, err

    data_warnings, latest_date, freshness_days = check_data_health(df_hist)
    for w in data_warnings:
        safe_warn(f"⚠️ 数据健康提醒: {w}")
        log_event("WARN", "data_health", {"msg": w})
    
    # Extract latest state
    last_row = df_hist.iloc[-1]
    state = last_row['State']
    
    # --- Fetch Portfolio Asset Trends (Dual Momentum) ---
    asset_trends = {}
    try:
        check_assets = ['G3B.SI', 'LVHI', 'SRT.SI', 'AJBU.SI', 'IWY', 'MBH.SI', 'GSD.SI']
        trend_start = datetime.date.today() - datetime.timedelta(days=400)
        data_raw = fetch_yf_with_retry(check_assets, start=trend_start, auto_adjust=False)
        
        df_assets = pd.DataFrame()
        if data_raw is not None and not data_raw.empty:
            df_assets = normalize_yf_prices(data_raw)

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
        log_event("ERROR", "asset_trend_fetch_failed", {"err": str(e)})

    # Logic helpers
    yc_series = df_hist['YieldCurve']
    yc_un_invert = False
    if len(yc_series) > 126:
        recent_min = yc_series.iloc[-126:].min()
        current_yc = yc_series.iloc[-1]
        yc_un_invert = (current_yc < 0.2) and (recent_min < -0.2)

    factor_cols = [c for c in ["VIX", "YieldCurve", "Corr", "Sahm", "RateShock"] if c in df_hist.columns]
    factor_trends = df_hist[factor_cols].tail(90) if factor_cols else pd.DataFrame()

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
        'asset_trends': asset_trends,
        'freshness_days': freshness_days,
        'latest_date': last_row.name.date() if hasattr(last_row, 'name') else None,
        'data_warnings': data_warnings,
        'factor_trends': factor_trends,
        'fetch_ts': datetime.datetime.now().isoformat(timespec='seconds')
    }
    
    return True, metrics


@st.cache_data(ttl=300, show_spinner=False)
def analyze_market_state_logic_cached():
    return analyze_market_state_logic()


def generate_email_risk_exposure(targets):
    """
    生成邮件用的风险暴露分析HTML
    """
    # 计算目标类别权重
    target_categories = {}
    for tkr, w in targets.items():
        cat = ASSET_CATEGORIES.get(tkr, {}).get('category', '其他')
        target_categories[cat] = target_categories.get(cat, 0) + w
    
    cat_colors = {
        '权益': '#f5222d', '固收': '#1890ff', '商品': '#faad14', 
        '对冲': '#52c41a', '另类': '#722ed1', '其他': '#999'
    }
    
    bars_html = ""
    for cat in ['权益', '固收', '商品', '对冲', '另类']:
        w = target_categories.get(cat, 0)
        if w > 0:
            bar_color = cat_colors.get(cat, '#999')
            bars_html += f"""
            <div style="margin-bottom:8px;">
                <span style="display:inline-block;width:70px;font-size:13px;color:#666;">{cat}</span>
                <span style="display:inline-block;width:150px;background:#e8e8e8;height:18px;border-radius:4px;vertical-align:middle;">
                    <span style="display:block;width:{w*100}%;height:100%;background:{bar_color};border-radius:4px;"></span>
                </span>
                <span style="font-size:13px;margin-left:10px;font-weight:600;">{w*100:.1f}%</span>
            </div>
            """
    
    return bars_html


def generate_email_v15_status(metrics, state, change_info=None):
    """
    生成v1.5优化机制状态的邮件HTML
    change_info: 包含 days_in_state 等信息的字典
    """
    vix = metrics.get('vix', 15)
    sahm = metrics.get('sahm', 0)
    corr = metrics.get('corr', 0)
    yc = metrics.get('yield_curve', 0)
    yc_un_invert = metrics.get('yc_un_invert', False)
    
    status_items = []
    
    # 0. 信号持续天数与确认状态
    days_in_state = change_info.get('days_in_state') if change_info else None
    if days_in_state is not None:
        if days_in_state <= SIGNAL_CONFIRM_DAYS:
            confirm_text = f"确认中 ({days_in_state}/{SIGNAL_CONFIRM_DAYS}天)"
            confirm_color = "#fa8c16"  # 橙色警示
        else:
            confirm_text = f"已确认 ({days_in_state}天)"
            confirm_color = "#52c41a"  # 绿色
        
        status_items.append({
            'name': '🔄 信号状态',
            'value': confirm_text,
            'color': confirm_color
        })
    
    # 1. 现金缓冲状态
    if state == "EXTREME_ACCUMULATION":
        cash_buffer = 0
        cash_color = "#52c41a"
    else:
        cash_buffer = CASH_BUFFER_BASE
        if vix > CASH_BUFFER_VIX_THRESHOLD:
            extra_cash = min((vix - CASH_BUFFER_VIX_THRESHOLD) / 5 * CASH_BUFFER_VIX_SCALE, 
                             CASH_BUFFER_MAX - CASH_BUFFER_BASE)
            cash_buffer = CASH_BUFFER_BASE + extra_cash
        cash_color = "#faad14" if cash_buffer > CASH_BUFFER_BASE else "#52c41a"
    
    status_items.append({
        'name': '💵 现金缓冲',
        'value': f"{cash_buffer*100:.1f}%",
        'color': cash_color
    })
    
    # 2. CAUTIOUS_VOL VIX分层
    if state == "CAUTIOUS_VOL":
        if vix >= 30:
            tier_text = "Tier3 (IWY↓10%)"
            tier_color = "#f5222d"
        elif vix >= 25:
            tier_text = "Tier2 (IWY↓20%)"
            tier_color = "#fa8c16"
        else:
            tier_text = "Tier1 (IWY 30%)"
            tier_color = "#faad14"
    else:
        tier_text = "N/A"
        tier_color = "#999"
    
    status_items.append({
        'name': '📊 VIX分层',
        'value': tier_text,
        'color': tier_color
    })
    
    # 3. 相关性渐进响应
    if corr > CORR_HIGH_THRESHOLD:
        corr_text = f"最大调整 {CORR_MAX_REALLOC*100:.0f}%"
        corr_color = "#f5222d"
    elif corr > CORR_MID_THRESHOLD:
        adjustment_pct = (corr - CORR_MID_THRESHOLD) / (CORR_HIGH_THRESHOLD - CORR_MID_THRESHOLD)
        realloc = adjustment_pct * CORR_MAX_REALLOC
        corr_text = f"渐进 {realloc*100:.1f}%"
        corr_color = "#fa8c16"
    else:
        corr_text = "正常"
        corr_color = "#52c41a"
    
    status_items.append({
        'name': '🔗 相关性响应',
        'value': corr_text,
        'color': corr_color
    })
    
    # 4. Sahm预警
    if sahm >= SAHM_EARLY_WARNING_HI:
        sahm_text = "衰退确认"
        sahm_color = "#f5222d"
    elif sahm >= SAHM_EARLY_WARNING_LO:
        reduction_pct = int((sahm - SAHM_EARLY_WARNING_LO) / (SAHM_EARLY_WARNING_HI - SAHM_EARLY_WARNING_LO) * SAHM_REDUCTION_RATE * 100)
        sahm_text = f"预警 -{reduction_pct}%"
        sahm_color = "#fa8c16"
    else:
        sahm_text = "正常"
        sahm_color = "#52c41a"
    
    status_items.append({
        'name': '📉 Sahm预警',
        'value': sahm_text,
        'color': sahm_color
    })
    
    # 5. 曲线保护
    if yc < 0:
        yc_text = "倒挂中"
        yc_color = "#f5222d"
    elif yc_un_invert:
        yc_text = f"解倒挂保护"
        yc_color = "#fa8c16"
    else:
        yc_text = "正常"
        yc_color = "#52c41a"
    
    status_items.append({
        'name': '📈 曲线保护',
        'value': yc_text,
        'color': yc_color
    })
    
    # 6. 市场广度估算
    asset_trends = metrics.get('asset_trends', {})
    if asset_trends:
        bullish_count = sum(1 for bear in asset_trends.values() if not bear)
        total_count = len(asset_trends)
        breadth = bullish_count / total_count if total_count > 0 else 0.5
    else:
        breadth = 0.5
    
    if breadth < MARKET_BREADTH_LOW:
        breadth_text = f"低 ({breadth*100:.0f}%)"
        breadth_color = "#f5222d"
    elif breadth < MARKET_BREADTH_MID:
        breadth_text = f"一般 ({breadth*100:.0f}%)"
        breadth_color = "#fa8c16"
    else:
        breadth_text = f"正常 ({breadth*100:.0f}%)"
        breadth_color = "#52c41a"
    
    status_items.append({
        'name': '📊 市场广度',
        'value': breadth_text,
        'color': breadth_color
    })
    
    # 构建HTML
    items_html = ""
    for i, item in enumerate(status_items):
        items_html += f"""
        <td style="padding:8px 12px;text-align:center;border-right:{'1px solid #e5e7eb' if i < len(status_items)-1 else 'none'};">
            <div style="font-size:11px;color:#666;">{item['name']}</div>
            <div style="font-size:13px;font-weight:600;color:{item['color']};margin-top:2px;">{item['value']}</div>
        </td>
        """
    
    return f"""
    <div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:12px;margin:12px 0;">
        <div style="font-weight:600;color:#374151;margin-bottom:8px;font-size:14px;">⚙️ v1.5 优化机制状态</div>
        <table style="width:100%;border-collapse:collapse;">
            <tr>{items_html}</tr>
        </table>
    </div>
    """


def generate_email_execution_tips(metrics, state):
    """
    生成邮件用的执行建议HTML
    """
    tips = []
    vix = metrics.get('vix')
    sahm = metrics.get('sahm')
    corr = metrics.get('corr')
    yc_val = metrics.get('yield_curve', 0)
    
    # 1. VIX相关提示
    if vix is not None:
        if vix > VIX_SMOOTH_END:
            reduction_pct = int(VIX_MAX_REDUCTION * 100)
            tips.append({
                'icon': '📊',
                'title': '高波动率警告',
                'content': f'VIX={vix:.1f} 处于高位，建议按目标配置的 {100-reduction_pct}% 执行，剩余资金持有现金或 WTMF。',
                'color': '#cf1322',
                'bg': '#fff2f0'
            })
        elif vix > VIX_SMOOTH_START:
            reduction = (vix - VIX_SMOOTH_START) / (VIX_SMOOTH_END - VIX_SMOOTH_START) * VIX_MAX_REDUCTION
            exec_pct = int((1 - reduction) * 100)
            tips.append({
                'icon': '📊',
                'title': '波动率偏高',
                'content': f'VIX={vix:.1f}，可考虑按目标配置的 {exec_pct}% 执行，留 {100-exec_pct}% 现金缓冲。',
                'color': '#ad6800',
                'bg': '#fffbe6'
            })
    
    # 2. Sahm Rule预警
    if sahm is not None and SAHM_EARLY_WARNING_LO <= sahm < SAHM_EARLY_WARNING_HI:
        reduction_pct = int((sahm - SAHM_EARLY_WARNING_LO) / (SAHM_EARLY_WARNING_HI - SAHM_EARLY_WARNING_LO) * SAHM_REDUCTION_RATE * 100)
        tips.append({
            'icon': '📉',
            'title': 'Sahm预警区间',
            'content': f'Sahm Rule={sahm:.2f} 处于预警区间 ({SAHM_EARLY_WARNING_LO}-{SAHM_EARLY_WARNING_HI})，IWY已预防性减仓{reduction_pct}%。',
            'color': '#ad6800',
            'bg': '#fffbe6'
        })
    
    # 3. 相关性渐进响应
    if corr is not None and corr > CORR_MID_THRESHOLD:
        if corr > CORR_HIGH_THRESHOLD:
            tips.append({
                'icon': '🔗',
                'title': '股债相关性失效',
                'content': f'Corr={corr:.2f} 超过阈值，债券对冲效果减弱，MBH已转移{CORR_MAX_REALLOC*100:.0f}%至WTMF/黄金。',
                'color': '#cf1322',
                'bg': '#fff2f0'
            })
        else:
            adjustment_pct = (corr - CORR_MID_THRESHOLD) / (CORR_HIGH_THRESHOLD - CORR_MID_THRESHOLD)
            realloc = adjustment_pct * CORR_MAX_REALLOC
            tips.append({
                'icon': '🔗',
                'title': '相关性渐进响应',
                'content': f'Corr={corr:.2f} 处于关注区间，MBH渐进转移{realloc*100:.1f}%至非相关资产。',
                'color': '#ad6800',
                'bg': '#fffbe6'
            })
    
    # 4. 收益率曲线提示
    if yc_val < 0:
        tips.append({
            'icon': '📈',
            'title': '收益率曲线倒挂',
            'content': f'10Y-2Y={yc_val:.2f}%，曲线倒挂中，债券配置需谨慎，优先选择短久期或WTMF。',
            'color': '#ad6800',
            'bg': '#fffbe6'
        })
    elif metrics.get('yc_un_invert', False):
        tips.append({
            'icon': '⚠️',
            'title': '解倒挂保护期',
            'content': f'收益率曲线刚转正，历史上此阶段衰退风险仍高，IWY已防御性减仓{YC_UNINVERT_REDUCTION*100:.0f}%。',
            'color': '#ad6800',
            'bg': '#fffbe6'
        })
    
    # 5. 极端状态提示
    if state == "EXTREME_ACCUMULATION":
        tips.append({
            'icon': '⚡',
            'title': '抄底状态注意',
            'content': '当前为极端抄底状态，现金缓冲已关闭。建议分批建仓：首次40% → 反弹确认后60% → 趋势确立后75%。',
            'color': '#ad6800',
            'bg': '#fffbe6'
        })
    elif state in ["DEFLATION_RECESSION", "INFLATION_SHOCK"]:
        tips.append({
            'icon': '🛡️',
            'title': '防御模式提醒',
            'content': '当前处于危机状态，建议严格执行目标配置，优先保护本金，避免抄底冲动。',
            'color': '#cf1322',
            'bg': '#fff2f0'
        })
    elif state == "CAUTIOUS_VOL":
        vix_tier = "Tier1" if vix < 25 else ("Tier2" if vix < 30 else "Tier3")
        tips.append({
            'icon': '⚡',
            'title': f'高波震荡 ({vix_tier})',
            'content': f'VIX={vix:.1f}，已启用分层配置。保持核心成长，WTMF对冲波动。',
            'color': '#ad6800',
            'bg': '#fffbe6'
        })
    
    # 6. 通用执行建议
    tips.append({
        'icon': '📏',
        'title': '再平衡建议',
        'content': f'单一资产偏离>{REBALANCE_THRESHOLD*100:.0f}%时再调仓；大幅调仓建议分{STATE_TRANSITION_DAYS}天执行，信号需连续{SIGNAL_CONFIRM_DAYS}天确认。',
        'color': '#0050b3',
        'bg': '#e6f7ff'
    })
    
    tips_html = ""
    for tip in tips:
        tips_html += f"""
        <div style="background:{tip['bg']};border-radius:8px;padding:10px 14px;margin-bottom:8px;">
            <div style="font-weight:600;color:{tip['color']};margin-bottom:2px;">{tip['icon']} {tip['title']}</div>
            <div style="color:#333;font-size:13px;line-height:1.4;">{tip['content']}</div>
        </div>
        """
    
    return tips_html


def render_email_html(metrics, targets, adjustments, s_conf, sent_at, report_date, change_info=None):
    target_rows = ""
    for t, w in targets.items():
        if w > 0:
            target_rows += f"<tr><td>{ASSET_NAMES.get(t, t)}</td><td style='color:#555'>{t}</td><td><b>{w*100:.1f}%</b></td></tr>"

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

    yc_val = metrics.get('yield_curve', 0)
    state = metrics.get('state', 'NEUTRAL')
    
    # 信号持续天数信息
    days_in_state = change_info.get('days_in_state') if change_info else None
    days_info = ""
    if days_in_state is not None:
        if days_in_state <= SIGNAL_CONFIRM_DAYS:
            days_info = f" (确认中 {days_in_state}/{SIGNAL_CONFIRM_DAYS}天)"
        else:
            days_info = f" (持续{days_in_state}天)"
    
    summary_points = [
        f"数据截至 {report_date}",
        f"状态: {s_conf['display']}{days_info}",
        f"VIX {metrics['vix']:.1f} ({'⚠️ 高波动' if metrics['fear'] else '✅ 正常'})",
        f"10Y-2Y {yc_val:.2f}% ({'⚠️ 倒挂/解倒挂' if (yc_val < 0 or metrics.get('yc_un_invert', False)) else '✅ 正常'})",
        f"Sahm {metrics['sahm']:.2f} ({'⚠️ 衰退信号' if metrics['recession'] else '✅ 未触发'})"
    ]
    summary_html = "".join([f"<span style='display:inline-block;background:#f0f4ff;color:#1a73e8;padding:6px 10px;border-radius:20px;margin:4px 4px 0 0;font-size:13px;'>{p}</span>" for p in summary_points])
    
    # 生成v1.5优化机制状态
    v15_status_html = generate_email_v15_status(metrics, state, change_info)
    
    # 生成风险暴露分析
    risk_exposure_html = generate_email_risk_exposure(targets)
    
    # 生成执行建议
    execution_tips_html = generate_email_execution_tips(metrics, state)

    return f"""
    <html>
    <body style=\"font-family: 'Helvetica Neue', Arial, sans-serif; color: #1f2937; background:#f7f8fa;\">
        <div style=\"max-width: 680px; margin: 24px auto; background:#fff; border:1px solid #e5e7eb; border-radius:14px; overflow:hidden; box-shadow:0 10px 30px rgba(0,0,0,0.05);\">
            <div style=\"padding:22px 24px; background: linear-gradient(135deg, {s_conf['border_color']} 0%, #1f1f1f 100%); color:#fff;\">
                <div style=\"font-size:13px; opacity:0.85;\">数据截至 {report_date}</div>
                <div style=\"font-size:12px; opacity:0.75;\">发送时间 {sent_at}</div>
                <h2 style=\"margin:6px 0 4px 0; font-weight:700; letter-spacing:0.3px;\">{s_conf['icon']} 宏观策略快报 v1.5</h2>
                <div style=\"opacity:0.9; line-height:1.5; font-size:14px;\">{s_conf['desc']}</div>
            </div>

            <div style=\"padding:22px 24px;\">
                <div style=\"margin-bottom:12px;\">{summary_html}</div>
                
                {v15_status_html}

                <h3 style=\"margin:18px 0 10px 0; font-size:16px;\">📈 核心指标 (Key Metrics)</h3>
                <table style=\"width:100%; border-collapse:separate; border-spacing:0 8px; font-size:14px;\">
                    <tr style=\"background:#f9fafb;\"><td style=\"padding:10px 12px; border-radius:10px 0 0 10px;\">利率冲击</td><td style=\"padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#d93025' if metrics['rate_shock'] else '#15803d'};\">{metrics['tnx_roc']:.1%} ({'⚠️ 触发' if metrics['rate_shock'] else '✅ 安全'})</td></tr>
                    <tr style=\"background:#f9fafb;\"><td style=\"padding:10px 12px; border-radius:10px 0 0 10px;\">Sahm Rule</td><td style=\"padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#d93025' if metrics['recession'] else '#15803d'};\">{metrics['sahm']:.2f} ({'⚠️ 触发' if metrics['recession'] else '✅ 安全'})</td></tr>
                    <tr style=\"background:#f9fafb;\"><td style=\"padding:10px 12px; border-radius:10px 0 0 10px;\">VIX</td><td style=\"padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#ea580c' if metrics['fear'] else '#15803d'};\">{metrics['vix']:.1f} ({'⚠️ 恐慌' if metrics['fear'] else '✅ 正常'})</td></tr>
                    <tr style=\"background:#f9fafb;\"><td style=\"padding:10px 12px; border-radius:10px 0 0 10px;\">股债相关性</td><td style=\"padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#d93025' if metrics['corr_broken'] else '#15803d'};\">{metrics['corr']:.2f} ({'⚠️ 失效' if metrics['corr_broken'] else '✅ 正常'})</td></tr>
                    <tr style=\"background:#f9fafb;\"><td style=\"padding:10px 12px; border-radius:10px 0 0 10px;\">收益率曲线 (10Y-2Y)</td><td style=\"padding:10px 12px; border-radius:0 10px 10px 0; font-weight:600; color:{'#d93025' if (yc_val < 0 or metrics.get('yc_un_invert', False)) else '#15803d'};\">{yc_val:.2f}%</td></tr>
                </table>

                <h3 style=\"margin:20px 0 10px 0; font-size:16px;\">🎯 战术概览 (Tactical)</h3>
                <ul style=\"line-height:1.6; margin-top:6px; padding-left:18px; color:#374151;\">
                    <li><b>黄金趋势:</b> {'🐻 回避' if metrics['gold_bear'] else '🐂 持有/增配'}</li>
                    <li><b>风格轮动:</b> {'🧱 Value 价值占优' if metrics['value_regime'] else '🚀 Growth 成长占优'}</li>
                </ul>

                {adj_html}

                <h3 style=\"margin:20px 0 10px 0; font-size:16px;\">📊 建议配置 (Target Allocation)</h3>
                <table border=\"0\" cellpadding=\"10\" cellspacing=\"0\" style=\"width: 100%; border-collapse: collapse; margin-top: 8px; font-size:14px;\">
                    <tr style=\"background-color: #f3f4f6; text-align: left;\">
                        <th style=\"border-bottom: 2px solid #e5e7eb;\">资产名称</th>
                        <th style=\"border-bottom: 2px solid #e5e7eb;\">代码</th>
                        <th style=\"border-bottom: 2px solid #e5e7eb;\">目标仓位</th>
                    </tr>
                    {target_rows}
                </table>
                
                <h3 style=\"margin:20px 0 10px 0; font-size:16px;\">🎯 风险暴露分析 (Risk Exposure)</h3>
                <div style=\"background:#f9fafb;border-radius:10px;padding:14px 16px;margin:8px 0;\">
                    {risk_exposure_html}
                </div>
                
                <h3 style=\"margin:20px 0 10px 0; font-size:16px;\">💡 执行建议 (Execution Tips)</h3>
                {execution_tips_html}

                <p style=\"font-size: 12px; color: #6b7280; margin-top: 26px; text-align: center; border-top: 1px solid #e5e7eb; padding-top: 10px;\">
                    此邮件由 Stock Strategy Analyzer v1.5 自动生成，供参考，不构成投资建议。
                </p>
            </div>
        </div>
    </body>
    </html>
    """

def send_strategy_email(metrics, config):
    """发送策略分析邮件，返回 (success, message)。"""
    ensure_fred_cached()
    email_to = str(config.get("email_to", "")).strip()
    email_from = str(config.get("email_from", "")).strip()
    email_pwd = config.get("email_pwd", "")
    smtp_server = str(config.get("smtp_server", "smtp.gmail.com")).strip() or "smtp.gmail.com"
    try:
        smtp_port = int(config.get("smtp_port", 587))
    except Exception:
        smtp_port = 587

    if not email_to or not email_from or not email_pwd:
        log_event("ERROR", "email config incomplete", {"to": email_to, "from": email_from})
        return False, "邮箱配置不完整"

    state = metrics['state']
    s_conf = MACRO_STATES.get(state, MACRO_STATES["NEUTRAL"])
    sent_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    report_date = metrics.get('date', sent_at.split(' ')[0])
    
    # 计算信号持续天数
    history = load_state_history()
    current_date = metrics.get('latest_date')
    if current_date is None:
        try:
            current_date = datetime.date.fromisoformat(report_date)
        except:
            current_date = datetime.date.today()
    change_info = get_state_change_info(history, state, current_date)
    
    targets = get_target_percentages(
        state,
        gold_bear=metrics['gold_bear'],
        value_regime=metrics['value_regime'],
        asset_trends=metrics.get('asset_trends', {}),
        vix=metrics.get('vix'),
        yield_curve=metrics.get('yield_curve'),
        sahm=metrics.get('sahm'),
        corr=metrics.get('corr'),
        yc_recently_inverted=metrics.get('yc_un_invert', False)
    )

    adjustments = get_adjustment_reasons(
        state,
        gold_bear=metrics['gold_bear'],
        value_regime=metrics['value_regime'],
        asset_trends=metrics.get('asset_trends', {}),
        vix=metrics.get('vix'),
        yield_curve=metrics.get('yield_curve'),
        sahm=metrics.get('sahm'),
        corr=metrics.get('corr'),
        yc_recently_inverted=metrics.get('yc_un_invert', False)
    )

    html_content = render_email_html(metrics, targets, adjustments, s_conf, sent_at, report_date, change_info)

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
        log_event("INFO", "email sent", {"to": email_to, "state": state, "report_date": report_date})
        return True, "邮件发送成功"
    except Exception as e:
        log_event("ERROR", "email send failed", {"err": str(e)})
        return False, f"邮件发送失败: {str(e)}"

# --- Background Scheduler (Lightweight) ---

scheduler_thread = None

@st.cache_resource
def start_scheduler_service():
    """
    Starts the background scheduler in a singleton thread.
    Uses @st.cache_resource to ensure only one thread runs per server process,
    preventing duplicate emails when multiple tabs are open.
    """
    global scheduler_thread
    if scheduler_thread:
        return scheduler_thread
    if st.session_state.get("_scheduler_started"):
        return scheduler_thread

    if not acquire_scheduler_lock(ttl_hours=6):
        print("[Scheduler] Lock held by another process/session; skip start.")
        return scheduler_thread

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
                            # Idempotent guard: prevent duplicate sends across threads/processes (24h)
                            if not acquire_daily_lock(today_str, ttl_minutes=1440):
                                log_event("WARN", f"Skip duplicate send for {today_str}")
                            else:
                                log_event("INFO", "Triggering auto-analysis", {"now": str(now)})
                                success, res = analyze_market_state_logic()
                                if success:
                                    email_ok, msg = send_strategy_email(res, cfg)
                                    if email_ok:
                                        log_event("INFO", "Email sent", {"to": cfg.get("email_to")})
                                        cfg = load_alert_config()
                                        cfg["last_run"] = today_str
                                        save_alert_config(cfg)
                                    else:
                                        log_event("ERROR", "Email failed", {"err": msg})
                                else:
                                    log_event("ERROR", "Analysis failed", {"err": res})
            except Exception as e:
                log_event("ERROR", "Scheduler loop error", {"err": str(e)})
            
            time.sleep(60) # Check every minute

    # Create and start the thread
    t = threading.Thread(target=run_scheduler_check, daemon=True)
    t.start()
    scheduler_thread = t
    st.session_state["_scheduler_started"] = True
    print("[System] Global background scheduler service started.")
    return scheduler_thread

# Start scheduler (Singleton)
if __name__ == "__main__":
    start_scheduler_service()

# --- Shared Logic for Backtest & State Machine ---

def base_allocation(s, value_regime=False, vix=None):
    """
    基础资产配置矩阵
    v1.5: CAUTIOUS_VOL 状态支持VIX分层配置
    """
    if s == "INFLATION_SHOCK":
        # v1.8: 通胀冲击 - 成长清零，WTMF+黄金为主，红利保留（股息抗通胀）
        # 基准IWY 60%→0%, WTMF 7%→45%, GSD 5%→20%, LVHI 15%→20%
        return {
            'IWY': 0.00, 'WTMF': 0.45, 'LVHI': 0.20,  # 红利抗通胀
            'G3B.SI': 0.00, 'MBH.SI': 0.00, 'GSD.SI': 0.20,  # 黄金抗通胀
            'SRT.SI': 0.05, 'AJBU.SI': 0.10  # REITs抗通胀
        }
    if s == "DEFLATION_RECESSION":
        # v1.8: 通缩衰退 - 债券为王，黄金避险，少量成长/红利
        # 基准IWY 60%→10%, MBH 5%→35%, GSD 5%→20%, LVHI 15%→10%
        return {
            'IWY': 0.10, 'WTMF': 0.15, 'LVHI': 0.10,  # 保留少量权益
            'G3B.SI': 0.00, 'MBH.SI': 0.35, 'GSD.SI': 0.20,  # 债券+黄金
            'SRT.SI': 0.05, 'AJBU.SI': 0.05  # REITs在衰退时可能被错杀
        }
    if s == "EXTREME_ACCUMULATION":
        # v2.0: 极端抄底，IWY最大化但保留波动缓冲
        # IWY 75%（不再极端到80%，保留缓冲空间）
        return {
            'IWY': 0.75, 'WTMF': 0.00, 'LVHI': 0.10,  # LVHI波动缓冲
            'G3B.SI': 0.05, 'MBH.SI': 0.00, 'GSD.SI': 0.03,
            'SRT.SI': 0.05, 'AJBU.SI': 0.02
        }
    if s == "CAUTIOUS_TREND":
        # v1.8: 趋势谨慎 - 红利为主，成长减半，WTMF对冲
        # 基准IWY 60%→25%, LVHI 15%→30%, WTMF 7%→20%
        growth_w = 0.25                # 成长减半以上
        value_w = 0.30                 # 红利为主（抗跌+分红）
        wtmf_w = 0.20                  # WTMF对冲趋势风险
        if value_regime:
            growth_w = 0.15            # 价值占优时进一步减少成长
            value_w = 0.38             # 更多红利
            wtmf_w = 0.18
        return {
            'IWY': growth_w, 'WTMF': wtmf_w, 'LVHI': value_w,
            'G3B.SI': 0.05, 'MBH.SI': 0.08, 'GSD.SI': 0.07,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }
    if s == "CAUTIOUS_VOL":
        # v2.0: 高波动VIX分层 - IWY仍为核心，但通过LVHI+WTMF控制波动
        # 核心: IWY始终占主导，但波动越高，WTMF越多
        iwy_w = 0.50   # 基准65%↓到50%（保持核心地位）
        wtmf_w = 0.15  # 基准5%↑到15%
        lvhi_w = 0.15  # LVHI作为波动缓冲
        mbh_w = 0.05
        
        if vix is not None:
            # VIX分层: 波动越高，IWY↓WTMF↑，但LVHI保持稳定
            if 20 <= vix < 25:
                iwy_w = 0.50   # 轻度高波
                wtmf_w = 0.15
                lvhi_w = 0.15
            elif 25 <= vix < 30:
                iwy_w = 0.42   # 中度高波
                wtmf_w = 0.22
                lvhi_w = 0.15  # LVHI保持（红利抗跌）
            elif 30 <= vix < 40:
                iwy_w = 0.35   # 重度高波（IWY仍占最大比重）
                wtmf_w = 0.30
                lvhi_w = 0.15
            elif vix >= 40:
                iwy_w = 0.28   # 极端高波（IWY仍为核心）
                wtmf_w = 0.35
                lvhi_w = 0.15
        
        return {
            'IWY': iwy_w, 'WTMF': wtmf_w, 'LVHI': lvhi_w,
            'G3B.SI': 0.03, 'MBH.SI': mbh_w, 'GSD.SI': 0.05,
            'SRT.SI': 0.03, 'AJBU.SI': 0.02
        }
    # NEUTRAL - v2.1 IWY核心配置 (IWY 68%, LVHI 10%, WTMF 5%)
    # 核心思想: IWY主导收益，减少低收益资产拖累
    growth_w = 0.68                   # IWY核心配置 (提高: 65%→68%)
    value_w = 0.10                    # LVHI最低缓冲
    wtmf_w = 0.05                     # WTMF基础危机保险
    if value_regime:
        growth_w = 0.58               # 价值占优时减少成长
        value_w = 0.18                # 增加红利
        wtmf_w = 0.05
    return {
        'IWY': growth_w, 'WTMF': wtmf_w, 'LVHI': value_w,
        'G3B.SI': 0.05, 'MBH.SI': 0.04, 'GSD.SI': 0.05,  # 减少债券配置
        'SRT.SI': 0.03, 'AJBU.SI': 0.00
    }


def apply_vix_adjustments(targets, state, vix):
    """v2.0: VIX驱动的动态调整（IWY核心优化）"""
    if vix is None:
        return
    
    if state == "NEUTRAL":
        if vix < VIX_BOOST_LO:
            # 极低VIX (<13): 从WTMF/避险资产转到IWY（最大化成长）
            # 但保留LVHI最低配置作为波动锚定
            wtmf_amt = max(targets.get('WTMF', 0) - WTMF_BASE_ALLOCATION, 0)
            mbh_amt = targets.get('MBH.SI', 0) * 0.5  # 减50%债券
            gsd_amt = targets.get('GSD.SI', 0) * 0.3  # 减30%黄金
            
            total_boost = wtmf_amt + mbh_amt + gsd_amt
            targets['WTMF'] = WTMF_BASE_ALLOCATION  # 保持基础WTMF
            targets['MBH.SI'] = targets.get('MBH.SI', 0) - mbh_amt
            targets['GSD.SI'] = targets.get('GSD.SI', 0) - gsd_amt
            targets['IWY'] = targets.get('IWY', 0) + total_boost
            
        elif vix > VIX_GROWTH_TO_VALUE_START:
            # VIX>22: 从IWY转向LVHI+WTMF（波动控制）
            # IWY仍保持核心地位，但增加波动缓冲
            shift_ratio = min((vix - VIX_GROWTH_TO_VALUE_START) / (VIX_GROWTH_TO_VALUE_FULL - VIX_GROWTH_TO_VALUE_START), 1.0)
            max_shift = min(targets.get('IWY', 0) * 0.20, 0.12 * shift_ratio)  # 最多转移12%
            
            if max_shift > 0:
                targets['IWY'] -= max_shift
                # 50%转LVHI（红利抗跌），50%转WTMF（危机对冲）
                targets['LVHI'] = targets.get('LVHI', 0) + max_shift * 0.5
                targets['WTMF'] = targets.get('WTMF', 0) + max_shift * 0.5
    
    elif state == "CAUTIOUS_VOL":
        # 高波动状态：动态调整已在base_allocation中处理
        pass


def apply_yield_curve_guard(targets, state, yield_curve):
    if state not in ["DEFLATION_RECESSION", "CAUTIOUS_TREND"]:
        return
    if yield_curve is None or yield_curve >= YIELD_CURVE_CUTOFF:
        return
    if targets.get('MBH.SI', 0) > 0:
        move_amt = targets['MBH.SI'] * 0.7
        targets['MBH.SI'] -= move_amt
        targets['WTMF'] = targets.get('WTMF', 0) + move_amt


def apply_trend_filters(targets, state, asset_trends):
    if state == "EXTREME_ACCUMULATION":
        return
    assets_to_check = ['G3B.SI', 'LVHI', 'MBH.SI', 'GSD.SI', 'SRT.SI', 'AJBU.SI']
    for asset in assets_to_check:
        if targets.get(asset, 0) > 0 and asset_trends.get(asset, False):
            weight_to_move = targets[asset]
            targets[asset] = 0.0
            if state == "NEUTRAL":
                if not asset_trends.get('IWY', False):
                    targets['IWY'] = targets.get('IWY', 0) + weight_to_move
                else:
                    targets['WTMF'] = targets.get('WTMF', 0) + weight_to_move
            else:
                targets['WTMF'] = targets.get('WTMF', 0) + weight_to_move


def apply_iwy_safety_valve(targets, state, asset_trends, vix):
    if state == "EXTREME_ACCUMULATION" or targets.get('IWY', 0) <= 0:
        return
    if asset_trends.get('IWY', False):
        severity = 0.5
        if vix is not None and vix > VIX_PANIC:
            severity = 0.8
        cut_amount = targets['IWY'] * severity
        targets['IWY'] -= cut_amount
        targets['WTMF'] = targets.get('WTMF', 0) + cut_amount


def apply_gold_filter(targets, gold_bear):
    if gold_bear and targets.get('GSD.SI', 0) > 0:
        cut_amount = targets['GSD.SI']
        targets['GSD.SI'] -= cut_amount
        targets['WTMF'] = targets.get('WTMF', 0) + cut_amount


def apply_momentum_intensity(targets, state, momentum_scores):
    """
    优化1: 动量强度分层配置
    根据价格距离MA的幅度分层调整权重，而非简单的二元判断
    momentum_scores: dict {ticker: score} where score = (price - ma) / ma
    """
    if state == "EXTREME_ACCUMULATION" or not momentum_scores:
        return
    
    # IWY动量强度调整
    iwy_score = momentum_scores.get('IWY')
    if iwy_score is not None and targets.get('IWY', 0) > 0:
        if iwy_score < (MOMENTUM_WEAK_THRESHOLD - 1):
            # 弱势区：已由趋势熔断处理，这里不重复
            pass
        elif iwy_score < (MOMENTUM_STRONG_THRESHOLD - 1):
            # 中性区 (-5% ~ +5%)：减仓一部分
            reduction = targets['IWY'] * MOMENTUM_NEUTRAL_REDUCTION
            targets['IWY'] -= reduction
            targets['WTMF'] = targets.get('WTMF', 0) + reduction


def apply_sahm_early_warning(targets, state, sahm):
    """
    优化2: Sahm Rule 预警增强
    在Sahm 0.30-0.50区间提前减仓，而非等到0.50才触发
    """
    if state not in ["NEUTRAL", "CAUTIOUS_VOL"] or sahm is None:
        return
    
    if SAHM_EARLY_WARNING_LO <= sahm < SAHM_EARLY_WARNING_HI:
        # 线性减仓: 0.30时减0%, 0.50时减50%
        reduction_pct = (sahm - SAHM_EARLY_WARNING_LO) / (SAHM_EARLY_WARNING_HI - SAHM_EARLY_WARNING_LO) * SAHM_REDUCTION_RATE
        iwy_current = targets.get('IWY', 0)
        if iwy_current > 0:
            move_amt = iwy_current * reduction_pct
            targets['IWY'] = iwy_current - move_amt
            targets['WTMF'] = targets.get('WTMF', 0) + move_amt


def apply_yield_curve_uninvert_protection(targets, state, yield_curve, yc_recently_inverted):
    """
    优化3: 收益率曲线解倒挂后延保护
    收益率曲线从负转正后12个月内保持防御配置
    yc_recently_inverted: bool, 过去12个月内是否曾深度倒挂
    """
    if state not in ["NEUTRAL", "CAUTIOUS_VOL"]:
        return
    
    # 当前曲线已转正但近期曾倒挂 -> 保护期
    if yield_curve is not None and yield_curve > 0 and yc_recently_inverted:
        iwy_current = targets.get('IWY', 0)
        if iwy_current > 0:
            move_amt = iwy_current * YC_UNINVERT_REDUCTION
            targets['IWY'] = iwy_current - move_amt
            targets['MBH.SI'] = targets.get('MBH.SI', 0) + move_amt * 0.5
            targets['WTMF'] = targets.get('WTMF', 0) + move_amt * 0.5


def apply_vix_mean_reversion(targets, state, vix, vix_recent_peak):
    """
    优化4: VIX均值回归加仓
    VIX从高位回落时触发温和加仓
    vix_recent_peak: 近期VIX最高值
    """
    if state not in ["NEUTRAL", "CAUTIOUS_VOL"] or vix is None or vix_recent_peak is None:
        return
    
    # 条件: 近期峰值>25，当前VIX已回落超过20%
    if vix_recent_peak >= VIX_MEAN_REVERSION_PEAK and vix < vix_recent_peak * VIX_MEAN_REVERSION_RATIO:
        # 从WTMF转移到IWY
        wtmf_current = targets.get('WTMF', 0)
        if wtmf_current > VIX_MEAN_REVERSION_BOOST:
            targets['WTMF'] = wtmf_current - VIX_MEAN_REVERSION_BOOST
            targets['IWY'] = targets.get('IWY', 0) + VIX_MEAN_REVERSION_BOOST


def apply_correlation_adjustment(targets, state, corr):
    """
    优化5: 相关性动态再配置（v1.5 渐进响应）
    股债相关性上升时渐进增配非相关资产
    """
    if state not in ["NEUTRAL", "CAUTIOUS_VOL", "CAUTIOUS_TREND"] or corr is None:
        return
    
    if corr > CORR_MID_THRESHOLD:
        # 渐进式调整：0.15-0.30区间线性增加调整幅度
        adjustment_pct = min((corr - CORR_MID_THRESHOLD) / (CORR_HIGH_THRESHOLD - CORR_MID_THRESHOLD), 1.0)
        realloc = adjustment_pct * CORR_MAX_REALLOC
        
        mbh_current = targets.get('MBH.SI', 0)
        if mbh_current > realloc:
            targets['MBH.SI'] = mbh_current - realloc
            targets['WTMF'] = targets.get('WTMF', 0) + realloc * 0.7
            targets['GSD.SI'] = targets.get('GSD.SI', 0) + realloc * 0.3  # 部分转黄金


def apply_cash_buffer(targets, state, vix):
    """
    优化6: 现金缓冲机制
    在不确定性高的时期保留战术现金
    """
    if state == "EXTREME_ACCUMULATION":
        return  # 抄底模式不留现金
    
    cash_buffer = CASH_BUFFER_BASE
    if vix is not None and vix > CASH_BUFFER_VIX_THRESHOLD:
        # VIX每升高5点，现金增加一定比例
        extra_cash = min((vix - CASH_BUFFER_VIX_THRESHOLD) / 5 * CASH_BUFFER_VIX_SCALE, 
                         CASH_BUFFER_MAX - CASH_BUFFER_BASE)
        cash_buffer = CASH_BUFFER_BASE + extra_cash
    
    # 按比例缩减所有资产
    if cash_buffer > 0:
        scale = 1 - cash_buffer
        for asset in targets:
            targets[asset] *= scale


def apply_dual_ma_trend_filter(targets, state, dual_ma_signals):
    """
    优化7: 双均线趋势确认
    使用50日和200日均线判断趋势强度
    dual_ma_signals: dict {ticker: 'STRONG_BEAR'|'WEAK_BEAR'|'BULLISH'}
    """
    if state == "EXTREME_ACCUMULATION" or not dual_ma_signals:
        return
    
    for asset, signal in dual_ma_signals.items():
        if asset not in targets or targets.get(asset, 0) <= 0:
            continue
        
        weight = targets[asset]
        if signal == "STRONG_BEAR":
            # 强熊市：大幅减仓
            cut_amount = weight * STRONG_BEAR_REDUCTION
            targets[asset] = weight - cut_amount
            targets['WTMF'] = targets.get('WTMF', 0) + cut_amount
        elif signal == "WEAK_BEAR":
            # 弱熊市（可能是回调）：小幅减仓
            cut_amount = weight * WEAK_BEAR_REDUCTION
            targets[asset] = weight - cut_amount
            targets['WTMF'] = targets.get('WTMF', 0) + cut_amount


def apply_market_breadth_adjustment(targets, state, breadth_score):
    """
    优化8: 跨资产动量（市场广度）
    根据整体市场动量调整权益配置
    breadth_score: 0-1，表示有多少比例的资产处于上升趋势
    """
    if state == "EXTREME_ACCUMULATION" or breadth_score is None:
        return
    
    reduction = 0
    if breadth_score < MARKET_BREADTH_LOW:
        # 市场广度很差，整体减仓权益
        reduction = BREADTH_LOW_REDUCTION
    elif breadth_score < MARKET_BREADTH_MID:
        # 市场广度一般，小幅减仓
        reduction = BREADTH_MID_REDUCTION
    
    if reduction > 0:
        # 只减仓高风险权益资产
        risk_assets = ['IWY', 'G3B.SI']
        total_cut = 0
        for asset in risk_assets:
            if targets.get(asset, 0) > 0:
                cut_amount = targets[asset] * reduction
                targets[asset] -= cut_amount
                total_cut += cut_amount
        
        # 差额补到WTMF
        if total_cut > 0:
            targets['WTMF'] = targets.get('WTMF', 0) + total_cut


def apply_trend_boost(targets, state, momentum_scores, vix):
    """
    v2.0: 趋势驱动的成长↔WTMF动态轮换（IWY核心优化）
    强牛市: 最大化IWY，但保留LVHI波动缓冲
    强熊市: 增加WTMF+LVHI对冲，但保持IWY核心
    """
    if state not in ["NEUTRAL", "CAUTIOUS_VOL"] or not momentum_scores or vix is None:
        return
    
    iwy_score = momentum_scores.get('IWY')
    if iwy_score is None:
        return
    
    # 计算价格相对MA的偏离度 (score = (price - ma) / ma)
    price_vs_ma = iwy_score + 1  # 转换为 price/ma 比值
    
    if price_vs_ma >= TREND_STRONG_BULL:
        # 强牛市 (>10%): 最大化IWY，但保留LVHI作为波动缓冲
        if vix < 18:
            # 从WTMF/MBH转移，但保留LVHI最低配置
            boost_from_wtmf = max(targets.get('WTMF', 0) - WTMF_BASE_ALLOCATION, 0)
            boost_from_mbh = targets.get('MBH.SI', 0) * 0.6
            
            total_boost = boost_from_wtmf + boost_from_mbh
            targets['WTMF'] = max(targets.get('WTMF', 0) - boost_from_wtmf, WTMF_BASE_ALLOCATION)
            targets['MBH.SI'] = targets.get('MBH.SI', 0) - boost_from_mbh
            targets['IWY'] = targets.get('IWY', 0) + total_boost
            
            # 确保LVHI不低于最低配置
            if targets.get('LVHI', 0) < LVHI_MIN_ALLOCATION:
                diff = LVHI_MIN_ALLOCATION - targets.get('LVHI', 0)
                targets['LVHI'] = LVHI_MIN_ALLOCATION
                targets['IWY'] = targets.get('IWY', 0) - diff
            
    elif price_vs_ma >= TREND_MILD_BULL:
        # 温和牛市 (2-10%): 适度倾斜IWY
        boost_amt = min(targets.get('WTMF', 0) - WTMF_BASE_ALLOCATION, BULL_IWY_BOOST)
        if boost_amt > 0:
            targets['WTMF'] = targets.get('WTMF', 0) - boost_amt
            targets['IWY'] = targets.get('IWY', 0) + boost_amt
            
    elif price_vs_ma < TREND_MILD_BEAR:
        # 温和熊市 (<-3%): IWY减仓，但分散到LVHI和WTMF
        shift_amt = min(targets.get('IWY', 0) * 0.15, BEAR_WTMF_BOOST)
        if shift_amt > 0:
            targets['IWY'] -= shift_amt
            # v2.0: 更多转向LVHI（红利抗跌），而非全部WTMF
            targets['LVHI'] = targets.get('LVHI', 0) + shift_amt * LVHI_VOL_BUFFER_RATIO
            targets['WTMF'] = targets.get('WTMF', 0) + shift_amt * (1 - LVHI_VOL_BUFFER_RATIO)


def apply_value_rotation(targets, state, momentum_scores):
    """
    v1.7 新增: 红利相对强弱轮换
    当红利相对成长跑赢时，增配红利；反之增配成长
    """
    if state not in ["NEUTRAL", "CAUTIOUS_VOL"] or not momentum_scores:
        return
    
    iwy_score = momentum_scores.get('IWY')
    lvhi_score = momentum_scores.get('LVHI')
    
    if iwy_score is None or lvhi_score is None:
        return
    
    # 计算相对强弱 (LVHI相对IWY的超额收益)
    relative_strength = lvhi_score - iwy_score
    
    if relative_strength > VALUE_OUTPERFORM_THRESHOLD:
        # 红利跑赢: 从成长转向红利
        shift_amt = min(targets.get('IWY', 0) * 0.3, VALUE_ROTATION_AMOUNT)
        if shift_amt > 0:
            targets['IWY'] -= shift_amt
            targets['LVHI'] = targets.get('LVHI', 0) + shift_amt
            
    elif relative_strength < VALUE_UNDERPERFORM_THRESHOLD:
        # 成长跑赢: 从红利转向成长（但保持LVHI最低配置）
        max_shift = targets.get('LVHI', 0) - LVHI_MIN_ALLOCATION
        shift_amt = min(max_shift, VALUE_ROTATION_AMOUNT)
        if shift_amt > 0:
            targets['LVHI'] -= shift_amt
            targets['IWY'] = targets.get('IWY', 0) + shift_amt


def apply_volatility_control(targets, state, vix, iwy_volatility=None):
    """
    v2.0 新增: IWY波动率自适应控制
    根据IWY自身波动率动态调整仓位，高波动时减仓到LVHI/WTMF
    
    参数:
    - iwy_volatility: IWY的20日年化波动率
    """
    if state == "EXTREME_ACCUMULATION" or iwy_volatility is None:
        return
    
    iwy_weight = targets.get('IWY', 0)
    if iwy_weight <= 0:
        return
    
    # 波动率自适应调整
    if iwy_volatility > IWY_VOL_HIGH:
        # 高波动: IWY减仓，分散到LVHI和WTMF
        excess_vol_ratio = min((iwy_volatility - IWY_VOL_HIGH) / (0.35 - IWY_VOL_HIGH), 1.0)
        reduction = iwy_weight * IWY_VOL_ADJUST_MAX * excess_vol_ratio
        
        targets['IWY'] -= reduction
        # 高波动时优先转WTMF（危机对冲），其次LVHI
        targets['WTMF'] = targets.get('WTMF', 0) + reduction * 0.6
        targets['LVHI'] = targets.get('LVHI', 0) + reduction * 0.4
        
    elif iwy_volatility < IWY_VOL_LOW and state == "NEUTRAL":
        # 低波动且牛市: 可以从WTMF/LVHI转回IWY
        low_vol_ratio = min((IWY_VOL_LOW - iwy_volatility) / (IWY_VOL_LOW - 0.10), 1.0)
        boost = IWY_VOL_ADJUST_MAX * 0.5 * low_vol_ratio  # 加仓幅度较小
        
        # 从WTMF转（保持最低WTMF配置）
        wtmf_available = max(targets.get('WTMF', 0) - WTMF_BASE_ALLOCATION, 0)
        from_wtmf = min(wtmf_available, boost * 0.7)
        
        if from_wtmf > 0:
            targets['WTMF'] -= from_wtmf
            targets['IWY'] += from_wtmf


def apply_wtmf_vol_scaling(targets, state, vix):
    """
    v2.0 新增: WTMF危机Alpha动态配置
    VIX驱动的WTMF配置，作为组合波动率缓冲器
    """
    if state == "EXTREME_ACCUMULATION" or vix is None:
        return
    
    current_wtmf = targets.get('WTMF', 0)
    
    if vix > WTMF_VOL_TRIGGER:
        # VIX高于触发阈值，增配WTMF
        vol_excess_ratio = min((vix - WTMF_VOL_TRIGGER) / (40 - WTMF_VOL_TRIGGER), 1.0)
        target_wtmf = WTMF_BASE_ALLOCATION + (WTMF_MAX_ALLOCATION - WTMF_BASE_ALLOCATION) * vol_excess_ratio
        
        if target_wtmf > current_wtmf:
            # 从IWY转移到WTMF（保持IWY核心地位）
            iwy_current = targets.get('IWY', 0)
            transfer_needed = target_wtmf - current_wtmf
            transfer_from_iwy = min(iwy_current * 0.20, transfer_needed)  # 最多从IWY转20%
            
            if transfer_from_iwy > 0:
                targets['IWY'] -= transfer_from_iwy
                targets['WTMF'] = current_wtmf + transfer_from_iwy
    
    # 确保WTMF不低于基础配置（危机保险）
    if targets.get('WTMF', 0) < WTMF_BASE_ALLOCATION and state != "EXTREME_ACCUMULATION":
        shortfall = WTMF_BASE_ALLOCATION - targets.get('WTMF', 0)
        # 从MBH或GSD补充
        for source in ['MBH.SI', 'GSD.SI']:
            if shortfall <= 0:
                break
            available = targets.get(source, 0) * 0.5
            transfer = min(available, shortfall)
            targets[source] = targets.get(source, 0) - transfer
            targets['WTMF'] = targets.get('WTMF', 0) + transfer
            shortfall -= transfer


def apply_lvhi_floor(targets, state):
    """
    v2.0 新增: LVHI波动缓冲最低配置
    确保LVHI（红利）作为波动缓冲器的最低配置
    """
    if state == "EXTREME_ACCUMULATION":
        return
    
    lvhi_current = targets.get('LVHI', 0)
    if lvhi_current < LVHI_MIN_ALLOCATION:
        shortfall = LVHI_MIN_ALLOCATION - lvhi_current
        # 优先从MBH补充，其次从GSD
        for source in ['MBH.SI', 'GSD.SI', 'SRT.SI']:
            if shortfall <= 0:
                break
            available = targets.get(source, 0) * 0.5
            transfer = min(available, shortfall)
            targets[source] = targets.get(source, 0) - transfer
            targets['LVHI'] = targets.get('LVHI', 0) + transfer
            shortfall -= transfer


def apply_momentum_acceleration(targets, state, momentum_scores, vix):
    """
    v2.1 新增: 动量加速因子
    强趋势+低波动时，额外加仓IWY
    """
    if not MOMENTUM_ACCELERATION_ENABLED:
        return
    if state != "NEUTRAL" or not momentum_scores or vix is None:
        return
    
    iwy_score = momentum_scores.get('IWY')
    if iwy_score is None:
        return
    
    price_vs_ma = iwy_score + 1  # price/ma 比值
    
    # 强趋势(>5%) + 低波动(VIX<18) = 动量加速
    if price_vs_ma >= MOMENTUM_ACCEL_THRESHOLD and vix < 18:
        # 计算加速幅度（趋势越强，加速越多）
        excess_momentum = price_vs_ma - MOMENTUM_ACCEL_THRESHOLD
        accel_ratio = min(excess_momentum / 0.05, 1.0)  # 5%-10%线性加速
        boost = MOMENTUM_ACCEL_BOOST * accel_ratio
        
        # 从低收益资产转移
        sources = [('MBH.SI', 0.6), ('GSD.SI', 0.4), ('G3B.SI', 0.3)]
        total_transfer = 0
        for asset, max_pct in sources:
            if boost <= 0:
                break
            available = targets.get(asset, 0) * max_pct
            transfer = min(available, boost)
            targets[asset] = targets.get(asset, 0) - transfer
            total_transfer += transfer
            boost -= transfer
        
        targets['IWY'] = targets.get('IWY', 0) + total_transfer


def apply_vix_reversion_acceleration(targets, state, vix, vix_recent_peak):
    """
    v2.1 新增: VIX均值回归加速
    VIX从峰值大幅回落时，加速加仓
    """
    if state not in ["NEUTRAL", "CAUTIOUS_VOL"] or vix is None or vix_recent_peak is None:
        return
    if vix_recent_peak < VIX_MEAN_REVERSION_PEAK:
        return
    
    # 计算回落比例
    reversion_ratio = (vix_recent_peak - vix) / vix_recent_peak
    
    if reversion_ratio >= VIX_REVERSION_ACCEL_THRESHOLD:
        # VIX回落超过35%，触发加速加仓
        excess_reversion = reversion_ratio - VIX_REVERSION_ACCEL_THRESHOLD
        accel_ratio = min(excess_reversion / 0.15, 1.0)  # 35%-50%线性加速
        boost = VIX_REVERSION_ACCEL_BOOST * accel_ratio
        
        # 从WTMF转移到IWY（恐慌结束，减少对冲）
        wtmf_available = max(targets.get('WTMF', 0) - WTMF_BASE_ALLOCATION, 0)
        transfer = min(wtmf_available, boost)
        
        if transfer > 0:
            targets['WTMF'] -= transfer
            targets['IWY'] = targets.get('IWY', 0) + transfer


def apply_iwy_cap(targets, state, momentum_scores, vix):
    """
    v2.1 新增: NEUTRAL状态IWY动态上限
    牛市中放开IWY上限，允许更高配置
    """
    if state != "NEUTRAL":
        return
    
    iwy_current = targets.get('IWY', 0)
    
    # 判断是否为强牛市
    is_strong_bull = False
    if momentum_scores and vix is not None:
        iwy_score = momentum_scores.get('IWY')
        if iwy_score is not None:
            price_vs_ma = iwy_score + 1
            is_strong_bull = (vix < 15 and price_vs_ma > 1.05)
    
    cap = NEUTRAL_IWY_CAP_BULL if is_strong_bull else NEUTRAL_IWY_CAP_NORMAL
    
    if iwy_current > cap:
        excess = iwy_current - cap
        targets['IWY'] = cap
        # 超出部分分配到LVHI
        targets['LVHI'] = targets.get('LVHI', 0) + excess


# === v3.0 新增优化函数 ===

def calculate_tsmom(price_series, lookback_fast=21, lookback_slow=126):
    """
    时序动量信号：结合短期和长期动量
    返回: -1 (空头), 0 (中性), +1 (多头)
    """
    if not TSMOM_ENABLED:
        return 0
    if price_series is None or len(price_series) < lookback_slow:
        return 0
    
    try:
        # 计算收益率动量
        ret_fast = (price_series.iloc[-1] / price_series.iloc[-lookback_fast] - 1)
        ret_slow = (price_series.iloc[-1] / price_series.iloc[-lookback_slow] - 1)
        
        # 加权组合
        combo_momentum = TSMOM_COMBO_WEIGHT * ret_fast + (1 - TSMOM_COMBO_WEIGHT) * ret_slow
        
        if combo_momentum > TSMOM_THRESHOLD:
            return 1
        elif combo_momentum < -TSMOM_THRESHOLD:
            return -1
        return 0
    except Exception:
        return 0


def apply_tsmom_boost(targets, state, tsmom_signal, vix):
    """
    时序动量增强：趋势强时更激进配置IWY
    - 正动量+低波动: 从MBH/GSD/WTMF转移到IWY
    - 负动量: 从IWY转移到WTMF
    """
    if not TSMOM_ENABLED:
        return
    if state not in ["NEUTRAL", "CAUTIOUS_VOL"] or tsmom_signal == 0 or vix is None:
        return
    
    if tsmom_signal == 1 and vix < 20:
        # 强动量+低波动：额外加仓IWY
        boost_sources = [('MBH.SI', 0.5), ('GSD.SI', 0.3), ('WTMF', 0.5)]
        total_boost = 0
        remaining_boost = TSMOM_BOOST_AMOUNT
        
        for asset, max_pct in boost_sources:
            if remaining_boost <= 0:
                break
            available = targets.get(asset, 0) * max_pct
            # 保持WTMF基础配置
            if asset == 'WTMF':
                available = max(targets.get('WTMF', 0) - WTMF_BASE_ALLOCATION, 0)
            transfer = min(available, remaining_boost)
            if transfer > 0:
                targets[asset] = targets.get(asset, 0) - transfer
                total_boost += transfer
                remaining_boost -= transfer
        
        targets['IWY'] = targets.get('IWY', 0) + total_boost
    
    elif tsmom_signal == -1:
        # 负动量：从IWY转移到WTMF（防御）
        iwy_cut = min(targets.get('IWY', 0) * 0.15, 0.10)
        if iwy_cut > 0:
            targets['IWY'] -= iwy_cut
            targets['WTMF'] = targets.get('WTMF', 0) + iwy_cut


def detect_vol_spike(vix_history, threshold=None):
    """检测VIX尖峰（波动率聚集的领先信号）"""
    if not TAIL_RISK_ENABLED:
        return False
    if vix_history is None or len(vix_history) < 2:
        return False
    
    threshold = threshold or TAIL_RISK_VIX_SPIKE
    try:
        daily_change = vix_history[-1] - vix_history[-2]
        return daily_change > threshold
    except Exception:
        return False


def apply_tail_risk_control(targets, state, vix_history, days_since_spike=None):
    """
    尾部风险控制：VIX急涨后的预防性减仓
    - 避免在恐慌初期被动止损
    - 分批恢复仓位
    
    返回: 更新后的 days_since_spike
    """
    if not TAIL_RISK_ENABLED:
        return days_since_spike
    if state == "EXTREME_ACCUMULATION":
        return days_since_spike  # 抄底模式不减仓
    
    if detect_vol_spike(vix_history):
        # 触发尾部风险减仓
        risk_assets = ['IWY', 'G3B.SI', 'LVHI']
        for asset in risk_assets:
            current_weight = targets.get(asset, 0)
            if current_weight > 0:
                reduction = current_weight * TAIL_RISK_REDUCTION
                targets[asset] -= reduction
                targets['WTMF'] = targets.get('WTMF', 0) + reduction
        return 0  # 重置恢复计数
    
    # 分批恢复逻辑（在调用端处理）
    if days_since_spike is not None and days_since_spike < TAIL_RISK_RECOVERY_DAYS:
        return days_since_spike + 1
    
    return None


def should_rebalance_smart(current_weights, target_weights, expected_returns=None, cost_bps=None):
    """
    智能再平衡：只在预期收益超过成本时才调仓
    expected_returns: dict {ticker: expected_daily_return}
    返回: (should_rebal, reason)
    """
    if not SMART_REBAL_ENABLED:
        return True, "智能再平衡已禁用"
    
    cost_bps = cost_bps or REBAL_COST_BPS
    expected_returns = expected_returns or {}
    
    total_deviation = 0
    expected_gain = 0
    
    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    
    for asset in all_assets:
        curr_w = current_weights.get(asset, 0)
        tgt_w = target_weights.get(asset, 0)
        deviation = tgt_w - curr_w
        total_deviation += abs(deviation)
        
        # 预期收益 = 偏离度 × 预期超额收益
        exp_ret = expected_returns.get(asset, REBAL_EXPECTED_ALPHA)
        expected_gain += deviation * exp_ret
    
    trading_cost = total_deviation * cost_bps / 10000
    one_side_turnover = total_deviation / 2
    
    # 硬阈值强制再平衡
    if one_side_turnover > SMART_REBAL_HARD_THRESHOLD:
        return True, f"单边换手{one_side_turnover*100:.1f}%超过硬阈值{SMART_REBAL_HARD_THRESHOLD*100:.0f}%"
    
    # 收益/成本比判断
    if trading_cost > 0:
        edge_ratio = expected_gain / trading_cost
        if edge_ratio > REBAL_MIN_EDGE:
            return True, f"收益/成本比{edge_ratio:.1f}>{REBAL_MIN_EDGE:.0f}"
    
    # 原有阈值作为备用
    if one_side_turnover > REBALANCE_THRESHOLD:
        return True, f"偏离{one_side_turnover*100:.1f}%超过阈值{REBALANCE_THRESHOLD*100:.0f}%"
    
    return False, f"偏离{one_side_turnover*100:.1f}%未达阈值，暂不调仓"


def get_smoothed_state(current_state, prev_state, prev_confidence, alpha=None):
    """
    指数平滑状态转换：减少噪音但保持响应性
    返回: (state, confidence)
    """
    if not PROB_STATE_ENABLED:
        return current_state, 1.0
    
    alpha = alpha or STATE_SMOOTHING_ALPHA
    
    if current_state == prev_state:
        # 同状态，增强置信度
        new_confidence = min(1.0, prev_confidence + alpha * (1 - prev_confidence))
    else:
        # 状态切换，降低置信度
        new_confidence = alpha  # 重置为初始置信度
    
    # 特例：EXTREME_ACCUMULATION 不受限制（快速响应抄底机会）
    if current_state in STATE_HYSTERESIS_EXCEPTION:
        return current_state, new_confidence
    
    # 只在置信度足够时才切换状态
    if current_state != prev_state and new_confidence < STATE_CONFIDENCE_THRESHOLD:
        return prev_state, new_confidence  # 维持原状态
    
    return current_state, new_confidence


def calculate_risk_parity_weights(returns_df, target_vol=None, min_weight=None):
    """
    逆波动率加权的简化风险平价
    returns_df: 各资产日收益率DataFrame
    返回: dict {ticker: weight}
    """
    if not RISK_PARITY_BLEND_ENABLED:
        return {}
    if returns_df is None or returns_df.empty:
        return {}
    
    target_vol = target_vol or RISK_PARITY_TARGET_VOL
    min_weight = min_weight or RISK_PARITY_MIN_WEIGHT
    
    try:
        vol = returns_df.std() * np.sqrt(252)  # 年化波动率
        inv_vol = 1 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], 0)
        inv_vol = inv_vol.fillna(0)
        
        # 逆波动率加权
        if inv_vol.sum() <= 0:
            return {}
        
        raw_weights = inv_vol / inv_vol.sum()
        
        # 应用最低权重约束
        weights = raw_weights.clip(lower=min_weight)
        weights = weights / weights.sum()
        
        return weights.to_dict()
    except Exception:
        return {}


def blend_with_risk_parity(strategic_weights, risk_parity_weights, blend_ratio=None):
    """
    策略权重与风险平价权重混合
    blend_ratio: 风险平价的混合比例
    """
    if not RISK_PARITY_BLEND_ENABLED or not risk_parity_weights:
        return strategic_weights
    
    blend_ratio = blend_ratio or RISK_PARITY_BLEND_RATIO
    blended = {}
    all_assets = set(strategic_weights.keys()) | set(risk_parity_weights.keys())
    
    for asset in all_assets:
        strat_w = strategic_weights.get(asset, 0)
        rp_w = risk_parity_weights.get(asset, 0)
        blended[asset] = (1 - blend_ratio) * strat_w + blend_ratio * rp_w
    
    # 归一化
    total = sum(blended.values())
    if total > 0:
        return {k: v / total for k, v in blended.items()}
    return strategic_weights


# === 组合优化器 (Portfolio Optimizer) ===

def portfolio_optimizer_metrics(weights, mean_returns, cov_matrix):
    """
    计算组合的收益、波动率和夏普比率
    weights: numpy array
    mean_returns: 年化收益率
    cov_matrix: 协方差矩阵
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe


def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.04, 
                        min_weight=0.0, max_weight=1.0, n_iterations=10000):
    """
    最大夏普比率优化 (蒙特卡洛模拟 + 梯度优化)
    使用无外部依赖的方法
    """
    n_assets = len(mean_returns)
    
    best_sharpe = -np.inf
    best_weights = np.ones(n_assets) / n_assets
    
    # 蒙特卡洛随机搜索
    for _ in range(n_iterations):
        weights = np.random.random(n_assets)
        weights = weights / weights.sum()
        
        # 应用权重约束
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()
        
        port_return, port_vol, sharpe = portfolio_optimizer_metrics(
            weights, mean_returns, cov_matrix
        )
        
        excess_return = port_return - risk_free_rate
        adj_sharpe = excess_return / port_vol if port_vol > 0 else 0
        
        if adj_sharpe > best_sharpe:
            best_sharpe = adj_sharpe
            best_weights = weights.copy()
    
    return best_weights, best_sharpe


def optimize_min_volatility(mean_returns, cov_matrix, 
                            min_weight=0.0, max_weight=1.0, n_iterations=10000):
    """
    最小波动率优化
    """
    n_assets = len(mean_returns)
    
    best_vol = np.inf
    best_weights = np.ones(n_assets) / n_assets
    
    for _ in range(n_iterations):
        weights = np.random.random(n_assets)
        weights = weights / weights.sum()
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()
        
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        if port_vol < best_vol:
            best_vol = port_vol
            best_weights = weights.copy()
    
    return best_weights, best_vol


def optimize_max_return_given_vol(mean_returns, cov_matrix, target_vol=0.12,
                                   min_weight=0.0, max_weight=1.0, n_iterations=10000):
    """
    给定波动率下最大化收益
    """
    n_assets = len(mean_returns)
    
    best_return = -np.inf
    best_weights = np.ones(n_assets) / n_assets
    tolerance = 0.02  # 允许2%误差
    
    for _ in range(n_iterations):
        weights = np.random.random(n_assets)
        weights = weights / weights.sum()
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()
        
        port_return, port_vol, _ = portfolio_optimizer_metrics(
            weights, mean_returns, cov_matrix
        )
        
        # 只接受波动率在目标附近的组合
        if abs(port_vol - target_vol) <= tolerance:
            if port_return > best_return:
                best_return = port_return
                best_weights = weights.copy()
    
    return best_weights, best_return


def calculate_efficient_frontier(mean_returns, cov_matrix, n_points=50, 
                                  min_weight=0.0, max_weight=1.0):
    """
    计算有效前沿
    返回: list of (volatility, return, weights)
    """
    n_assets = len(mean_returns)
    
    # 先找到最小波动率和最大收益的边界
    min_vol_weights, min_vol = optimize_min_volatility(
        mean_returns, cov_matrix, min_weight, max_weight, 5000
    )
    min_vol_ret = np.dot(min_vol_weights, mean_returns)
    
    # 最大收益（全押最高收益资产，受max_weight限制）
    max_ret_idx = np.argmax(mean_returns)
    max_possible_ret = mean_returns[max_ret_idx]
    
    # 生成目标收益序列
    target_returns = np.linspace(min_vol_ret, max_possible_ret * 0.95, n_points)
    
    frontier = []
    for target_ret in target_returns:
        best_vol = np.inf
        best_weights = None
        
        # 搜索满足目标收益的最小波动率组合
        for _ in range(3000):
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()
            
            port_return, port_vol, _ = portfolio_optimizer_metrics(
                weights, mean_returns, cov_matrix
            )
            
            # 收益需要接近目标
            if abs(port_return - target_ret) <= 0.02:
                if port_vol < best_vol:
                    best_vol = port_vol
                    best_weights = weights.copy()
        
        if best_weights is not None:
            frontier.append((best_vol, target_ret, best_weights))
    
    return frontier


def calculate_risk_contribution(weights, cov_matrix):
    """
    计算各资产的风险贡献
    """
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if portfolio_vol == 0:
        return np.zeros(len(weights))
    
    # 边际风险贡献
    marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
    
    # 风险贡献 = 权重 × 边际风险
    risk_contribution = weights * marginal_risk
    
    # 归一化为百分比
    total_risk = risk_contribution.sum()
    if total_risk > 0:
        return risk_contribution / total_risk
    return risk_contribution


def optimize_risk_parity_full(cov_matrix, min_weight=0.02, max_weight=0.50, n_iterations=10000):
    """
    真正的风险平价优化：使各资产风险贡献相等
    """
    n_assets = cov_matrix.shape[0]
    target_risk = 1.0 / n_assets  # 每个资产的目标风险贡献
    
    best_weights = np.ones(n_assets) / n_assets
    best_error = np.inf
    
    for _ in range(n_iterations):
        weights = np.random.random(n_assets)
        weights = weights / weights.sum()
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()
        
        risk_contrib = calculate_risk_contribution(weights, cov_matrix)
        
        # 计算与目标的误差
        error = np.sum((risk_contrib - target_risk) ** 2)
        
        if error < best_error:
            best_error = error
            best_weights = weights.copy()
    
    return best_weights, calculate_risk_contribution(best_weights, cov_matrix)


def run_portfolio_optimization(price_df, tickers, risk_free_rate=0.04, 
                                min_weight=0.0, max_weight=0.50):
    """
    运行完整的组合优化
    price_df: 价格DataFrame (日期为index，ticker为columns)
    tickers: 资产列表
    
    返回: dict with optimization results
    """
    # 计算日收益率
    returns = price_df[tickers].pct_change().dropna()
    
    if len(returns) < 60:  # 至少需要60天数据
        return {"error": "数据不足，需要至少60个交易日"}
    
    # 年化收益和协方差
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    results = {
        "tickers": tickers,
        "mean_returns": mean_returns.to_dict(),
        "volatility": (returns.std() * np.sqrt(252)).to_dict(),
        "correlation": returns.corr().to_dict(),
    }
    
    # 1. 等权重基准
    equal_weights = np.ones(len(tickers)) / len(tickers)
    eq_ret, eq_vol, eq_sharpe = portfolio_optimizer_metrics(
        equal_weights, mean_returns.values, cov_matrix.values
    )
    results["equal_weight"] = {
        "weights": dict(zip(tickers, equal_weights)),
        "return": eq_ret,
        "volatility": eq_vol,
        "sharpe": (eq_ret - risk_free_rate) / eq_vol if eq_vol > 0 else 0
    }
    
    # 2. 最大夏普比率
    max_sharpe_w, max_sharpe = optimize_max_sharpe(
        mean_returns.values, cov_matrix.values, risk_free_rate, min_weight, max_weight
    )
    ms_ret, ms_vol, _ = portfolio_optimizer_metrics(
        max_sharpe_w, mean_returns.values, cov_matrix.values
    )
    results["max_sharpe"] = {
        "weights": dict(zip(tickers, max_sharpe_w)),
        "return": ms_ret,
        "volatility": ms_vol,
        "sharpe": max_sharpe
    }
    
    # 3. 最小波动率
    min_vol_w, min_vol = optimize_min_volatility(
        mean_returns.values, cov_matrix.values, min_weight, max_weight
    )
    mv_ret, mv_vol, _ = portfolio_optimizer_metrics(
        min_vol_w, mean_returns.values, cov_matrix.values
    )
    results["min_volatility"] = {
        "weights": dict(zip(tickers, min_vol_w)),
        "return": mv_ret,
        "volatility": mv_vol,
        "sharpe": (mv_ret - risk_free_rate) / mv_vol if mv_vol > 0 else 0
    }
    
    # 4. 风险平价
    rp_weights, rp_risk_contrib = optimize_risk_parity_full(
        cov_matrix.values, max(min_weight, 0.02), max_weight
    )
    rp_ret, rp_vol, _ = portfolio_optimizer_metrics(
        rp_weights, mean_returns.values, cov_matrix.values
    )
    results["risk_parity"] = {
        "weights": dict(zip(tickers, rp_weights)),
        "return": rp_ret,
        "volatility": rp_vol,
        "sharpe": (rp_ret - risk_free_rate) / rp_vol if rp_vol > 0 else 0,
        "risk_contribution": dict(zip(tickers, rp_risk_contrib))
    }
    
    # 5. 目标波动率12%
    target_vol_w, target_vol_ret = optimize_max_return_given_vol(
        mean_returns.values, cov_matrix.values, target_vol=0.12, 
        min_weight=min_weight, max_weight=max_weight
    )
    tv_ret, tv_vol, _ = portfolio_optimizer_metrics(
        target_vol_w, mean_returns.values, cov_matrix.values
    )
    results["target_vol_12"] = {
        "weights": dict(zip(tickers, target_vol_w)),
        "return": tv_ret,
        "volatility": tv_vol,
        "sharpe": (tv_ret - risk_free_rate) / tv_vol if tv_vol > 0 else 0
    }
    
    # 6. 有效前沿
    frontier = calculate_efficient_frontier(
        mean_returns.values, cov_matrix.values, n_points=30, 
        min_weight=min_weight, max_weight=max_weight
    )
    results["efficient_frontier"] = [
        {"volatility": f[0], "return": f[1]} for f in frontier
    ]
    
    return results


def render_optimization_results(opt_results, price_df, tickers):
    """
    渲染优化结果的UI
    """
    if "error" in opt_results:
        st.error(opt_results["error"])
        return
    
    st.markdown("### 📊 资产统计")
    
    # 资产收益/波动率表
    stats_data = []
    for t in tickers:
        stats_data.append({
            "资产": t,
            "年化收益": f"{opt_results['mean_returns'].get(t, 0)*100:.1f}%",
            "年化波动": f"{opt_results['volatility'].get(t, 0)*100:.1f}%",
            "收益/波动": f"{opt_results['mean_returns'].get(t, 0)/opt_results['volatility'].get(t, 1):.2f}"
        })
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    # 相关性矩阵
    with st.expander("🔗 相关性矩阵", expanded=False):
        corr_df = pd.DataFrame(opt_results['correlation'])
        st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn_r', vmin=-1, vmax=1).format("{:.2f}"))
    
    st.markdown("### 🎯 优化结果对比")
    
    # 优化方案对比表
    strategies = [
        ("equal_weight", "等权重", "📊"),
        ("max_sharpe", "最大夏普", "⭐"),
        ("min_volatility", "最小波动", "🛡️"),
        ("risk_parity", "风险平价", "⚖️"),
        ("target_vol_12", "目标波动12%", "🎯"),
    ]
    
    compare_data = []
    for key, name, icon in strategies:
        if key in opt_results:
            res = opt_results[key]
            compare_data.append({
                "策略": f"{icon} {name}",
                "年化收益": f"{res['return']*100:.1f}%",
                "年化波动": f"{res['volatility']*100:.1f}%",
                "夏普比率": f"{res['sharpe']:.2f}",
            })
    
    st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)
    
    # 最优配置推荐
    st.markdown("### ⭐ 推荐配置")
    
    # 选择推荐策略
    best_strategy = "max_sharpe"
    if opt_results.get("max_sharpe", {}).get("sharpe", 0) < opt_results.get("risk_parity", {}).get("sharpe", 0):
        best_strategy = "risk_parity"
    
    rec = opt_results.get(best_strategy, opt_results.get("equal_weight", {}))
    rec_weights = rec.get("weights", {})
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**推荐策略: {best_strategy.replace('_', ' ').title()}**")
        st.metric("预期年化收益", f"{rec.get('return', 0)*100:.1f}%")
        st.metric("预期年化波动", f"{rec.get('volatility', 0)*100:.1f}%")
        st.metric("夏普比率", f"{rec.get('sharpe', 0):.2f}")
    
    with col2:
        # 配置饼图
        fig = go.Figure(data=[go.Pie(
            labels=list(rec_weights.keys()),
            values=[v*100 for v in rec_weights.values()],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )])
        fig.update_layout(
            title="推荐配置比例",
            height=300,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 详细权重表
    st.markdown("### 📋 各策略详细权重")
    
    weights_df_data = {"资产": tickers}
    for key, name, _ in strategies:
        if key in opt_results:
            weights = opt_results[key].get("weights", {})
            weights_df_data[name] = [f"{weights.get(t, 0)*100:.1f}%" for t in tickers]
    
    st.dataframe(pd.DataFrame(weights_df_data), use_container_width=True, hide_index=True)
    
    # 有效前沿图
    st.markdown("### 📈 有效前沿")
    
    frontier = opt_results.get("efficient_frontier", [])
    if frontier:
        fig = go.Figure()
        
        # 有效前沿线
        fig.add_trace(go.Scatter(
            x=[f["volatility"]*100 for f in frontier],
            y=[f["return"]*100 for f in frontier],
            mode='lines',
            name='有效前沿',
            line=dict(color='blue', width=2)
        ))
        
        # 各策略点
        colors = {'equal_weight': 'gray', 'max_sharpe': 'gold', 
                  'min_volatility': 'green', 'risk_parity': 'purple', 'target_vol_12': 'red'}
        names = {'equal_weight': '等权重', 'max_sharpe': '最大夏普', 
                 'min_volatility': '最小波动', 'risk_parity': '风险平价', 'target_vol_12': '目标波动12%'}
        
        for key in ['equal_weight', 'max_sharpe', 'min_volatility', 'risk_parity', 'target_vol_12']:
            if key in opt_results:
                res = opt_results[key]
                fig.add_trace(go.Scatter(
                    x=[res['volatility']*100],
                    y=[res['return']*100],
                    mode='markers+text',
                    name=names[key],
                    marker=dict(size=12, color=colors[key]),
                    text=[names[key]],
                    textposition='top center'
                ))
        
        # 单个资产点
        for t in tickers:
            fig.add_trace(go.Scatter(
                x=[opt_results['volatility'].get(t, 0)*100],
                y=[opt_results['mean_returns'].get(t, 0)*100],
                mode='markers+text',
                name=t,
                marker=dict(size=8, symbol='diamond'),
                text=[t],
                textposition='bottom center',
                showlegend=False
            ))
        
        fig.update_layout(
            title="风险-收益有效前沿",
            xaxis_title="年化波动率 (%)",
            yaxis_title="年化收益率 (%)",
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 风险贡献图 (风险平价)
    if "risk_parity" in opt_results and "risk_contribution" in opt_results["risk_parity"]:
        st.markdown("### ⚖️ 风险平价 - 风险贡献分解")
        
        rp = opt_results["risk_parity"]
        rc = rp["risk_contribution"]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(rc.keys()),
            y=[v*100 for v in rc.values()],
            name='风险贡献',
            marker_color='purple'
        ))
        fig.add_trace(go.Bar(
            x=list(rp["weights"].keys()),
            y=[v*100 for v in rp["weights"].values()],
            name='权重',
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="风险平价: 权重 vs 风险贡献",
            xaxis_title="资产",
            yaxis_title="百分比 (%)",
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


def get_adaptive_vix_thresholds(vix_history, lookback=None):
    """
    基于滚动历史计算VIX阈值，避免前视偏差
    返回: (panic_threshold, calm_threshold)
    """
    if not ADAPTIVE_VIX_ENABLED:
        return VIX_EXTREME_THRESHOLD, VIX_BOOST_LO
    
    lookback = lookback or VIX_ADAPTIVE_LOOKBACK
    
    if vix_history is None or len(vix_history) < lookback:
        return VIX_EXTREME_THRESHOLD, VIX_BOOST_LO
    
    try:
        rolling_vix = vix_history[-lookback:]
        panic_threshold = np.percentile(rolling_vix, VIX_PANIC_PERCENTILE * 100)
        calm_threshold = np.percentile(rolling_vix, VIX_CALM_PERCENTILE * 100)
        return panic_threshold, calm_threshold
    except Exception:
        return VIX_EXTREME_THRESHOLD, VIX_BOOST_LO


def get_target_percentages(s, gold_bear=False, value_regime=False, asset_trends=None, vix=None, yield_curve=None,
                           sahm=None, corr=None, momentum_scores=None, yc_recently_inverted=False, vix_recent_peak=None,
                           dual_ma_signals=None, breadth_score=None, iwy_volatility=None,
                           tsmom_signal=None, vix_history=None, risk_parity_weights=None):
    """
    Returns target asset allocation based on macro state.
    Shared by State Machine Diagnosis and Backtest.
    
    v3.0 新增参数:
    - tsmom_signal: 时序动量信号 (-1, 0, +1)
    - vix_history: VIX历史序列，用于尾部风险控制
    - risk_parity_weights: 风险平价权重字典
    
    v2.1 新增:
    - 动量加速因子
    - VIX均值回归加速
    - NEUTRAL状态IWY动态上限
    
    v2.0 新增参数:
    - iwy_volatility: IWY的20日年化波动率，用于波动率自适应控制
    
    v1.5 新增参数:
    - dual_ma_signals: dict {ticker: 'STRONG_BEAR'|'WEAK_BEAR'|'BULLISH'}
    - breadth_score: 0-1，跨资产动量分数
    """
    asset_trends = asset_trends or {}

    # v1.5: base_allocation 支持 VIX 分层
    targets = base_allocation(s, value_regime, vix)

    # 原有调整
    apply_vix_adjustments(targets, s, vix)
    apply_yield_curve_guard(targets, s, yield_curve)
    
    # v1.5: 双均线趋势过滤（替代原有简单趋势过滤）
    if dual_ma_signals:
        apply_dual_ma_trend_filter(targets, s, dual_ma_signals)
    else:
        apply_trend_filters(targets, s, asset_trends)
    
    apply_iwy_safety_valve(targets, s, asset_trends, vix)
    apply_gold_filter(targets, gold_bear)
    
    # 新增优化调整（按影响程度排序，后执行的优先级更高）
    apply_momentum_intensity(targets, s, momentum_scores)
    apply_sahm_early_warning(targets, s, sahm)
    
    # v3.0: 时序动量增强（在其他调整之后，波动率控制之前）
    if tsmom_signal is not None:
        apply_tsmom_boost(targets, s, tsmom_signal, vix)
    
    # v3.0: 尾部风险控制
    if vix_history is not None and len(vix_history) >= 2:
        apply_tail_risk_control(targets, s, vix_history)
    
    apply_yield_curve_uninvert_protection(targets, s, yield_curve, yc_recently_inverted)
    apply_correlation_adjustment(targets, s, corr)
    apply_vix_mean_reversion(targets, s, vix, vix_recent_peak)
    
    # v1.5: 新增优化
    apply_market_breadth_adjustment(targets, s, breadth_score)
    
    # v1.7: 动态轮换（核心收益增强）
    apply_trend_boost(targets, s, momentum_scores, vix)
    apply_value_rotation(targets, s, momentum_scores)
    
    # v2.0: 波动率控制（IWY为核心，控制整体波动）
    apply_volatility_control(targets, s, vix, iwy_volatility)
    apply_wtmf_vol_scaling(targets, s, vix)
    apply_lvhi_floor(targets, s)
    
    # v2.1: 收益增强优化
    apply_momentum_acceleration(targets, s, momentum_scores, vix)
    apply_vix_reversion_acceleration(targets, s, vix, vix_recent_peak)
    apply_iwy_cap(targets, s, momentum_scores, vix)
    
    # v3.0: 风险平价混合（最后执行，作为整体权重调整）
    if risk_parity_weights and RISK_PARITY_BLEND_ENABLED:
        targets = blend_with_risk_parity(targets, risk_parity_weights)

    return targets


def generate_execution_tips(metrics, change_info, current_holdings=None, targets=None, total_value=None):
    """
    生成执行建议提示，帮助用户在实际操作时参考回测中的优化机制。
    
    改进点:
    1. 信号确认: 明确显示"待确认"状态，建议观望
    2. 波动率: 基于收盘数据（已确定），给出明确执行比例
    3. 止损: 基于用户实际持仓计算回撤，给出具体操作
    4. 可执行性: 给出具体的资产和金额建议
    """
    tips = []
    
    vix = metrics.get('vix')
    state = metrics.get('state')
    days_in_state = change_info.get('days_in_state') if change_info else None
    prev_state = change_info.get('prev_state') if change_info else None
    
    # === 0. 止损状态检查（最高优先级）===
    if total_value and total_value > 0:
        stop_loss_status = get_stop_loss_status(total_value)
        if stop_loss_status.get('in_stop_loss'):
            drawdown = stop_loss_status['drawdown_pct']
            recovery_ratio = stop_loss_status['recovery_ratio']
            reduce_pct = (1 - recovery_ratio) * 100
            tips.append({
                'type': 'error',
                'icon': '🚨',
                'title': f'止损触发 (回撤 {drawdown*100:.1f}%)',
                'content': f'当前组合回撤已超过{abs(DRAWDOWN_STOP_LOSS)*100:.0f}%止损线！'
                           f'建议立即将风险资产（IWY, G3B.SI等）减仓至目标的{recovery_ratio*100:.0f}%，'
                           f'释放的资金转入WTMF或现金。回撤恢复至{abs(DRAWDOWN_RECOVERY_THRESHOLD)*100:.0f}%内后再逐步恢复。'
            })
        elif stop_loss_status['drawdown_pct'] < -0.05:  # 接近止损线
            drawdown = stop_loss_status['drawdown_pct']
            tips.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': f'接近止损线 (回撤 {drawdown*100:.1f}%)',
                'content': f'当前组合回撤{drawdown*100:.1f}%，距离止损线({abs(DRAWDOWN_STOP_LOSS)*100:.0f}%)较近。'
                           f'建议降低风险敞口，或设置盘中价格提醒，若继续下跌{(DRAWDOWN_STOP_LOSS - drawdown)*100:.1f}%即触发止损。'
            })
    
    # === 1. 信号确认提示（重要提醒）===
    if days_in_state is not None and days_in_state <= SIGNAL_CONFIRM_DAYS:
        remaining_days = SIGNAL_CONFIRM_DAYS - days_in_state + 1
        if prev_state and prev_state != state:
            tips.append({
                'type': 'warning',
                'icon': '🔄',
                'title': f'状态待确认 ({days_in_state}/{SIGNAL_CONFIRM_DAYS}天)',
                'content': f'状态刚从 {prev_state} 切换到 {state}，需连续{SIGNAL_CONFIRM_DAYS}天确认。'
                           f'【建议】暂不执行大幅调仓，等待{remaining_days}天确认后再行动。'
                           f'若急需操作，可先执行目标配置的50%。'
            })
        else:
            tips.append({
                'type': 'info',
                'icon': '🔄',
                'title': f'信号确认中 ({days_in_state}/{SIGNAL_CONFIRM_DAYS}天)',
                'content': f'当前状态 {state} 持续{days_in_state}天，还需{remaining_days}天确认。可按目标的50-70%先行配置。'
            })
    elif days_in_state is not None and days_in_state > SIGNAL_CONFIRM_DAYS:
        tips.append({
            'type': 'success',
            'icon': '✅',
            'title': f'信号已确认 (持续{days_in_state}天)',
            'content': f'状态 {state} 已确认，可按目标配置全额执行。'
        })
    
    # === 2. 波动率执行建议（基于收盘VIX，已确定）===
    if vix is not None:
        if vix > VIX_SMOOTH_END:
            reduction_pct = int(VIX_MAX_REDUCTION * 100)
            exec_pct = 100 - reduction_pct
            tips.append({
                'type': 'error',
                'icon': '📊',
                'title': f'高波动警告 (VIX={vix:.1f})',
                'content': f'VIX超过{VIX_SMOOTH_END:.0f}，市场波动剧烈。'
                           f'【执行】按目标配置的{exec_pct}%建仓，{reduction_pct}%留作现金/WTMF。'
                           f'例如目标IWY 55%，实际执行IWY {55*exec_pct/100:.0f}%。'
            })
        elif vix > VIX_SMOOTH_START:
            reduction = (vix - VIX_SMOOTH_START) / (VIX_SMOOTH_END - VIX_SMOOTH_START) * VIX_MAX_REDUCTION
            exec_pct = int((1 - reduction) * 100)
            tips.append({
                'type': 'warning',
                'icon': '📊',
                'title': f'波动偏高 (VIX={vix:.1f})',
                'content': f'VIX处于{VIX_SMOOTH_START:.0f}-{VIX_SMOOTH_END:.0f}区间，建议保守执行。'
                           f'【执行】按目标配置的{exec_pct}%建仓，留{100-exec_pct}%现金缓冲。'
            })
        elif vix < VIX_BOOST_LO:
            tips.append({
                'type': 'success',
                'icon': '🚀',
                'title': f'低波动机会 (VIX={vix:.1f})',
                'content': f'VIX<{VIX_BOOST_LO:.0f}，市场极度平稳。可全额执行目标配置，甚至考虑减少WTMF/债券，增加权益。'
            })
    
    # === 3. 具体调仓建议 ===
    if targets and current_holdings and total_value and total_value > 0:
        # 计算各资产偏离
        deviations = []
        for ticker, target_w in targets.items():
            current_val = current_holdings.get(ticker, 0)
            current_w = current_val / total_value if isinstance(current_val, (int, float)) else 0
            deviation = target_w - current_w
            diff_val = deviation * total_value
            if abs(deviation) > 0.02:  # 超过2%才显示
                deviations.append({
                    'ticker': ticker,
                    'name': ASSET_NAMES.get(ticker, ticker),
                    'deviation': deviation,
                    'diff_val': diff_val,
                    'action': '买入' if deviation > 0 else '卖出'
                })
        
        # 检查需要清仓的资产
        for ticker, current_val in current_holdings.items():
            if ticker not in targets and isinstance(current_val, (int, float)) and current_val > 100:
                deviations.append({
                    'ticker': ticker,
                    'name': ASSET_NAMES.get(ticker, ticker),
                    'deviation': -current_val / total_value,
                    'diff_val': -current_val,
                    'action': '清仓'
                })
        
        # 按偏离大小排序
        deviations.sort(key=lambda x: abs(x['deviation']), reverse=True)
        
        total_change = sum(abs(d['deviation']) for d in deviations) / 2  # 单边换手
        max_deviation = max(abs(d['deviation']) for d in deviations) if deviations else 0
        
        if max_deviation < REBALANCE_THRESHOLD:
            tips.append({
                'type': 'success',
                'icon': '📏',
                'title': '无需调仓',
                'content': f'所有资产偏离均<{REBALANCE_THRESHOLD*100:.0f}%，可暂不调仓以节省交易成本（预估0.1-0.3%）。'
            })
        elif total_change > 0.20:
            # 大幅调仓，建议分步
            top_actions = deviations[:3]
            action_text = "; ".join([
                f"{d['action']}{d['name'][:6]}约${abs(d['diff_val']):,.0f}" for d in top_actions
            ])
            tips.append({
                'type': 'info',
                'icon': '🔀',
                'title': f'分步调仓 (换手{total_change*100:.0f}%)',
                'content': f'调仓幅度较大，建议分{STATE_TRANSITION_DAYS}天执行。'
                           f'【今日操作】{action_text}。每天调整约{total_change/STATE_TRANSITION_DAYS*100:.0f}%。'
            })
        elif deviations:
            top_actions = deviations[:2]
            action_text = "; ".join([
                f"{d['action']}{d['name'][:6]}约${abs(d['diff_val']):,.0f}" for d in top_actions
            ])
            tips.append({
                'type': 'info',
                'icon': '📋',
                'title': '调仓建议',
                'content': f'【操作】{action_text}。'
            })
    
    # === 4. 极端状态提示 ===
    if state == "EXTREME_ACCUMULATION":
        tips.append({
            'type': 'warning',
            'icon': '⚡',
            'title': '抄底状态',
            'content': '极端抄底模式，风险与机会并存。【执行】分批建仓：首次40% → 反弹5%后加至60% → 突破MA50后加至75%。'
        })
    elif state in ["DEFLATION_RECESSION", "INFLATION_SHOCK"]:
        tips.append({
            'type': 'error',
            'icon': '🛡️',
            'title': '危机防御模式',
            'content': '当前为危机状态，优先保本。严格执行目标配置，避免抄底冲动。WTMF和黄金是主要避险工具。'
        })
    
    # === 5. Sahm Rule 预警提示 ===
    sahm = metrics.get('sahm')
    if sahm is not None and SAHM_EARLY_WARNING_LO <= sahm < SAHM_EARLY_WARNING_HI:
        reduction_pct = int((sahm - SAHM_EARLY_WARNING_LO) / (SAHM_EARLY_WARNING_HI - SAHM_EARLY_WARNING_LO) * SAHM_REDUCTION_RATE * 100)
        tips.append({
            'type': 'warning',
            'icon': '📉',
            'title': f'Sahm预警 ({sahm:.2f})',
            'content': f'Sahm Rule处于预警区间({SAHM_EARLY_WARNING_LO}-{SAHM_EARLY_WARNING_HI})。'
                       f'【影响】IWY目标已自动减少{reduction_pct}%，转入WTMF。'
        })
    
    # === 6. 收益率曲线解倒挂提示 ===
    yc_un_invert = metrics.get('yc_un_invert', False)
    yield_curve = metrics.get('yield_curve')
    if yc_un_invert and yield_curve is not None and yield_curve > 0:
        tips.append({
            'type': 'warning',
            'icon': '📈',
            'title': '解倒挂保护期',
            'content': f'收益率曲线已转正({yield_curve:.2f}%)，但近期曾深度倒挂。'
                       f'【历史规律】解倒挂后6-18个月易发生衰退。'
                       f'【影响】IWY目标已减少{int(YC_UNINVERT_REDUCTION*100)}%，维持防御配置。'
        })
    
    # === 7. 相关性警告 ===
    corr = metrics.get('corr')
    if corr is not None and corr > CORR_HIGH_THRESHOLD:
        tips.append({
            'type': 'info',
            'icon': '🔗',
            'title': f'相关性偏高 ({corr:.2f})',
            'content': f'股债相关性>{CORR_HIGH_THRESHOLD}，债券对冲效果减弱。'
                       f'【影响】MBH.SI目标已转移{int(CORR_MAX_REALLOC*100)}%至WTMF/黄金。'
        })
    
    # === 8. 跨市场执行提醒 ===
    if targets:
        sg_assets = [t for t in targets.keys() if '.SI' in t and targets[t] > 0.02]
        us_assets = [t for t in targets.keys() if '.SI' not in t and t != 'OTHERS' and targets[t] > 0.02]
        if sg_assets and us_assets:
            tips.append({
                'type': 'info',
                'icon': '🌏',
                'title': '跨市场执行',
                'content': f'涉及新加坡({", ".join(sg_assets[:3])})和美股({", ".join(us_assets[:3])})。'
                           f'【时区】SGX 9:00-17:00(+8), NYSE 21:30-04:00(+8)。建议先执行SGX，次日再执行US。'
            })
    
    return tips


def calculate_dual_ma_signals(price_data, ma_short=TREND_MA_SHORT, ma_long=TREND_MA_LONG):
    """
    计算双均线趋势信号
    返回: dict {ticker: 'STRONG_BEAR'|'WEAK_BEAR'|'BULLISH'}
    - STRONG_BEAR: 价格 < MA200 且 MA50 < MA200 (强熊市)
    - WEAK_BEAR: 价格 < MA200 但 MA50 > MA200 (可能是回调)
    - BULLISH: 价格 > MA200
    """
    signals = {}
    
    if price_data is None or price_data.empty:
        return signals
    
    for ticker in price_data.columns:
        try:
            prices = price_data[ticker].dropna()
            if len(prices) < ma_long:
                continue
            
            ma50 = prices.rolling(ma_short).mean().iloc[-1]
            ma200 = prices.rolling(ma_long).mean().iloc[-1]
            price = prices.iloc[-1]
            
            if pd.isna(ma50) or pd.isna(ma200) or pd.isna(price):
                continue
            
            if price < ma200 and ma50 < ma200:
                signals[ticker] = "STRONG_BEAR"
            elif price < ma200:
                signals[ticker] = "WEAK_BEAR"
            else:
                signals[ticker] = "BULLISH"
        except Exception:
            continue
    
    return signals


def calculate_market_breadth(price_data, ma_window=200):
    """
    计算跨资产动量（市场广度）
    返回: 0-1 之间的分数，表示有多少比例的资产处于上升趋势
    """
    if price_data is None or price_data.empty:
        return None
    
    above_ma_count = 0
    total_count = 0
    
    for ticker in price_data.columns:
        try:
            prices = price_data[ticker].dropna()
            if len(prices) < ma_window:
                continue
            
            ma = prices.rolling(ma_window).mean().iloc[-1]
            price = prices.iloc[-1]
            
            if pd.notna(ma) and pd.notna(price):
                total_count += 1
                if price > ma:
                    above_ma_count += 1
        except Exception:
            continue
    
    if total_count == 0:
        return None
    
    return above_ma_count / total_count


def calculate_portfolio_health(current_holdings, targets, total_value):
    """
    计算持仓健康度评分 (0-100分)
    返回: (score, details_dict)
    """
    if total_value <= 0:
        return 0, {'reason': '总市值为零'}
    
    # 1. 权重偏离度 (40分)
    total_deviation = 0
    max_single_deviation = 0
    deviations = {}
    
    all_tickers = set(targets.keys()).union(current_holdings.keys())
    for tkr in all_tickers:
        target_w = targets.get(tkr, 0)
        current_val = current_holdings.get(tkr, 0)
        current_w = current_val / total_value if total_value > 0 else 0
        dev = abs(target_w - current_w)
        deviations[tkr] = {'target': target_w, 'current': current_w, 'deviation': dev}
        total_deviation += dev
        max_single_deviation = max(max_single_deviation, dev)
    
    # 偏离度评分: 总偏离<10%得满分，>50%得0分
    deviation_score = max(0, 40 * (1 - total_deviation / 0.5))
    
    # 2. 单一资产集中度 (20分)
    max_weight = max([v / total_value for v in current_holdings.values()]) if current_holdings else 0
    # 单一资产<40%得满分，>70%得0分
    concentration_score = max(0, 20 * (1 - (max_weight - 0.4) / 0.3)) if max_weight > 0.4 else 20
    
    # 3. 资产类别多样性 (20分)
    category_weights = {}
    for tkr, val in current_holdings.items():
        if val <= 0:
            continue
        cat = ASSET_CATEGORIES.get(tkr, {}).get('category', '其他')
        category_weights[cat] = category_weights.get(cat, 0) + val / total_value
    
    # 至少覆盖3个类别得满分
    diversity_score = min(20, len([c for c, w in category_weights.items() if w > 0.05]) * 5)
    
    # 4. 现金/对冲覆盖 (20分) - 检查防御性配置
    defensive_weight = category_weights.get('固收', 0) + category_weights.get('对冲', 0) + category_weights.get('商品', 0)
    # 防御配置在15-40%之间得满分
    if 0.15 <= defensive_weight <= 0.40:
        defensive_score = 20
    elif defensive_weight < 0.15:
        defensive_score = max(0, 20 * defensive_weight / 0.15)
    else:
        defensive_score = max(0, 20 * (1 - (defensive_weight - 0.40) / 0.30))
    
    total_score = deviation_score + concentration_score + diversity_score + defensive_score
    
    return total_score, {
        'deviation_score': deviation_score,
        'concentration_score': concentration_score,
        'diversity_score': diversity_score,
        'defensive_score': defensive_score,
        'total_deviation': total_deviation,
        'max_single_deviation': max_single_deviation,
        'max_weight': max_weight,
        'category_weights': category_weights,
        'deviations': deviations
    }


def generate_rebalance_priority(current_holdings, targets, total_value, metrics):
    """
    生成调仓优先级列表，按紧迫程度排序
    返回: [(ticker, priority_score, reason, action_detail), ...]
    """
    priorities = []
    
    if total_value <= 0:
        return priorities
    
    vix = metrics.get('vix', 15)
    state = metrics.get('state', 'NEUTRAL')
    
    all_tickers = set(targets.keys()).union(current_holdings.keys())
    
    for tkr in all_tickers:
        target_w = targets.get(tkr, 0)
        current_val = current_holdings.get(tkr, 0)
        current_w = current_val / total_value
        diff_w = target_w - current_w
        diff_val = diff_w * total_value
        
        if abs(diff_w) < 0.02:  # 偏离<2%忽略
            continue
        
        # 基础优先级分数 (0-100)
        priority = abs(diff_w) * 100  # 偏离越大越紧急
        reason = []
        
        # 加权因子
        cat_info = ASSET_CATEGORIES.get(tkr, {})
        risk_level = cat_info.get('risk_level', 'medium')
        
        # 1. 风险资产在高波动期优先减仓
        if diff_w < 0 and risk_level == 'high' and vix > 20:
            priority *= 1.5
            reason.append(f"高风险资产+VIX={vix:.0f}")
        
        # 2. 目标为0的资产优先清仓
        if target_w == 0 and current_val > 0:
            priority *= 1.3
            reason.append("目标清仓")
        
        # 3. 防御状态下优先增配防御资产
        if state in ['DEFLATION_RECESSION', 'CAUTIOUS_VOL', 'CAUTIOUS_TREND']:
            if diff_w > 0 and cat_info.get('category') in ['固收', '对冲', '商品']:
                priority *= 1.2
                reason.append("防御态势增配")
        
        # 4. 极端抄底状态优先增配权益
        if state == 'EXTREME_ACCUMULATION':
            if diff_w > 0 and cat_info.get('category') == '权益':
                priority *= 1.2
                reason.append("抄底增配")
        
        action = "买入" if diff_w > 0 else "卖出"
        action_detail = f"{action} ${abs(diff_val):,.0f} ({abs(diff_w)*100:.1f}%)"
        
        priorities.append({
            'ticker': tkr,
            'name': ASSET_NAMES.get(tkr, tkr),
            'priority': priority,
            'reasons': reason,
            'action': action,
            'action_detail': action_detail,
            'diff_val': diff_val,
            'diff_w': diff_w,
            'current_w': current_w,
            'target_w': target_w
        })
    
    # 按优先级降序排序
    priorities.sort(key=lambda x: x['priority'], reverse=True)
    return priorities


def estimate_rebalance_cost(priorities, cost_bps=10):
    """
    估算调仓成本
    cost_bps: 交易成本 (基点, 默认10bps = 0.1%)
    """
    total_turnover = sum(abs(p['diff_val']) for p in priorities)
    estimated_cost = total_turnover * cost_bps / 10000
    return total_turnover, estimated_cost


def generate_stepwise_plan(priorities, total_value, days=3):
    """
    生成分步调仓计划
    """
    if not priorities:
        return []
    
    # 按天分配操作
    plan = []
    total_change = sum(abs(p['diff_val']) for p in priorities)
    
    if total_change / total_value < 0.10:
        # 变化<10%，一次性调整
        plan.append({
            'day': 1,
            'description': '一次性完成调仓',
            'actions': [(p['ticker'], p['action_detail']) for p in priorities]
        })
    else:
        # 分步执行
        # 第1天: 卖出操作 + 紧急买入
        day1_actions = []
        day2_actions = []
        day3_actions = []
        
        for p in priorities:
            if p['diff_val'] < 0:  # 卖出优先
                day1_actions.append((p['ticker'], p['action_detail']))
            elif p['priority'] > 30:  # 高优先级买入
                day2_actions.append((p['ticker'], p['action_detail']))
            else:
                day3_actions.append((p['ticker'], p['action_detail']))
        
        if day1_actions:
            plan.append({'day': 1, 'description': '执行卖出操作，回收资金', 'actions': day1_actions})
        if day2_actions:
            plan.append({'day': 2, 'description': '高优先级买入', 'actions': day2_actions})
        if day3_actions:
            plan.append({'day': 3, 'description': '完成剩余调整', 'actions': day3_actions})
    
    return plan


def render_portfolio_health_card(score, details, state):
    """渲染持仓健康度卡片"""
    st.markdown("### 📊 持仓健康度评估")
    
    # 健康度颜色
    if score >= 80:
        color, status = '#52c41a', '优秀'
    elif score >= 60:
        color, status = '#1890ff', '良好'
    elif score >= 40:
        color, status = '#faad14', '需调整'
    else:
        color, status = '#f5222d', '需重配'
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="text-align:center;padding:20px;background:#f9fafb;border-radius:12px;">
            <div style="font-size:48px;font-weight:700;color:{color};">{score:.0f}</div>
            <div style="font-size:16px;color:#666;margin-top:4px;">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 分项评分
        items = [
            ('权重偏离', details['deviation_score'], 40),
            ('集中度', details['concentration_score'], 20),
            ('多样性', details['diversity_score'], 20),
            ('防御配置', details['defensive_score'], 20),
        ]
        for name, score_item, max_score in items:
            pct = score_item / max_score * 100
            bar_color = '#52c41a' if pct >= 70 else ('#faad14' if pct >= 40 else '#f5222d')
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:13px;">
                    <span>{name}</span><span>{score_item:.0f}/{max_score}</span>
                </div>
                <div style="background:#e8e8e8;height:6px;border-radius:3px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_risk_exposure_chart(details, targets):
    """渲染风险暴露分析"""
    st.markdown("### 🎯 风险暴露分析")
    
    category_weights = details.get('category_weights', {})
    
    # 计算目标类别权重
    target_categories = {}
    for tkr, w in targets.items():
        cat = ASSET_CATEGORIES.get(tkr, {}).get('category', '其他')
        target_categories[cat] = target_categories.get(cat, 0) + w
    
    # 所有类别
    all_cats = ['权益', '固收', '商品', '对冲', '另类', '其他']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**当前配置**")
        for cat in all_cats:
            w = category_weights.get(cat, 0)
            if w > 0 or target_categories.get(cat, 0) > 0:
                bar_color = {'权益': '#f5222d', '固收': '#1890ff', '商品': '#faad14', '对冲': '#52c41a', '另类': '#722ed1'}.get(cat, '#999')
                st.markdown(f"""
                <div style="margin-bottom:6px;">
                    <span style="display:inline-block;width:60px;font-size:13px;">{cat}</span>
                    <span style="display:inline-block;width:120px;background:#e8e8e8;height:16px;border-radius:4px;vertical-align:middle;">
                        <span style="display:block;width:{w*100}%;height:100%;background:{bar_color};border-radius:4px;"></span>
                    </span>
                    <span style="font-size:13px;margin-left:8px;">{w*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**目标配置**")
        for cat in all_cats:
            w = target_categories.get(cat, 0)
            if w > 0 or category_weights.get(cat, 0) > 0:
                bar_color = {'权益': '#f5222d', '固收': '#1890ff', '商品': '#faad14', '对冲': '#52c41a', '另类': '#722ed1'}.get(cat, '#999')
                st.markdown(f"""
                <div style="margin-bottom:6px;">
                    <span style="display:inline-block;width:60px;font-size:13px;">{cat}</span>
                    <span style="display:inline-block;width:120px;background:#e8e8e8;height:16px;border-radius:4px;vertical-align:middle;">
                        <span style="display:block;width:{w*100}%;height:100%;background:{bar_color};border-radius:4px;"></span>
                    </span>
                    <span style="font-size:13px;margin-left:8px;">{w*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)


def render_rebalance_priority_table(priorities, turnover, cost):
    """渲染调仓优先级表格"""
    st.markdown("### 🔥 调仓优先级")
    st.caption(f"预估换手: ${turnover:,.0f} | 交易成本: ~${cost:,.0f}")
    
    if not priorities:
        st.info("当前持仓与目标配置偏离较小，无需调整")
        return
    
    # 构建表格数据
    data = []
    for i, p in enumerate(priorities, 1):
        urgency = '🔴 紧急' if p['priority'] > 30 else ('🟡 建议' if p['priority'] > 15 else '🟢 可选')
        reasons = ', '.join(p['reasons']) if p['reasons'] else '-'
        data.append({
            '优先级': i,
            '紧迫度': urgency,
            '资产': f"{p['name']} ({p['ticker']})",
            '操作': p['action_detail'],
            '当前→目标': f"{p['current_w']*100:.1f}% → {p['target_w']*100:.1f}%",
            '触发因素': reasons
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_stepwise_plan(plan):
    """渲染分步调仓计划"""
    if not plan:
        return
    
    st.markdown("### 📅 分步执行计划")
    
    for step in plan:
        day = step['day']
        desc = step['description']
        actions = step['actions']
        
        with st.expander(f"**第{day}天**: {desc}", expanded=(day == 1)):
            for tkr, action in actions:
                st.markdown(f"- **{ASSET_NAMES.get(tkr, tkr)}** ({tkr}): {action}")


def render_enhanced_diagnosis(metrics, current_holdings, total_value, targets, change_info):
    """渲染增强版持仓诊断"""
    st.markdown("---")
    st.markdown("## 🔬 深度持仓诊断")
    
    # 0. v1.5 优化机制实时状态
    st.markdown("### ⚙️ v1.5 优化机制状态")
    
    vix = metrics.get('vix', 15)
    sahm = metrics.get('sahm', 0)
    corr = metrics.get('corr', 0)
    yc = metrics.get('yield_curve', 0)
    state = metrics.get('state', 'NEUTRAL')
    
    # 计算各机制当前状态
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    
    with col_v1:
        # 现金缓冲状态
        if state == "EXTREME_ACCUMULATION":
            cash_buffer = 0
            cash_status = "抄底模式-不留现金"
        else:
            cash_buffer = CASH_BUFFER_BASE
            if vix > CASH_BUFFER_VIX_THRESHOLD:
                extra_cash = min((vix - CASH_BUFFER_VIX_THRESHOLD) / 5 * CASH_BUFFER_VIX_SCALE, 
                                 CASH_BUFFER_MAX - CASH_BUFFER_BASE)
                cash_buffer = CASH_BUFFER_BASE + extra_cash
            cash_status = "正常" if cash_buffer <= CASH_BUFFER_BASE else "增强"
        
        st.metric(
            "💵 现金缓冲",
            f"{cash_buffer*100:.1f}%",
            cash_status,
            delta_color="normal" if cash_status == "正常" else "off"
        )
    
    with col_v2:
        # VIX分层状态
        if state == "CAUTIOUS_VOL":
            if vix >= 30:
                vix_tier = "Tier3 (IWY 10%)"
            elif vix >= 25:
                vix_tier = "Tier2 (IWY 20%)"
            else:
                vix_tier = "Tier1 (IWY 30%)"
        else:
            vix_tier = "不适用"
        
        st.metric(
            "📊 VIX分层",
            f"VIX={vix:.1f}",
            vix_tier,
            delta_color="off"
        )
    
    with col_v3:
        # 相关性渐进响应
        if corr > CORR_HIGH_THRESHOLD:
            corr_status = f"最大调整 {CORR_MAX_REALLOC*100:.0f}%"
            corr_delta = "inverse"
        elif corr > CORR_MID_THRESHOLD:
            adjustment_pct = (corr - CORR_MID_THRESHOLD) / (CORR_HIGH_THRESHOLD - CORR_MID_THRESHOLD)
            realloc = adjustment_pct * CORR_MAX_REALLOC
            corr_status = f"渐进调整 {realloc*100:.1f}%"
            corr_delta = "off"
        else:
            corr_status = "正常"
            corr_delta = "normal"
        
        st.metric(
            "🔗 相关性响应",
            f"Corr={corr:.2f}",
            corr_status,
            delta_color=corr_delta
        )
    
    with col_v4:
        # Sahm预警状态
        if sahm >= SAHM_EARLY_WARNING_HI:
            sahm_status = "衰退确认"
            sahm_delta = "inverse"
        elif sahm >= SAHM_EARLY_WARNING_LO:
            reduction_pct = int((sahm - SAHM_EARLY_WARNING_LO) / (SAHM_EARLY_WARNING_HI - SAHM_EARLY_WARNING_LO) * SAHM_REDUCTION_RATE * 100)
            sahm_status = f"预警 -{reduction_pct}%"
            sahm_delta = "off"
        else:
            sahm_status = "正常"
            sahm_delta = "normal"
        
        st.metric(
            "📉 Sahm预警",
            f"Sahm={sahm:.2f}",
            sahm_status,
            delta_color=sahm_delta
        )
    
    # 第二行优化状态
    col_v5, col_v6, col_v7, col_v8 = st.columns(4)
    
    with col_v5:
        # 收益率曲线保护
        yc_un_invert = metrics.get('yc_un_invert', False)
        if yc < 0:
            yc_status = "倒挂中"
            yc_delta = "inverse"
        elif yc_un_invert:
            yc_status = f"解倒挂保护 -{YC_UNINVERT_REDUCTION*100:.0f}%"
            yc_delta = "off"
        else:
            yc_status = "正常"
            yc_delta = "normal"
        
        st.metric(
            "📈 曲线保护",
            f"10Y-2Y={yc:.2f}%",
            yc_status,
            delta_color=yc_delta
        )
    
    with col_v6:
        # 市场广度（估算）
        asset_trends = metrics.get('asset_trends', {})
        if asset_trends:
            bullish_count = sum(1 for bear in asset_trends.values() if not bear)
            total_count = len(asset_trends)
            breadth = bullish_count / total_count if total_count > 0 else 0.5
        else:
            breadth = 0.5  # 默认中性
        
        if breadth < MARKET_BREADTH_LOW:
            breadth_status = f"低广度 -{BREADTH_LOW_REDUCTION*100:.0f}%"
        elif breadth < MARKET_BREADTH_MID:
            breadth_status = f"一般 -{BREADTH_MID_REDUCTION*100:.0f}%"
        else:
            breadth_status = "正常"
        
        st.metric(
            "📊 市场广度",
            f"{breadth*100:.0f}%",
            breadth_status
        )
    
    with col_v7:
        # 信号确认
        days_in_state = change_info.get('days_in_state') if change_info else None
        if days_in_state is not None and days_in_state <= SIGNAL_CONFIRM_DAYS:
            confirm_status = f"确认中 ({days_in_state}/{SIGNAL_CONFIRM_DAYS})"
        else:
            confirm_status = "已确认"
        
        st.metric(
            "🔄 信号确认",
            f"{days_in_state or 0}天",
            confirm_status
        )
    
    with col_v8:
        # 再平衡状态
        max_dev = 0
        for tkr in set(targets.keys()).union(current_holdings.keys()):
            target_w = targets.get(tkr, 0)
            current_val = current_holdings.get(tkr, 0)
            current_w = current_val / total_value if total_value > 0 else 0
            dev = abs(target_w - current_w)
            max_dev = max(max_dev, dev)
        
        if max_dev > REBALANCE_THRESHOLD:
            rebal_status = "需要调仓"
        else:
            rebal_status = "无需调仓"
        
        st.metric(
            "📏 再平衡带",
            f"最大偏离 {max_dev*100:.1f}%",
            rebal_status,
            delta_color="inverse" if max_dev > REBALANCE_THRESHOLD else "normal"
        )
    
    # === 新增：止损状态面板 ===
    if total_value > 0:
        st.markdown("---")
        st.markdown("### 🛡️ 止损状态监控")
        
        stop_loss_status = get_stop_loss_status(total_value)
        drawdown_pct = stop_loss_status['drawdown_pct']
        peak_value = stop_loss_status['peak_value']
        days_since_peak = stop_loss_status['days_since_peak']
        in_stop_loss = stop_loss_status['in_stop_loss']
        recovery_ratio = stop_loss_status['recovery_ratio']
        
        col_sl1, col_sl2, col_sl3, col_sl4 = st.columns(4)
        
        with col_sl1:
            if drawdown_pct < DRAWDOWN_STOP_LOSS:
                dd_color = "inverse"
            elif drawdown_pct < DRAWDOWN_RECOVERY_THRESHOLD:
                dd_color = "off"
            else:
                dd_color = "normal"
            st.metric(
                "📉 当前回撤",
                f"{drawdown_pct*100:.1f}%",
                f"止损线: {DRAWDOWN_STOP_LOSS*100:.0f}%",
                delta_color=dd_color
            )
        
        with col_sl2:
            st.metric(
                "📈 历史最高",
                f"${peak_value:,.0f}",
                f"当前: ${total_value:,.0f}"
            )
        
        with col_sl3:
            st.metric(
                "📅 距峰值",
                f"{days_since_peak} 天",
                "持续时间"
            )
        
        with col_sl4:
            stage = stop_loss_status.get('stage', '正常')
            stage_color = stop_loss_status.get('stage_color', '#52c41a')
            if in_stop_loss:
                st.metric(
                    "🚨 止损阶段",
                    f"减仓至{recovery_ratio*100:.0f}%",
                    stage,
                    delta_color="inverse"
                )
            else:
                st.metric(
                    "✅ 止损状态",
                    stage,
                    stop_loss_status.get('advice', ''),
                    delta_color="normal" if stage == "正常" else "off"
                )
        
        # 止损操作建议
        if in_stop_loss:
            st.error(f"""
            **⚠️ 止损触发！** 当前回撤 {drawdown_pct*100:.1f}% 已超过止损线 {DRAWDOWN_STOP_LOSS*100:.0f}%。
            
            **建议操作：**
            - 将风险资产（IWY, G3B.SI, LVHI等）减仓至目标的 **{recovery_ratio*100:.0f}%**
            - 释放资金转入 WTMF 或 现金
            - 回撤恢复至 **{DRAWDOWN_RECOVERY_THRESHOLD*100:.0f}%** 内后再逐步恢复仓位
            """)
        elif drawdown_pct < -0.05:
            st.warning(f"""
            **⚠️ 接近止损线！** 当前回撤 {drawdown_pct*100:.1f}%，距止损线 {DRAWDOWN_STOP_LOSS*100:.0f}% 仅差 {(DRAWDOWN_STOP_LOSS - drawdown_pct)*100:.1f}%。
            
            建议：降低风险敞口或设置盘中价格提醒。
            """)
        
        # 重置峰值按钮（用于注入新资金后）
        with st.expander("⚙️ 止损设置", expanded=False):
            st.caption("如果您新注入资金，可重置历史最高净值以避免误触止损。")
            col_reset1, col_reset2 = st.columns(2)
            with col_reset1:
                if st.button("🔄 重置为当前净值", help="将历史最高净值重置为当前总市值"):
                    reset_portfolio_peak(total_value)
                    st.success(f"已重置历史最高净值为 ${total_value:,.0f}")
                    st.rerun()
            with col_reset2:
                new_peak = st.number_input("手动设置峰值", value=float(peak_value), step=1000.0, key="manual_peak")
                if st.button("确认设置"):
                    reset_portfolio_peak(new_peak)
                    st.success(f"已设置历史最高净值为 ${new_peak:,.0f}")
                    st.rerun()
    
    st.markdown("---")
    
    # 1. 健康度评估
    score, details = calculate_portfolio_health(current_holdings, targets, total_value)
    render_portfolio_health_card(score, details, metrics.get('state'))
    
    st.markdown("---")
    
    # 2. 风险暴露分析
    render_risk_exposure_chart(details, targets)
    
    st.markdown("---")
    
    # 3. 调仓优先级
    priorities = generate_rebalance_priority(current_holdings, targets, total_value, metrics)
    turnover, cost = estimate_rebalance_cost(priorities)
    render_rebalance_priority_table(priorities, turnover, cost)
    
    # 4. 分步执行计划
    plan = generate_stepwise_plan(priorities, total_value)
    render_stepwise_plan(plan)


def render_execution_tips(tips):
    """渲染执行建议提示卡片"""
    if not tips:
        return
    
    st.markdown("### 💡 执行建议 (Execution Tips)")
    st.caption("基于回测优化机制的实时操作参考")
    
    for tip in tips:
        tip_type = tip.get('type', 'info')
        icon = tip.get('icon', '💡')
        title = tip.get('title', '')
        content = tip.get('content', '')
        
        if tip_type == 'error':
            bg_color = '#fff2f0'
            border_color = '#ffccc7'
            text_color = '#cf1322'
        elif tip_type == 'warning':
            bg_color = '#fffbe6'
            border_color = '#ffe58f'
            text_color = '#ad6800'
        elif tip_type == 'success':
            bg_color = '#f6ffed'
            border_color = '#b7eb8f'
            text_color = '#389e0d'
        else:  # info
            bg_color = '#e6f7ff'
            border_color = '#91d5ff'
            text_color = '#0050b3'
        
        st.markdown(f"""
        <div style="background:{bg_color};border:1px solid {border_color};border-radius:8px;padding:12px 16px;margin-bottom:10px;">
            <div style="font-weight:600;color:{text_color};margin-bottom:4px;">{icon} {title}</div>
            <div style="color:#333;font-size:14px;line-height:1.5;">{content}</div>
        </div>
        """, unsafe_allow_html=True)


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
    # 转换为 DatetimeIndex 以正确计算日期差
    peak_dates_dt = pd.to_datetime(peak_dates)
    dd_days = pd.to_datetime(series.index) - peak_dates_dt
    max_dd_days = dd_days.max().days if len(dd_days) > 0 and pd.notna(dd_days.max()) else 0
    
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

def run_dynamic_backtest(df_states, start_date, end_date, initial_capital=10000.0, ma_window=200, use_proxies=False, rebal_freq='Daily', transaction_cost_bps=10):
    """
    Simulates the strategy over historical states.
    df_states: DataFrame with 'State', 'Gold_Bear', 'Value_Regime' columns, indexed by Date.
    rebal_freq: 'Daily', 'Weekly', 'Monthly', 'Quarterly'
    transaction_cost_bps: 交易成本（基点），默认10bps=0.1%，包含佣金+滑点
    
    关键改进（v1.6）：
    - 使用T-1日信号决定T日配置，避免前视偏差
    - 计入交易成本
    """
    ensure_fred_cached()
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
    
    price_data = None
    try:
        raw_data = yf.download(assets, start=fetch_start, end=end_date, progress=False, auto_adjust=False)
        if raw_data is not None and not raw_data.empty:
            price_data = normalize_yf_prices(raw_data)
    except Exception as e:
        return None, None, f"数据下载失败: {e}"
    
    if price_data is None or (hasattr(price_data, 'empty') and price_data.empty):
        return None, None, f"无法获取价格数据。资产列表: {assets}，日期范围: {fetch_start.date()} ~ {end_date}"

    # 转换为DataFrame（如果是Series）
    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame(name=assets[0] if assets else 'Price')

    # Fill missing
    price_data = price_data.ffill().bfill()
    
    # 检查有效列
    valid_cols = [c for c in price_data.columns if price_data[c].notna().sum() > 10]
    if not valid_cols:
        return None, None, f"所有资产数据均无效或数据点不足"
    price_data = price_data[valid_cols]
    
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
    
    # === 优化机制状态变量 ===
    # 波动率目标机制
    portfolio_returns_history = []  # 用于计算实现波动率
    
    # 动态止损机制
    peak_nav = initial_capital  # 历史最高净值
    in_stop_loss_mode = False  # 是否处于止损模式
    
    # 信号确认延迟机制
    pending_state = None  # 待确认的新状态
    pending_state_days = 0  # 待确认状态的连续天数
    confirmed_state = None  # 已确认的状态
    
    # 状态转换平滑机制
    transition_from_weights = None  # 过渡起始权重
    transition_day = 0  # 当前过渡天数
    is_in_transition = False  # 是否正在过渡
    
    # We iterate daily. To speed up, we could vectorise, but logic is complex.
    # Logic: Daily return = Sum(Weight_i * Return_i)
    # Rebalancing frequency controls when we update target weights.
    
    returns_df = price_data.pct_change().fillna(0)
    
    # Determine rebalancing dates based on frequency
    def is_rebalance_day(date, freq, prev_date=None):
        """Check if current date is a rebalancing day."""
        if freq == 'Daily':
            return True
        elif freq == 'Weekly':
            # Rebalance on Monday (weekday=0)
            return date.weekday() == 0
        elif freq == 'Monthly':
            # Rebalance on first trading day of month
            if prev_date is None:
                return True
            return date.month != prev_date.month
        elif freq == 'Quarterly':
            # Rebalance on first trading day of quarter
            if prev_date is None:
                return True
            return (date.month - 1) // 3 != (prev_date.month - 1) // 3
        return True
    
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

    prev_date = None
    
    # === 关键修复：使用T-1日信号决定T日配置（避免前视偏差）===
    # 预存前一天的状态信息用于当天决策
    prev_row_state = None  # T-1日的状态
    prev_row_gb = None     # T-1日的Gold_Bear
    prev_row_vr = None     # T-1日的Value_Regime
    prev_row_date = None   # T-1日的日期
    
    for date, row in df_states.iterrows():
        # ===【重要】使用T-1日的状态来决定T日配置 ===
        # 这模拟了真实交易：T-1收盘后看到数据，T日开盘执行
        if prev_row_state is None:
            # 第一天：无前一天数据，使用当天（这是不可避免的）
            raw_state = row['State']
            gb = row['Gold_Bear']
            vr = row['Value_Regime']
            decision_date = date  # 用于获取趋势等辅助信息
        else:
            # 使用T-1日的状态信息做T日决策
            raw_state = prev_row_state
            gb = prev_row_gb
            vr = prev_row_vr
            decision_date = prev_row_date  # 使用T-1日的趋势信息
        
        # 保存当天状态供下一天使用
        prev_row_state = row['State']
        prev_row_gb = row['Gold_Bear']
        prev_row_vr = row['Value_Regime']
        prev_row_date = date
        
        # === 优化1: 信号确认延迟机制 ===
        # 状态切换需连续 SIGNAL_CONFIRM_DAYS 天确认才生效
        # v1.6: EXTREME_ACCUMULATION使用更快确认（抄底机会稍纵即逝）
        if confirmed_state is None:
            # 首日直接确认
            confirmed_state = raw_state
            pending_state = None
            pending_state_days = 0
        elif raw_state != confirmed_state:
            # 检测到状态变化
            # v1.6: 根据目标状态选择确认天数
            required_confirm_days = EXTREME_CONFIRM_DAYS if raw_state == "EXTREME_ACCUMULATION" else SIGNAL_CONFIRM_DAYS
            
            if pending_state == raw_state:
                # 继续确认同一个待切换状态
                pending_state_days += 1
                if pending_state_days >= required_confirm_days:
                    # 确认切换！启动过渡
                    confirmed_state = raw_state
                    pending_state = None
                    pending_state_days = 0
                    # 标记开始状态过渡
                    is_in_transition = True
                    transition_day = 0
                    transition_from_weights = prev_targets.copy() if prev_targets else None
            else:
                # 新的待切换状态
                pending_state = raw_state
                pending_state_days = 1
        else:
            # 状态回归到已确认状态，取消待确认
            pending_state = None
            pending_state_days = 0
        
        # 使用确认后的状态
        s = confirmed_state
        
        # Check if this is a rebalancing day
        should_rebalance = is_rebalance_day(date, rebal_freq, prev_date)
        
        # Get trends for this date - 使用decision_date（T-1日）来获取趋势信息
        # 这确保决策基于前一天的信息
        daily_trends = {}
        trend_lookup_date = decision_date if decision_date in trend_bear_all.index else date
        
        if use_proxies:
            proxy_trend_bear = False
            if '^GSPC' in trend_bear_all.columns and trend_lookup_date in trend_bear_all.index:
                proxy_trend_bear = trend_bear_all.loc[trend_lookup_date]['^GSPC']
            
            for t in ['IWY', 'G3B.SI', 'LVHI', 'SRT.SI', 'AJBU.SI']:
                daily_trends[t] = proxy_trend_bear
                
            gold_proxy = 'GLD'
            if trend_lookup_date < pd.Timestamp('2004-11-18') and 'GC=F' in trend_bear_all.columns:
                 gold_proxy = 'GC=F'
            
            if gold_proxy in trend_bear_all.columns and trend_lookup_date in trend_bear_all.index:
                daily_trends['GSD.SI'] = trend_bear_all.loc[trend_lookup_date][gold_proxy]

            bond_proxy = 'TLT'
            if 'VUSTX' in trend_bear_all.columns:
                bond_proxy = 'VUSTX'
            
            if bond_proxy in trend_bear_all.columns and trend_lookup_date in trend_bear_all.index:
                daily_trends['MBH.SI'] = trend_bear_all.loc[trend_lookup_date][bond_proxy]
        else:
            if trend_lookup_date in trend_bear_all.index:
                daily_trends = trend_bear_all.loc[trend_lookup_date].to_dict()
        
        # === 使用T-1日的指标数据做决策 ===
        # 从df_states中获取decision_date对应的指标（如果可用）
        if decision_date in df_states.index:
            decision_row = df_states.loc[decision_date]
            vix_val = decision_row.get('VIX')
            yc_val = decision_row.get('YieldCurve')
            sahm_val = decision_row.get('Sahm')
            corr_val = decision_row.get('Corr')
        else:
            vix_val = row.get('VIX')
            yc_val = row.get('YieldCurve')
            sahm_val = row.get('Sahm')
            corr_val = row.get('Corr')
        
        # 计算动量强度分数 (price - ma) / ma - 使用T-1日数据
        momentum_scores = {}
        momentum_date = decision_date if decision_date in price_data.index else date
        if momentum_date in price_data.index and momentum_date in ma_all.index:
            for ticker in price_data.columns:
                try:
                    p = price_data.loc[momentum_date, ticker]
                    m = ma_all.loc[momentum_date, ticker]
                    if pd.notna(p) and pd.notna(m) and m > 0:
                        momentum_scores[ticker] = (p - m) / m
                except:
                    pass
            # 映射代理资产的动量到原始资产
            if use_proxies and '^GSPC' in momentum_scores:
                momentum_scores['IWY'] = momentum_scores['^GSPC']
        
        # 检查近期VIX峰值（用于均值回归加仓）- 基于decision_date
        vix_recent_peak = None
        if 'VIX' in df_states.columns and decision_date in df_states.index:
            lookback_start = max(0, df_states.index.get_loc(decision_date) - 60)
            vix_history = df_states['VIX'].iloc[lookback_start:df_states.index.get_loc(decision_date)+1]
            if len(vix_history) > 0:
                vix_recent_peak = vix_history.max()
        
        # 检查近12个月是否曾深度倒挂 - 基于decision_date
        yc_recently_inverted = False
        if 'YieldCurve' in df_states.columns and decision_date in df_states.index:
            lookback_start = max(0, df_states.index.get_loc(decision_date) - 252)
            yc_history = df_states['YieldCurve'].iloc[lookback_start:df_states.index.get_loc(decision_date)+1]
            if len(yc_history) > 0:
                yc_recently_inverted = (yc_history.min() < -0.20)
        
        # v2.0: 计算IWY波动率（20日年化）用于波动率自适应控制
        iwy_vol = None
        iwy_proxy = 'IWY'
        if use_proxies:
            iwy_proxy = '^NDX' if '^NDX' in price_data.columns else '^GSPC'
        if iwy_proxy in returns_df.columns and decision_date in returns_df.index:
            vol_start = max(0, returns_df.index.get_loc(decision_date) - VOL_LOOKBACK)
            iwy_rets = returns_df[iwy_proxy].iloc[vol_start:returns_df.index.get_loc(decision_date)+1]
            if len(iwy_rets) > 5:  # 至少需要5天数据
                iwy_vol = iwy_rets.std() * np.sqrt(252)  # 年化波动率
        
        # Calculate base target weights (with new optimization parameters)
        targets = get_target_percentages(
            s, gold_bear=gb, value_regime=vr, asset_trends=daily_trends, 
            vix=vix_val, yield_curve=yc_val,
            sahm=sahm_val, corr=corr_val, momentum_scores=momentum_scores,
            yc_recently_inverted=yc_recently_inverted, vix_recent_peak=vix_recent_peak,
            iwy_volatility=iwy_vol
        )
        
        # === 优化4: VIX响应平滑化 ===
        # 替代原有的阶梯式VIX调整，使用连续函数
        if s == "NEUTRAL" and vix_val is not None and vix_val > VIX_SMOOTH_START:
            # 线性平滑响应: VIX从15到30线性减仓0到40%
            smooth_reduction = min((vix_val - VIX_SMOOTH_START) / (VIX_SMOOTH_END - VIX_SMOOTH_START) * VIX_MAX_REDUCTION, VIX_MAX_REDUCTION)
            iwy_current = targets.get('IWY', 0)
            move_amt = iwy_current * smooth_reduction
            targets['IWY'] = iwy_current - move_amt
            targets['WTMF'] = targets.get('WTMF', 0) + move_amt
        
        # --- Map Targets to Available Assets (Proxy Translation) ---
        new_target_weights = {}
        for t, w in targets.items():
            mapped_asset = map_target_to_asset(t, date)
            if mapped_asset != 'CASH' and mapped_asset in price_data.columns:
                new_target_weights[mapped_asset] = new_target_weights.get(mapped_asset, 0.0) + w
        
        # --- Calculate Drifted Weights from previous day ---
        drifted_weights = {}
        if prev_targets:
            drifted_values = {}
            total_drifted_val = 0.0
            
            for t, w in prev_targets.items():
                r = 0.0
                if prev_rets is not None and t in prev_rets:
                    r = prev_rets[t]
                val = w * (1 + r)
                drifted_values[t] = val
                total_drifted_val += val
                
            prev_cash_w = max(0.0, 1.0 - sum(prev_targets.values()))
            drifted_cash_val = prev_cash_w * 1.0
            total_drifted_val += drifted_cash_val
            
            if total_drifted_val > 0:
                drifted_weights = {t: v / total_drifted_val for t, v in drifted_values.items()}
            else:
                drifted_weights = prev_targets.copy()
        
        # === 优化6: 状态转换平滑过渡 ===
        # 新旧权重按过渡天数加权混合
        if is_in_transition and transition_from_weights:
            transition_day += 1
            transition_progress = min(transition_day / STATE_TRANSITION_DAYS, 1.0)
            
            # 混合权重
            blended_weights = {}
            all_assets = set(new_target_weights.keys()) | set(transition_from_weights.keys())
            for asset in all_assets:
                old_w = transition_from_weights.get(asset, 0.0)
                new_w = new_target_weights.get(asset, 0.0)
                blended_weights[asset] = old_w * (1 - transition_progress) + new_w * transition_progress
            
            new_target_weights = blended_weights
            
            if transition_day >= STATE_TRANSITION_DAYS:
                is_in_transition = False
                transition_from_weights = None
                transition_day = 0
        
        # === 优化5: 再平衡容忍带 ===
        # 只有当权重偏离超过阈值时才再平衡
        needs_rebalance_by_threshold = False
        if drifted_weights and new_target_weights:
            all_assets = set(new_target_weights.keys()) | set(drifted_weights.keys())
            for asset in all_assets:
                target_w = new_target_weights.get(asset, 0.0)
                drifted_w = drifted_weights.get(asset, 0.0)
                if abs(target_w - drifted_w) > REBALANCE_THRESHOLD:
                    needs_rebalance_by_threshold = True
                    break
        
        # 综合判断是否再平衡
        should_actually_rebalance = (should_rebalance and needs_rebalance_by_threshold) or not prev_targets or is_in_transition
        
        # --- Determine actual weights for today ---
        if should_actually_rebalance:
            final_weights = new_target_weights
        else:
            final_weights = drifted_weights if drifted_weights else new_target_weights
        
        # === 优化2: 波动率目标机制 (使用T-1数据，避免前视偏差) ===
        # 根据实现波动率调整仓位
        # 关键修复：使用portfolio_returns_history[:-1]，即不包含当天的收益（当天收益尚未发生）
        # 这样确保在t日做决策时，只使用t-1及之前的信息
        vol_history_for_calc = portfolio_returns_history[:-1] if len(portfolio_returns_history) > 1 else []
        if len(vol_history_for_calc) >= VOL_LOOKBACK:
            realized_vol = np.std(vol_history_for_calc[-VOL_LOOKBACK:]) * np.sqrt(252)
            if realized_vol > 0:
                vol_scalar = TARGET_VOL / realized_vol
                vol_scalar = max(VOL_SCALAR_MIN, min(vol_scalar, VOL_SCALAR_MAX))
                
                # 应用波动率缩放
                scaled_weights = {}
                total_weight = sum(final_weights.values())
                if total_weight > 0:
                    for asset, w in final_weights.items():
                        scaled_weights[asset] = w * vol_scalar
                    # 确保总权重不超过1（超出部分变为现金）
                    total_scaled = sum(scaled_weights.values())
                    if total_scaled > 1.0:
                        for asset in scaled_weights:
                            scaled_weights[asset] /= total_scaled
                    final_weights = scaled_weights
        
        # === 优化3: 动态止损机制（v1.5 分阶段恢复）===
        # 组合回撤超过阈值时减仓，恢复时分阶段渐进
        current_drawdown = (current_val - peak_nav) / peak_nav if peak_nav > 0 else 0
        
        if not in_stop_loss_mode and current_drawdown < DRAWDOWN_STOP_LOSS:
            # 触发止损
            in_stop_loss_mode = True
        elif in_stop_loss_mode:
            # v1.5: 分阶段恢复检查
            # 找到当前回撤对应的恢复阶段
            recovery_ratio = DRAWDOWN_REDUCE_RATIO  # 默认维持止损减仓
            for threshold, ratio in STOP_LOSS_RECOVERY_STAGES:
                if current_drawdown > threshold:
                    recovery_ratio = 1 - ratio  # 转换为减仓比例
                    if ratio >= 1.0:
                        in_stop_loss_mode = False  # 完全恢复
                    break
        
        if in_stop_loss_mode:
            # 止损模式：所有风险资产按恢复阶段减仓
            stop_loss_weights = {}
            # 计算当前恢复比例
            current_recovery_ratio = 1 - DRAWDOWN_REDUCE_RATIO  # 默认50%仓位
            for threshold, ratio in STOP_LOSS_RECOVERY_STAGES:
                if current_drawdown > threshold:
                    current_recovery_ratio = ratio
                    break
            
            for asset, w in final_weights.items():
                # WTMF和GSD视为避险资产，不减仓
                if asset in ['WTMF', 'GSD.SI']:
                    stop_loss_weights[asset] = w
                else:
                    stop_loss_weights[asset] = w * current_recovery_ratio
            final_weights = stop_loss_weights
        
        # --- Calculate Turnover (Trading Volume) ---
        daily_turnover = 0.0
        
        if not prev_targets:
            daily_turnover = sum(final_weights.values())
        elif should_actually_rebalance:
            diff_sum = 0.0
            all_assets = set(final_weights.keys()) | set(drifted_weights.keys())
            
            for t in all_assets:
                w_tgt = final_weights.get(t, 0.0)
                w_drift = drifted_weights.get(t, 0.0)
                diff_sum += abs(w_tgt - w_drift)
            
            curr_cash_w = max(0.0, 1.0 - sum(final_weights.values()))
            prev_cash_w = max(0.0, 1.0 - sum(drifted_weights.values())) if drifted_weights else 0
            diff_sum += abs(curr_cash_w - prev_cash_w)
            
            daily_turnover = diff_sum / 2.0
        
        # === 交易成本扣减 ===
        # transaction_cost_bps 是基点，1bps = 0.01% = 0.0001
        # 交易成本 = 换手率 * 成本率
        trading_cost = daily_turnover * (transaction_cost_bps / 10000.0)
            
        # Record history (with enhanced info)
        rec = targets.copy()
        rec['Date'] = date
        rec['State'] = s
        rec['RawState'] = raw_state  # 原始未确认状态
        rec['Turnover'] = daily_turnover
        rec['TradingCost'] = trading_cost  # 新增：记录交易成本
        rec['Rebalanced'] = should_actually_rebalance
        rec['InStopLoss'] = in_stop_loss_mode
        rec['Drawdown'] = current_drawdown
        rec['InTransition'] = is_in_transition
        history_records.append(rec)
        
        # Calculate Portfolio Return for this day
        daily_ret = 0.0
        current_rets = pd.Series(dtype=float)
        
        if date in returns_df.index:
            current_rets = returns_df.loc[date]
            for t, w in final_weights.items():
                if t in current_rets:
                    daily_ret += w * current_rets[t]
        
        # === 扣除交易成本 ===
        daily_ret -= trading_cost
        
        # 记录收益用于波动率计算
        portfolio_returns_history.append(daily_ret)
        
        current_val = current_val * (1 + daily_ret)
        portfolio_values.append(current_val)
        
        # 更新历史最高净值
        if current_val > peak_nav:
            peak_nav = current_val
        
        # Prepare for next iteration
        prev_targets = final_weights
        prev_rets = current_rets
        prev_date = date

        
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

# --- Utility Functions for Backtest ---
def safe_div(a, b, default=0.0):
    """Safe division to avoid ZeroDivisionError."""
    return a / b if b != 0 else default

def get_state_segments(df, state_col='State'):
    """
    Extract state segments from a DataFrame with state column.
    Returns DataFrame with columns: grp, State, Start, End, Duration
    """
    if df is None or df.empty or state_col not in df.columns:
        return pd.DataFrame()
    df_copy = df.copy()
    df_copy['state_grp'] = (df_copy[state_col] != df_copy[state_col].shift()).cumsum()
    segments = df_copy.groupby(['state_grp', state_col]).agg(
        Start=(state_col, lambda x: x.index[0]),
        End=(state_col, lambda x: x.index[-1])
    ).reset_index()
    segments.columns = ['grp', 'State', 'Start', 'End']
    segments['Duration'] = (segments['End'] - segments['Start']).dt.days + 1
    return segments

def validate_date_range(start_date, end_date, min_days=30):
    """
    Validate date range for backtest.
    Returns (is_valid, error_message)
    """
    if start_date is None or end_date is None:
        return False, "请选择有效的日期范围"
    if start_date >= end_date:
        return False, "结束日期必须晚于开始日期"
    if (end_date - start_date).days < min_days:
        return False, f"回测周期至少需要 {min_days} 天"
    return True, None

def normalize_weights(weights_dict):
    """
    Normalize weights to sum to 1.0, handling edge cases.
    """
    if not weights_dict:
        return {}
    total = sum(weights_dict.values())
    if total <= 0:
        return {k: 0.0 for k in weights_dict}
    return {k: v / total for k, v in weights_dict.items()}

def calculate_state_transition_matrix(df_states, state_col='State'):
    """
    Calculate state transition matrix from state history.
    Returns DataFrame with transition counts and probabilities.
    """
    if df_states is None or df_states.empty or state_col not in df_states.columns:
        return None, None
    states = df_states[state_col]
    transitions = pd.crosstab(states.shift(1), states, dropna=True)
    # Normalize to probabilities
    trans_prob = transitions.div(transitions.sum(axis=1), axis=0).fillna(0)
    return transitions, trans_prob

def calculate_state_statistics(df_history, state_col='State'):
    """
    Calculate statistics for each state: count, avg duration, total days.
    """
    segments = get_state_segments(df_history, state_col)
    if segments.empty:
        return pd.DataFrame()
    stats = segments.groupby('State').agg(
        Occurrences=('grp', 'count'),
        AvgDuration=('Duration', 'mean'),
        TotalDays=('Duration', 'sum'),
        MinDuration=('Duration', 'min'),
        MaxDuration=('Duration', 'max')
    ).round(1)
    return stats

def calculate_state_returns(df_history, returns_series, state_col='State'):
    """
    Calculate return statistics by state.
    """
    if df_history is None or df_history.empty or returns_series is None or returns_series.empty:
        return pd.DataFrame()
    # Align indices
    common_idx = df_history.index.intersection(returns_series.index)
    if len(common_idx) == 0:
        return pd.DataFrame()
    states = df_history.loc[common_idx, state_col]
    rets = returns_series.loc[common_idx]
    
    result = rets.groupby(states).agg(['mean', 'std', 'sum', 'count'])
    result.columns = ['AvgDailyRet', 'StdDev', 'CumulativeRet', 'Days']
    result['AvgDailyRet'] = result['AvgDailyRet'] * 100  # Convert to %
    result['StdDev'] = result['StdDev'] * 100
    result['CumulativeRet'] = result['CumulativeRet'] * 100
    result['AnnualizedRet'] = result['AvgDailyRet'] * 252
    return result.round(2)

def determine_macro_state(row, params=None):
    """
    Determines macro state based on a row of indicators.
    Expected row keys: Sahm, RateShock, Corr, VIX, Trend_Bear
    v2.1: 优化EXTREME_ACCUMULATION触发条件，更早抄底
    """
    if params is None:
        params = {
            'sahm_threshold': 0.50,
            'rate_shock_threshold': 0.20,
            'corr_threshold': 0.30,
            'vix_panic': VIX_EXTREME_THRESHOLD,  # v2.1: 使用优化后的阈值
            'vix_recession': 35,
            'vix_elevated': 20
        }
        
    is_rec = row['Sahm'] >= params['sahm_threshold']
    is_shock = row['RateShock'] > params['rate_shock_threshold']
    is_c_broken = row['Corr'] > params['corr_threshold']
    is_f = row['VIX'] > params['vix_panic']
    is_down = row['Trend_Bear']
    is_vol_elevated = row['VIX'] > params['vix_elevated']
    
    # v2.1: 优化状态判断顺序，让EXTREME_ACCUMULATION更容易触发
    if is_shock or (is_rec and is_c_broken):
        return "INFLATION_SHOCK"
    elif is_rec or (is_down and row['VIX'] > params['vix_recession']):
        return "DEFLATION_RECESSION"
    elif is_f and not is_shock and not is_rec:
        # v2.1: VIX>30即触发抄底（原32），且不要求趋势下跌
        return "EXTREME_ACCUMULATION"
    elif is_down:
        return "CAUTIOUS_TREND"
    elif is_vol_elevated:
        return "CAUTIOUS_VOL"
    else:
        return "NEUTRAL"

@st.cache_data(ttl=900, show_spinner=False)
def fetch_yf_with_retry(tickers, start=None, end=None, auto_adjust=False, attempts: int = 2, backoff: int = 3, interval: str = "1d"):
    tickers_list = list(tickers) if isinstance(tickers, (list, tuple, set)) else [tickers]
    last_err = None
    for i in range(attempts):
        try:
            data_raw = yf.download(
                tickers_list,
                start=start,
                end=end,
                progress=False,
                auto_adjust=auto_adjust,
                timeout=12,
                interval=interval,
            )
            if data_raw is not None and not data_raw.empty:
                return data_raw
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff * (i + 1))
    log_event("ERROR", "yfinance download failed", {"tickers": tickers_list, "err": last_err})
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def get_live_prices(tickers):
    tickers_list = [t for t in (list(tickers) if isinstance(tickers, (list, tuple, set)) else [tickers]) if t]
    if not tickers_list:
        return {}
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=7)
    df_raw = fetch_yf_with_retry(tickers_list, start=start, end=end, auto_adjust=False)
    if df_raw is None or df_raw.empty:
        return {}
    df = normalize_yf_prices(df_raw).ffill().tail(2)
    if df.empty:
        return {}
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    out = {}
    for t in tickers_list:
        try:
            p = latest.get(t)
            prev_p = prev.get(t)
            if pd.isna(p):
                continue
            change_pct = None
            if prev_p and not pd.isna(prev_p) and prev_p != 0:
                change_pct = (p - prev_p) / prev_p * 100
            out[t] = {"price": float(p), "change_pct": float(change_pct) if change_pct is not None else None}
        except Exception:
            continue
    return out


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
    df_all = fetch_yf_with_retry(tickers, start=fetch_start, end=fetch_end, auto_adjust=False)
    if df_all is None or df_all.empty:
        return pd.DataFrame(), f"Market data fetch failed. 请检查网络连接或稍后重试。(tickers: {tickers})"

    data = normalize_yf_prices(df_all)
    
    if data.empty:
        # 提供更详细的诊断信息
        col_info = str(df_all.columns.tolist())[:200] if df_all is not None else "None"
        return pd.DataFrame(), f"数据格式解析失败。原始列: {col_info}"
    
    # 诊断: 检查各列有效数据量
    col_stats = {}
    for c in data.columns:
        valid_count = data[c].dropna().shape[0]
        total_count = len(data)
        col_stats[c] = f"{valid_count}/{total_count}"
    
    valid_cols = [c for c in data.columns if data[c].dropna().shape[0] > 0]
    if not valid_cols:
        return pd.DataFrame(), f"所有列数据为空。列统计: {col_stats}"
    
    # 如果 IWY 为空但 ^GSPC 有数据，记录警告
    if 'IWY' in data.columns and data['IWY'].dropna().shape[0] == 0:
        if '^GSPC' in data.columns and data['^GSPC'].dropna().shape[0] > 0:
            log_event("WARN", "IWY data empty, will use ^GSPC as proxy", {"col_stats": col_stats})

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
        
        # 检查 ETF 是否有有效数据（不只是列存在，至少要有10%的数据不为空）
        min_valid_ratio = 0.1  # 至少10%的数据有效
        iwy_has_data = ('IWY' in data.columns) and (data['IWY'].dropna().shape[0] > len(data) * min_valid_ratio)
        tlt_has_data = ('TLT' in data.columns) and (data['TLT'].dropna().shape[0] > len(data) * min_valid_ratio)
        prefer_etfs = iwy_has_data and tlt_has_data and (not use_proxies)

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
        # Only use IWY/IWD if NOT using proxies and both have valid data
        iwd_has_data = ('IWD' in data.columns) and (data['IWD'].dropna().shape[0] > len(data) * min_valid_ratio)
        if not use_proxies and iwy_has_data and iwd_has_data:
            pair_ratio = data['IWY'] / data['IWD']
            pair_ma = pair_ratio.rolling(ma_window).mean()
            style_value_regime = pair_ratio < pair_ma 
        else:
            # Fallback for style if ETFs missing
            style_value_regime = pd.Series(False, index=data.index)

        # Assemble DataFrame
        # 获取 VIX 数据，如果缺失则用默认值15（中性水平）
        vix_series = data.get('^VIX', None)
        if vix_series is None or vix_series.dropna().empty:
            vix_series = pd.Series(15.0, index=data.index)
        else:
            vix_series = vix_series.ffill().bfill().fillna(15.0)
        
        # 确保 iwy_series 有数据（前向/后向填充）
        if iwy_series is not None and not iwy_series.dropna().empty:
            iwy_series = iwy_series.ffill().bfill()
        else:
            # 最后兜底：如果 iwy_series 仍然全空，使用 ^GSPC
            if '^GSPC' in data.columns and data['^GSPC'].dropna().shape[0] > 0:
                iwy_series = data['^GSPC'].ffill().bfill()
                log_event("WARN", "Using ^GSPC as IWY proxy due to empty IWY data")
            else:
                return pd.DataFrame(), "无法获取 IWY 或 ^GSPC 价格数据"
        
        df_hist = pd.DataFrame({
            'IWY': iwy_series,
            'Sahm': sahm_series,
            'RateShock': tnx_roc,
            'Corr': corr,
            'VIX': vix_series,
            'Trend_Bear': trend_bear,
            'YieldCurve': yc_daily['T10Y2Y'],
            'Gold_Bear': gold_trend_bear,
            'Value_Regime': style_value_regime
        })
        
        # 用前向填充处理缺失值，而不是直接删除（避免丢失过多数据）
        df_hist = df_hist.ffill().bfill()
        
        # 只删除关键列仍为空的行
        essential_cols = ['IWY', 'VIX', 'Trend_Bear']
        
        # 诊断: 检查关键列的缺失情况
        missing_info = {}
        for col in essential_cols:
            if col in df_hist.columns:
                na_count = df_hist[col].isna().sum()
                total = len(df_hist)
                missing_info[col] = f"{na_count}/{total}"
            else:
                missing_info[col] = "列不存在"
        
        df_hist = df_hist.dropna(subset=essential_cols)
        
        if df_hist.empty:
            diag_msg = f"关键列缺失情况: {missing_info}; 可用列: {list(data.columns)}; 日期范围: {data.index.min()} ~ {data.index.max()}"
            return pd.DataFrame(), f"数据处理后为空。{diag_msg}"
        
        # 4. Determine States
        # Pass params to the state determinator
        df_hist['State'] = df_hist.apply(lambda row: determine_macro_state(row, params), axis=1)
        
        # Filter Output
        df_final = df_hist.loc[(df_hist.index >= pd.to_datetime(start_date)) & (df_hist.index <= pd.to_datetime(end_date))]
        
        if df_final.empty:
            return pd.DataFrame(), f"所选日期范围 ({start_date} ~ {end_date}) 内无有效数据，请尝试更近的日期或启用 Use Proxies"
        
        return df_final, None

    except Exception as e:
        import traceback
        return pd.DataFrame(), f"计算错误: {str(e)}\n{traceback.format_exc()}"

# --- UI Components ---

def render_manual_data_import():
    """Renders the manual data import expander."""
    with st.expander("📂 手动导入宏观数据 (网络受限时使用)", expanded=False):
        st.info("如果网络受限导致 FRED 数据 (UNRATE, T10Y2Y) 获取失败，请手动下载并上传 CSV 文件。")
        col_u1, col_u2 = st.columns(2)

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
        cols = st.columns(2)
        
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
            with cols[i % 2]:
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
    """Renders the metrics dashboard with mini trendlines."""
    st.markdown("### 📊 核心宏观因子 (Macro Factors)")
    hist = metrics.get('factor_trends')
    if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
        hist = pd.DataFrame()

    def get_series(col):
        if isinstance(hist, pd.DataFrame) and col in hist.columns:
            return hist[col].dropna()
        return pd.Series(dtype=float)

    def sparkline_fig(series, color="#2962FF"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", line=dict(color=color, width=2), hovertemplate="%{y:.2f}<extra></extra>"))
        fig.update_layout(
            height=140,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(title=None, zeroline=False, showgrid=True, tickfont=dict(size=10)),
        )
        return fig

    factor_items = [
        {
            "title": "利率冲击 (TNX ROC)",
            "value": f"{metrics['tnx_roc']:+.1%}",
            "status": "⚠️ 触发" if metrics['rate_shock'] else "✅ 安全",
            "color": "#7c3aed",
            "series": get_series("RateShock"),
            "delta": metrics['tnx_roc'],
        },
        {
            "title": "衰退信号 (Sahm)",
            "value": f"{metrics['sahm']:.2f}",
            "status": "⚠️ 触发" if metrics['recession'] else "✅ 安全",
            "color": "#0ea5e9",
            "series": get_series("Sahm"),
            "delta": metrics['sahm'],
        },
        {
            "title": "股债相关性 (Corr)",
            "value": f"{metrics['corr']:.2f}",
            "status": "⚠️ 失效" if metrics['corr_broken'] else "✅ 正常",
            "color": "#fb923c",
            "series": get_series("Corr"),
            "delta": metrics['corr'],
        },
        {
            "title": "恐慌指数 (VIX)",
            "value": f"{metrics['vix']:.1f}",
            "status": "⚠️ 恐慌" if metrics['fear'] else "✅ 正常",
            "color": "#ef4444",
            "series": get_series("VIX"),
            "delta": metrics['vix'],
        },
    ]

    for i in range(0, len(factor_items), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j >= len(factor_items):
                continue
            item = factor_items[i + j]
            with cols[j]:
                st.metric(item["title"], item["value"], item["status"], delta_color="inverse" if "⚠️" in item["status"] else "normal")
                if not item["series"].empty:
                    st.plotly_chart(sparkline_fig(item["series"], item["color"]), use_container_width=True)

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
        yield_curve=metrics.get('yield_curve'),
        sahm=metrics.get('sahm'),
        corr=metrics.get('corr'),
        yc_recently_inverted=metrics.get('yc_un_invert', False)
    )
    
    if adjustments:
        with st.expander("🔧 动态风控触发 (Active Strategy Adjustments)", expanded=True):
            for adj in adjustments:
                st.markdown(f"- {adj}")


def render_data_health_badges(metrics):
    freshness_days = metrics.get('freshness_days')
    latest_date = metrics.get('date')
    warnings = metrics.get('data_warnings', []) or []
    fetch_ts = metrics.get('fetch_ts')
    badge = "🟢 数据最新"
    note = f"数据截至 {latest_date}" if latest_date else "数据时间未知"
    if freshness_days is not None:
        note += f" ｜ 滞后 {freshness_days} 天" if freshness_days > 0 else " ｜ 当日数据"
    if fetch_ts:
        note += f" ｜ 上次拉取: {fetch_ts}"
    if freshness_days is not None and freshness_days > 5:
        badge = "🔴 数据已过期"
    elif freshness_days is not None and freshness_days > 2:
        badge = "🟡 数据待更新"

    st.markdown(
        f"""
        <div style="padding:12px;border-radius:8px;border:1px solid #e5e7eb;background:#f8fafc;margin-bottom:12px;">
            <div style="font-weight:700;color:#0f172a;">{badge}</div>
            <div style="color:#475467;font-size:13px;">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if warnings:
        with st.expander("⚠️ 数据健康提醒", expanded=False):
            for w in warnings:
                st.markdown(f"- {w}")

def render_rebalancing_table(state, current_holdings, total_value, is_gold_bear, is_value_regime, asset_trends=None, vix=None, yield_curve=None, price_info=None, sahm=None, corr=None, yc_recently_inverted=False):
    """Renders the rebalancing table with live prices."""
    if asset_trends is None: asset_trends = {}
    targets = get_target_percentages(state, gold_bear=is_gold_bear, value_regime=is_value_regime, asset_trends=asset_trends, vix=vix, yield_curve=yield_curve, sahm=sahm, corr=corr, yc_recently_inverted=yc_recently_inverted)
    
    # Add Current Holdings not in targets
    all_tickers = set(targets.keys()).union(current_holdings.keys())
    if price_info is None:
        price_info = get_live_prices(all_tickers)
    
    data = []
    if total_value == 0:
        st.warning("⚠️ 请输入持仓市值以获取建议。")
        return

    for tkr in all_tickers:
        tgt_pct = targets.get(tkr, 0.0)
        curr_val = current_holdings.get(tkr, 0.0)
        curr_pct = curr_val / total_value if total_value > 0 else 0
        
        diff_val = (tgt_pct - curr_pct) * total_value
        price = price_info.get(tkr, {}).get("price") if price_info else None
        chg = price_info.get(tkr, {}).get("change_pct") if price_info else None
        price_text = f"${price:,.2f}" if price is not None else "-"
        chg_text = f"{chg:+.2f}%" if chg is not None else "-"
        
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
            "最新价": price_text,
            "日变动": chg_text,
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
                "日变动": st.column_config.TextColumn(help="相对前一交易日的涨跌幅"),
            },
            hide_index=True,
            use_container_width=True
        )


def render_export_options(metrics, adjustments, targets):
    state = metrics.get('state')
    report_date = metrics.get('date')
    lines = [
        f"诊断时间: {metrics.get('fetch_ts', '')}",
        f"数据截至: {report_date}",
        f"当前状态: {state} ({MACRO_STATES.get(state, {}).get('display', '')})",
        "",
        "关键因子:",
        f"- 利率冲击: {metrics.get('tnx_roc', 0):+.1%}",
        f"- Sahm: {metrics.get('sahm', 0):.2f}",
        f"- 股债相关性: {metrics.get('corr', 0):.2f}",
        f"- VIX: {metrics.get('vix', 0):.1f}",
        f"- 收益率曲线: {metrics.get('yield_curve', 0):.2f}%",
        "",
        "动态风控触发:",
    ]
    lines.extend([f"- {a}" for a in adjustments] or ["- 无"])
    lines.append("")
    lines.append("目标配置:")
    for k, v in targets.items():
        if v > 0:
            lines.append(f"- {ASSET_NAMES.get(k, k)} ({k}): {v*100:.1f}%")
    summary = "\n".join(lines)

    st.markdown("#### 📤 导出诊断结果")
    st.text_area("诊断摘要 (可复制)", summary, height=160)
    st.download_button(
        label="下载诊断摘要 (.txt)",
        data=summary.encode('utf-8'),
        file_name=f"diagnosis_{report_date or 'latest'}.txt",
        mime="text/plain",
        use_container_width=True,
    )


def render_historical_backtest_section():
    """Renders the independent historical backtest section."""
    st.markdown("---")
    st.markdown("### 🕰️ 历史状态回溯与策略仿真")
    
    # --- Initialize all session state at the top ---
    session_defaults = {
        "bt_use_proxies": False,
        "bt_ma_window": 200,
        "bt_p_sahm": 0.50,
        "bt_p_vix_panic": 32,
        "bt_p_vix_rec": 35,
        "bt_rebal_freq": "Daily",
        "bt_cost_bps": 10,
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    
    # --- Advanced Settings (Sensitivity & Proxies) ---
    with st.expander("⚙️ 高级回测设置 (参数敏感性与样本外测试)", expanded=False):
        # Reset Button
        if st.button("🔄 恢复默认设置"):
            for key, val in session_defaults.items():
                st.session_state[key] = val
            st.rerun()

        c_adv1, c_adv2, c_adv3 = st.columns(3)
        with c_adv1:
            st.markdown("**1. 样本外测试 (Out-of-Sample)**")
            use_proxies = st.checkbox("启用代理资产 (Use Proxies)", help="使用 S&P500, VUSTX(1986+), GC=F 等替代 ETF 以支持更长历史回测 (1990+)。", key="bt_use_proxies")
            ma_window = st.number_input("动量窗口 (MA Window)", step=10, help="默认 200 日均线。尝试 150 或 250 测试敏感性。", key="bt_ma_window")
            
        with c_adv2:
            st.markdown("**2. 阈值敏感性 (Sensitivity)**")
            p_sahm = st.number_input("Sahm Rule", step=0.01, format="%.2f", key="bt_p_sahm")
            p_vix_panic = st.number_input("VIX Panic", step=1, key="bt_p_vix_panic")
            p_vix_rec = st.number_input("VIX Recession", step=1, key="bt_p_vix_rec")
        
        with c_adv3:
            st.markdown("**3. 交易参数 (Trading)**")
            rebal_freq = st.selectbox("再平衡频率", ["Daily", "Weekly", "Monthly", "Quarterly"], key="bt_rebal_freq", help="Daily=每日, Weekly=每周一, Monthly=每月初, Quarterly=每季度初")
            cost_bps = st.number_input("交易成本 (bps)", min_value=0, max_value=100, step=5, key="bt_cost_bps", help="单边交易成本，默认 10bps = 0.1%")
    
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
    
    # --- Date Validation ---
    if run:
        if not isinstance(dates, (tuple, list)) or len(dates) != 2:
            st.error("请选择有效的日期范围（开始日期和结束日期）")
            return
        is_valid, err_msg = validate_date_range(dates[0], dates[1], min_days=30)
        if not is_valid:
            st.error(err_msg)
            return
        
    if run and isinstance(dates, (tuple, list)) and len(dates)==2:
        with st.spinner("回测中..."):
            df_states, err = get_historical_macro_data(dates[0], dates[1], ma_window=int(ma_window), params=custom_params, use_proxies=use_proxies)
            if err:
                st.error(f"获取宏观数据失败: {err}")
                st.info("💡 提示：如果日期范围较早，请尝试勾选 '启用代理资产 (Use Proxies)' 选项")
                return
            if df_states.empty:
                st.error("宏观数据为空，请检查日期范围或网络连接")
                return
            
            res, df_history, bt_err = run_dynamic_backtest(df_states, dates[0], dates[1], cap, ma_window=int(ma_window), use_proxies=use_proxies, rebal_freq=rebal_freq, transaction_cost_bps=cost_bps)
            if res is None:
                st.error(f"回测执行失败: {bt_err}")
                st.info("💡 提示：请尝试缩短日期范围或勾选 'Use Proxies' 选项")
                return
            
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
                    
                    # Est Cost (use user-defined cost_bps)
                    total_cost_est = total_turnover * (cost_bps / 10000)
                    annual_cost_est = annual_turnover * (cost_bps / 10000)
                    
                    # Avg Holding Period (Days)
                    avg_hold_days = safe_div(1, avg_daily_turnover, 0)
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
                    freq_pct = safe_div(active_days, total_days, 0)
                    st.metric("活跃交易频率", f"{freq_pct:.1%}", help="日换手率超过 1% 的天数比例")

                # Cost Sensitivity Analysis
                st.markdown("**交易成本敏感性分析 (Cost Sensitivity)**")
                cost_levels = [5, 10, 15, 20, 30]
                cost_impact = []
                for c_bps in cost_levels:
                    annual_drag = annual_turnover * (c_bps / 10000) * 100  # Convert to %
                    cost_impact.append({'成本(bps)': c_bps, '年化拖累%': annual_drag})
                df_cost = pd.DataFrame(cost_impact)
                
                c_sens1, c_sens2 = st.columns([1, 2])
                with c_sens1:
                    st.dataframe(df_cost.style.format({'年化拖累%': '{:.2f}%'}), use_container_width=True, hide_index=True)
                with c_sens2:
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Bar(x=df_cost['成本(bps)'].astype(str) + ' bps', y=df_cost['年化拖累%'], marker_color='#ff7043'))
                    fig_cost.update_layout(title="不同成本水平下的年化拖累", yaxis_title="年化拖累%", template="plotly_white", height=250)
                    st.plotly_chart(fig_cost, use_container_width=True)

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
            
            # --- 4.5 v1.5 优化机制效果分析 ---
            if df_history is not None and not df_history.empty:
                st.markdown("---")
                st.markdown("#### ⚙️ v1.5 优化机制效果 (Optimization Impact)")
                st.caption("展示各优化模块在回测期间的触发情况与效果")
                
                # 止损触发统计
                if 'InStopLoss' in df_history.columns:
                    stop_loss_days = df_history['InStopLoss'].sum()
                    stop_loss_pct = stop_loss_days / len(df_history) * 100
                    
                    # 计算止损保护效果 (止损期间的平均回撤恢复)
                    if 'Drawdown' in df_history.columns:
                        sl_drawdowns = df_history[df_history['InStopLoss']]['Drawdown']
                        avg_sl_drawdown = sl_drawdowns.mean() * 100 if len(sl_drawdowns) > 0 else 0
                else:
                    stop_loss_days = 0
                    stop_loss_pct = 0
                    avg_sl_drawdown = 0
                
                # 状态过渡统计
                if 'InTransition' in df_history.columns:
                    transition_days = df_history['InTransition'].sum()
                else:
                    transition_days = 0
                
                # 实际再平衡统计
                if 'Rebalanced' in df_history.columns:
                    rebal_days = df_history['Rebalanced'].sum()
                    rebal_pct = rebal_days / len(df_history) * 100
                else:
                    rebal_days = 0
                    rebal_pct = 0
                
                # 显示统计卡片
                col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
                with col_opt1:
                    st.metric(
                        "🛡️ 止损保护天数", 
                        f"{stop_loss_days} 天 ({stop_loss_pct:.1f}%)",
                        help=f"触发止损机制的天数（回撤>{abs(DRAWDOWN_STOP_LOSS)*100:.0f}%）"
                    )
                with col_opt2:
                    st.metric(
                        "📉 止损期平均回撤",
                        f"{avg_sl_drawdown:.2f}%" if stop_loss_days > 0 else "N/A",
                        help="止损保护期间的平均回撤水平"
                    )
                with col_opt3:
                    st.metric(
                        "🔀 状态过渡天数",
                        f"{transition_days} 天",
                        help=f"状态切换时的平滑过渡期（{STATE_TRANSITION_DAYS}天渐进）"
                    )
                with col_opt4:
                    st.metric(
                        "📊 实际再平衡",
                        f"{rebal_days} 次 ({rebal_pct:.1f}%)",
                        help=f"超过容忍带{REBALANCE_THRESHOLD*100:.0f}%才再平衡"
                    )
                
                # 详细优化机制说明
                with st.expander("📖 v1.5 优化机制详解", expanded=False):
                    st.markdown(f"""
**当前启用的优化机制：**

| 机制 | 参数 | 说明 |
|------|------|------|
| 🛡️ 动态止损 | 回撤>{abs(DRAWDOWN_STOP_LOSS)*100:.0f}%触发 | 减仓{DRAWDOWN_REDUCE_RATIO*100:.0f}%，分阶段恢复 |
| 🔄 信号确认 | {SIGNAL_CONFIRM_DAYS}天确认期 | 状态切换需连续{SIGNAL_CONFIRM_DAYS}天确认 |
| 📊 波动率目标 | 目标{TARGET_VOL*100:.0f}%年化 | 根据实现波动率动态调整仓位 |
| 📏 再平衡带 | >{REBALANCE_THRESHOLD*100:.0f}%才调仓 | 减少频繁交易成本 |
| 🔀 平滑过渡 | {STATE_TRANSITION_DAYS}天过渡 | 状态切换时渐进调仓 |
| 💵 现金缓冲 | VIX>{CASH_BUFFER_VIX_THRESHOLD:.0f}时增加 | 基础{CASH_BUFFER_BASE*100:.0f}%，最高{CASH_BUFFER_MAX*100:.0f}% |
| 📈 双均线趋势 | MA{TREND_MA_SHORT}/MA{TREND_MA_LONG} | 强熊减{STRONG_BEAR_REDUCTION*100:.0f}%，弱熊减{WEAK_BEAR_REDUCTION*100:.0f}% |
| 🔗 相关性渐进 | {CORR_MID_THRESHOLD}-{CORR_HIGH_THRESHOLD}区间 | 股债相关性升高时渐进转移 |
| 📊 市场广度 | <{MARKET_BREADTH_LOW*100:.0f}%时保守 | 跨资产动量共振检测 |
""")
                
                # 止损触发时间线
                if 'InStopLoss' in df_history.columns and stop_loss_days > 0:
                    st.markdown("**🛡️ 止损保护时间线**")
                    
                    # 找出止损区间
                    df_sl = df_history.copy()
                    df_sl['sl_change'] = df_sl['InStopLoss'].astype(int).diff().fillna(0)
                    
                    sl_starts = df_sl[df_sl['sl_change'] == 1].index.tolist()
                    sl_ends = df_sl[df_sl['sl_change'] == -1].index.tolist()
                    
                    # 匹配止损区间
                    sl_periods = []
                    for i, start in enumerate(sl_starts):
                        # 找到对应的结束点
                        end = None
                        for e in sl_ends:
                            if e > start:
                                end = e
                                break
                        if end is None:
                            end = df_history.index[-1]
                        duration = (end - start).days
                        sl_periods.append({
                            '开始': start.strftime('%Y-%m-%d'),
                            '结束': end.strftime('%Y-%m-%d'),
                            '持续(天)': duration
                        })
                    
                    if sl_periods:
                        st.dataframe(pd.DataFrame(sl_periods), hide_index=True, use_container_width=True)
                    
                    # --- 5. State Transition Analysis ---
                    st.markdown("---")
                    st.markdown("#### 🔄 状态转换分析 (State Transition Analysis)")
                    
                    if df_history is not None and not df_history.empty and 'State' in df_history.columns:
                        tab_trans, tab_attr, tab_yearly = st.tabs(["状态转换矩阵", "收益归因", "分年度收益"])
                        
                        with tab_trans:
                            # State Transition Matrix
                            trans_counts, trans_probs = calculate_state_transition_matrix(df_history, 'State')
                            if trans_counts is not None and trans_probs is not None and not trans_counts.empty:
                                c_mat1, c_mat2 = st.columns(2)
                                with c_mat1:
                                    st.markdown("**转换次数 (Counts)**")
                                    st.dataframe(trans_counts.style.background_gradient(cmap='Blues'), use_container_width=True)
                                with c_mat2:
                                    st.markdown("**转换概率 (Probabilities)**")
                                    st.dataframe(trans_probs.style.background_gradient(cmap='Greens', vmin=0, vmax=1).format("{:.1%}"), use_container_width=True)
                                
                                # State Statistics
                                st.markdown("**状态统计 (State Statistics)**")
                                state_stats = calculate_state_statistics(df_history, 'State')
                                if not state_stats.empty:
                                    state_stats_display = state_stats.copy()
                                    state_stats_display.columns = ['出现次数', '平均持续(天)', '总天数', '最短(天)', '最长(天)']
                                    st.dataframe(state_stats_display, use_container_width=True)
                                    
                                    # Duration Distribution Chart
                                    segments = get_state_segments(df_history, 'State')
                                    if not segments.empty:
                                        fig_dur = go.Figure()
                                        for state in segments['State'].unique():
                                            durations = segments[segments['State'] == state]['Duration']
                                            s_conf = MACRO_STATES.get(state, MACRO_STATES["NEUTRAL"])
                                            fig_dur.add_trace(go.Box(y=durations, name=f"{s_conf['icon']} {state}", marker_color=s_conf['border_color']))
                                        fig_dur.update_layout(title="状态持续时间分布 (Duration Distribution)", yaxis_title="天数", template="plotly_white", height=350)
                                        st.plotly_chart(fig_dur, use_container_width=True)
                            else:
                                st.info("状态数据不足以生成转换矩阵")
                        
                        with tab_attr:
                            # Attribution Analysis by State
                            if 'Dynamic Strategy' in res.columns:
                                daily_rets = res['Dynamic Strategy'].pct_change().dropna()
                                state_rets = calculate_state_returns(df_history, daily_rets, 'State')
                                if not state_rets.empty:
                                    st.markdown("**按状态收益归因 (Returns by State)**")
                                    state_rets_display = state_rets.copy()
                                    state_rets_display.columns = ['日均收益%', '标准差%', '累计收益%', '天数', '年化收益%']
                                    st.dataframe(state_rets_display.style.background_gradient(subset=['累计收益%'], cmap='RdYlGn'), use_container_width=True)
                                    
                                    # Contribution Bar Chart
                                    fig_attr = go.Figure()
                                    for state in state_rets.index:
                                        s_conf = MACRO_STATES.get(state, MACRO_STATES["NEUTRAL"])
                                        fig_attr.add_trace(go.Bar(
                                            x=[f"{s_conf['icon']} {state}"],
                                            y=[state_rets.loc[state, 'CumulativeRet']],
                                            name=state,
                                            marker_color=s_conf['border_color']
                                        ))
                                    fig_attr.update_layout(title="各状态收益贡献 (Contribution by State)", yaxis_title="累计收益%", template="plotly_white", showlegend=False, height=350)
                                    st.plotly_chart(fig_attr, use_container_width=True)
                                else:
                                    st.info("无法计算收益归因")
                            else:
                                st.info("需要 Dynamic Strategy 列来计算归因")
                        
                        with tab_yearly:
                            # Yearly Returns Table
                            if 'Dynamic Strategy' in res.columns:
                                # Calculate yearly returns
                                yearly_data = []
                                for col in res.columns:
                                    yearly_rets = res[col].resample('Y').last().pct_change().dropna() * 100
                                    for yr, ret in yearly_rets.items():
                                        yearly_data.append({'策略': col, '年份': yr.year, '收益率%': ret})
                                
                                if yearly_data:
                                    df_yearly = pd.DataFrame(yearly_data)
                                    df_yearly_pivot = df_yearly.pivot(index='年份', columns='策略', values='收益率%')
                                    
                                    st.markdown("**分年度收益率 (Yearly Returns)**")
                                    st.dataframe(df_yearly_pivot.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}%"), use_container_width=True)
                                    
                                    # Yearly Bar Chart
                                    fig_yearly = go.Figure()
                                    for col in df_yearly_pivot.columns:
                                        fig_yearly.add_trace(go.Bar(x=df_yearly_pivot.index.astype(str), y=df_yearly_pivot[col], name=col))
                                    fig_yearly.update_layout(title="分年度收益对比 (Yearly Returns Comparison)", yaxis_title="收益率%", barmode='group', template="plotly_white", height=400)
                                    st.plotly_chart(fig_yearly, use_container_width=True)
                                    
                                    # Monthly Heatmap for Dynamic Strategy
                                    st.markdown("**月度收益热力图 (Monthly Returns Heatmap)**")
                                    daily_s = res['Dynamic Strategy']
                                    if len(daily_s) > 30:
                                        monthly_rets = daily_s.resample('M').last().pct_change().dropna() * 100
                                        if len(monthly_rets) > 0:
                                            monthly_df = pd.DataFrame({
                                                'Year': monthly_rets.index.year,
                                                'Month': monthly_rets.index.month,
                                                'Return': monthly_rets.values
                                            })
                                            monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
                                            monthly_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_pivot.columns)]
                                            
                                            fig_heat = go.Figure(data=go.Heatmap(
                                                z=monthly_pivot.values,
                                                x=monthly_pivot.columns,
                                                y=monthly_pivot.index.astype(str),
                                                colorscale='RdYlGn',
                                                zmid=0,
                                                text=np.round(monthly_pivot.values, 1),
                                                texttemplate="%{text}%",
                                                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>"
                                            ))
                                            fig_heat.update_layout(title="Dynamic Strategy 月度收益", template="plotly_white", height=max(300, len(monthly_pivot) * 30))
                                            st.plotly_chart(fig_heat, use_container_width=True)
                                else:
                                    st.info("回测周期不足一年，无法生成年度数据")
                            else:
                                st.info("需要策略数据来生成年度收益")
                    
                    # --- 6. Export Options ---
                    st.markdown("---")
                    st.markdown("#### 📤 导出回测结果 (Export Results)")
                    
                    c_exp1, c_exp2, c_exp3 = st.columns(3)
                    with c_exp1:
                        # Export Net Value Curve
                        csv_nv = res.to_csv()
                        st.download_button(
                            label="📈 下载净值曲线 (CSV)",
                            data=csv_nv,
                            file_name=f"backtest_nav_{dates[0]}_{dates[1]}.csv",
                            mime="text/csv"
                        )
                    with c_exp2:
                        # Export Allocation History
                        if df_history is not None and not df_history.empty:
                            csv_alloc = df_history.to_csv()
                            st.download_button(
                                label="📊 下载持仓历史 (CSV)",
                                data=csv_alloc,
                                file_name=f"backtest_allocation_{dates[0]}_{dates[1]}.csv",
                                mime="text/csv"
                            )
                    with c_exp3:
                        # Export Summary Report
                        report_lines = [
                            f"回测报告 - {dates[0]} 至 {dates[1]}",
                            f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            "",
                            "=== 参数设置 ===",
                            f"初始资金: {cap:,.0f}",
                            f"动量窗口: {ma_window}",
                            f"使用代理资产: {'是' if use_proxies else '否'}",
                            f"再平衡频率: {rebal_freq}",
                            f"交易成本: {cost_bps}bps",
                            "",
                            "=== 性能指标 ===",
                        ]
                        for _, row in df_metrics.iterrows():
                            report_lines.append(f"\n{row['Strategy']}:")
                            for col in df_metrics.columns:
                                if col != 'Strategy':
                                    val = row[col]
                                    if isinstance(val, float):
                                        report_lines.append(f"  {col}: {val:.2f}")
                                    else:
                                        report_lines.append(f"  {col}: {val}")
                        
                        report_text = "\n".join(report_lines)
                        st.download_button(
                            label="📝 下载摘要报告 (TXT)",
                            data=report_text,
                            file_name=f"backtest_report_{dates[0]}_{dates[1]}.txt",
                            mime="text/plain"
                        )
                    
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

                st.markdown("**实时风控提醒**")
                state_change_alert = st.checkbox("状态变化时立即提醒", value=bool(config.get("state_change_alert", False)))
                vix_alert_enabled = st.checkbox("VIX 超阈值提醒", value=bool(config.get("vix_alert_enabled", False)))
                vix_alert_threshold = st.number_input("VIX 阈值", value=float(config.get("vix_alert_threshold", 35)), step=1.0)

                st.info("仅在应用运行时触发；默认新加坡时间 09:30，请根据本地/服务器时区自行调整。")

            channels_cfg = config.get("channels", {}) or {}
            with st.expander("📡 多渠道占位 (Telegram / 企业微信)", expanded=False):
                telegram_bot_token = st.text_input("Telegram Bot Token", value=str(channels_cfg.get("telegram_bot_token", "")))
                telegram_chat_id = st.text_input("Telegram Chat ID", value=str(channels_cfg.get("telegram_chat_id", "")))
                wechat_webhook = st.text_input("企业微信 Webhook", value=str(channels_cfg.get("wechat_webhook", "")))

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
                        "last_run": config.get("last_run", ""),
                        "state_change_alert": state_change_alert,
                        "vix_alert_enabled": vix_alert_enabled,
                        "vix_alert_threshold": vix_alert_threshold,
                        "channels": {
                            "telegram_bot_token": telegram_bot_token,
                            "telegram_chat_id": telegram_chat_id,
                            "wechat_webhook": wechat_webhook,
                        }
                    }
                    merged, issues, warns = validate_alert_config(new_config)
                    if issues:
                        for i in issues:
                            st.error(i)
                    else:
                        save_alert_config(merged)
                        for w in warns:
                            st.warning(w)
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
    
    use_cache = st.toggle("⚡ 5分钟缓存 (减少重复拉取)", value=True)
    
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
            
            # Use the shared logic with optional cache
            success, metrics = analyze_market_state_logic_cached() if use_cache else analyze_market_state_logic()
            
            if not success:
                status.update(label="诊断失败", state="error")
                st.error(metrics) # metrics is error msg here
            else:
                st.write("✅ 数据获取与计算完成")
                status.update(label="诊断完成", state="complete", expanded=False)

                # State history & alerts
                history = record_state_history(metrics['state'], metrics)
                change_info = get_state_change_info(history, metrics['state'], metrics.get('latest_date'))
                cfg = load_alert_config()
                if change_info:
                    prev_state = change_info.get('prev_state')
                    days_in_state = change_info.get('days_in_state')
                    changed_on = change_info.get('changed_on')
                    msg = f"当前状态已持续 {days_in_state} 天" if days_in_state else "状态持续时间未知"
                    if prev_state:
                        msg = f"上次状态：{prev_state} → 当前：{metrics['state']}，自 {changed_on} 起 {days_in_state} 天"
                    st.info(msg)
                if cfg.get("state_change_alert") and change_info and change_info.get('prev_state') and change_info.get('prev_state') != metrics['state']:
                    st.warning("状态发生变化，已触发提醒 (占位)。")
                if cfg.get("vix_alert_enabled") and metrics.get('vix') is not None and metrics['vix'] >= cfg.get('vix_alert_threshold', 35):
                    st.error(f"VIX 达到 {metrics['vix']:.1f}，超过阈值 {cfg.get('vix_alert_threshold', 35)}。")

                # 记录持仓快照（用于回撤计算）
                if total_value > 0:
                    record_portfolio_snapshot(total_value, current_holdings, metrics['state'])

                # Render Results
                render_data_health_badges(metrics)
                render_status_card(metrics['state'])
                render_factor_dashboard(metrics)

                adjustments = get_adjustment_reasons(
                    metrics['state'],
                    gold_bear=metrics['gold_bear'],
                    value_regime=metrics['value_regime'],
                    asset_trends=metrics.get('asset_trends', {}),
                    vix=metrics.get('vix'),
                    yield_curve=metrics.get('yield_curve'),
                    sahm=metrics.get('sahm'),
                    corr=metrics.get('corr'),
                    yc_recently_inverted=metrics.get('yc_un_invert', False)
                )
                targets = get_target_percentages(
                    metrics['state'],
                    gold_bear=metrics['gold_bear'],
                    value_regime=metrics['value_regime'],
                    asset_trends=metrics.get('asset_trends', {}),
                    vix=metrics.get('vix'),
                    yield_curve=metrics.get('yield_curve'),
                    sahm=metrics.get('sahm'),
                    corr=metrics.get('corr'),
                    yc_recently_inverted=metrics.get('yc_un_invert', False)
                )
                price_info = get_live_prices(set(targets.keys()).union(current_holdings.keys()))
                
                st.markdown("---")
                render_rebalancing_table(
                    metrics['state'], 
                    current_holdings, 
                    total_value, 
                    metrics['gold_bear'], 
                    metrics['value_regime'], 
                    metrics.get('asset_trends', {}),
                    vix=metrics.get('vix'),
                    yield_curve=metrics.get('yield_curve'),
                    price_info=price_info,
                    sahm=metrics.get('sahm'),
                    corr=metrics.get('corr'),
                    yc_recently_inverted=metrics.get('yc_un_invert', False)
                )

                # 执行建议提示
                st.markdown("---")
                execution_tips = generate_execution_tips(
                    metrics, 
                    change_info, 
                    current_holdings=current_holdings,
                    targets=targets,
                    total_value=total_value
                )
                render_execution_tips(execution_tips)

                # 增强版深度诊断
                render_enhanced_diagnosis(metrics, current_holdings, total_value, targets, change_info)

                render_export_options(metrics, adjustments, targets)
                if history:
                    hist_df = pd.DataFrame(history).tail(10)
                    with st.expander("📜 最近状态变更记录", expanded=False):
                        st.dataframe(hist_df, use_container_width=True, hide_index=True)
                
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
                col_bt, col_opt = st.columns(2)
                with col_bt:
                    run_backtest = st.button("🚀 回测", type="primary", use_container_width=True)
                with col_opt:
                    run_optimize = st.button("⚡ 优化", type="secondary", use_container_width=True)
        else:
            run_backtest = False
            run_optimize = False

    st.markdown("---")
    
    # === 组合优化器 ===
    if 'run_optimize' in dir() and run_optimize and tickers:
        st.header("⚡ 组合优化器 (Portfolio Optimizer)")
        st.caption("基于历史数据找到最优资产配置比例，平衡收益、波动和回撤")
        
        with st.spinner("正在优化组合配置..."):
            try:
                # 获取数据
                opt_data_raw = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
                
                if opt_data_raw is None or opt_data_raw.empty:
                    st.error("无法获取数据，请检查网络或ticker")
                else:
                    opt_prices = normalize_yf_prices(opt_data_raw)
                    
                    if isinstance(opt_prices, pd.Series):
                        opt_prices = opt_prices.to_frame(name=tickers[0])
                    
                    opt_prices = opt_prices.dropna(axis=1, how='all').ffill().bfill()
                    
                    valid_tickers = [t for t in tickers if t in opt_prices.columns]
                    
                    if len(valid_tickers) < 2:
                        st.error("需要至少2个有效资产进行优化")
                    else:
                        # 优化参数
                        with st.expander("⚙️ 优化参数", expanded=False):
                            opt_col1, opt_col2, opt_col3 = st.columns(3)
                            with opt_col1:
                                opt_rf = st.number_input("无风险利率 (%)", value=4.0, min_value=0.0, max_value=10.0, step=0.5) / 100
                            with opt_col2:
                                opt_min_w = st.number_input("单资产最小权重 (%)", value=0.0, min_value=0.0, max_value=50.0, step=5.0) / 100
                            with opt_col3:
                                opt_max_w = st.number_input("单资产最大权重 (%)", value=50.0, min_value=10.0, max_value=100.0, step=10.0) / 100
                        
                        # 运行优化
                        opt_results = run_portfolio_optimization(
                            opt_prices, valid_tickers, 
                            risk_free_rate=opt_rf,
                            min_weight=opt_min_w,
                            max_weight=opt_max_w
                        )
                        
                        # 渲染结果
                        render_optimization_results(opt_results, opt_prices, valid_tickers)
                        
                        # 应用最优配置按钮
                        st.markdown("---")
                        st.markdown("### 📥 应用优化结果")
                        
                        apply_col1, apply_col2, apply_col3 = st.columns(3)
                        
                        with apply_col1:
                            if st.button("应用最大夏普配置", use_container_width=True):
                                if "max_sharpe" in opt_results:
                                    new_weights = opt_results["max_sharpe"]["weights"]
                                    for t, w in new_weights.items():
                                        st.session_state[f"w_{t}"] = w * 100
                                    st.toast("已应用最大夏普配置", icon="✅")
                                    st.rerun()
                        
                        with apply_col2:
                            if st.button("应用最小波动配置", use_container_width=True):
                                if "min_volatility" in opt_results:
                                    new_weights = opt_results["min_volatility"]["weights"]
                                    for t, w in new_weights.items():
                                        st.session_state[f"w_{t}"] = w * 100
                                    st.toast("已应用最小波动配置", icon="✅")
                                    st.rerun()
                        
                        with apply_col3:
                            if st.button("应用风险平价配置", use_container_width=True):
                                if "risk_parity" in opt_results:
                                    new_weights = opt_results["risk_parity"]["weights"]
                                    for t, w in new_weights.items():
                                        st.session_state[f"w_{t}"] = w * 100
                                    st.toast("已应用风险平价配置", icon="✅")
                                    st.rerun()
                        
            except Exception as e:
                st.error(f"优化过程出错: {e}")
                import traceback
                st.code(traceback.format_exc())

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
                data_raw_full = yf.download(download_tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
                
                if data_raw_full is None or data_raw_full.empty:
                    st.error("No data found. Check tickers or internet connection.")
                    return
                
                # 使用 normalize_yf_prices 处理不同版本 yfinance 的格式
                data_raw = normalize_yf_prices(data_raw_full)
                
                if data_raw is None or (hasattr(data_raw, 'empty') and data_raw.empty):
                    st.error(f"无法解析价格数据。原始列: {data_raw_full.columns.tolist()[:5]}")
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
