"""
Built-in strategy templates for user reference.
"""

# Simple Moving Average Strategy
MA_CROSSOVER_TEMPLATE = '''"""
均线交叉策略 (Moving Average Crossover)
当短期均线上穿长期均线时增加仓位，下穿时减少仓位。
"""

def strategy():
    # 获取当前权重
    weights = ctx.get_current_weights()
    
    # 策略参数
    short_period = 20   # 短期均线周期
    long_period = 50    # 长期均线周期
    
    # 遍历所有标的
    for ticker in ctx.tickers:
        current_weight = weights.get(ticker, 0)
        
        # 计算均线
        short_ma = ctx.ma(ticker, short_period)
        long_ma = ctx.ma(ticker, long_period)
        
        if len(short_ma) < 2 or len(long_ma) < 2:
            continue
        
        # 金叉: 增加仓位
        if ctx.ma_cross_up(ticker, short_period, long_period):
            weights[ticker] = min(current_weight + 10, 50)
            ctx.log(f"🟢 {ticker} 金叉信号, 增加仓位")
        
        # 死叉: 减少仓位
        elif ctx.ma_cross_down(ticker, short_period, long_period):
            weights[ticker] = max(current_weight - 10, 0)
            ctx.log(f"🔴 {ticker} 死叉信号, 减少仓位")
    
    # 设置目标权重
    ctx.set_target_weights(weights)
'''

# Momentum Strategy
MOMENTUM_TEMPLATE = '''"""
动量策略 (Momentum Strategy)
根据过去一段时间的涨幅调整仓位，追涨杀跌。
"""

def strategy():
    weights = ctx.get_current_weights()
    
    # 策略参数
    lookback = 20       # 动量计算周期
    threshold = 5       # 动量阈值 (%)
    
    for ticker in ctx.tickers:
        current_weight = weights.get(ticker, 0)
        
        # 计算动量
        mom = ctx.momentum(ticker, lookback)
        if mom.empty:
            continue
        
        current_momentum = mom.iloc[-1]
        
        # 正向动量: 增加仓位
        if current_momentum > threshold:
            weights[ticker] = min(current_weight + 5, 40)
            ctx.log(f"📈 {ticker} 动量 {current_momentum:.1f}% > {threshold}%, 增仓")
        
        # 负向动量: 减少仓位
        elif current_momentum < -threshold:
            weights[ticker] = max(current_weight - 5, 0)
            ctx.log(f"📉 {ticker} 动量 {current_momentum:.1f}% < -{threshold}%, 减仓")
    
    ctx.set_target_weights(weights)
'''

# VIX-Based Strategy
VIX_STRATEGY_TEMPLATE = '''"""
VIX 波动率策略 (Volatility Strategy)
根据 VIX 指数调整风险资产仓位。
"""

def strategy():
    weights = ctx.get_current_weights()
    
    # 策略参数
    vix_low = 15        # 低波动阈值
    vix_high = 25       # 高波动阈值
    vix_panic = 35      # 恐慌阈值
    
    # 获取当前 VIX
    current_vix = ctx.current_vix()
    ctx.log(f"📊 当前 VIX: {current_vix:.1f}")
    
    # 定义风险资产和避险资产
    risk_assets = ['IWY', 'LVHI', 'G3B.SI']  # 根据你的组合调整
    safe_assets = ['GSD.SI', 'MBH.SI']       # 黄金、债券等
    
    if current_vix < vix_low:
        # 低波动: 激进配置
        ctx.log("🚀 低波动环境，增加风险资产")
        for ticker in risk_assets:
            if ticker in weights:
                weights[ticker] = weights.get(ticker, 0) * 1.2
        for ticker in safe_assets:
            if ticker in weights:
                weights[ticker] = weights.get(ticker, 0) * 0.8
    
    elif current_vix > vix_panic:
        # 恐慌: 避险配置
        ctx.log("🛡️ 恐慌环境，大幅减少风险资产")
        for ticker in risk_assets:
            if ticker in weights:
                weights[ticker] = weights.get(ticker, 0) * 0.5
        for ticker in safe_assets:
            if ticker in weights:
                weights[ticker] = weights.get(ticker, 0) * 1.5
    
    elif current_vix > vix_high:
        # 高波动: 谨慎配置
        ctx.log("⚠️ 高波动环境，适度减少风险资产")
        for ticker in risk_assets:
            if ticker in weights:
                weights[ticker] = weights.get(ticker, 0) * 0.85
    
    ctx.set_target_weights(weights)
'''

# RSI Mean Reversion Strategy
RSI_STRATEGY_TEMPLATE = '''"""
RSI 均值回归策略 (Mean Reversion)
RSI 超买/超卖时反向操作。
"""

def strategy():
    weights = ctx.get_current_weights()
    
    # 策略参数
    rsi_period = 14
    oversold = 30       # 超卖阈值
    overbought = 70     # 超买阈值
    
    for ticker in ctx.tickers:
        current_weight = weights.get(ticker, 0)
        
        # 计算 RSI
        rsi = ctx.rsi(ticker, rsi_period)
        if rsi.empty:
            continue
        
        current_rsi = rsi.iloc[-1]
        
        # 超卖: 买入信号
        if current_rsi < oversold:
            weights[ticker] = min(current_weight + 10, 50)
            ctx.log(f"🟢 {ticker} RSI={current_rsi:.0f} 超卖，增仓")
        
        # 超买: 卖出信号
        elif current_rsi > overbought:
            weights[ticker] = max(current_weight - 10, 5)
            ctx.log(f"🔴 {ticker} RSI={current_rsi:.0f} 超买，减仓")
    
    ctx.set_target_weights(weights)
'''

# Trend Following Strategy
TREND_FOLLOWING_TEMPLATE = '''"""
趋势跟踪策略 (Trend Following)
只持有处于上升趋势的资产（价格在均线上方）。
"""

def strategy():
    weights = ctx.get_current_weights()
    
    # 策略参数
    ma_period = 200     # 长期均线
    trend_ma = 50       # 趋势确认均线
    
    total_weight = 100
    trending_assets = []
    
    # 识别趋势资产
    for ticker in ctx.tickers:
        if ctx.price_above_ma(ticker, ma_period):
            trending_assets.append(ticker)
    
    ctx.log(f"📊 趋势向上的资产: {trending_assets}")
    
    if not trending_assets:
        # 没有趋势资产，保守配置
        ctx.log("⚠️ 无趋势资产，保持现有配置")
        return
    
    # 平均分配权重给趋势资产
    weight_per_asset = total_weight / len(trending_assets)
    
    for ticker in ctx.tickers:
        if ticker in trending_assets:
            # 趋势资产: 分配权重
            old_weight = weights.get(ticker, 0)
            weights[ticker] = weight_per_asset
            if weights[ticker] > old_weight:
                ctx.log(f"🟢 {ticker} 趋势向上，增仓至 {weight_per_asset:.1f}%")
        else:
            # 非趋势资产: 清仓
            if weights.get(ticker, 0) > 0:
                ctx.log(f"🔴 {ticker} 趋势转弱，清仓")
            weights[ticker] = 0
    
    ctx.set_target_weights(weights)
'''

# Risk Parity Inspired Strategy
RISK_PARITY_TEMPLATE = '''"""
风险平价策略 (Simplified Risk Parity)
根据波动率反比调整仓位，低波动资产配置更多。
"""

def strategy():
    weights = ctx.get_current_weights()
    
    # 策略参数
    vol_period = 20     # 波动率计算周期
    target_vol = 0.15   # 目标波动率 15%
    
    volatilities = {}
    
    # 计算各资产波动率
    for ticker in ctx.tickers:
        vol = ctx.volatility(ticker, vol_period, annualize=True)
        if not vol.empty and vol.iloc[-1] > 0:
            volatilities[ticker] = vol.iloc[-1]
    
    if not volatilities:
        ctx.log("⚠️ 无法计算波动率")
        return
    
    # 计算反波动率权重
    inv_vol = {t: 1/v for t, v in volatilities.items()}
    total_inv_vol = sum(inv_vol.values())
    
    # 归一化权重
    for ticker in ctx.tickers:
        if ticker in inv_vol:
            weights[ticker] = (inv_vol[ticker] / total_inv_vol) * 100
            ctx.log(f"📊 {ticker}: 波动率={volatilities[ticker]:.1%}, 权重={weights[ticker]:.1f}%")
        else:
            weights[ticker] = 0
    
    ctx.set_target_weights(weights)
'''

# Basic Rebalance Strategy
REBALANCE_TEMPLATE = '''"""
定期再平衡策略 (Periodic Rebalancing)
保持目标配置比例，超过阈值时触发再平衡。
注意：此模板会自动使用当前组合中的标的进行等权重分配。
你可以修改 target 字典来自定义目标配置。
"""

def strategy():
    # 获取当前权重作为基础
    current = ctx.get_current_weights()
    
    # 目标配置 - 你可以自定义每个标的的目标权重
    # 如果不在下面定义，会使用等权重分配
    custom_target = {
        # 例如:
        # 'IWY': 40,      # 美股成长 40%
        # 'LVHI': 15,     # 美股红利 15%
    }
    
    # 如果没有自定义目标，使用等权重分配
    if not custom_target:
        n_assets = len(ctx.tickers)
        if n_assets > 0:
            equal_weight = 100.0 / n_assets
            target = {ticker: equal_weight for ticker in ctx.tickers}
            ctx.log(f"📊 使用等权重分配: {equal_weight:.1f}% x {n_assets} 个标的")
        else:
            ctx.log("⚠️ 组合中没有标的")
            return
    else:
        target = custom_target
    
    # 再平衡阈值
    rebalance_threshold = 5  # 偏离超过 5% 触发
    
    needs_rebalance = False
    
    # 检查是否需要再平衡
    for ticker in ctx.tickers:
        target_weight = target.get(ticker, 0)
        current_weight = current.get(ticker, 0)
        deviation = abs(current_weight - target_weight)
        
        if deviation > rebalance_threshold:
            needs_rebalance = True
            ctx.log(f"⚖️ {ticker}: 当前 {current_weight:.1f}% vs 目标 {target_weight:.1f}%, 偏离 {deviation:.1f}%")
    
    if needs_rebalance:
        ctx.log("🔄 触发再平衡")
        ctx.set_target_weights(target)
    else:
        ctx.log("✅ 无需再平衡，配置在容忍范围内")
'''

# Strategy templates dictionary
STRATEGY_TEMPLATES = {
    "均线交叉策略": MA_CROSSOVER_TEMPLATE,
    "动量策略": MOMENTUM_TEMPLATE,
    "VIX 波动率策略": VIX_STRATEGY_TEMPLATE,
    "RSI 均值回归策略": RSI_STRATEGY_TEMPLATE,
    "趋势跟踪策略": TREND_FOLLOWING_TEMPLATE,
    "风险平价策略": RISK_PARITY_TEMPLATE,
    "定期再平衡策略": REBALANCE_TEMPLATE,
}

# API Documentation for users
STRATEGY_API_DOCS = '''
# 策略 API 文档

## 上下文对象 `ctx`

### 数据获取

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `ctx.get_price(ticker, lookback)` | 获取价格序列 | pd.Series |
| `ctx.get_prices(tickers, lookback)` | 获取多个标的价格 | pd.DataFrame |
| `ctx.get_returns(ticker, lookback)` | 获取收益率序列 | pd.Series |
| `ctx.current_price(ticker)` | 获取当前价格 | float |
| `ctx.vix(lookback)` | 获取 VIX 序列 | pd.Series |
| `ctx.current_vix()` | 获取当前 VIX | float |

### 技术指标

| 方法 | 说明 | 参数 |
|------|------|------|
| `ctx.ma(ticker, period)` | 简单移动平均 | period: 周期 |
| `ctx.ema(ticker, period)` | 指数移动平均 | period: 周期 |
| `ctx.rsi(ticker, period)` | RSI 指标 | period: 默认 14 |
| `ctx.macd(ticker, fast, slow, signal)` | MACD | 默认 12, 26, 9 |
| `ctx.bollinger(ticker, period, std)` | 布林带 | 默认 20, 2.0 |
| `ctx.atr(ticker, period)` | 平均真实波幅 | period: 默认 14 |
| `ctx.volatility(ticker, period)` | 波动率 | 年化波动率 |
| `ctx.momentum(ticker, period)` | 动量 | 百分比变化 |
| `ctx.drawdown(ticker)` | 回撤分析 | 返回 DataFrame |

### 信号检测

| 方法 | 说明 |
|------|------|
| `ctx.price_above_ma(ticker, period)` | 价格是否在均线上方 |
| `ctx.price_below_ma(ticker, period)` | 价格是否在均线下方 |
| `ctx.ma_cross_up(ticker, short, long)` | 短均线是否上穿长均线 |
| `ctx.ma_cross_down(ticker, short, long)` | 短均线是否下穿长均线 |

### 仓位管理

| 方法 | 说明 |
|------|------|
| `ctx.get_current_weights()` | 获取当前权重 (dict) |
| `ctx.set_target_weights(weights)` | 设置目标权重 |
| `ctx.log(message)` | 记录信号/日志 |

### 属性

| 属性 | 说明 |
|------|------|
| `ctx.tickers` | 可用标的列表 |
| `ctx.current_date` | 当前日期 |

## 示例

```python
def strategy():
    weights = ctx.get_current_weights()
    
    # 获取 VIX
    vix = ctx.current_vix()
    
    # 检查均线
    if ctx.price_above_ma('IWY', 200):
        weights['IWY'] = 50
        ctx.log("IWY 在 200 日均线上方")
    
    # 设置目标权重
    ctx.set_target_weights(weights)
```
'''
