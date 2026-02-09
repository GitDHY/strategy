"""
Strategy API documentation generator.
"""

from strategy.templates import STRATEGY_API_DOCS

def get_api_documentation() -> str:
    """Get the full API documentation."""
    return STRATEGY_API_DOCS


def get_quick_reference() -> str:
    """Get a quick reference card for common operations."""
    return """
## 快速参考

### 基本结构
```python
def strategy():
    weights = ctx.get_current_weights()
    # 你的策略逻辑
    ctx.set_target_weights(weights)
```

### 常用操作
```python
# 获取当前价格
price = ctx.current_price('IWY')

# 计算均线
ma20 = ctx.ma('IWY', 20)
ma50 = ctx.ma('IWY', 50)

# 检查趋势
if ctx.price_above_ma('IWY', 200):
    ctx.log("上升趋势")

# 获取 VIX
vix = ctx.current_vix()

# 调整仓位
weights['IWY'] = 50
ctx.set_target_weights(weights)
```

### 可用标的属性
- `ctx.tickers` - 组合中的标的列表
- `ctx.current_date` - 当前日期
"""
