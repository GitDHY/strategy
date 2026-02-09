"""
Strategy editing page.
Allows users to write, test, and save dynamic trading strategies.
"""

import streamlit as st
from datetime import date

from strategy.engine import StrategyEngine
from strategy.templates import STRATEGY_TEMPLATES, STRATEGY_API_DOCS
from portfolio.manager import PortfolioManager
from ui.components.code_editor import render_code_editor_with_toolbar, render_code_viewer


def render_strategy_page():
    """Render the strategy editing page."""
    
    st.title("🧠 策略编辑器")
    st.caption("编写和测试你的动态调仓策略")
    
    # Initialize
    strategy_engine = StrategyEngine()
    portfolio_manager = PortfolioManager()
    
    # Initialize session state for tracking selected strategy
    if "last_selected_strategy" not in st.session_state:
        st.session_state["last_selected_strategy"] = "新建策略"
    
    # Sidebar: Saved strategies
    with st.sidebar:
        st.subheader("📁 已保存策略")
        
        strategies = strategy_engine.get_all()
        strategy_names = list(strategies.keys())
        
        if strategy_names:
            selected_strategy = st.selectbox(
                "选择策略",
                ["新建策略"] + strategy_names,
                key="strategy_select"
            )
            
            if selected_strategy != "新建策略":
                # Load and Edit buttons
                col_load, col_del = st.columns(2)
                with col_load:
                    if st.button("📖 加载编辑", use_container_width=True, help="加载策略到编辑器"):
                        # Force load strategy into editor
                        st.session_state["load_strategy_trigger"] = selected_strategy
                        st.rerun()
                with col_del:
                    if st.button("🗑️ 删除", use_container_width=True):
                        strategy_engine.delete_strategy(selected_strategy)
                        st.success(f"已删除 {selected_strategy}")
                        st.session_state["last_selected_strategy"] = "新建策略"
                        st.rerun()
                
                # Show strategy info
                strategy_info = strategies[selected_strategy]
                st.caption(f"📝 {strategy_info.get('description', '无描述')}")
                if strategy_info.get('updated_at'):
                    st.caption(f"🕐 更新: {strategy_info['updated_at'][:10]}")
        else:
            selected_strategy = "新建策略"
            st.info("暂无保存的策略")
        
        st.divider()
        
        # Portfolio selection for testing
        st.subheader("🎯 测试组合")
        portfolios = portfolio_manager.get_all()
        portfolio_names = list(portfolios.keys())
        
        if portfolio_names:
            test_portfolio = st.selectbox(
                "选择组合",
                portfolio_names,
                key="test_portfolio_select"
            )
        else:
            test_portfolio = None
            st.warning("请先创建组合")
    
    # Handle strategy loading trigger
    load_trigger = st.session_state.pop("load_strategy_trigger", None)
    
    # Main content
    col_editor, col_docs = st.columns([3, 2])
    
    with col_editor:
        st.subheader("📝 策略代码")
        
        # Determine if we need to load a saved strategy
        if load_trigger and load_trigger in strategies:
            # User clicked "Load" - force load strategy code
            default_code = strategies[load_trigger].get('code', '')
            strategy_name = load_trigger
            strategy_desc = strategies[load_trigger].get('description', '')
            # Update editor content
            st.session_state["strategy_code_code"] = default_code
            st.session_state["strategy_code_editor_version"] = st.session_state.get("strategy_code_editor_version", 0) + 1
            st.session_state["last_selected_strategy"] = load_trigger
            st.session_state["editing_strategy_name"] = strategy_name
            st.session_state["editing_strategy_desc"] = strategy_desc
            st.success(f"✅ 已加载策略: {strategy_name}")
        elif st.session_state.get("editing_strategy_name"):
            # Currently editing a loaded strategy
            strategy_name = st.session_state.get("editing_strategy_name", "")
            strategy_desc = st.session_state.get("editing_strategy_desc", "")
            default_code = st.session_state.get("strategy_code_code", "")
        else:
            # Default: new strategy with template
            default_code = STRATEGY_TEMPLATES.get("均线交叉策略", "")
            strategy_name = ""
            strategy_desc = ""
        
        # Strategy name and description
        # Use session state keys directly for text inputs
        if load_trigger:
            # When loading, set the session state for text inputs
            st.session_state["strat_name"] = strategy_name
            st.session_state["strat_desc"] = strategy_desc
        
        col_name, col_desc = st.columns([1, 2])
        with col_name:
            # Initialize if not exists
            if "strat_name" not in st.session_state:
                st.session_state["strat_name"] = strategy_name
            new_name = st.text_input("策略名称", key="strat_name")
        with col_desc:
            if "strat_desc" not in st.session_state:
                st.session_state["strat_desc"] = strategy_desc
            new_desc = st.text_input("描述", key="strat_desc")
        
        # New strategy button
        if st.session_state.get("editing_strategy_name"):
            if st.button("➕ 新建策略", type="secondary"):
                st.session_state.pop("editing_strategy_name", None)
                st.session_state.pop("editing_strategy_desc", None)
                st.session_state["strategy_code_code"] = STRATEGY_TEMPLATES.get("均线交叉策略", "")
                st.session_state["strategy_code_editor_version"] = st.session_state.get("strategy_code_editor_version", 0) + 1
                # Clear name and description inputs
                st.session_state["strat_name"] = ""
                st.session_state["strat_desc"] = ""
                st.rerun()
        
        # Code editor with templates
        code = render_code_editor_with_toolbar(
            default_code=default_code,
            key="strategy_code",
            height=450,
            templates=STRATEGY_TEMPLATES,
        )
        
        # Action buttons
        st.divider()
        col_validate, col_test, col_save = st.columns(3)
        
        with col_validate:
            if st.button("✅ 验证代码", use_container_width=True):
                result = strategy_engine.validate_strategy(code)
                
                if result['valid']:
                    st.success("✅ 代码验证通过")
                else:
                    st.error("❌ 代码有错误:")
                    for err in result['errors']:
                        st.error(err)
                
                if result['warnings']:
                    for warn in result['warnings']:
                        st.warning(warn)
        
        with col_test:
            if st.button("🧪 测试运行", type="primary", use_container_width=True):
                if test_portfolio is None:
                    st.error("请先选择测试组合")
                else:
                    run_strategy_test(strategy_engine, portfolio_manager, test_portfolio, code)
        
        with col_save:
            # Determine if we're editing an existing strategy
            editing_name = st.session_state.get("editing_strategy_name")
            is_editing = editing_name is not None
            
            # Save button label
            save_label = "💾 更新策略" if is_editing and new_name == editing_name else "💾 保存策略"
            
            if st.button(save_label, use_container_width=True):
                if not new_name:
                    st.error("请输入策略名称")
                else:
                    if strategy_engine.save_strategy(
                        name=new_name,
                        code=code,
                        description=new_desc,
                        portfolio_name=test_portfolio or ""
                    ):
                        # Update editing state
                        st.session_state["editing_strategy_name"] = new_name
                        st.session_state["editing_strategy_desc"] = new_desc
                        
                        if is_editing and new_name == editing_name:
                            st.success(f"✅ 策略 '{new_name}' 已更新")
                        else:
                            st.success(f"✅ 策略 '{new_name}' 已保存")
                        st.rerun()
                    else:
                        st.error("保存失败")
    
    with col_docs:
        render_api_documentation()


def run_strategy_test(
    strategy_engine: StrategyEngine,
    portfolio_manager: PortfolioManager,
    portfolio_name: str,
    code: str
):
    """Run a quick strategy test."""
    
    portfolio = portfolio_manager.get(portfolio_name)
    if portfolio is None:
        st.error("组合不存在")
        return
    
    st.subheader("🧪 测试结果")
    
    with st.spinner("正在执行策略..."):
        result = strategy_engine.execute(
            code=code,
            tickers=portfolio.tickers,
            current_weights=portfolio.weights,
            current_date=date.today(),
        )
    
    if result.success:
        st.success(f"✅ 执行成功 (耗时 {result.execution_time:.3f}s)")
        
        # Show signals
        if result.signals:
            st.write("**📢 策略信号:**")
            for signal in result.signals:
                st.info(signal)
        
        # Show weight changes
        st.write("**📊 仓位变化:**")
        
        changes_data = []
        all_tickers = set(portfolio.weights.keys()) | set(result.target_weights.keys())
        
        for ticker in sorted(all_tickers):
            current = portfolio.weights.get(ticker, 0)
            target = result.target_weights.get(ticker, 0)
            diff = target - current
            
            changes_data.append({
                "标的": ticker,
                "当前 (%)": current,
                "目标 (%)": target,
                "变化 (%)": diff,
            })
        
        import pandas as pd
        changes_df = pd.DataFrame(changes_data)
        
        st.dataframe(
            changes_df,
            column_config={
                "变化 (%)": st.column_config.NumberColumn(
                    "变化 (%)",
                    format="%+.1f"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Highlight significant changes
        significant = [c for c in changes_data if abs(c["变化 (%)"]) > 1]
        if significant:
            st.write("**⚠️ 显著调仓:**")
            for c in significant:
                arrow = "⬆️" if c["变化 (%)"] > 0 else "⬇️"
                st.write(f"{arrow} {c['标的']}: {c['当前 (%)']:.1f}% → {c['目标 (%)']:.1f}%")
        else:
            st.info("无显著仓位变化")
    
    else:
        st.error(f"❌ 执行失败: {result.message}")


def render_api_documentation():
    """Render the API documentation panel."""
    
    st.subheader("📖 API 文档")
    
    with st.expander("快速参考", expanded=True):
        st.markdown("""
### 基本结构
```python
def strategy():
    weights = ctx.get_current_weights()
    # 你的策略逻辑
    ctx.set_target_weights(weights)
```

### 常用方法
| 方法 | 说明 |
|------|------|
| `ctx.get_current_weights()` | 获取当前权重 |
| `ctx.set_target_weights(w)` | 设置目标权重 |
| `ctx.log(msg)` | 记录信号 |
| `ctx.current_price(t)` | 获取当前价格 |
| `ctx.current_vix()` | 获取当前VIX |
""")
    
    with st.expander("数据获取"):
        st.markdown("""
### 价格数据
```python
# 单个标的
price = ctx.get_price('IWY', lookback=252)

# 多个标的
prices = ctx.get_prices(['IWY', 'LVHI'])

# 当前价格
current = ctx.current_price('IWY')
```

### VIX 指数
```python
vix_series = ctx.vix()
current_vix = ctx.current_vix()
```
""")
    
    with st.expander("技术指标"):
        st.markdown("""
### 均线
```python
ma20 = ctx.ma('IWY', 20)    # 简单均线
ema20 = ctx.ema('IWY', 20)  # 指数均线
```

### 动量/波动
```python
rsi = ctx.rsi('IWY', 14)
mom = ctx.momentum('IWY', 10)
vol = ctx.volatility('IWY', 20)
```

### 其他指标
```python
macd = ctx.macd('IWY')
bb = ctx.bollinger('IWY')
atr = ctx.atr('IWY')
```
""")
    
    with st.expander("信号检测"):
        st.markdown("""
### 趋势判断
```python
# 价格在均线上方
if ctx.price_above_ma('IWY', 200):
    ctx.log("上升趋势")

# 均线交叉
if ctx.ma_cross_up('IWY', 20, 50):
    ctx.log("金叉信号")
```
""")
    
    with st.expander("完整示例"):
        st.markdown("""
```python
def strategy():
    weights = ctx.get_current_weights()
    vix = ctx.current_vix()
    
    # VIX 低于 15: 激进配置
    if vix < 15:
        weights['IWY'] = 60
        weights['LVHI'] = 20
        ctx.log(f"低波动 VIX={vix:.1f}")
    
    # VIX 高于 30: 保守配置
    elif vix > 30:
        weights['IWY'] = 20
        weights['GSD.SI'] = 30
        ctx.log(f"高波动 VIX={vix:.1f}")
    
    ctx.set_target_weights(weights)
```
""")
