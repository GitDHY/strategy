"""
Backtesting page.
Run and analyze strategy backtests with detailed metrics.
Supports multi-strategy comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional

from backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from portfolio.manager import PortfolioManager
from strategy.engine import StrategyEngine
from ui.components.charts import (
    render_equity_curve,
    render_drawdown_chart,
    render_monthly_returns_heatmap,
    render_correlation_matrix,
    render_metrics_cards,
    render_trade_history,
)


def render_backtest_page():
    """Render the backtesting page."""
    
    st.title("📈 策略回测")
    st.caption("回测你的投资组合和动态策略，支持多策略对比分析")
    
    # Initialize managers
    portfolio_manager = PortfolioManager()
    strategy_engine = StrategyEngine()
    
    # Sidebar: Configuration
    with st.sidebar:
        st.subheader("⚙️ 回测设置")
        
        # Date range
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "开始日期",
                value=date.today() - timedelta(days=365*3),
                key="bt_start"
            )
        with col_end:
            end_date = st.date_input(
                "结束日期",
                value=date.today(),
                key="bt_end"
            )
        
        # Initial capital
        initial_capital = st.number_input(
            "初始资金 ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
            key="bt_capital"
        )
        
        st.divider()
        
        # Rebalancing
        rebalance_freq = st.selectbox(
            "调仓频率",
            ["daily", "weekly", "monthly", "quarterly"],
            index=2,
            format_func=lambda x: {
                "daily": "每日",
                "weekly": "每周",
                "monthly": "每月",
                "quarterly": "每季度"
            }[x],
            key="bt_rebal"
        )
        
        st.divider()
        
        # Costs
        st.write("**交易成本**")
        commission_pct = st.slider(
            "佣金 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            key="bt_comm"
        ) / 100
        
        slippage_pct = st.slider(
            "滑点 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            key="bt_slip"
        ) / 100
    
    # Main content - Three tabs
    tab1, tab2, tab3 = st.tabs(["📊 静态回测", "🧠 策略回测", "⚔️ 多策略对比"])
    
    with tab1:
        render_static_backtest(
            portfolio_manager,
            start_date, end_date,
            initial_capital,
            rebalance_freq,
            commission_pct, slippage_pct
        )
    
    with tab2:
        render_dynamic_backtest(
            portfolio_manager,
            strategy_engine,
            start_date, end_date,
            initial_capital,
            rebalance_freq,
            commission_pct, slippage_pct
        )
    
    with tab3:
        render_multi_strategy_comparison(
            portfolio_manager,
            strategy_engine,
            start_date, end_date,
            initial_capital,
            rebalance_freq,
            commission_pct, slippage_pct
        )


def render_static_backtest(
    portfolio_manager: PortfolioManager,
    start_date: date,
    end_date: date,
    initial_capital: float,
    rebalance_freq: str,
    commission_pct: float,
    slippage_pct: float
):
    """Render static (buy-and-hold) backtest."""
    
    st.subheader("📊 静态回测 (买入持有)")
    
    # Portfolio selection
    portfolios = portfolio_manager.get_all()
    portfolio_names = list(portfolios.keys())
    
    if not portfolio_names:
        st.warning("请先创建投资组合")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_portfolio = st.selectbox(
            "选择组合",
            portfolio_names,
            key="static_portfolio"
        )
    
    with col2:
        # Benchmark selection
        benchmarks = ["无", "SPY (标普500)", "QQQ (纳斯达克)", "60/40 组合"]
        benchmark = st.selectbox("对比基准", benchmarks, key="static_bench")
    
    # Run backtest button
    if st.button("🚀 运行回测", type="primary", key="run_static"):
        portfolio = portfolio_manager.get(selected_portfolio)
        
        if portfolio is None or not portfolio.tickers:
            st.error("组合无效或没有资产")
            return
        
        # Create config
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            rebalance_freq=rebalance_freq,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        
        # Run backtest
        with st.spinner("正在回测..."):
            engine = BacktestEngine(config)
            result = engine.run_static(
                tickers=portfolio.tickers,
                weights=portfolio.weights,
                config=config
            )
        
        if not result.success:
            st.error(f"回测失败: {result.message}")
            return
        
        # Get benchmark if selected
        benchmark_values = None
        if benchmark != "无":
            benchmark_values = get_benchmark_values(engine, benchmark, config)
        
        # Display results
        display_backtest_results(
            result, 
            benchmark_values,
            portfolio_name=selected_portfolio
        )


def render_dynamic_backtest(
    portfolio_manager: PortfolioManager,
    strategy_engine: StrategyEngine,
    start_date: date,
    end_date: date,
    initial_capital: float,
    rebalance_freq: str,
    commission_pct: float,
    slippage_pct: float
):
    """Render dynamic strategy backtest."""
    
    st.subheader("🧠 策略回测 (动态调仓)")
    
    # Portfolio selection
    portfolios = portfolio_manager.get_all()
    portfolio_names = list(portfolios.keys())
    
    if not portfolio_names:
        st.warning("请先创建投资组合")
        return
    
    # Strategy selection
    strategies = strategy_engine.get_all()
    strategy_names = list(strategies.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_portfolio = st.selectbox(
            "选择组合",
            portfolio_names,
            key="dynamic_portfolio"
        )
    
    with col2:
        if strategy_names:
            selected_strategy = st.selectbox(
                "选择策略",
                strategy_names,
                key="dynamic_strategy"
            )
        else:
            st.warning("请先创建策略")
            return
    
    # Benchmark selection - more options
    st.write("**📊 基准对比**")
    col_bench1, col_bench2 = st.columns(2)
    
    with col_bench1:
        benchmarks = ["无", "买入持有 (同组合)", "SPY (标普500)", "QQQ (纳斯达克)", "60/40 组合"]
        benchmark = st.selectbox("对比基准", benchmarks, key="dynamic_bench")
    
    with col_bench2:
        show_weights = st.checkbox("显示仓位变化", value=True, key="show_weights_dynamic")
    
    # Run backtest button
    if st.button("🚀 运行策略回测", type="primary", key="run_dynamic"):
        portfolio = portfolio_manager.get(selected_portfolio)
        strategy = strategy_engine.get(selected_strategy)
        
        if portfolio is None or strategy is None:
            st.error("组合或策略无效")
            return
        
        # Create config
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            rebalance_freq=rebalance_freq,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        
        # Run dynamic backtest
        with st.spinner("正在回测策略 (这可能需要一些时间)..."):
            engine = BacktestEngine(config)
            result = engine.run_with_code(
                tickers=portfolio.tickers,
                initial_weights=portfolio.weights,
                strategy_code=strategy['code'],
                config=config
            )
        
        if not result.success:
            st.error(f"回测失败: {result.message}")
            return
        
        # Get benchmark
        benchmark_values = None
        if benchmark == "买入持有 (同组合)":
            # Run static backtest for comparison
            static_result = engine.run_static(
                tickers=portfolio.tickers,
                weights=portfolio.weights,
                config=config
            )
            if static_result.success:
                benchmark_values = {"买入持有": static_result.portfolio_values}
        elif benchmark == "SPY (标普500)":
            benchmark_values = get_benchmark_values(engine, "SPY (标普500)", config)
        elif benchmark == "QQQ (纳斯达克)":
            benchmark_values = get_benchmark_values(engine, "QQQ (纳斯达克)", config)
        elif benchmark == "60/40 组合":
            benchmark_values = get_benchmark_values(engine, "60/40 组合", config)
        
        # Display results
        display_backtest_results(
            result, 
            benchmark_values, 
            show_trades=True,
            strategy_name=selected_strategy,
            portfolio_name=selected_portfolio
        )


def render_multi_strategy_comparison(
    portfolio_manager: PortfolioManager,
    strategy_engine: StrategyEngine,
    start_date: date,
    end_date: date,
    initial_capital: float,
    rebalance_freq: str,
    commission_pct: float,
    slippage_pct: float
):
    """Render multi-strategy comparison backtest."""
    
    st.subheader("⚔️ 多策略对比回测")
    st.caption("选择多个策略进行对比分析，找出最优策略")
    
    # Portfolio selection
    portfolios = portfolio_manager.get_all()
    portfolio_names = list(portfolios.keys())
    
    if not portfolio_names:
        st.warning("请先创建投资组合")
        return
    
    # Strategy selection
    strategies = strategy_engine.get_all()
    strategy_names = list(strategies.keys())
    
    if not strategy_names:
        st.warning("请先在策略编辑器中创建并保存策略")
        return
    
    # Selection UI
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_portfolio = st.selectbox(
            "选择组合",
            portfolio_names,
            key="compare_portfolio"
        )
    
    with col2:
        selected_strategies = st.multiselect(
            "选择要对比的策略 (可多选)",
            strategy_names,
            default=strategy_names[:min(3, len(strategy_names))],
            key="compare_strategies",
            help="选择 2-5 个策略进行对比"
        )
    
    # Additional options
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        include_buyhold = st.checkbox("包含买入持有基准", value=True, key="compare_buyhold")
    
    with col_opt2:
        include_spy = st.checkbox("包含 SPY 基准", value=False, key="compare_spy")
    
    # Validate selection
    if len(selected_strategies) < 1:
        st.info("请至少选择 1 个策略进行回测")
        return
    
    if len(selected_strategies) > 5:
        st.warning("建议选择不超过 5 个策略，以保证可读性")
    
    # Run comparison button
    if st.button("🚀 运行对比回测", type="primary", key="run_compare"):
        portfolio = portfolio_manager.get(selected_portfolio)
        
        if portfolio is None:
            st.error("组合无效")
            return
        
        # Create config
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            rebalance_freq=rebalance_freq,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        
        engine = BacktestEngine(config)
        
        # Store all results
        all_results: Dict[str, BacktestResult] = {}
        all_values: Dict[str, pd.Series] = {}
        all_metrics: List[Dict] = []
        
        # Progress tracking
        total_backtests = len(selected_strategies)
        if include_buyhold:
            total_backtests += 1
        if include_spy:
            total_backtests += 1
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        completed = 0
        
        # Run buy-and-hold benchmark first
        if include_buyhold:
            status_text.text("回测: 买入持有基准...")
            buyhold_result = engine.run_static(
                tickers=portfolio.tickers,
                weights=portfolio.weights,
                config=config
            )
            if buyhold_result.success:
                all_results["📦 买入持有"] = buyhold_result
                all_values["📦 买入持有"] = buyhold_result.portfolio_values
                metrics = buyhold_result.metrics.copy()
                metrics['策略名称'] = "📦 买入持有"
                metrics['类型'] = "基准"
                all_metrics.append(metrics)
            completed += 1
            progress_bar.progress(completed / total_backtests)
        
        # Run SPY benchmark
        if include_spy:
            status_text.text("回测: SPY 基准...")
            spy_result = engine.run_static(
                tickers=["SPY"],
                weights={"SPY": 100},
                config=config
            )
            if spy_result.success:
                all_results["📊 SPY"] = spy_result
                all_values["📊 SPY"] = spy_result.portfolio_values
                metrics = spy_result.metrics.copy()
                metrics['策略名称'] = "📊 SPY"
                metrics['类型'] = "基准"
                all_metrics.append(metrics)
            completed += 1
            progress_bar.progress(completed / total_backtests)
        
        # Run each strategy
        for strategy_name in selected_strategies:
            status_text.text(f"回测: {strategy_name}...")
            
            strategy = strategy_engine.get(strategy_name)
            if strategy is None:
                continue
            
            try:
                result = engine.run_with_code(
                    tickers=portfolio.tickers,
                    initial_weights=portfolio.weights,
                    strategy_code=strategy['code'],
                    config=config
                )
                
                if result.success:
                    display_name = f"🧠 {strategy_name}"
                    all_results[display_name] = result
                    all_values[display_name] = result.portfolio_values
                    
                    metrics = result.metrics.copy()
                    metrics['策略名称'] = display_name
                    metrics['类型'] = "策略"
                    metrics['交易次数'] = len(result.trades)
                    metrics['总交易成本'] = sum(t.cost for t in result.trades)
                    all_metrics.append(metrics)
            
            except Exception as e:
                st.warning(f"策略 '{strategy_name}' 回测失败: {str(e)}")
            
            completed += 1
            progress_bar.progress(completed / total_backtests)
        
        progress_bar.empty()
        status_text.empty()
        
        # Check if any backtests succeeded
        if len(all_results) < 1:
            st.error("没有成功完成的回测")
            return
        
        # Store results in session state for persistence across reruns
        st.session_state.comparison_results = {
            'all_results': all_results,
            'all_values': all_values,
            'all_metrics': all_metrics,
            'config': config
        }
    
    # Display results if available in session state
    if 'comparison_results' in st.session_state and st.session_state.comparison_results:
        cached = st.session_state.comparison_results
        display_comparison_results(
            cached['all_results'],
            cached['all_values'],
            cached['all_metrics'],
            cached['config']
        )


def display_comparison_results(
    all_results: Dict[str, BacktestResult],
    all_values: Dict[str, pd.Series],
    all_metrics: List[Dict],
    config: BacktestConfig
):
    """Display multi-strategy comparison results."""
    
    st.divider()
    st.subheader("📊 对比结果")
    
    # Summary metrics table
    st.write("**📋 关键指标对比**")
    
    # Select key metrics for comparison
    key_metrics = [
        '策略名称', '类型',
        'Total Return (%)', 'CAGR (%)', 'Volatility (%)',
        'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)',
        'Win Rate (%)', 'Calmar Ratio'
    ]
    
    # Build comparison dataframe
    comparison_data = []
    for metrics in all_metrics:
        row = {}
        for key in key_metrics:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    row[key] = round(value, 2)
                else:
                    row[key] = value
            else:
                row[key] = '-'
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display with highlighting
    if not comparison_df.empty:
        # Rename columns to Chinese
        column_names = {
            '策略名称': '策略',
            '类型': '类型',
            'Total Return (%)': '总收益率 (%)',
            'CAGR (%)': '年化收益 (%)',
            'Volatility (%)': '波动率 (%)',
            'Sharpe Ratio': '夏普比率',
            'Sortino Ratio': '索提诺比率',
            'Max Drawdown (%)': '最大回撤 (%)',
            'Win Rate (%)': '胜率 (%)',
            'Calmar Ratio': '卡玛比率',
        }
        comparison_df = comparison_df.rename(columns=column_names)
        
        st.dataframe(
            comparison_df,
            hide_index=True,
            width="stretch",
            column_config={
                "年化收益 (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "波动率 (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "最大回撤 (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "总收益率 (%)": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
    
    # Find best strategy
    st.divider()
    col_best1, col_best2, col_best3 = st.columns(3)
    
    with col_best1:
        best_return = max(all_metrics, key=lambda x: x.get('CAGR (%)', -999))
        st.metric(
            "🏆 最高年化收益",
            best_return.get('策略名称', 'N/A'),
            f"{best_return.get('CAGR (%)', 0):.2f}%"
        )
    
    with col_best2:
        best_sharpe = max(all_metrics, key=lambda x: x.get('Sharpe Ratio', -999))
        st.metric(
            "🎯 最高夏普比率",
            best_sharpe.get('策略名称', 'N/A'),
            f"{best_sharpe.get('Sharpe Ratio', 0):.2f}"
        )
    
    with col_best3:
        best_dd = min(all_metrics, key=lambda x: abs(x.get('Max Drawdown (%)', -999)))
        st.metric(
            "🛡️ 最小回撤",
            best_dd.get('策略名称', 'N/A'),
            f"{best_dd.get('Max Drawdown (%)', 0):.2f}%"
        )
    
    st.divider()
    
    # Charts
    tab_equity, tab_dd, tab_weights, tab_monthly, tab_risk = st.tabs([
        "💰 净值曲线对比",
        "📉 回撤对比",
        "📊 仓位变化",
        "📊 收益分布",
        "⚖️ 风险收益"
    ])
    
    with tab_equity:
        render_multi_equity_curve(all_values)
    
    with tab_dd:
        render_multi_drawdown_chart(all_results)
    
    with tab_weights:
        render_multi_weights_comparison(all_results)
    
    with tab_monthly:
        render_returns_distribution(all_values)
    
    with tab_risk:
        render_risk_return_scatter(all_metrics)
    
    # Export comparison with settings
    st.divider()
    st.write("**📥 导出数据**")
    
    # Generate export with settings
    comparison_export = generate_comparison_export(
        comparison_df=comparison_df,
        all_values=all_values,
        config=config
    )
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        if not comparison_df.empty:
            st.download_button(
                "📥 完整对比报告 (CSV)",
                comparison_export['full_report'],
                file_name="strategy_comparison_report.csv",
                mime="text/csv",
                help="包含回测设置和对比指标"
            )
    
    with col_exp2:
        st.download_button(
            "📥 净值对比数据 (CSV)",
            comparison_export['values_csv'],
            file_name="strategy_values_comparison.csv",
            mime="text/csv",
            help="各策略每日净值对比"
        )


def generate_comparison_export(comparison_df, all_values, config):
    """Generate comparison export data with settings."""
    from datetime import datetime
    
    # Rebalance frequency mapping
    freq_map = {
        "daily": "每日",
        "weekly": "每周", 
        "monthly": "每月",
        "quarterly": "每季度"
    }
    
    # Build full report
    report_lines = [
        "=== 多策略对比报告 ===",
        f"生成时间,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== 回测设置 ===",
        f"开始日期,{config.start_date}",
        f"结束日期,{config.end_date}",
        f"初始资金,${config.initial_capital:,.0f}",
        f"调仓频率,{freq_map.get(config.rebalance_freq, config.rebalance_freq)}",
        f"佣金费率,{config.commission_pct * 100:.2f}%",
        f"滑点费率,{config.slippage_pct * 100:.2f}%",
        "",
        "=== 策略对比指标 ===",
    ]
    
    # Add comparison table
    if not comparison_df.empty:
        report_lines.append(",".join(comparison_df.columns.tolist()))
        for _, row in comparison_df.iterrows():
            report_lines.append(",".join([str(v) for v in row.values]))
    
    full_report = "\n".join(report_lines)
    
    # Build values comparison CSV
    values_lines = [
        "# 多策略净值对比数据",
        f"# 开始日期: {config.start_date}",
        f"# 结束日期: {config.end_date}",
        f"# 初始资金: ${config.initial_capital:,.0f}",
        f"# 调仓频率: {freq_map.get(config.rebalance_freq, config.rebalance_freq)}",
        "#",
    ]
    
    # Combine all values into single DataFrame
    if all_values:
        combined_df = pd.DataFrame(all_values)
        # Normalize to 100
        combined_df = (combined_df / combined_df.iloc[0]) * 100
        
        # Add header
        values_lines.append("日期," + ",".join(combined_df.columns.tolist()))
        
        for date_idx, row in combined_df.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d") if hasattr(date_idx, 'strftime') else str(date_idx)
            values = [f"{v:.2f}" for v in row.values]
            values_lines.append(f"{date_str}," + ",".join(values))
    
    values_csv = "\n".join(values_lines)
    
    return {
        'full_report': full_report,
        'values_csv': values_csv,
    }


def render_multi_equity_curve(all_values: Dict[str, pd.Series]):
    """Render multi-strategy equity curves."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Color palette
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    for i, (name, values) in enumerate(all_values.items()):
        if values is None or values.empty:
            continue
        
        # Normalize to 100 for comparison
        normalized = (values / values.iloc[0]) * 100
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f'{name}<br>日期: %{{x}}<br>净值: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="策略净值对比 (初始 = 100)",
        xaxis_title="日期",
        yaxis_title="净值",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_multi_drawdown_chart(all_results: Dict[str, BacktestResult]):
    """Render multi-strategy drawdown comparison."""
    import plotly.graph_objects as go
    from backtest.metrics import PerformanceMetrics
    
    fig = go.Figure()
    metrics = PerformanceMetrics()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    for i, (name, result) in enumerate(all_results.items()):
        if result.drawdown_series is None or result.drawdown_series.empty:
            continue
        
        dd = result.drawdown_series
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values * 100,  # Convert to percentage
            name=name,
            fill='tozeroy',
            line=dict(color=color, width=1),
            fillcolor=color.replace(')', ', 0.1)').replace('rgb', 'rgba'),
            hovertemplate=f'{name}<br>日期: %{{x}}<br>回撤: %{{y:.2f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        title="回撤对比",
        xaxis_title="日期",
        yaxis_title="回撤 (%)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_multi_weights_comparison(all_results: Dict[str, BacktestResult]):
    """Render weights comparison for multiple strategies."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Filter strategies with weights history (exclude benchmarks)
    strategies_with_weights = {
        name: result for name, result in all_results.items()
        if result.weights_history is not None and not result.weights_history.empty
        and "🧠" in name  # Only show strategy weights, not benchmarks
    }
    
    if not strategies_with_weights:
        st.info("仓位历史仅对动态策略可用（基准为静态仓位）")
        return
    
    # Let user select which strategy to view
    strategy_names = list(strategies_with_weights.keys())
    
    selected_strategy = st.selectbox(
        "选择要查看仓位变化的策略",
        strategy_names,
        key="weights_strategy_select"
    )
    
    if not selected_strategy or selected_strategy not in strategies_with_weights:
        return
    
    result = strategies_with_weights[selected_strategy]
    weights_history = result.weights_history
    
    # Get tickers
    tickers = [col for col in weights_history.columns if col not in ['date', 'index']]
    
    if not tickers:
        st.info("无仓位数据")
        return
    
    colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    
    # Stacked area chart
    fig = go.Figure()
    
    for i, ticker in enumerate(tickers):
        if ticker in weights_history.columns:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=weights_history.index,
                y=weights_history[ticker],
                name=ticker,
                mode='lines',
                stackgroup='one',
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate=f'{ticker}<br>日期: %{{x}}<br>权重: %{{y:.1f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"{selected_strategy} - 仓位变化",
        xaxis_title="日期",
        yaxis_title="权重 (%)",
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=450,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats table
    st.write("**📋 仓位统计**")
    stats_data = []
    for ticker in tickers:
        if ticker in weights_history.columns:
            series = weights_history[ticker]
            stats_data.append({
                '资产': ticker,
                '平均权重 (%)': round(series.mean(), 2),
                '最大权重 (%)': round(series.max(), 2),
                '最小权重 (%)': round(series.min(), 2),
                '权重波动 (%)': round(series.std(), 2),
            })
    
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, width="stretch")


def render_returns_distribution(all_values: Dict[str, pd.Series]):
    """Render returns distribution comparison."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    for i, (name, values) in enumerate(all_values.items()):
        if values is None or values.empty:
            continue
        
        # Calculate monthly returns (use 'ME' instead of deprecated 'M')
        monthly = values.resample('ME').last().pct_change().dropna() * 100
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Box(
            y=monthly.values,
            name=name,
            marker_color=color,
            boxmean=True,
        ))
    
    fig.update_layout(
        title="月度收益分布对比",
        yaxis_title="月度收益 (%)",
        showlegend=False,
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly stats table
    stats_data = []
    for name, values in all_values.items():
        if values is None or values.empty:
            continue
        monthly = values.resample('ME').last().pct_change().dropna() * 100
        stats_data.append({
            '策略': name,
            '平均月收益 (%)': round(monthly.mean(), 2),
            '月收益标准差 (%)': round(monthly.std(), 2),
            '最佳月 (%)': round(monthly.max(), 2),
            '最差月 (%)': round(monthly.min(), 2),
            '正收益月数': (monthly > 0).sum(),
            '负收益月数': (monthly < 0).sum(),
        })
    
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, width="stretch")


def render_risk_return_scatter(all_metrics: List[Dict]):
    """Render risk-return scatter plot."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    for i, metrics in enumerate(all_metrics):
        annual_return = metrics.get('CAGR (%)', 0)
        volatility = metrics.get('Volatility (%)', 1)
        sharpe = metrics.get('Sharpe Ratio', 0)
        name = metrics.get('策略名称', f'策略{i+1}')
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=[volatility],
            y=[annual_return],
            mode='markers+text',
            name=name,
            marker=dict(
                size=max(15, min(50, sharpe * 20)),  # Size based on Sharpe
                color=color,
            ),
            text=[name],
            textposition="top center",
            hovertemplate=(
                f'{name}<br>'
                f'年化收益: {annual_return:.2f}%<br>'
                f'波动率: {volatility:.2f}%<br>'
                f'夏普比率: {sharpe:.2f}'
                '<extra></extra>'
            ),
        ))
    
    # Add efficient frontier reference line (简化)
    min_vol = min(m.get('Volatility (%)', 10) for m in all_metrics) * 0.8
    max_vol = max(m.get('Volatility (%)', 20) for m in all_metrics) * 1.2
    
    fig.update_layout(
        title="风险-收益散点图 (点大小 = 夏普比率)",
        xaxis_title="波动率 (%)",
        yaxis_title="年化收益 (%)",
        showlegend=False,
        height=450,
    )
    
    # Add quadrant lines
    avg_return = np.mean([m.get('CAGR (%)', 0) for m in all_metrics])
    avg_vol = np.mean([m.get('Volatility (%)', 15) for m in all_metrics])
    
    fig.add_hline(y=avg_return, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=avg_vol, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add annotations for quadrants
    fig.add_annotation(
        x=max_vol * 0.9, y=avg_return * 1.5,
        text="高收益<br>高风险",
        showarrow=False,
        font=dict(color="gray", size=10)
    )
    fig.add_annotation(
        x=min_vol * 1.1, y=avg_return * 1.5,
        text="高收益<br>低风险 ⭐",
        showarrow=False,
        font=dict(color="green", size=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def get_benchmark_values(engine: BacktestEngine, benchmark_name: str, config: BacktestConfig):
    """Get benchmark portfolio values."""
    
    benchmark_map = {
        "SPY (标普500)": (["SPY"], {"SPY": 100}),
        "QQQ (纳斯达克)": (["QQQ"], {"QQQ": 100}),
        "60/40 组合": (["SPY", "TLT"], {"SPY": 60, "TLT": 40}),
    }
    
    if benchmark_name not in benchmark_map:
        return None
    
    tickers, weights = benchmark_map[benchmark_name]
    
    result = engine.run_static(
        tickers=tickers,
        weights=weights,
        config=config
    )
    
    if result.success:
        return {benchmark_name: result.portfolio_values}
    
    return None


def display_backtest_results(result, benchmark_values=None, show_trades=False, 
                            strategy_name=None, portfolio_name=None):
    """Display backtest results with charts and metrics."""
    
    st.divider()
    st.subheader("📊 回测结果")
    
    # Key metrics
    st.write("**关键指标**")
    render_metrics_cards(result.metrics)
    
    # Comparison with benchmark if available
    if benchmark_values:
        render_benchmark_comparison(result, benchmark_values)
    
    st.divider()
    
    # Charts - Add weights tab
    tabs = ["💰 净值曲线", "📉 回撤分析", "📊 仓位变化", "📅 月度收益", "📋 交易记录"]
    tab_equity, tab_dd, tab_weights, tab_monthly, tab_trades = st.tabs(tabs)
    
    with tab_equity:
        render_equity_curve(
            result.portfolio_values,
            benchmark_values=benchmark_values,
            title="组合净值走势"
        )
    
    with tab_dd:
        benchmark_dd = None
        if benchmark_values:
            from backtest.metrics import PerformanceMetrics
            metrics = PerformanceMetrics()
            benchmark_dd = {
                name: metrics.drawdown_series(values)
                for name, values in benchmark_values.items()
            }
        
        render_drawdown_chart(
            result.drawdown_series,
            benchmark_drawdowns=benchmark_dd,
            title="回撤分析"
        )
    
    with tab_weights:
        render_weights_history_chart(result.weights_history)
    
    with tab_monthly:
        render_monthly_returns_heatmap(
            result.portfolio_values,
            title="月度收益热力图"
        )
    
    with tab_trades:
        if show_trades and result.trades:
            st.write(f"**共 {len(result.trades)} 笔交易**")
            render_trade_history([t.to_dict() for t in result.trades])
            
            # Trading costs summary
            total_costs = sum(t.cost for t in result.trades)
            st.metric("总交易成本", f"${total_costs:,.2f}")
        else:
            st.info("静态回测无交易记录")
    
    # Detailed metrics table
    st.divider()
    with st.expander("📋 详细指标"):
        metrics_df = pd.DataFrame([
            {"指标": k, "值": f"{v:.2f}" if isinstance(v, float) else str(v)}
            for k, v in result.metrics.items()
        ])
        st.dataframe(metrics_df, hide_index=True, width="stretch")
    
    # Export section
    st.divider()
    st.write("**📥 导出数据**")
    
    # Generate export data with backtest settings
    export_data = generate_export_data(
        result=result,
        benchmark_values=benchmark_values,
        strategy_name=strategy_name,
        portfolio_name=portfolio_name
    )
    
    col_export1, col_export2, col_export3, col_export4 = st.columns(4)
    
    with col_export1:
        # Export complete report (includes settings)
        st.download_button(
            "📥 完整报告 (CSV)",
            export_data['full_report'],
            file_name="backtest_report.csv",
            mime="text/csv",
            help="包含回测设置、指标和净值数据"
        )
    
    with col_export2:
        # Export daily values with settings header
        st.download_button(
            "📥 净值数据 (CSV)",
            export_data['values_csv'],
            file_name="backtest_values.csv",
            mime="text/csv"
        )
    
    with col_export3:
        # Export trades
        if result.trades:
            st.download_button(
                "📥 交易记录 (CSV)",
                export_data['trades_csv'],
                file_name="backtest_trades.csv",
                mime="text/csv"
            )
    
    with col_export4:
        # Export weights history
        if result.weights_history is not None and not result.weights_history.empty:
            st.download_button(
                "📥 仓位历史 (CSV)",
                export_data['weights_csv'],
                file_name="backtest_weights.csv",
                mime="text/csv"
            )


def generate_export_data(result, benchmark_values=None, strategy_name=None, portfolio_name=None):
    """Generate export data with backtest settings included."""
    from datetime import datetime
    import io
    
    config = result.config
    
    # Rebalance frequency mapping
    freq_map = {
        "daily": "每日",
        "weekly": "每周", 
        "monthly": "每月",
        "quarterly": "每季度"
    }
    
    # Build settings info
    settings_info = [
        ["=== 回测报告 ===", ""],
        ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["", ""],
        ["=== 回测设置 ===", ""],
        ["开始日期", str(config.start_date)],
        ["结束日期", str(config.end_date)],
        ["初始资金", f"${config.initial_capital:,.0f}"],
        ["调仓频率", freq_map.get(config.rebalance_freq, config.rebalance_freq)],
        ["佣金费率", f"{config.commission_pct * 100:.2f}%"],
        ["滑点费率", f"{config.slippage_pct * 100:.2f}%"],
        ["无风险利率", f"{config.risk_free_rate * 100:.1f}%"],
    ]
    
    if strategy_name:
        settings_info.append(["策略名称", strategy_name])
    if portfolio_name:
        settings_info.append(["投资组合", portfolio_name])
    
    settings_info.append(["", ""])
    settings_info.append(["=== 业绩指标 ===", ""])
    
    # Add metrics
    for key, value in result.metrics.items():
        if isinstance(value, float):
            settings_info.append([key, f"{value:.4f}"])
        else:
            settings_info.append([key, str(value)])
    
    # Generate full report CSV
    full_report_lines = []
    for row in settings_info:
        full_report_lines.append(f"{row[0]},{row[1]}")
    
    full_report_lines.append("")
    full_report_lines.append("=== 每日净值 ===,")
    full_report_lines.append("日期,组合净值")
    
    for date_idx, value in result.portfolio_values.items():
        date_str = date_idx.strftime("%Y-%m-%d") if hasattr(date_idx, 'strftime') else str(date_idx)
        full_report_lines.append(f"{date_str},{value:.2f}")
    
    full_report = "\n".join(full_report_lines)
    
    # Generate values CSV with header
    values_lines = [
        "# 回测净值数据",
        f"# 开始日期: {config.start_date}",
        f"# 结束日期: {config.end_date}",
        f"# 初始资金: ${config.initial_capital:,.0f}",
        f"# 调仓频率: {freq_map.get(config.rebalance_freq, config.rebalance_freq)}",
        f"# 佣金: {config.commission_pct * 100:.2f}%",
        f"# 滑点: {config.slippage_pct * 100:.2f}%",
    ]
    if strategy_name:
        values_lines.append(f"# 策略: {strategy_name}")
    if portfolio_name:
        values_lines.append(f"# 组合: {portfolio_name}")
    values_lines.append("#")
    values_lines.append("日期,净值")
    
    for date_idx, value in result.portfolio_values.items():
        date_str = date_idx.strftime("%Y-%m-%d") if hasattr(date_idx, 'strftime') else str(date_idx)
        values_lines.append(f"{date_str},{value:.2f}")
    
    values_csv = "\n".join(values_lines)
    
    # Generate trades CSV with header
    trades_csv = ""
    if result.trades:
        trades_lines = [
            "# 交易记录",
            f"# 开始日期: {config.start_date}",
            f"# 结束日期: {config.end_date}",
            f"# 总交易次数: {len(result.trades)}",
            f"# 总交易成本: ${sum(t.cost for t in result.trades):,.2f}",
        ]
        if strategy_name:
            trades_lines.append(f"# 策略: {strategy_name}")
        trades_lines.append("#")
        trades_lines.append("日期,标的,操作,股数,价格,金额,成本")
        
        for trade in result.trades:
            trade_dict = trade.to_dict()
            trades_lines.append(
                f"{trade_dict['date']},{trade_dict['ticker']},{trade_dict['action']},"
                f"{trade_dict['shares']:.4f},{trade_dict['price']:.2f},"
                f"{trade_dict['value']:.2f},{trade_dict['cost']:.2f}"
            )
        
        trades_csv = "\n".join(trades_lines)
    
    # Generate weights CSV with header
    weights_csv = ""
    if result.weights_history is not None and not result.weights_history.empty:
        weights_lines = [
            "# 仓位历史",
            f"# 开始日期: {config.start_date}",
            f"# 结束日期: {config.end_date}",
            f"# 调仓频率: {freq_map.get(config.rebalance_freq, config.rebalance_freq)}",
        ]
        if strategy_name:
            weights_lines.append(f"# 策略: {strategy_name}")
        if portfolio_name:
            weights_lines.append(f"# 组合: {portfolio_name}")
        weights_lines.append("#")
        
        # Add column headers
        tickers = [col for col in result.weights_history.columns if col not in ['date', 'index']]
        weights_lines.append("日期," + ",".join(tickers))
        
        for date_idx, row in result.weights_history.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d") if hasattr(date_idx, 'strftime') else str(date_idx)
            values = [f"{row[t]:.2f}" for t in tickers]
            weights_lines.append(f"{date_str}," + ",".join(values))
        
        weights_csv = "\n".join(weights_lines)
    
    return {
        'full_report': full_report,
        'values_csv': values_csv,
        'trades_csv': trades_csv,
        'weights_csv': weights_csv,
    }


def render_benchmark_comparison(result, benchmark_values):
    """Render benchmark comparison metrics."""
    import plotly.graph_objects as go
    from backtest.metrics import PerformanceMetrics
    
    st.divider()
    st.write("**📊 基准对比**")
    
    metrics_calc = PerformanceMetrics()
    
    # Calculate metrics for strategy and benchmark
    strategy_metrics = result.metrics
    
    comparison_data = []
    
    # Strategy data
    comparison_data.append({
        '名称': '📈 策略',
        '总收益 (%)': strategy_metrics.get('Total Return (%)', 0),
        '年化收益 (%)': strategy_metrics.get('CAGR (%)', 0),
        '波动率 (%)': strategy_metrics.get('Volatility (%)', 0),
        '夏普比率': strategy_metrics.get('Sharpe Ratio', 0),
        '最大回撤 (%)': strategy_metrics.get('Max Drawdown (%)', 0),
        '卡玛比率': strategy_metrics.get('Calmar Ratio', 0),
    })
    
    # Benchmark data
    for bench_name, bench_values in benchmark_values.items():
        if bench_values is not None and not bench_values.empty:
            bench_metrics = metrics_calc.calculate_all(bench_values)
            comparison_data.append({
                '名称': f'📊 {bench_name}',
                '总收益 (%)': bench_metrics.get('Total Return (%)', 0),
                '年化收益 (%)': bench_metrics.get('CAGR (%)', 0),
                '波动率 (%)': bench_metrics.get('Volatility (%)', 0),
                '夏普比率': bench_metrics.get('Sharpe Ratio', 0),
                '最大回撤 (%)': bench_metrics.get('Max Drawdown (%)', 0),
                '卡玛比率': bench_metrics.get('Calmar Ratio', 0),
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(
        comparison_df,
        hide_index=True,
        width="stretch",
        column_config={
            "总收益 (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "年化收益 (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "波动率 (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "最大回撤 (%)": st.column_config.NumberColumn(format="%.2f%%"),
        }
    )
    
    # Calculate excess returns (Alpha)
    if len(comparison_data) >= 2:
        strategy_return = comparison_data[0]['年化收益 (%)']
        bench_return = comparison_data[1]['年化收益 (%)']
        excess_return = strategy_return - bench_return
        
        # Display alpha metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "normal" if excess_return >= 0 else "inverse"
            st.metric(
                "超额收益 (Alpha)",
                f"{excess_return:.2f}%",
                delta=f"vs {comparison_data[1]['名称']}"
            )
        
        with col2:
            strategy_vol = comparison_data[0]['波动率 (%)']
            bench_vol = comparison_data[1]['波动率 (%)']
            vol_diff = strategy_vol - bench_vol
            st.metric(
                "波动率差异",
                f"{strategy_vol:.2f}%",
                delta=f"{vol_diff:+.2f}%"
            )
        
        with col3:
            strategy_dd = comparison_data[0]['最大回撤 (%)']
            bench_dd = comparison_data[1]['最大回撤 (%)']
            dd_diff = strategy_dd - bench_dd
            st.metric(
                "最大回撤差异",
                f"{strategy_dd:.2f}%",
                delta=f"{dd_diff:+.2f}%"
            )
        
        with col4:
            strategy_sharpe = comparison_data[0]['夏普比率']
            bench_sharpe = comparison_data[1]['夏普比率']
            sharpe_diff = strategy_sharpe - bench_sharpe
            st.metric(
                "夏普比率差异",
                f"{strategy_sharpe:.2f}",
                delta=f"{sharpe_diff:+.2f}"
            )


def render_weights_history_chart(weights_history):
    """Render portfolio weights history as stacked area chart."""
    import plotly.graph_objects as go
    
    if weights_history is None or weights_history.empty:
        st.info("无仓位历史数据")
        return
    
    st.write("**📊 仓位变化历史**")
    
    # Get tickers (columns except 'date' if present)
    tickers = [col for col in weights_history.columns if col not in ['date', 'index']]
    
    if not tickers:
        st.info("无仓位数据")
        return
    
    # Color palette
    colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    
    # Create stacked area chart
    fig = go.Figure()
    
    for i, ticker in enumerate(tickers):
        if ticker in weights_history.columns:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=weights_history.index,
                y=weights_history[ticker],
                name=ticker,
                mode='lines',
                stackgroup='one',
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate=f'{ticker}<br>日期: %{{x}}<br>权重: %{{y:.1f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        title="组合仓位变化 (堆叠面积图)",
        xaxis_title="日期",
        yaxis_title="权重 (%)",
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=450,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show line chart for better readability
    st.write("**📈 各资产权重走势**")
    
    fig2 = go.Figure()
    
    for i, ticker in enumerate(tickers):
        if ticker in weights_history.columns:
            color = colors[i % len(colors)]
            fig2.add_trace(go.Scatter(
                x=weights_history.index,
                y=weights_history[ticker],
                name=ticker,
                mode='lines',
                line=dict(width=2, color=color),
                hovertemplate=f'{ticker}<br>日期: %{{x}}<br>权重: %{{y:.1f}}%<extra></extra>'
            ))
    
    fig2.update_layout(
        title="各资产权重变化",
        xaxis_title="日期",
        yaxis_title="权重 (%)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Summary statistics
    st.write("**📋 仓位统计**")
    
    stats_data = []
    for ticker in tickers:
        if ticker in weights_history.columns:
            series = weights_history[ticker]
            stats_data.append({
                '资产': ticker,
                '平均权重 (%)': round(series.mean(), 2),
                '最大权重 (%)': round(series.max(), 2),
                '最小权重 (%)': round(series.min(), 2),
                '权重标准差 (%)': round(series.std(), 2),
                '变化次数': int((series.diff().abs() > 0.5).sum()),
            })
    
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, width="stretch")
