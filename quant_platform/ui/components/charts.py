"""
Chart components using Plotly.
Provides reusable chart functions for backtest visualization.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Any
import streamlit as st


def render_equity_curve(
    portfolio_values: pd.Series,
    benchmark_values: Optional[Dict[str, pd.Series]] = None,
    title: str = "Portfolio Value",
    height: int = 500,
) -> go.Figure:
    """
    Render equity curve chart.
    
    Args:
        portfolio_values: Main portfolio value series
        benchmark_values: Optional dict of benchmark name -> value series
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Main portfolio line
    fig.add_trace(go.Scatter(
        x=portfolio_values.index,
        y=portfolio_values.values,
        name="Portfolio",
        line=dict(width=2.5, color='#2962FF'),
        hovertemplate='%{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))
    
    # Benchmark lines
    if benchmark_values:
        colors = ['#FF6D00', '#00C853', '#AA00FF', '#FFD600', '#D50000']
        for i, (name, values) in enumerate(benchmark_values.items()):
            fig.add_trace(go.Scatter(
                x=values.index,
                y=values.values,
                name=name,
                line=dict(width=1.5, color=colors[i % len(colors)], dash='dot'),
                hovertemplate='%{x}<br>' + name + ': $%{y:,.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


def render_drawdown_chart(
    drawdown_series: pd.Series,
    benchmark_drawdowns: Optional[Dict[str, pd.Series]] = None,
    title: str = "Drawdown",
    height: int = 400,
) -> go.Figure:
    """
    Render drawdown chart.
    
    Args:
        drawdown_series: Main drawdown series (in percentage)
        benchmark_drawdowns: Optional dict of benchmark drawdowns
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Main drawdown (filled area)
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        name="Portfolio",
        fill='tozeroy',
        line=dict(width=1.5, color='#2962FF'),
        fillcolor='rgba(41, 98, 255, 0.3)',
        hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Benchmark drawdowns
    if benchmark_drawdowns:
        colors = ['#FF6D00', '#00C853', '#AA00FF']
        for i, (name, dd) in enumerate(benchmark_drawdowns.items()):
            fig.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                name=name,
                line=dict(width=1, color=colors[i % len(colors)]),
                hovertemplate='%{x}<br>' + name + ': %{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(
            ticksuffix="%",
            range=[min(drawdown_series.min() * 1.1, -5), 1]
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


def render_monthly_returns_heatmap(
    values: pd.Series,
    title: str = "Monthly Returns",
    height: int = 400,
) -> go.Figure:
    """
    Render monthly returns heatmap.
    
    Args:
        values: Portfolio value series
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    # Calculate monthly returns (use 'ME' instead of deprecated 'M')
    monthly = values.resample('ME').last().pct_change() * 100
    
    if monthly.empty:
        st.info("Not enough data for monthly analysis")
        return None
    
    # Create pivot table
    monthly_df = monthly.to_frame(name='Return')
    monthly_df['Year'] = monthly_df.index.year
    monthly_df['Month'] = monthly_df.index.month_name().str[:3]
    
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    pivot = monthly_df.pivot_table(index='Year', columns='Month', values='Return')
    pivot = pivot.reindex(columns=month_order)
    
    # Add yearly total (use 'YE' instead of deprecated 'Y')
    yearly = values.resample('YE').last().pct_change() * 100
    yearly.index = yearly.index.year
    pivot['YTD'] = yearly
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        colorscale='RdBu',
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont={"size": 10},
        showscale=True,
        colorbar=dict(title="Return %"),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=max(height, len(pivot) * 35 + 100),
        yaxis=dict(autorange="reversed", type='category'),
        xaxis=dict(side='top'),
        margin=dict(l=60, r=30, t=80, b=30),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


def render_correlation_matrix(
    prices: pd.DataFrame,
    title: str = "Correlation Matrix",
    height: int = 500,
) -> go.Figure:
    """
    Render asset correlation matrix heatmap.
    
    Args:
        prices: DataFrame of asset prices
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    if prices.shape[1] < 2:
        st.info("Need at least 2 assets for correlation analysis")
        return None
    
    # Calculate correlation
    returns = prices.pct_change().dropna()
    corr = returns.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11},
        showscale=True,
        colorbar=dict(title="Correlation"),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(side='bottom'),
        margin=dict(l=80, r=30, t=60, b=80),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


def render_allocation_pie(
    weights: Dict[str, float],
    title: str = "Portfolio Allocation",
    height: int = 400,
) -> go.Figure:
    """
    Render portfolio allocation pie chart.
    
    Args:
        weights: Dictionary of ticker -> weight
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure
    """
    # Filter out zero weights
    weights = {k: v for k, v in weights.items() if v > 0}
    
    if not weights:
        st.info("No allocation data")
        return None
    
    labels = list(weights.keys())
    values = list(weights.values())
    
    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='%{label}<br>Weight: %{value:.1f}%<br>(%{percent})<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(l=30, r=30, t=60, b=60),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


def render_trade_history(
    trades: List[Dict[str, Any]],
    height: int = 400,
):
    """
    Render trade history table.
    
    Args:
        trades: List of trade dictionaries
        height: Table height
    """
    if not trades:
        st.info("No trades recorded")
        return
    
    df = pd.DataFrame(trades)
    
    st.dataframe(
        df,
        use_container_width=True,
        height=height,
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "action": st.column_config.TextColumn("Action"),
            "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "value": st.column_config.NumberColumn("Value", format="$%.2f"),
            "cost": st.column_config.NumberColumn("Cost", format="$%.2f"),
        }
    )


def render_metrics_cards(metrics: Dict[str, float]):
    """
    Render key performance metrics as cards.
    
    Args:
        metrics: Dictionary of metric name -> value
    """
    # Define metric display config
    metric_config = {
        'Total Return (%)': {'format': '{:.2f}%', 'icon': '📈'},
        'CAGR (%)': {'format': '{:.2f}%', 'icon': '📊'},
        'Max Drawdown (%)': {'format': '{:.2f}%', 'icon': '📉'},
        'Sharpe Ratio': {'format': '{:.2f}', 'icon': '⚖️'},
        'Sortino Ratio': {'format': '{:.2f}', 'icon': '📐'},
        'Calmar Ratio': {'format': '{:.2f}', 'icon': '🎯'},
        'Volatility (%)': {'format': '{:.2f}%', 'icon': '🌊'},
        'Win Rate (%)': {'format': '{:.1f}%', 'icon': '✅'},
    }
    
    # Display in columns
    cols = st.columns(4)
    
    for i, (name, value) in enumerate(metrics.items()):
        col = cols[i % 4]
        config = metric_config.get(name, {'format': '{:.2f}', 'icon': '📌'})
        
        with col:
            formatted_value = config['format'].format(value)
            st.metric(
                label=f"{config['icon']} {name}",
                value=formatted_value,
            )
