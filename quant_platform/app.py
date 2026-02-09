"""
Quant Platform - 量化投资管理平台
主应用入口

功能:
- 投资组合管理 (美股/新加坡股市)
- 动态策略编写与测试
- 历史回测 (支持交易成本、滑点)
- 调仓提醒 (邮件/微信)

运行方式:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Quant Platform - 量化投资平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # Quant Platform 📈
        
        量化投资管理平台 v1.0
        
        功能特性:
        - 📊 投资组合管理
        - 🧠 动态策略编写
        - 📈 历史回测分析
        - 🔔 调仓提醒通知
        """
    }
)

# Import pages
from ui.pages.portfolio_page import render_portfolio_page
from ui.pages.strategy_page import render_strategy_page
from ui.pages.backtest_page import render_backtest_page
from ui.pages.notification_page import render_notification_page


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("📈 Quant Platform")
    st.sidebar.caption("量化投资管理平台")
    
    st.sidebar.divider()
    
    # Navigation menu
    pages = {
        "📊 投资组合": "portfolio",
        "🧠 策略编辑": "strategy",
        "📈 回测分析": "backtest",
        "🔔 通知设置": "notification",
    }
    
    selected_page = st.sidebar.radio(
        "功能导航",
        list(pages.keys()),
        key="nav_radio"
    )
    
    st.sidebar.divider()
    
    # Quick info
    with st.sidebar.expander("ℹ️ 使用说明"):
        st.markdown("""
        **快速开始:**
        
        1. 📊 **创建组合** - 添加投资标的和配置权重
        2. 🧠 **编写策略** - 使用 Python 编写动态调仓逻辑
        3. 📈 **回测验证** - 用历史数据验证策略效果
        4. 🔔 **配置提醒** - 设置邮件/微信通知
        
        **支持市场:**
        - 🇺🇸 美股 (NYSE, NASDAQ)
        - 🇸🇬 新加坡 (SGX)
        
        **数据来源:**
        - Yahoo Finance
        """)
    
    # Version info
    st.sidebar.caption("v1.0.0 | Powered by Streamlit")
    
    # Render selected page
    page_key = pages[selected_page]
    
    if page_key == "portfolio":
        render_portfolio_page()
    elif page_key == "strategy":
        render_strategy_page()
    elif page_key == "backtest":
        render_backtest_page()
    elif page_key == "notification":
        render_notification_page()


if __name__ == "__main__":
    main()
