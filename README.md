# Macro Strategy Analyzer (宏观策略分析器)

## 项目简介
这是一个基于 Streamlit 的宏观策略分析应用，旨在帮助投资者根据宏观经济指标（如失业率、利率、恐慌指数等）自动判断当前的市场状态，并提供相应的资产配置建议。

该项目实现了“全天候”策略的变体，将市场划分为 6 种宏观状态：
1.  **INFLATION_SHOCK (滞胀/加息冲击)**: 现金为王
2.  **DEFLATION_RECESSION (衰退/崩盘)**: 债券与黄金
3.  **EXTREME_ACCUMULATION (极度贪婪/抄底)**: 成长股抄底
4.  **CAUTIOUS_TREND (谨慎/趋势破位)**: 防御配置
5.  **CAUTIOUS_VOL (谨慎/高波震荡)**: 对冲配置
6.  **NEUTRAL (常态/牛市)**: 标准增长配置

## 功能特性
*   **自动宏观数据获取**: 集成 FRED 和 Yahoo Finance API。
*   **状态机诊断**: 根据萨姆规则、TNX 冲击、VIX 等指标判断经济周期。
*   **动态资产配置**: 针对不同状态推荐具体的 ETF/REITs 组合。
*   **历史回测**: 仿真策略在过去时间段的表现，并与 SPY/60:40 组合对比。
*   **邮件预警**: 支持定时检查并通过 SMTP 发送策略变更通知。

## 安装与运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动应用
```bash
streamlit run app.py
```

## 数据源说明
*   **FRED**: 失业率 (UNRATE), 10年-2年国债利差 (T10Y2Y)
*   **Yahoo Finance**: 标普500, 纳斯达克, 黄金, 波动率 (VIX) 等价格数据

## 目录结构
*   `app.py`: 主程序入口
*   `strategy_doc.md`: 策略逻辑详细文档
*   `requirements.txt`: 项目依赖
*   `data/` & `*.csv`: 本地缓存的宏观数据文件
