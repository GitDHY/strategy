"""
Notification settings page.
Configure email and WeChat push notifications.
"""

import streamlit as st
from datetime import datetime

from config.settings import get_settings, NotificationDefaults
from notification.email_sender import EmailSender
from notification.wechat_push import WeChatPush
from notification.scheduler import AlertScheduler


def render_notification_page():
    """Render the notification settings page."""
    
    st.title("🔔 通知设置")
    st.caption("配置邮件和微信推送提醒")
    
    settings = get_settings()
    config = settings.load_notification_config()
    
    tab1, tab2, tab3 = st.tabs(["📧 邮件设置", "💬 微信推送", "⏰ 定时任务"])
    
    with tab1:
        render_email_settings(settings, config)
    
    with tab2:
        render_wechat_settings(settings, config)
    
    with tab3:
        render_scheduler_settings()


def render_email_settings(settings, config: NotificationDefaults):
    """Render email configuration settings."""
    
    st.subheader("📧 邮件通知配置")
    
    st.info("💡 推荐使用 Gmail，需要开启「应用专用密码」功能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        smtp_server = st.text_input(
            "SMTP 服务器",
            value=config.smtp_server,
            placeholder="smtp.gmail.com",
            key="smtp_server"
        )
        
        email_from = st.text_input(
            "发件邮箱",
            value=config.email_from,
            placeholder="your-email@gmail.com",
            key="email_from"
        )
    
    with col2:
        smtp_port = st.number_input(
            "SMTP 端口",
            value=config.smtp_port,
            min_value=1,
            max_value=65535,
            key="smtp_port"
        )
        
        email_to = st.text_input(
            "收件邮箱",
            value=config.email_to,
            placeholder="recipient@gmail.com",
            key="email_to"
        )
    
    email_pwd = st.text_input(
        "邮箱密码 / 应用专用密码",
        value=config.email_pwd,
        type="password",
        key="email_pwd"
    )
    
    st.divider()
    
    col_save, col_test = st.columns(2)
    
    with col_save:
        if st.button("💾 保存邮件设置", use_container_width=True):
            config.smtp_server = smtp_server
            config.smtp_port = smtp_port
            config.email_from = email_from
            config.email_to = email_to
            config.email_pwd = email_pwd
            
            if settings.save_notification_config(config):
                st.success("✅ 邮件设置已保存")
            else:
                st.error("保存失败")
    
    with col_test:
        if st.button("🧪 发送测试邮件", type="primary", use_container_width=True):
            # Create sender with current config
            test_config = NotificationDefaults(
                smtp_server=smtp_server,
                smtp_port=smtp_port,
                email_from=email_from,
                email_to=email_to,
                email_pwd=email_pwd,
            )
            
            sender = EmailSender(test_config)
            
            if not sender.is_configured():
                st.error("请先填写完整的邮件配置")
            else:
                with st.spinner("正在发送测试邮件..."):
                    result = sender.send_test_email()
                
                if result.success:
                    st.success(f"✅ {result.message}")
                else:
                    st.error(f"❌ {result.message}")
    
    # Gmail setup guide
    with st.expander("📖 Gmail 设置指南"):
        st.markdown("""
### 如何获取 Gmail 应用专用密码

1. **开启两步验证**
   - 登录 Google 账户 → 安全性 → 两步验证 → 开启

2. **创建应用专用密码**
   - 安全性 → 两步验证 → 应用专用密码
   - 选择「邮件」和设备类型
   - 点击「生成」，复制 16 位密码

3. **填入配置**
   - SMTP 服务器: `smtp.gmail.com`
   - 端口: `587`
   - 密码: 使用上面生成的 16 位应用专用密码

### 常见问题

- **连接超时**: 检查网络是否可以访问 Gmail
- **认证失败**: 确认使用的是应用专用密码，不是 Google 账户密码
- **发送失败**: 检查发件邮箱是否正确
""")


def render_wechat_settings(settings, config: NotificationDefaults):
    """Render WeChat push notification settings."""
    
    st.subheader("💬 微信推送配置")
    
    st.info("💡 支持 Server酱 和 PushPlus 两种推送服务，配置任一即可")
    
    # Server酱
    st.write("**Server酱 (ServerChan)**")
    
    serverchan_key = st.text_input(
        "SendKey",
        value=config.serverchan_key,
        placeholder="SCT...",
        type="password",
        key="serverchan_key",
        help="在 https://sct.ftqq.com 获取"
    )
    
    col_sc1, col_sc2 = st.columns([3, 1])
    with col_sc2:
        if st.button("🧪 测试", key="test_sc", use_container_width=True):
            test_config = NotificationDefaults(serverchan_key=serverchan_key)
            push = WeChatPush(test_config)
            
            if not push.is_serverchan_configured():
                st.error("请先填写 SendKey")
            else:
                with st.spinner("发送测试消息..."):
                    result = push.send_serverchan("测试", "这是一条来自 Quant Platform 的测试消息")
                
                if result.success:
                    st.success("✅ 发送成功")
                else:
                    st.error(f"❌ {result.message}")
    
    st.divider()
    
    # PushPlus
    st.write("**PushPlus**")
    
    pushplus_token = st.text_input(
        "Token",
        value=config.pushplus_token,
        placeholder="xxxxxxxxxxxxxx",
        type="password",
        key="pushplus_token",
        help="在 https://www.pushplus.plus 获取"
    )
    
    col_pp1, col_pp2 = st.columns([3, 1])
    with col_pp2:
        if st.button("🧪 测试", key="test_pp", use_container_width=True):
            test_config = NotificationDefaults(pushplus_token=pushplus_token)
            push = WeChatPush(test_config)
            
            if not push.is_pushplus_configured():
                st.error("请先填写 Token")
            else:
                with st.spinner("发送测试消息..."):
                    result = push.send_pushplus("测试", "这是一条来自 Quant Platform 的测试消息")
                
                if result.success:
                    st.success("✅ 发送成功")
                else:
                    st.error(f"❌ {result.message}")
    
    st.divider()
    
    # Save button
    if st.button("💾 保存微信推送设置", use_container_width=True):
        config.serverchan_key = serverchan_key
        config.pushplus_token = pushplus_token
        
        if settings.save_notification_config(config):
            st.success("✅ 微信推送设置已保存")
        else:
            st.error("保存失败")
    
    # Setup guides
    with st.expander("📖 Server酱 设置指南"):
        st.markdown("""
### Server酱 配置步骤

1. 访问 [https://sct.ftqq.com](https://sct.ftqq.com)
2. 使用微信扫码登录
3. 进入「SendKey」页面
4. 复制你的 SendKey（以 SCT 开头）
5. 在微信中关注「方糖」公众号以接收消息

**特点**: 
- 免费额度充足
- 支持企业微信、邮箱等多通道
""")
    
    with st.expander("📖 PushPlus 设置指南"):
        st.markdown("""
### PushPlus 配置步骤

1. 访问 [https://www.pushplus.plus](https://www.pushplus.plus)
2. 使用微信扫码登录
3. 在首页获取你的 Token
4. 微信会自动关注 PushPlus 公众号

**特点**:
- 支持 Markdown 格式
- 支持群组推送
- 免费版有每日限制
""")


def render_scheduler_settings():
    """Render scheduler settings."""
    
    st.subheader("⏰ 定时检查设置")
    
    scheduler = AlertScheduler()
    config = scheduler.load_config()
    status = scheduler.get_status()
    
    # Status display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("调度器状态", "运行中 🟢" if status['running'] else "未运行 🔴")
    
    with col2:
        st.metric("上次运行", config.last_run or "从未运行")
    
    with col3:
        st.metric("检查时间", config.check_time)
    
    st.divider()
    
    # Configuration
    col_enable, col_time, col_freq = st.columns(3)
    
    with col_enable:
        enabled = st.checkbox("启用定时检查", value=config.enabled, key="sched_enabled")
    
    with col_time:
        check_time = st.time_input(
            "检查时间",
            value=datetime.strptime(config.check_time, "%H:%M").time(),
            key="sched_time"
        )
    
    with col_freq:
        frequency = st.selectbox(
            "检查频率",
            ["daily", "weekly"],
            index=0 if config.frequency == "daily" else 1,
            format_func=lambda x: "每日" if x == "daily" else "每周",
            key="sched_freq"
        )
    
    st.divider()
    
    col_save, col_run = st.columns(2)
    
    with col_save:
        if st.button("💾 保存定时设置", use_container_width=True):
            config.enabled = enabled
            config.check_time = check_time.strftime("%H:%M")
            config.frequency = frequency
            
            if scheduler.save_config(config):
                st.success("✅ 定时设置已保存")
            else:
                st.error("保存失败")
    
    with col_run:
        if st.button("▶️ 立即运行检查", type="primary", use_container_width=True):
            with st.spinner("正在检查..."):
                results = scheduler.run_now()
            
            st.success(f"检查完成 ({results['timestamp']})")
            
            if results['checks']:
                for check in results['checks']:
                    if check['success']:
                        st.info(f"✅ {check['callback']}: 成功")
                    else:
                        st.error(f"❌ {check['callback']}: {check.get('error', '失败')}")
            else:
                st.info("未配置检查回调函数")
    
    # Help text
    with st.expander("📖 定时检查说明"):
        st.markdown("""
### 工作原理

定时检查功能会在指定时间自动运行你保存的策略，并在检测到调仓信号时发送通知。

### 设置步骤

1. **创建策略**: 在「策略编辑器」中创建并保存你的策略
2. **配置通知**: 在「邮件设置」或「微信推送」中配置通知渠道
3. **启用定时**: 开启定时检查并设置检查时间

### 注意事项

- 检查时间基于你的本地时区
- 策略检查需要网络连接获取市场数据
- 建议设置在美股收盘后的时间（如北京时间 9:00）
- 确保应用保持运行状态
""")
