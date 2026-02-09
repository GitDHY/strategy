"""
WeChat push notification module.
Supports Server酱 and PushPlus services for WeChat notifications.
"""

import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from config.settings import get_settings, NotificationDefaults


@dataclass
class PushResult:
    """Result of push notification operation."""
    success: bool
    message: str
    service: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class WeChatPush:
    """
    WeChat push notification sender.
    Supports Server酱 (ServerChan) and PushPlus services.
    """
    
    # API endpoints
    SERVERCHAN_API = "https://sctapi.ftqq.com/{key}.send"
    PUSHPLUS_API = "http://www.pushplus.plus/send"
    
    def __init__(self, config: Optional[NotificationDefaults] = None):
        """
        Initialize WeChat push sender.
        
        Args:
            config: Notification configuration
        """
        if config is None:
            settings = get_settings()
            config = settings.load_notification_config()
        
        self.serverchan_key = config.serverchan_key
        self.pushplus_token = config.pushplus_token
    
    def is_serverchan_configured(self) -> bool:
        """Check if Server酱 is configured."""
        return bool(self.serverchan_key)
    
    def is_pushplus_configured(self) -> bool:
        """Check if PushPlus is configured."""
        return bool(self.pushplus_token)
    
    def is_configured(self) -> bool:
        """Check if any push service is configured."""
        return self.is_serverchan_configured() or self.is_pushplus_configured()
    
    def send_serverchan(
        self,
        title: str,
        content: str = "",
        channel: str = ""
    ) -> PushResult:
        """
        Send notification via Server酱.
        
        Args:
            title: Message title (required, max 32 chars)
            content: Message content (optional, supports Markdown)
            channel: Push channel (optional: 9=iOS, 98=邮箱, 66=企业微信群机器人)
            
        Returns:
            PushResult
        """
        if not self.is_serverchan_configured():
            return PushResult(
                success=False,
                message="Server酱未配置，请设置 SendKey",
                service="serverchan"
            )
        
        url = self.SERVERCHAN_API.format(key=self.serverchan_key)
        
        data = {
            'title': title[:32],  # Max 32 chars
            'desp': content,
        }
        
        if channel:
            data['channel'] = channel
        
        try:
            response = requests.post(url, data=data, timeout=10)
            result = response.json()
            
            if result.get('code') == 0:
                return PushResult(
                    success=True,
                    message="Server酱推送成功",
                    service="serverchan"
                )
            else:
                error_msg = result.get('message', '未知错误')
                return PushResult(
                    success=False,
                    message=f"Server酱推送失败: {error_msg}",
                    service="serverchan"
                )
                
        except requests.Timeout:
            return PushResult(
                success=False,
                message="Server酱请求超时",
                service="serverchan"
            )
        except Exception as e:
            return PushResult(
                success=False,
                message=f"Server酱推送异常: {str(e)}",
                service="serverchan"
            )
    
    def send_pushplus(
        self,
        title: str,
        content: str = "",
        template: str = "html",
        topic: str = ""
    ) -> PushResult:
        """
        Send notification via PushPlus.
        
        Args:
            title: Message title
            content: Message content
            template: Content template (html, txt, json, markdown)
            topic: Topic ID for group push (optional)
            
        Returns:
            PushResult
        """
        if not self.is_pushplus_configured():
            return PushResult(
                success=False,
                message="PushPlus未配置，请设置 Token",
                service="pushplus"
            )
        
        data = {
            'token': self.pushplus_token,
            'title': title,
            'content': content,
            'template': template,
        }
        
        if topic:
            data['topic'] = topic
        
        try:
            response = requests.post(
                self.PUSHPLUS_API,
                json=data,
                timeout=10
            )
            result = response.json()
            
            if result.get('code') == 200:
                return PushResult(
                    success=True,
                    message="PushPlus推送成功",
                    service="pushplus"
                )
            else:
                error_msg = result.get('msg', '未知错误')
                return PushResult(
                    success=False,
                    message=f"PushPlus推送失败: {error_msg}",
                    service="pushplus"
                )
                
        except requests.Timeout:
            return PushResult(
                success=False,
                message="PushPlus请求超时",
                service="pushplus"
            )
        except Exception as e:
            return PushResult(
                success=False,
                message=f"PushPlus推送异常: {str(e)}",
                service="pushplus"
            )
    
    def send(
        self,
        title: str,
        content: str = "",
        prefer_service: str = "pushplus"
    ) -> PushResult:
        """
        Send notification via available service.
        
        Args:
            title: Message title
            content: Message content
            prefer_service: Preferred service ('pushplus' or 'serverchan')
            
        Returns:
            PushResult
        """
        # Try preferred service first
        if prefer_service == "serverchan" and self.is_serverchan_configured():
            result = self.send_serverchan(title, content)
            if result.success:
                return result
        
        if prefer_service == "pushplus" and self.is_pushplus_configured():
            result = self.send_pushplus(title, content)
            if result.success:
                return result
        
        # Try fallback service
        if self.is_pushplus_configured():
            result = self.send_pushplus(title, content)
            if result.success:
                return result
        
        if self.is_serverchan_configured():
            result = self.send_serverchan(title, content)
            if result.success:
                return result
        
        return PushResult(
            success=False,
            message="未配置任何微信推送服务",
            service="none"
        )
    
    def send_strategy_alert(
        self,
        strategy_name: str,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        reason: str = ""
    ) -> PushResult:
        """
        Send strategy rebalancing alert via WeChat.
        
        Args:
            strategy_name: Strategy name
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            reason: Reason for rebalancing
            
        Returns:
            PushResult
        """
        title = f"🔄 调仓提醒: {strategy_name}"
        
        # Build Markdown content
        lines = [
            f"### 📊 策略调仓提醒",
            f"",
            f"**策略名称:** {strategy_name}",
            f"**检测时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
        ]
        
        if reason:
            lines.extend([
                f"**调仓原因:**",
                f"> {reason}",
                f"",
            ])
        
        lines.append("**建议调仓:**")
        lines.append("")
        lines.append("| 标的 | 当前 | 目标 | 变化 |")
        lines.append("|------|------|------|------|")
        
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        for ticker in sorted(all_tickers):
            curr = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            diff = target - curr
            if abs(diff) > 0.1:
                arrow = "⬆️" if diff > 0 else "⬇️"
                lines.append(f"| {ticker} | {curr:.1f}% | {target:.1f}% | {arrow} {diff:+.1f}% |")
        
        lines.extend([
            "",
            "---",
            "*此消息由 Quant Platform 自动发送*",
        ])
        
        content = "\n".join(lines)
        
        # Try PushPlus with markdown template
        if self.is_pushplus_configured():
            return self.send_pushplus(title, content, template="markdown")
        
        # Fallback to Server酱
        if self.is_serverchan_configured():
            return self.send_serverchan(title, content)
        
        return PushResult(
            success=False,
            message="未配置任何微信推送服务",
            service="none"
        )
    
    def send_test(self) -> PushResult:
        """
        Send test notification to verify configuration.
        
        Returns:
            PushResult
        """
        title = "🧪 Quant Platform 测试消息"
        content = f"""
### ✅ 推送测试成功！

**发送时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

如果您收到此消息，说明微信推送功能已正确配置。

---
*此消息由 Quant Platform 发送*
"""
        
        return self.send(title, content)
