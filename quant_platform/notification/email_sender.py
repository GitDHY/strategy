"""
Email notification sender module.
Supports SMTP-based email delivery for strategy alerts.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from config.settings import get_settings, NotificationDefaults


@dataclass
class EmailResult:
    """Result of email sending operation."""
    success: bool
    message: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EmailSender:
    """
    SMTP-based email sender for strategy notifications.
    """
    
    def __init__(self, config: Optional[NotificationDefaults] = None):
        """
        Initialize email sender.
        
        Args:
            config: Notification configuration (loads from settings if not provided)
        """
        if config is None:
            settings = get_settings()
            config = settings.load_notification_config()
        
        self.smtp_server = config.smtp_server
        self.smtp_port = config.smtp_port
        self.email_from = config.email_from
        self.email_to = config.email_to
        self.email_pwd = config.email_pwd
    
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(
            self.smtp_server and
            self.email_from and
            self.email_to and
            self.email_pwd
        )
    
    def send(
        self,
        subject: str,
        body: str,
        to_email: Optional[str] = None,
        html: bool = False
    ) -> EmailResult:
        """
        Send an email.
        
        Args:
            subject: Email subject
            body: Email body (plain text or HTML)
            to_email: Recipient email (uses config default if not provided)
            html: Whether body is HTML
            
        Returns:
            EmailResult with success status
        """
        if not self.is_configured():
            return EmailResult(
                success=False,
                message="Email not configured. Please set SMTP settings."
            )
        
        recipient = to_email or self.email_to
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_from
            msg['To'] = recipient
            
            # Add body
            content_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, content_type, 'utf-8'))
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.email_from, self.email_pwd)
                server.sendmail(self.email_from, recipient, msg.as_string())
            
            return EmailResult(
                success=True,
                message=f"Email sent successfully to {recipient}"
            )
            
        except smtplib.SMTPAuthenticationError:
            return EmailResult(
                success=False,
                message="SMTP authentication failed. Check email/password."
            )
        except smtplib.SMTPException as e:
            return EmailResult(
                success=False,
                message=f"SMTP error: {str(e)}"
            )
        except Exception as e:
            return EmailResult(
                success=False,
                message=f"Failed to send email: {str(e)}"
            )
    
    def send_strategy_alert(
        self,
        strategy_name: str,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        reason: str = "",
        additional_info: Dict[str, Any] = None
    ) -> EmailResult:
        """
        Send a strategy rebalancing alert email.
        
        Args:
            strategy_name: Name of the strategy
            current_weights: Current portfolio weights
            target_weights: Recommended target weights
            reason: Reason for rebalancing
            additional_info: Additional data to include
            
        Returns:
            EmailResult
        """
        # Build HTML email content
        subject = f"🔄 策略调仓提醒: {strategy_name}"
        
        # Calculate weight changes
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        changes = []
        for ticker in sorted(all_tickers):
            curr = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            diff = target - curr
            if abs(diff) > 0.1:  # Only show significant changes
                arrow = "⬆️" if diff > 0 else "⬇️"
                changes.append(f"<tr><td>{ticker}</td><td>{curr:.1f}%</td><td>{target:.1f}%</td><td>{arrow} {diff:+.1f}%</td></tr>")
        
        changes_html = "".join(changes) if changes else "<tr><td colspan='4'>无显著调仓</td></tr>"
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h2 {{ color: #2962FF; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                .reason {{ background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h2>📊 策略调仓提醒</h2>
            <p><strong>策略名称:</strong> {strategy_name}</p>
            <p><strong>检测时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="reason">
                <strong>调仓原因:</strong><br/>
                {reason or '策略信号触发'}
            </div>
            
            <h3>📋 建议调仓明细</h3>
            <table>
                <tr>
                    <th>标的</th>
                    <th>当前仓位</th>
                    <th>目标仓位</th>
                    <th>变化</th>
                </tr>
                {changes_html}
            </table>
            
            <div class="footer">
                <p>此邮件由 Quant Platform 自动发送，请勿直接回复。</p>
                <p>风险提示：以上内容仅供参考，不构成投资建议。</p>
            </div>
        </body>
        </html>
        """
        
        return self.send(subject, html_body, html=True)
    
    def send_test_email(self) -> EmailResult:
        """
        Send a test email to verify configuration.
        
        Returns:
            EmailResult
        """
        subject = "🧪 Quant Platform 测试邮件"
        body = f"""
        <html>
        <body>
            <h2>✅ 邮件配置测试成功！</h2>
            <p>这是一封来自 Quant Platform 的测试邮件。</p>
            <p>发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>如果您收到此邮件，说明邮件通知功能已正确配置。</p>
        </body>
        </html>
        """
        return self.send(subject, body, html=True)
