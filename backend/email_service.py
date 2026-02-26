"""Email service for AuthChecker using Resend API.

Handles:
- Email verification OTPs
- Password reset tokens
"""
import os
import resend
import logging

logger = logging.getLogger("authchecker")

# Configure Resend
resend.api_key = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "AuthChecker <onboarding@resend.dev>")


def send_verification_email(to_email: str, otp_code: str, username: str) -> bool:
    """Send a 6-digit OTP for email verification."""
    try:
        html_body = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 32px; background: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb;">
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="color: #1a1a2e; font-size: 24px; margin: 0;">🛡️ AuthChecker</h1>
                <p style="color: #6b7280; font-size: 14px; margin-top: 4px;">Medicine Verification Platform</p>
            </div>
            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 16px 0;" />
            <p style="color: #374151; font-size: 16px;">Hi <strong>{username}</strong>,</p>
            <p style="color: #374151; font-size: 16px;">Welcome to AuthChecker! Please verify your email address using the code below:</p>
            <div style="text-align: center; margin: 32px 0;">
                <div style="display: inline-block; background: #f0f4ff; border: 2px dashed #4f46e5; border-radius: 12px; padding: 16px 32px;">
                    <span style="font-size: 36px; font-weight: bold; letter-spacing: 8px; color: #4f46e5;">{otp_code}</span>
                </div>
            </div>
            <p style="color: #6b7280; font-size: 14px; text-align: center;">This code expires in <strong>15 minutes</strong>.</p>
            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 24px 0;" />
            <p style="color: #9ca3af; font-size: 12px; text-align: center;">If you didn't create an account with AuthChecker, you can safely ignore this email.</p>
        </div>
        """

        params = {
            "from": FROM_EMAIL,
            "to": [to_email],
            "subject": f"AuthChecker — Your verification code is {otp_code}",
            "html": html_body,
        }
        email = resend.Emails.send(params)
        logger.info(f"[Email] Verification sent to {to_email}, id={email.get('id', 'n/a')}")
        return True

    except Exception as e:
        logger.error(f"[Email] Failed to send verification to {to_email}: {e}")
        return False


def send_password_reset_email(to_email: str, reset_token: str, username: str) -> bool:
    """Send a password reset link/token via email."""
    try:
        html_body = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 32px; background: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb;">
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="color: #1a1a2e; font-size: 24px; margin: 0;">🛡️ AuthChecker</h1>
                <p style="color: #6b7280; font-size: 14px; margin-top: 4px;">Password Reset Request</p>
            </div>
            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 16px 0;" />
            <p style="color: #374151; font-size: 16px;">Hi <strong>{username}</strong>,</p>
            <p style="color: #374151; font-size: 16px;">We received a request to reset your password. Use the code below in the app:</p>
            <div style="text-align: center; margin: 32px 0;">
                <div style="display: inline-block; background: #fef2f2; border: 2px dashed #dc2626; border-radius: 12px; padding: 16px 32px;">
                    <span style="font-size: 28px; font-weight: bold; letter-spacing: 4px; color: #dc2626;">{reset_token}</span>
                </div>
            </div>
            <p style="color: #6b7280; font-size: 14px; text-align: center;">This code expires in <strong>30 minutes</strong>.</p>
            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 24px 0;" />
            <p style="color: #9ca3af; font-size: 12px; text-align: center;">If you didn't request a password reset, please ignore this email. Your password will remain unchanged.</p>
        </div>
        """

        params = {
            "from": FROM_EMAIL,
            "to": [to_email],
            "subject": "AuthChecker — Password Reset Request",
            "html": html_body,
        }
        email = resend.Emails.send(params)
        logger.info(f"[Email] Password reset sent to {to_email}, id={email.get('id', 'n/a')}")
        return True

    except Exception as e:
        logger.error(f"[Email] Failed to send password reset to {to_email}: {e}")
        return False
