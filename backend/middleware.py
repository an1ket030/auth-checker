"""Security and logging middleware for AuthChecker backend."""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time
import logging

logger = logging.getLogger("authchecker")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with timing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({duration_ms:.0f}ms)"
        )
        return response


class HMACSigningMiddleware(BaseHTTPMiddleware):
    """Validate HMAC-SHA256 signatures on critical endpoints.
    
    The client must send an `X-HMAC-Signature` header with:
        HMAC-SHA256(secret_key, request_path + timestamp)
    And an `X-HMAC-Timestamp` header with the Unix timestamp used.
    
    Protected paths: POST /api/v1/scan, POST /api/v1/reports
    """
    PROTECTED_PATHS = {"/api/v1/scan", "/api/v1/reports"}

    def __init__(self, app, secret_key: str = ""):
        super().__init__(app)
        self.secret_key = secret_key
        import os
        self.enabled = os.getenv("HMAC_ENABLED", "false").lower() == "true"

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        method = request.method.upper()

        if method == "POST" and path in self.PROTECTED_PATHS:
            import hmac
            import hashlib

            signature = request.headers.get("X-HMAC-Signature", "")
            timestamp = request.headers.get("X-HMAC-Timestamp", "")

            if not signature or not timestamp:
                from starlette.responses import JSONResponse
                return JSONResponse(
                    {"detail": "Missing HMAC signature headers"},
                    status_code=403
                )

            # Validate timestamp is within 5 minutes
            try:
                ts = int(timestamp)
                now = int(time.time())
                if abs(now - ts) > 300:
                    from starlette.responses import JSONResponse
                    return JSONResponse(
                        {"detail": "HMAC timestamp expired"},
                        status_code=403
                    )
            except ValueError:
                from starlette.responses import JSONResponse
                return JSONResponse(
                    {"detail": "Invalid HMAC timestamp"},
                    status_code=403
                )

            # Compute expected signature
            message = f"{path}{timestamp}"
            expected = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected):
                from starlette.responses import JSONResponse
                return JSONResponse(
                    {"detail": "Invalid HMAC signature"},
                    status_code=403
                )

        return await call_next(request)
