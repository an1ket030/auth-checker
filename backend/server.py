# backend/server.py (FULL - patched)
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import shutil
import os
import uuid
from datetime import date
from typing import List, Optional
import re
import traceback
import io
import httpx

from .middleware import SecurityHeadersMiddleware, RequestLoggingMiddleware
from .email_service import send_verification_email, send_password_reset_email
from .models import (
    Users, ValidMedicine, ScanHistory, ScanStatus,
    EmailVerification, PasswordReset, UserSession,
    DrugInformation, CounterfeitReport
)
import random
import string
import secrets

# ML Inference is handled by HuggingFace Space (external service)
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:7860")

# pyzbar removed due to deployment constraints (libzbar0 missing on Render Python env)

from .database import engine, get_db, Base

# Security Imports
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer
from fastapi import Request

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# SECURITY CONFIG
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if os.getenv("ENVIRONMENT") == "production":
        raise ValueError("No SECRET_KEY set for production environment.")
    print("WARNING: Using default insecure key for development.")
    SECRET_KEY = "supersecretkey123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Reduced from 600; refresh tokens will handle re-auth

# pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
from passlib.hash import pbkdf2_sha256

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    print(f"DEBUG: verify_password called")
    try:
        return pbkdf2_sha256.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"DEBUG: verify_password FAILED: {e}")
        return False

def get_password_hash(password):
    return pbkdf2_sha256.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(Users).filter(Users.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Legacy imports removed# Force table creation (models must match DB; migrations preferred)
# Force table creation (models must match DB; migrations preferred)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AuthChecker Production API")

# -- Background Cleanup Task --
import asyncio
import time

async def cleanup_old_uploads():
    """Delete files in uploads/ older than 24 hours."""
    while True:
        try:
            upload_dir = "uploads"
            if os.path.exists(upload_dir):
                now = time.time()
                for f in os.listdir(upload_dir):
                    fpath = os.path.join(upload_dir, f)
                    if os.path.isfile(fpath):
                        # Delete if older than 24h (86400 seconds)
                        if os.stat(fpath).st_mtime < now - 86400:
                            os.remove(fpath)
                            print(f"[Cleanup] Deleted old file: {f}")
            await asyncio.sleep(3600) # Check every hour
        except Exception as e:
            print(f"[Cleanup Error]: {e}")
            await asyncio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_uploads())

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers (nosniff, HSTS, X-Frame-Options, etc.)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Add Rate Limiter Exception Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

async def call_ml_inference(image_bytes: bytes, filename: str = "scan.jpg"):
    """Forward image to HuggingFace Space for ML inference."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (filename, image_bytes, "image/jpeg")}
            response = await client.post(f"{HF_SPACE_URL}/predict", files=files)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        print("[ML] HuggingFace Space timed out (cold start?)")
        return {"label": "ERROR", "confidence": 0.0, "reason": "ML service timed out. Please try again."}
    except httpx.HTTPStatusError as e:
        print(f"[ML] HuggingFace Space returned {e.response.status_code}: {e.response.text}")
        return {"label": "ERROR", "confidence": 0.0, "reason": f"ML service error: {e.response.status_code}"}
    except Exception as e:
        print(f"[ML] Failed to reach HuggingFace Space: {e}")
        return {"label": "ERROR", "confidence": 0.0, "reason": str(e)}

from pydantic import BaseModel, validator

# ---- Pydantic models ----
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

    @validator("email")
    def validate_email(cls, v):
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
        return v

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one number")
        return v

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_verified: bool = False
    access_token: str
    refresh_token: str
    token_type: str

class HistoryItem(BaseModel):
    product: str
    status: str
    date: str
    score: float

class VerifyEmailRequest(BaseModel):
    email: str
    otp_code: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email: str
    reset_token: str
    new_password: str

    @validator("new_password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one number")
        return v

class RefreshRequest(BaseModel):
    refresh_token: str

# ---- Helpers ----
def generate_otp(length: int = 6) -> str:
    """Generate a random numeric OTP."""
    return ''.join(random.choices(string.digits, k=length))

def generate_refresh_token(length: int = 40) -> str:
    """Generate a long, secure refresh token."""
    return secrets.token_urlsafe(length)

def generate_reset_token(length: int = 8) -> str:
    """Generate a short alphanumeric reset token."""
    return secrets.token_urlsafe(length)[:length].upper()

# ---- Routes ----
@app.post("/api/v1/register", response_model=UserResponse)
@limiter.limit("5/minute")
def register_user(request: Request, user: UserRegister, db: Session = Depends(get_db)):
    # Check username uniqueness
    if db.query(Users).filter(Users.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    # Check email uniqueness
    if db.query(Users).filter(Users.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = Users(
        username=user.username,
        email=user.email,
        password_hash=hashed_password,
        is_verified=False
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Send verification OTP
    otp = generate_otp()
    verification = EmailVerification(
        user_id=new_user.id,
        otp_code=otp,
        expires_at=datetime.utcnow() + timedelta(minutes=15)
    )
    db.add(verification)
    db.commit()
    send_verification_email(new_user.email, otp, new_user.username)
    print(f"[Auth] Verification OTP sent to {new_user.email}")

    # Create token pair
    access_token = create_access_token(data={"sub": new_user.username})
    refresh_token = generate_refresh_token()
    
    # Store session
    session = UserSession(
        user_id=new_user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get("user-agent", "Unknown Device"),
        ip_address=request.client.host if request.client else "Unknown IP",
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(session)
    db.commit()

    return {
        "id": new_user.id,
        "username": new_user.username,
        "email": new_user.email,
        "is_verified": False,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@app.post("/api/v1/verify-email")
@limiter.limit("10/minute")
def verify_email(request: Request, body: VerifyEmailRequest, db: Session = Depends(get_db)):
    """Verify user email with OTP code."""
    user = db.query(Users).filter(Users.email == body.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_verified:
        return {"message": "Email already verified"}

    # Find the latest non-expired, unused OTP
    verification = (
        db.query(EmailVerification)
        .filter(
            EmailVerification.user_id == user.id,
            EmailVerification.otp_code == body.otp_code,
            EmailVerification.verified == False,
            EmailVerification.expires_at > datetime.utcnow()
        )
        .order_by(EmailVerification.created_at.desc())
        .first()
    )
    if not verification:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP code")

    # Mark as verified
    verification.verified = True
    user.is_verified = True
    db.commit()
    print(f"[Auth] Email verified for {user.email}")
    return {"message": "Email verified successfully"}


@app.post("/api/v1/resend-verification")
@limiter.limit("3/hour")
def resend_verification(request: Request, body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Resend verification OTP. Rate limited to 3 per hour."""
    user = db.query(Users).filter(Users.email == body.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_verified:
        return {"message": "Email already verified"}

    otp = generate_otp()
    verification = EmailVerification(
        user_id=user.id,
        otp_code=otp,
        expires_at=datetime.utcnow() + timedelta(minutes=15)
    )
    db.add(verification)
    db.commit()
    send_verification_email(user.email, otp, user.username)
    return {"message": "Verification code sent"}


@app.post("/api/v1/forgot-password")
@limiter.limit("3/hour")
def forgot_password(request: Request, body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Send password reset token via email."""
    user = db.query(Users).filter(Users.email == body.email).first()
    if not user:
        # Don't reveal if email exists — always return success
        return {"message": "If this email is registered, you will receive a reset code."}

    token = generate_reset_token()
    reset = PasswordReset(
        user_id=user.id,
        reset_token=token,
        expires_at=datetime.utcnow() + timedelta(minutes=30)
    )
    db.add(reset)
    db.commit()
    send_password_reset_email(user.email, token, user.username)
    print(f"[Auth] Password reset token sent to {user.email}")
    return {"message": "If this email is registered, you will receive a reset code."}


@app.post("/api/v1/reset-password")
@limiter.limit("5/minute")
def reset_password(request: Request, body: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Reset password using token from email."""
    user = db.query(Users).filter(Users.email == body.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Find valid reset token
    reset = (
        db.query(PasswordReset)
        .filter(
            PasswordReset.user_id == user.id,
            PasswordReset.reset_token == body.reset_token,
            PasswordReset.used == False,
            PasswordReset.expires_at > datetime.utcnow()
        )
        .order_by(PasswordReset.created_at.desc())
        .first()
    )
    if not reset:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # Update password
    user.password_hash = get_password_hash(body.new_password)
    reset.used = True
    db.commit()
    print(f"[Auth] Password reset successful for {user.email}")
    return {"message": "Password reset successful. You can now log in."}


@app.post("/api/v1/login", response_model=UserResponse)
@limiter.limit("10/minute")
def login_user(request: Request, user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(Users).filter(
        (Users.username == user.username) | (Users.email == user.username)
    ).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # Update last login timestamp
    db_user.last_login_at = datetime.utcnow()

    # Create token pair
    access_token = create_access_token(data={"sub": db_user.username})
    refresh_token = generate_refresh_token()
    
    # Store session
    session = UserSession(
        user_id=db_user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get("user-agent", "Unknown Device"),
        ip_address=request.client.host if request.client else "Unknown IP",
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(session)
    db.commit()

    return {
        "id": db_user.id,
        "username": db_user.username,
        "email": db_user.email,
        "is_verified": db_user.is_verified or False,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@app.post("/api/v1/refresh")
def refresh_token(request: Request, body: RefreshRequest, db: Session = Depends(get_db)):
    """Exchange a valid refresh token for a new access+refresh token pair."""
    # Find session
    session = db.query(UserSession).filter(
        UserSession.refresh_token == body.refresh_token,
        UserSession.revoked == False
    ).first()
    
    if not session or session.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    # Revoke old token (rotation)
    session.revoked = True
    
    # Get user
    user = session.user
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Generate new token pair
    access_token = create_access_token(data={"sub": user.username})
    new_refresh_token = generate_refresh_token()
    
    # Create new session
    new_session = UserSession(
        user_id=user.id,
        refresh_token=new_refresh_token,
        device_info=request.headers.get("user-agent", "Unknown Device"),
        ip_address=request.client.host if request.client else "Unknown IP",
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(new_session)
    db.commit()

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@app.post("/api/v1/logout")
def logout_user(request: Request, body: RefreshRequest, db: Session = Depends(get_db)):
    """Revoke the current refresh token."""
    session = db.query(UserSession).filter(
        UserSession.refresh_token == body.refresh_token
    ).first()
    if session:
        session.revoked = True
        db.commit()
    return {"message": "Logged out successfully"}


@app.get("/api/v1/sessions")
def get_sessions(current_user: Users = Depends(get_current_user), db: Session = Depends(get_db)):
    """List active sessions for the user."""
    sessions = db.query(UserSession).filter(
        UserSession.user_id == current_user.id,
        UserSession.revoked == False,
        UserSession.expires_at > datetime.utcnow()
    ).order_by(UserSession.created_at.desc()).all()
    
    return [
        {
            "id": s.id,
            "device_info": s.device_info,
            "ip_address": s.ip_address,
            "created_at": s.created_at,
            "expires_at": s.expires_at
        } for s in sessions
    ]


@app.delete("/api/v1/sessions/{session_id}")
def revoke_session(session_id: int, current_user: Users = Depends(get_current_user), db: Session = Depends(get_db)):
    """Revoke a specific session."""
    session = db.query(UserSession).filter(
        UserSession.id == session_id,
        UserSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.revoked = True
    db.commit()
    return {"message": "Session revoked"}

@app.get("/api/v1/history", response_model=List[HistoryItem])
def get_history(current_user: Users = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Return the 10 most recent scans for user_id.
    Compute status deterministically from authenticity_score to avoid mismatches.
    """
    scans = db.query(ScanHistory).filter(ScanHistory.user_id == current_user.id).order_by(ScanHistory.scanned_at.desc()).limit(10).all()
    results = []
    for s in scans:
        # Use the stored authenticity_score as the canonical value
        score = float(s.authenticity_score) if s.authenticity_score is not None else 0.0

        # Determine label deterministically (mirror compute_trust_score thresholds)
        # UNIFIED THRESHOLD: 75%
        if score >= 75:
            label = ScanStatus.AUTHENTIC.value
        elif score >= 50:
            label = ScanStatus.SUSPICIOUS.value
        elif score >= 25:
            label = "UNKNOWN"
        else:
            label = ScanStatus.FAKE.value

        product_name = "Unknown"
        if s.scanned_batch_number and s.scanned_batch_number != "UNKNOWN":
            med = db.query(ValidMedicine).filter(ValidMedicine.batch_number == s.scanned_batch_number).first()
            if med:
                product_name = med.brand_name

        results.append({
            "product": product_name if product_name != "Unknown" else f"Batch {s.scanned_batch_number}",
            "status": label,
            "date": s.scanned_at.strftime("%Y-%m-%d"),
            "score": round(score, 2)
        })
    return results


@app.post("/api/v1/scan")
@limiter.limit("20/minute") # Allow frequent scanning but prevent abuse
async def scan_medicine(
    request: Request,
    file: UploadFile = File(...),
    barcode: Optional[str] = Form(None),   # client may supply barcode separately
    current_user: Users = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Accepts multipart/form-data:
      - file: image file
      - barcode (optional): barcode string scanned on device (helps when barcode region is separate)
    """
    try:
        # Validate file size (max 5MB)
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 5MB)")
            
        # Validate extension
        filename = file.filename.lower()
        if not filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
             raise HTTPException(status_code=400, detail="Invalid file type. Only JPG, PNG, WEBP allowed.")

        # Save upload to temp file for debugging/logging (optional)
        file_location = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        # Read file bytes
        image_bytes = await file.read()
        print(f"[Scan] Received image: {filename}, size: {len(image_bytes)} bytes")
        
        with open(file_location, "wb+") as file_object:
            file_object.write(image_bytes)

        # 1) ML Inference via HuggingFace Space
        print(f"[Scan] Calling HF Space at {HF_SPACE_URL}/predict")
        ml_result = await call_ml_inference(image_bytes, filename)
        print(f"[Scan] ML result: {ml_result}")
        
        # Check if ML service returned an error
        if ml_result.get('label') == 'ERROR':
            reason = ml_result.get('reason', 'ML inference failed')
            raise HTTPException(status_code=503, detail=reason)
        
        final_status = ml_result['label'] # "AUTHENTIC" or "FAKE"
        confidence = ml_result['confidence']
        
        # 2) Trust Score & Reason
        # Simple mapping: 
        # If Authentic with high confidence -> High Score
        # If Authentic with low confidence -> Medium Score
        # If Fake -> Low Score
        
        score = int(confidence * 100)
        if final_status == 'FAKE':
            score = 100 - score # Invert confidence for FAKE (e.g. 99% confident it's fake = 1% trust)
        
        # Reason generation
        if final_status == 'AUTHENTIC':
            if confidence > 0.9:
                reason = "High confidence authentic packaging detected."
            else:
                reason = "Packaging looks authentic but with lower confidence."
        else:
            reason = "Potential counterfeit detected based on packaging analysis."

        product_name = "Unknown Product" # Placeholder, later can limit scope or use other metadata
        
        # Persist scan history
        scan = ScanHistory(
            user_id=current_user.id,
            scanned_batch_number="ML_SCAN" , # Legacy field
            authenticity_score=score,
            status=final_status,
            image_path=file_location,
            # Store full ML breakdown if possible, for now just status
        )
        
        db.add(scan)
        db.commit()

        return {
            "status": final_status,
            "label": final_status, 
            "score": score,
            "reason": reason,
            "product": product_name,
            "breakdown": ml_result
        }
        


    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        print(f"[Scan] UNEXPECTED ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
