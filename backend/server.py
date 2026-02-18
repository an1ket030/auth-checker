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

from ml.inference.engine import MLInferenceEngine

# Initialize ML Engine
# ML Engine initialized lazily below

# pyzbar removed due to deployment constraints (libzbar0 missing on Render Python env)

from .database import engine, get_db, Base
from .database import engine, get_db, Base
from .models import ValidMedicine, ScanHistory, ScanStatus, Users

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
ACCESS_TOKEN_EXPIRE_MINUTES = 600

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

# Add Rate Limiter Exception Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ml_engine initialized lazily
_ml_engine_instance = None

def get_ml_engine():
    global _ml_engine_instance
    if _ml_engine_instance is None:
        print("Lazy loading ML Engine...")
        _ml_engine_instance = MLInferenceEngine()
    return _ml_engine_instance

from pydantic import BaseModel, validator, EmailStr

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
        # Basic regex or use EmailStr if email-validator installed
        # Using regex for dependency-free robustness
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
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    access_token: str
    token_type: str

class HistoryItem(BaseModel):
    product: str
    status: str
    date: str
    score: float

# ---- Helpers ----
MONTH_MAP = {
    'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6,
    'JUL':7, 'AUG':8, 'SEP':9, 'SEPT':9, 'OCT':10, 'NOV':11, 'DEC':12
}



# --- barcode decode helper
# decode_barcode_from_bytes removed

# ---- Routes ----
@app.post("/api/v1/register", response_model=UserResponse)
@limiter.limit("5/minute") # Rate limit: 5 per minute per IP
def register_user(request: Request, user: UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(Users).filter(Users.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = Users(username=user.username, email=user.email, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": new_user.username})
    return {
        "id": new_user.id, 
        "username": new_user.username, 
        "email": new_user.email,
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/api/v1/login", response_model=UserResponse)
@limiter.limit("10/minute") # Slightly higher for login
def login_user(request: Request, user: UserLogin, db: Session = Depends(get_db)):
    # Try finding by username OR email (if user sends email as username)
    db_user = db.query(Users).filter((Users.username == user.username) | (Users.email == user.username)).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": db_user.username})
    return {
        "id": db_user.id, 
        "username": db_user.username,
        "email": db_user.email,
        "access_token": access_token, 
        "token_type": "bearer"
    }

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
        
        with open(file_location, "wb+") as file_object:
            file_object.write(image_bytes)

        # 1) ML Inference
        # Predict using the new ML Engine (Lazy Load)
        ml_engine = get_ml_engine()
        ml_result = ml_engine.predict(image_bytes)
        
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
            scan_image_path=file_location, # Changed from image_path to scan_image_path
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
        


    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
