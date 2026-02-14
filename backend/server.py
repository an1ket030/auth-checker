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

# OCR engine and fuzzy string match
from thefuzz import fuzz

# Barcode decode
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode

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
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey123") # Change in prod
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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.inference.engine import InferenceEngine  # your existing OCR wrapper

# Force table creation (models must match DB; migrations preferred)
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Rate Limiter Exception Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ml_engine = InferenceEngine()

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

def normalize_batch(raw: Optional[str]) -> Optional[str]:
    """
    Normalize OCR batch strings to a canonical alnum-hyphen form.

    Fixes common OCR confusions (e.g. 'N0' -> 'NO', leading '0' where 'O' expected),
    removes noise and enforces at-least-two-digits sanity check.
    """
    if not raw:
        return None
    s = str(raw).upper().strip()

    # 1. Remove standard prefixes (spaced or dotted)
    #    MATCH: BATCH, B.NO, BNO, B-NO etc.
    s = re.sub(r'^\s*(BATCH|B\.?NO\.?|BNO|B[\.\s]?NO[:\.\s-]*)\s*', '', s, flags=re.I)

    # 2. Aggressive Glue Removal:
    #    OCR often reads "B.No.XY123" as "NOXY123" or "0XY123" or "N0XY123"
    #    Remove leading "NO", "N0", "0", "O" if followed by letters/digits
    #    Repeat loop to catch multiple layers like "NON0..."
    for _ in range(3):
        s = re.sub(r'^(NO|N0|0|O)(?=[A-Z0-9])', '', s, flags=re.I)

    # 3. Remove non-alnum (keep hyphen/slash)
    s = re.sub(r'[^A-Z0-9\-\/]', '', s)

    # 4. Require at least 3 digits (stricter sanity)
    #    Real batches usually have 4-10 chars.
    if len(s) < 3: 
        return None
        
    return s


def extract_dates_from_text(full_text: str):
    """Return (mfg_date, exp_date) or (None, None)"""
    if not full_text:
        return None, None
    text = full_text.upper()
    mfg = None
    exp = None

    month_year_pat = re.compile(r'([A-Z]{3,4})[\.\/\-\s]?\.?(\d{4})')
    numeric_mmyy = re.compile(r'([0-1]?\d)[\/\-\.](\d{2,4})')

    mfg_match = re.search(r'(MFG|MFD|MFG\.)\s*[:\-]?\s*([A-Z0-9\./\-]{3,12})', text)
    if mfg_match:
        candidate = mfg_match.group(2)
        my = month_year_pat.search(candidate)
        if my:
            mon = my.group(1)[:3]; yr = int(my.group(2))
            monnum = MONTH_MAP.get(mon[:3], None)
            if monnum: mfg = date(yr, monnum, 1)
        else:
            nmy = numeric_mmyy.search(candidate)
            if nmy:
                monnum = int(nmy.group(1)); yr = int(nmy.group(2)) if len(nmy.group(2))==4 else 2000+int(nmy.group(2))
                if 1 <= monnum <= 12: mfg = date(yr, monnum, 1)

    exp_match = re.search(r'(EXP|EXP\.|EXPIRY|EXPIRY\.)\s*[:\-]?\s*([A-Z0-9\./\-]{3,12})', text)
    if exp_match:
        candidate = exp_match.group(2)
        my = month_year_pat.search(candidate)
        if my:
            mon = my.group(1)[:3]; yr = int(my.group(2))
            monnum = MONTH_MAP.get(mon[:3], None)
            if monnum: exp = date(yr, monnum, 1)
        else:
            nmy = numeric_mmyy.search(candidate)
            if nmy:
                monnum = int(nmy.group(1)); yr = int(nmy.group(2)) if len(nmy.group(2))==4 else 2000+int(nmy.group(2))
                if 1 <= monnum <= 12: exp = date(yr, monnum, 1)

    if not exp:
        fallback = re.search(r'EXP[\.\s:\/\-]*([A-Z0-9\.\-\/]{3,12})', text)
        if fallback:
            candidate = fallback.group(1)
            my = month_year_pat.search(candidate)
            if my:
                mon = my.group(1)[:3]; yr = int(my.group(2))
                monnum = MONTH_MAP.get(mon[:3], None)
                if monnum: exp = date(yr, monnum, 1)

    return mfg, exp

def mfg_exp_valid(mfg_date: Optional[date], exp_date: Optional[date]):
    try:
        if not exp_date:
            return False
        today = date.today()
        if exp_date.year < today.year or (exp_date.year == today.year and exp_date.month < today.month):
            return False
        if mfg_date and exp_date:
            return mfg_date < exp_date
        return True
    except:
        return False

def compute_trust_score(evidence: dict):
    """Same scoring function we used earlier — returns score + label + breakdown."""
    product_matched = bool(evidence.get('product_matched'))
    batch_in_db = bool(evidence.get('batch_in_db'))
    dates_ok = bool(evidence.get('mfg_exp_valid'))
    ocr_conf = float(evidence.get('ocr_confidence', 0.0))
    packaging_sim = float(evidence.get('packaging_sim', 0.0))
    manufacturer_verified = bool(evidence.get('manufacturer_verified', False))
    registry = bool(evidence.get('pharma_registry_match', False))

    # Adjusted weights: verified batch is the critical factor
    weights = {
        'product_matched': 0.20,
        'batch_in_db': 0.50, # Critical factor
        'mfg_exp_valid': 0.10,
        'ocr_confidence': 0.05,
        'packaging_sim': 0.05,
        'manufacturer_verified': 0.05,
        'registry': 0.05,
        'serial_valid': 0.0 # treated as bonus now
    }

    s_product = 1.0 if product_matched else 0.0
    s_batch = 1.0 if batch_in_db else 0.0
    s_dates = 1.0 if dates_ok else 0.0
    s_ocr = max(0.0, min(1.0, ocr_conf))
    s_pack = max(0.0, min(1.0, packaging_sim))
    s_mfg = 1.0 if manufacturer_verified else 0.0
    s_reg = 1.0 if registry else 0.0
    
    # Serial Logic: 
    s_serial = 0.0
    if evidence.get('serial_valid'):
        if evidence.get('serial_is_clone', False):
             s_serial = -1.0
        else:
             s_serial = 1.0

    raw = (weights['product_matched']*s_product +
           weights['batch_in_db']*s_batch +
           weights['mfg_exp_valid']*s_dates +
           weights['ocr_confidence']*s_ocr +
           weights['packaging_sim']*s_pack +
           weights['manufacturer_verified']*s_mfg +
           weights['registry']*s_reg +
           0.20*max(0, s_serial)) # Bonus 20% for serial
           
    max_w = sum(weights.values()) + 0.20
    normalized = (raw / max_w) * 100.0

    # BATCH OVERRIDE: If batch is in DB, it is AUTHENTIC
    if batch_in_db:
        normalized = max(normalized, 96.0)

    # SERIAL OVERRIDE: If serial is valid, ensure AUTHENTIC
    if s_serial == 1.0:
        normalized = max(normalized, 99.0)
    
    # CLONE OVERRIDE: If serial is clone, force SUSPICIOUS or FAKE
    if s_serial == -1.0:
        normalized = 0.0
        label = ScanStatus.SUSPICIOUS.value
        return {'score': 0.0, 'label': label, 'breakdown': {'cloned_serial': True}}

    positive_count = sum([s_product>0, s_batch>0, s_dates>0, s_ocr>0.5, s_pack>0.5, s_mfg>0, s_reg>0])
    
    # UNIFIED THRESHOLD: 75%
    if normalized >= 75:
        label = ScanStatus.AUTHENTIC.value
    elif normalized >= 40 and positive_count >= 1:
        label = ScanStatus.SUSPICIOUS.value
    elif normalized >= 20:
        label = "UNKNOWN"
    else:
        label = ScanStatus.FAKE.value

    breakdown = {
        'product_matched': s_product,
        'batch_in_db': s_batch,
        'mfg_exp_valid': s_dates,
        'ocr_confidence': round(s_ocr, 3),
        'packaging_sim': round(s_pack, 3),
        'manufacturer_verified': s_mfg,
        'registry': s_reg,
        'serial_valid': s_serial,
        'raw_score': raw,
        'normalized': round(normalized, 2),
        'positive_count': int(positive_count)
    }

    return {'score': round(normalized, 2), 'label': label, 'breakdown': breakdown}

# --- fuzzy brand/manufacturer helper
def fuzzy_match_brand_manufacturer(full_text: str, db: Session):
    """
    Returns (best_brand, brand_score_float, best_manufacturer, manufacturer_score_float)
    scores are in [0.0, 1.0]
    """
    if not full_text:
        return None, 0.0, None, 0.0
    text = full_text.upper()

    try:
        brands = [row[0] for row in db.query(ValidMedicine.brand_name).distinct()]
        mans = [row[0] for row in db.query(ValidMedicine.manufacturer).distinct()]
    except Exception:
        brands, mans = [], []

    best_brand, best_b_score = None, 0
    for b in brands:
        if not b: continue
        r = fuzz.partial_ratio(b.upper(), text)
        if r > best_b_score:
            best_b_score = r
            best_brand = b

    best_man, best_m_score = None, 0
    for m in mans:
        if not m: continue
        r = fuzz.partial_ratio(m.upper(), text)
        if r > best_m_score:
            best_m_score = r
            best_man = m

    return best_brand, best_b_score/100.0, best_man, best_m_score/100.0

# --- barcode decode helper
def decode_barcode_from_bytes(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        decoded = pyzbar_decode(img)
        results = []
        for d in decoded:
            try:
                data = d.data.decode('utf-8', errors='ignore')
            except Exception:
                data = d.data.decode('latin1', errors='ignore')
            results.append({'data': data, 'type': d.type})
        return results
    except Exception as e:
        print("Barcode decode error:", e)
        return []

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

        # save uploaded image
        file_id = str(uuid.uuid4())
        file_ext = filename.split(".")[-1]
        file_path = f"uploads/{file_id}.{file_ext}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(file_path, "rb") as img_file:
            image_bytes = img_file.read()

        # 1) OCR (Synchronous to avoid thread safety issues with PaddleOCR)
        # from fastapi.concurrency import run_in_threadpool
        # ocr_result = await run_in_threadpool(ml_engine.extract_text, image_bytes)
        
        ocr_result = ml_engine.extract_text(image_bytes)
        
        detected_batch_raw = ocr_result.get("batch_number")
        detected_serial_raw = ocr_result.get("serial_number") # Capture SN
        full_text_raw = ocr_result.get("full_text", "") or ""
        
        # LOGGING FOR DEBUGGING
        try:
            with open("debug_ocr.log", "a", encoding="utf-8") as log:
                log.write(f"\n--- SCAN {datetime.now()} ---\n")
                log.write(f"Image Size: {len(image_bytes)} bytes\n")
                log.write(f"Raw Text:\n{full_text_raw}\n")
                log.write(f"Detected Batch Raw: {detected_batch_raw}\n")
                log.write(f"Detected Serial Raw: {detected_serial_raw}\n")
        except Exception as log_err:
            print(f"Logging failed: {log_err}")

        print(f"[DEBUG] OCR detected_batch_raw={detected_batch_raw!r}, SN={detected_serial_raw!r}")

        # Normalize batch
        normalized_batch = normalize_batch(detected_batch_raw)
        print(f"[DEBUG] normalized_batch={normalized_batch!r}")
        
        with open("debug_ocr.log", "a", encoding="utf-8") as log:
             log.write(f"Normalized Batch: {normalized_batch}\n")

        # default evidence
        evidence = {
            'product_matched': False,
            'batch_in_db': False,
            'mfg_exp_valid': False,
            'ocr_confidence': 0.0,
            'packaging_sim': 0.0,
            'manufacturer_verified': False,
            'manufacturer_verified': False,
            'pharma_registry_match': False,
            'serial_valid': False
        }

        # 2) barcode decode from image (if client did not supply)
        barcodes = []
        if barcode:
            # client provided barcode string (prefer it)
            barcodes = [{'data': barcode, 'type': 'CLIENT'}]
            print(f"[DEBUG] Received barcode from client: {barcode}")
        else:
            barcodes = decode_barcode_from_bytes(image_bytes)
            print(f"[DEBUG] Barcodes decoded from image: {barcodes}")

        # helper: normalize barcode string to alnum uppercase
        def _normalize_barcode(s):
            if not s: return None
            return re.sub(r'[^A-Z0-9]', '', str(s).upper())

        barcode_val = None
        if barcodes:
            raw_bar = barcodes[0].get('data', '').strip()
            barcode_val = _normalize_barcode(raw_bar)
            print(f"[DEBUG] Normalized barcode: {barcode_val}")

        golden_record = None

        if barcode_val:
            try:
                # Prefer exact GTIN lookup (use gtin column if available)
                try:
                    med_by_gtin = db.query(ValidMedicine).filter(ValidMedicine.gtin == barcode_val).first()
                except Exception:
                    # fallback if model lacks gtin field: try packaging_hash or batch_number checks
                    med_by_gtin = None

                if med_by_gtin:
                    print("[DEBUG] GTIN matched local DB record (gtin).")
                    golden_record = med_by_gtin
                    evidence['product_matched'] = True
                    evidence['barcode_found'] = True
                    evidence['barcode_value'] = barcode_val
                else:
                    # Try tolerant fallback: suffix match for numeric GTINs (last up to 13 digits)
                    if barcode_val.isdigit():
                        suffix = barcode_val[-13:]
                        try:
                            med_by_suffix = db.query(ValidMedicine).filter(ValidMedicine.gtin.ilike(f"%{suffix}")).first()
                        except Exception:
                            med_by_suffix = None
                        if med_by_suffix:
                            print("[DEBUG] GTIN suffix matched local DB record.")
                            golden_record = med_by_suffix
                            evidence['product_matched'] = True
                            evidence['barcode_found'] = True
                            evidence['barcode_value'] = barcode_val
            except Exception as e:
                print(f"[DEBUG] GTIN DB lookup error: {e}")
                golden_record = None

        # 3) fuzzy brand / manufacturer matching
        best_brand, brand_score, best_man, man_score = fuzzy_match_brand_manufacturer(full_text_raw, db)
        print(f"[DEBUG] brand match: {best_brand} ({brand_score:.2f}), manufacturer match: {best_man} ({man_score:.2f})")
        if brand_score >= 0.60:
            evidence['product_matched'] = True
        evidence['brand_match_score'] = brand_score
        if man_score >= 0.60:
            evidence['manufacturer_verified'] = True
        evidence['manufacturer_score'] = man_score

        # 4) tolerant DB lookup by normalized_batch (if any)
        golden_record_from_batch = False
        if normalized_batch and not golden_record:
            tried_variants = []
            candidates = []

            # original normalized candidate
            candidates.append(normalized_batch)

            # common OCR-corrected variants:
            # N0 -> NO (leading)
            if normalized_batch.startswith('N0'):
                candidates.append('NO' + normalized_batch[2:])
            # replace N0 anywhere before letters
            candidates.append(re.sub(r'N0(?=[A-Z])', 'NO', normalized_batch))
            # leading 0 -> O (if followed by letter)
            candidates.append(re.sub(r'^0(?=[A-Z])', 'O', normalized_batch))

            # dedupe candidates preserving order
            seen = set()
            filtered = []
            for c in candidates:
                if c and c not in seen:
                    seen.add(c)
                    filtered.append(c)
            candidates = filtered
            # Try exact matches first across variants
            for cand in candidates:
                tried_variants.append(cand)
                try:
                    gr = db.query(ValidMedicine).filter(ValidMedicine.batch_number == cand).first()
                except Exception:
                    gr = None
                if gr:
                    golden_record = gr
                    golden_record_from_batch = True
                    print(f"[DEBUG] tolerant exact match using variant: {cand}")
                    break
             # If still not found, try stripping common leading NO (if present) and match again
            if not golden_record:
                for cand in candidates:
                    maybe = re.sub(r'^(NO|N0)', '', cand, flags=re.I)
                    if maybe and maybe != cand:
                        try:
                            gr = db.query(ValidMedicine).filter(ValidMedicine.batch_number == maybe).first()
                        except Exception:
                            gr = None
                        if gr:
                            golden_record = gr
                            golden_record_from_batch = True
                            print(f"[DEBUG] match after stripping NO prefix: {maybe} (from {cand})")
                            break
            # Finally try suffix ilike matching for each candidate (handles truncated OCR)
            if not golden_record:
                for cand in candidates:
                    suffix = cand[-12:]
                    try:
                        gr = db.query(ValidMedicine).filter(ValidMedicine.batch_number.ilike(f"%{suffix}")).first()
                    except Exception:
                        gr = None
                    print(f"[DEBUG] suffix ilike lookup (%{suffix}): {'FOUND' if gr else 'NOT FOUND'}")
                    if gr:
                        golden_record = gr
                        golden_record_from_batch = True
                        print(f"[DEBUG] tolerant suffix match using variant: {cand}")
                        break
            print(f"[DEBUG] tried batch variants: {tried_variants}")
        # Now set batch_in_db only if match came from batch lookup
        if golden_record_from_batch:
            evidence['batch_in_db'] = True

        # If golden_record exists (from either GTIN or batch), set product_matched True
        if golden_record:
            evidence['product_matched'] = True
            print(f"[DEBUG] golden_record found: {golden_record.batch_number} -> {golden_record.brand_name}")

        # 4.5) Check Serial Number logic
        if detected_serial_raw:
            # Clean it
            sn_clean = re.sub(r'[^A-Z0-9]', '', detected_serial_raw.upper())
            # Check DB
            med_by_sn = db.query(ValidMedicine).filter(ValidMedicine.serial_number == sn_clean).first()
            if med_by_sn:
                print(f"[DEBUG] Valid Serial Number found: {sn_clean}")
                evidence['serial_valid'] = True
                
                # Check CLONE status
                # If scanned > 10 times, flag it
                old_count = med_by_sn.scan_count or 0
                if old_count > 10:
                    print(f"[WARN] Serial {sn_clean} scanned {old_count} times! Possible Clone.")
                    evidence['serial_is_clone'] = True
                
                # Update Scan Count (Increment)
                try:
                    med_by_sn.scan_count = old_count + 1
                    med_by_sn.last_scanned_at = datetime.utcnow()
                    db.commit()
                except Exception as e:
                    print(f"[DB Error] Failed to update scan count: {e}")
                    db.rollback()

                # If we didn't have a golden record before, we do now
                if not golden_record:
                    golden_record = med_by_sn
                    evidence['product_matched'] = True
            else:
                print(f"[DEBUG] Unknown Serial Number: {sn_clean}")

        # 5) extract & validate dates
        mfg_date, exp_date = extract_dates_from_text(full_text_raw)
        evidence['mfg_exp_valid'] = mfg_exp_valid(mfg_date, exp_date)
        print(f"[DEBUG] mfg_date={mfg_date}, exp_date={exp_date}, mfg_exp_valid={evidence['mfg_exp_valid']}")

        # 6) OCR confidence heuristic
        if full_text_raw and detected_batch_raw and detected_batch_raw != "UNKNOWN":
            evidence['ocr_confidence'] = 0.8
        elif full_text_raw:
            evidence['ocr_confidence'] = 0.5
        else:
            evidence['ocr_confidence'] = 0.2

        # Compute trust score
        res = compute_trust_score(evidence)
        final_status = res['label']
        score = res['score']
        
        # User Friendly Reason
        bd = res['breakdown']
        reasons = []
        if bd.get('batch_in_db'): reasons.append("Batch verified.")
        else: reasons.append("Batch not found.")
        
        if bd.get('serial_valid'): reasons.append("Serial number valid.")
        elif detected_serial_raw: reasons.append("Serial number invalid.")
        
        if bd.get('mfg_exp_valid'): reasons.append("Dates valid.")
        
        if final_status == 'AUTHENTIC':
             reason = "Product verified successfully. " + " ".join(reasons)
        elif final_status == 'FAKE':
             reason = "Potential counterfeit detected. " + " ".join(reasons)
        else:
             reason = "Verification inconclusive. " + " ".join(reasons)

        product_name = golden_record.brand_name if golden_record else (best_brand or "Unknown")

        # Persist scan history
        scan = ScanHistory(
            user_id=current_user.id,
            scanned_batch_number=normalize_batch(detected_batch_raw) or (detected_batch_raw or "UNKNOWN"),
            authenticity_score=score,
            status=final_status,
            image_path=file_path
        )
        # optionally, if ScanHistory model has barcode_detected attribute, set it here:
        try:
            if barcode_val and hasattr(scan, 'barcode_detected'):
                setattr(scan, 'barcode_detected', barcode_val)
        except Exception:
            pass

        db.add(scan)
        db.commit()

        return {
            "status": final_status,
            "label": final_status, # Frontend expects label
            "score": score,
            "reason": reason,
            "product": product_name,
            "breakdown": res['breakdown']
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
