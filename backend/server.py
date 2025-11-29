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
from .models import ValidMedicine, ScanHistory, ScanStatus, Users

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.inference.engine import InferenceEngine  # your existing OCR wrapper

# Force table creation (models must match DB; migrations preferred)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AuthChecker Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_engine = InferenceEngine()

# ---- Pydantic models ----
class UserLogin(BaseModel):
    username: str
    email: str

class UserResponse(BaseModel):
    id: int
    username: str

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

    # Remove common prefixes like BATCH, B.NO, BNO etc.
    s = re.sub(r'^\s*(BATCH|B\.?NO\.?|BNO|B[\.\s]?NO[:\.\s-]*)\s*', '', s, flags=re.I)

    # Remove leading 'NO' artifact (if exact letters). Keep this AFTER the prefix removal above.
    s = re.sub(r'^\s*NO(?=[A-Z0-9])', '', s, flags=re.I)

    # Remove common non-alnum characters but keep hyphen and slash (some batches use them)
    s = re.sub(r'[^A-Z0-9\-\/]', '', s)

    # --- Heuristic corrections for OCR confusions ---
    # 1) Leading "N0" (N + zero) often should be "NO" (N + letter O)
    if s.startswith('N0'):
        s = 'NO' + s[2:]

    # 2) Leading zero followed by a letter (e.g. "0YMS..." which likely was "OYMS...") -> replace leading 0 -> O
    s = re.sub(r'^0(?=[A-Z])', 'O', s)

    # 3) Replace occurrences of 'N0' when followed by letters (covers glued forms like 'N0YMS2584')
    s = re.sub(r'N0(?=[A-Z])', 'NO', s)

    # Keep only A-Z,0-9 and - / now (again, defensive)
    s = re.sub(r'[^A-Z0-9\-\/]', '', s)

    # require at least 2 digits in the candidate (basic sanity)
    if not re.search(r'\d{2,}', s):
        return None

    return s if s else None


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

    # Adjusted weights: smaller penalty for missing packaging_sim
    weights = {
        'product_matched': 0.30,
        'batch_in_db': 0.35,
        'mfg_exp_valid': 0.12,
        'ocr_confidence': 0.08,
        'packaging_sim': 0.05,
        'manufacturer_verified': 0.05,
        'registry': 0.05
    }

    s_product = 1.0 if product_matched else 0.0
    s_batch = 1.0 if batch_in_db else 0.0
    s_dates = 1.0 if dates_ok else 0.0
    s_ocr = max(0.0, min(1.0, ocr_conf))
    s_pack = max(0.0, min(1.0, packaging_sim))
    s_mfg = 1.0 if manufacturer_verified else 0.0
    s_reg = 1.0 if registry else 0.0

    raw = (weights['product_matched']*s_product +
           weights['batch_in_db']*s_batch +
           weights['mfg_exp_valid']*s_dates +
           weights['ocr_confidence']*s_ocr +
           weights['packaging_sim']*s_pack +
           weights['manufacturer_verified']*s_mfg +
           weights['registry']*s_reg)
    max_w = sum(weights.values())
    normalized = (raw / max_w) * 100.0

    positive_count = sum([s_product>0, s_batch>0, s_dates>0, s_ocr>0.6, s_pack>0.6, s_mfg>0, s_reg>0])

    if normalized >= 80 and (s_product or s_batch or s_mfg or s_reg):
        label = ScanStatus.AUTHENTIC.value
    elif normalized >= 50 and positive_count >= 2:
        label = ScanStatus.SUSPICIOUS.value
    elif normalized >= 30 and positive_count >= 1:
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
@app.post("/api/v1/login", response_model=UserResponse)
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(Users).filter(Users.email == user.email).first()
    if not db_user:
        db_user = Users(username=user.username, email=user.email)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    return {"id": db_user.id, "username": db_user.username}

@app.get("/api/v1/history/{user_id}", response_model=List[HistoryItem])
def get_history(user_id: int, db: Session = Depends(get_db)):
    """
    Return the 10 most recent scans for user_id.
    Compute status deterministically from authenticity_score to avoid mismatches.
    """
    scans = db.query(ScanHistory).filter(ScanHistory.user_id == user_id).order_by(ScanHistory.scanned_at.desc()).limit(10).all()
    results = []
    for s in scans:
        # Use the stored authenticity_score as the canonical value
        score = float(s.authenticity_score) if s.authenticity_score is not None else 0.0

        # Determine label deterministically (mirror compute_trust_score thresholds)
        if score >= 80:
            label = ScanStatus.AUTHENTIC.value
        elif score >= 60:
            label = ScanStatus.AUTHENTIC.value   # keep 60+ as AUTHENTIC (adjust if you use 80+)
        elif score >= 50:
            label = ScanStatus.SUSPICIOUS.value
        elif score >= 30:
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
async def scan_medicine(
    user_id: int,
    file: UploadFile = File(...),
    barcode: Optional[str] = Form(None),   # client may supply barcode separately
    db: Session = Depends(get_db)
):
    """
    Accepts multipart/form-data:
      - file: image file
      - barcode (optional): barcode string scanned on device (helps when barcode region is separate)
    """
    try:
        # save uploaded image
        file_id = str(uuid.uuid4())
        file_ext = file.filename.split(".")[-1] if file.filename else "jpg"
        file_path = f"uploads/{file_id}.{file_ext}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(file_path, "rb") as img_file:
            image_bytes = img_file.read()

        # 1) OCR
        ocr_result = ml_engine.extract_text(image_bytes)
        detected_batch_raw = ocr_result.get("batch_number")
        full_text_raw = ocr_result.get("full_text", "") or ""
        print(f"[DEBUG] OCR detected_batch_raw={detected_batch_raw!r}")

        # Normalize batch
        normalized_batch = normalize_batch(detected_batch_raw)
        print(f"[DEBUG] normalized_batch={normalized_batch!r}")

        # default evidence
        evidence = {
            'product_matched': False,
            'batch_in_db': False,
            'mfg_exp_valid': False,
            'ocr_confidence': 0.0,
            'packaging_sim': 0.0,
            'manufacturer_verified': False,
            'pharma_registry_match': False
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
        reason = f"AI Read: '{full_text_raw[:160]}' | breakdown: {res['breakdown']}"
        product_name = golden_record.brand_name if golden_record else (best_brand or "Unknown")

        # Persist scan history
        scan = ScanHistory(
            user_id=user_id,
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
            "score": score,
            "reason": reason,
            "product": product_name
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
