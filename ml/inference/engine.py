import cv2
import numpy as np
import re
import os
import logging
import traceback

# Suppress Paddle warnings
os.environ['FLAGS_minloglevel'] = '2'

from paddleocr import PaddleOCR
from thefuzz import fuzz

# Initialize PaddleOCR
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

class InferenceEngine:
    def __init__(self, model_path=None, debug_dir="ocr_debug"):
        print("\n--- INITIALIZING PADDLEOCR ENGINE (IMPROVED) ---")
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)

    def preprocess_image(self, image_bytes):
        """Simple, reliable preprocessing"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print("Failed to decode image")
                return None

            height, width = img.shape[:2]
            print(f"Original image size: {width}x{height}")

            # Resize if too large
            if width > 2000:
                scale = 2000 / width
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                print(f"Resized to: {img.shape[1]}x{img.shape[0]}")

            return img

        except Exception as e:
            print(f"Preprocessing error: {e}")
            traceback.print_exc()
            return None

    def preprocess_image_enhanced(self, image_bytes):
        """Enhanced preprocessing for small/faint text"""
        try:
            img = self.preprocess_image(image_bytes)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # CLAHE for contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Bilateral filter to preserve edges
            bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Convert back to BGR for consistency
            result = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
            
            print("Applied enhancement preprocessing")
            return result

        except Exception as e:
            print(f"Enhanced preprocessing error: {e}")
            traceback.print_exc()
            return None

    def preprocess_image_aggressive(self, image_bytes):
        """Aggressive preprocessing for very challenging images"""
        try:
            img = self.preprocess_image(image_bytes)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Upscale if small
            if gray.shape[1] < 800:
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            result = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
            
            print("Applied aggressive preprocessing")
            return result

        except Exception as e:
            print(f"Aggressive preprocessing error: {e}")
            traceback.print_exc()
            return None

    def extract_text(self, image_bytes, known_batches=[], debug_name="img"):
        """Extract text with fallback methods"""
        try:
            print(f"\n[SCAN] Starting OCR for {debug_name}")
            
            # Try original image first
            print("Attempt 1: Original image")
            img1 = self.preprocess_image(image_bytes)
            
            if img1 is not None:
                try:
                    print(f"  → Image shape: {img1.shape}, dtype: {img1.dtype}")
                    result1 = ocr_engine.ocr(img1)
                    # debug: save raw result
                    print("OCR raw output (attempt1):", repr(result1))
                    with open(os.path.join(self.debug_dir, f"{debug_name}_ocr_raw_attempt1.txt"), "w", encoding="utf-8") as f:
                        f.write(repr(result1))
                    print(f"  → Raw result type: {type(result1)}, length: {len(result1) if result1 else 0}")
                    texts1 = self._parse_ocr_result(result1)
                    print(f"  → Detected {len(texts1)} text segments")
                    
                    if len(texts1) > 5:  # If we got good results, use them
                        return self._extract_batch_from_texts(texts1, known_batches)
                except Exception as e:
                    print(f"  → Failed: {e}")

            # Try enhanced preprocessing
            print("Attempt 2: Enhanced preprocessing")
            img2 = self.preprocess_image_enhanced(image_bytes)
            
            if img2 is not None:
                try:
                    print(f"  → Image shape: {img2.shape}, dtype: {img2.dtype}")
                    result2 = ocr_engine.ocr(img2)
                    print("OCR raw output (attempt2):", repr(result2))
                    with open(os.path.join(self.debug_dir, f"{debug_name}_ocr_raw_attempt2.txt"), "w", encoding="utf-8") as f:
                        f.write(repr(result2))
                    print(f"  → Raw result type: {type(result2)}, length: {len(result2) if result2 else 0}")
                    texts2 = self._parse_ocr_result(result2)
                    print(f"  → Detected {len(texts2)} text segments")
                    
                    if len(texts2) > 0:
                        return self._extract_batch_from_texts(texts2, known_batches)
                except Exception as e:
                    print(f"  → Failed: {e}")

            # Try aggressive preprocessing
            print("Attempt 3: Aggressive preprocessing")
            img3 = self.preprocess_image_aggressive(image_bytes)
            
            if img3 is not None:
                try:
                    print(f"  → Image shape: {img3.shape}, dtype: {img3.dtype}")
                    result3 = ocr_engine.ocr(img3)
                    print("OCR raw output (attempt3):", repr(result3))
                    with open(os.path.join(self.debug_dir, f"{debug_name}_ocr_raw_attempt3.txt"), "w", encoding="utf-8") as f:
                        f.write(repr(result3))
                    print(f"  → Raw result type: {type(result3)}, length: {len(result3) if result3 else 0}")
                    texts3 = self._parse_ocr_result(result3)
                    print(f"  → Detected {len(texts3)} text segments")
                    
                    if len(texts3) > 0:
                        return self._extract_batch_from_texts(texts3, known_batches)
                except Exception as e:
                    print(f"  → Failed: {e}")

            print("All OCR attempts failed")
            return {"full_text": "OCR Error", "batch_number": "UNKNOWN"}

        except Exception as e:
            print(f"[ERROR] Unhandled exception: {e}")
            traceback.print_exc()
            return {"full_text": "Processing Error", "batch_number": "UNKNOWN"}

    def _parse_ocr_result(self, result):
        """Parse PaddleOCR result - handles new OCRResult object format"""
        detected_texts = []
        
        try:
            if result is None:
                return detected_texts
            
            # NEW FORMAT: result is a list containing OCRResult objects with rec_texts
            if isinstance(result, list) and len(result) > 0:
                page = result[0]
                
                # Check if it's the new OCRResult format with rec_texts
                if isinstance(page, dict) and 'rec_texts' in page:
                    print(f"  → Using new OCRResult format")
                    rec_texts = page.get('rec_texts', [])
                    rec_scores = page.get('rec_scores', [])
                    
                    # Filter by confidence threshold
                    for text, score in zip(rec_texts, rec_scores):
                        if score > 0.3 and text and text.strip():
                            detected_texts.append(text.strip())
                    
                    print(f"  → Extracted {len(detected_texts)} texts with confidence > 0.3")
                    return detected_texts
                
                # FALLBACK: Old format - list of line detections
                print(f"  → Using fallback format")
                if hasattr(page, '__iter__'):
                    for line in page:
                        if line is None or len(line) < 2:
                            continue
                        try:
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                                text = str(text_info[0]).strip()
                                if text:
                                    detected_texts.append(text)
                        except Exception:
                            continue
        
        except Exception as e:
            print(f"Parse error: {e}")
            traceback.print_exc()

        return detected_texts

    def _extract_batch_from_texts(self, detected_texts, known_batches):
        """Extract batch number from detected texts"""
        
        # Build the combined full text and cleaned version up-front
        full_text = " ".join(detected_texts).upper()
        clean_text = re.sub(r'[^A-Z0-9\s\.\-\/:]', '', full_text)

        # Define ignore tokens
        IGNORE_TOKENS = {'MRP', 'EXP', 'MFG', 'MFD', 'TABS', 'TABLETS', 'RS', 'INCL', 'ALL', 'TAXES', 'PER'}

        # First: neighbor-based check (improved)
        for i, text in enumerate(detected_texts):
            text_upper = text.upper().strip()
            # quick keyword check
            if re.search(r'\b(BATCH|B\.?NO\.?|BNO|B[\.\s]?NO)\b', text_upper) or 'BATCH' in text_upper:
                # check next 2 tokens for a candidate
                for j in (1, 2):
                    if i + j < len(detected_texts):
                        cand_raw = detected_texts[i + j].upper().strip()
                        # sanitize: keep alnum and hyphen only
                        cand = re.sub(r'[^A-Z0-9\-]', '', cand_raw)
                        # skip if cand is short or matches ignore tokens
                        if not cand or len(cand) < 3:
                            continue
                        if any(tok in cand for tok in IGNORE_TOKENS):
                            continue
                        # require at least two digits to qualify as batch
                        if re.search(r'\d{2,}', cand):
                            extracted_batch = cand.strip().strip(".-")
                            print(f"[Strategy 0 improved] Found neighbor batch: {extracted_batch}")
                            return {"full_text": clean_text, "batch_number": extracted_batch}

            # attempt combined token e.g. 'B.No.EA25049' or 'B.NO.25238MPUL'
            combined_match = re.search(r'\b(?:B|B\.|B\.?NO|BNO|BATCH)[\.\s:\/\-]*(.+)', text, flags=re.I)
            if combined_match:
                tail = combined_match.group(1)
                # sanitize: keep alnum and hyphen only
                candidate = re.sub(r'[^A-Z0-9\-]', '', tail.upper() if isinstance(tail, str) else str(tail).upper())
                # ignore pure words and tokens w/out digits
                if candidate and re.search(r'\d{2,}', candidate) and not any(tok in candidate for tok in IGNORE_TOKENS):
                    # if trailing month letters were glued, strip trailing alpha sequences after digits
                    candidate = re.sub(r'([0-9]+)[A-Z]+$', r'\1', candidate)
                    candidate = candidate.strip().strip(".-")
                    if len(candidate) >= 3:
                        print(f"[Strategy 0b improved] Found combined keyword+value: {candidate}")
                        return {"full_text": clean_text, "batch_number": candidate}
        
        # Debug print of all texts if we reach here
        print(f"\n--- OCR RESULTS ---")
        print(f"ALL DETECTED TEXTS:")
        for i, text in enumerate(detected_texts):
            print(f"  [{i}] {text}")
        print(f"Clean text (first 500 chars): {clean_text[:500]}")
        print(f"-------------------\n")

        # STRATEGY 1: Keyword Search (IMPROVED)
        match_keyword = re.search(
            r'(?:BATCH|B\.?[\s\.]*NO|BNO|B[\s\.]*NO|LOT)[\s\.:\-]*([A-Z0-9]{3,20})',
            clean_text,
            flags=re.I
        )
        if match_keyword:
            extracted_batch = match_keyword.group(1).strip().strip(".-")
            # small sanity: require 2 digits
            if re.search(r'\d{2,}', extracted_batch):
                print(f"[Strategy 1] Keyword match: {extracted_batch}")
                return {"full_text": clean_text, "batch_number": extracted_batch}

        # STRATEGY 2: Fuzzy match against known batches (if provided)
        if known_batches:
            best_match = None
            best_ratio = 0
            
            for db_batch in known_batches:
                clean_db = db_batch.upper().strip()
                if clean_db in clean_text:
                    print(f"[Strategy 2] Exact match: {db_batch}")
                    return {"full_text": clean_text, "batch_number": db_batch}
                try:
                    ratio = fuzz.partial_ratio(clean_db, clean_text)
                except Exception:
                    from difflib import SequenceMatcher
                    ratio = int(SequenceMatcher(None, clean_db, clean_text).ratio() * 100)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = db_batch
            
            if best_ratio > 80:
                print(f"[Strategy 2] Fuzzy match: {best_match} ({best_ratio}%)")
                return {"full_text": clean_text, "batch_number": best_match}

        # STRATEGY 3: Pattern matching
        patterns = [
            r'(?:^|\s)([A-Z]{2,4}[0-9]{3,6})(?:\s|$)',
            r'(?:^|\s)([0-9]{6,8})(?:\s|$)',
            r'(?:^|\s)([A-Z][0-9]{4,7})(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                extracted_batch = match.group(1)
                extracted_batch = extracted_batch.strip().strip(".-")
                # sanity: require at least two digits
                if re.search(r'\d{2,}', extracted_batch):
                    print(f"[Strategy 3] Pattern match: {extracted_batch}")
                    return {"full_text": clean_text, "batch_number": extracted_batch}

        print(f"[No match] Returning UNKNOWN")
        return {
            "full_text": clean_text if clean_text else "NO_TEXT_DETECTED",
            "batch_number": "UNKNOWN"
        }

    def compare_visuals(self, image_bytes, reference_vector_str):
        return 0.95
