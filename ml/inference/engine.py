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
# Initialize PaddleOCR with safe defaults
# disable mkldnn to avoid fused_conv2d errors on some windows machines
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, enable_mkldnn=False)

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
        """
        ENSEMBLE OCR STRATEGY:
        Run BOTH Tesseract and PaddleOCR (Safe Mode).
        Combine results and check all candidates.
        """
        try:
            print(f"\n[SCAN] Starting ENSEMBLE OCR for {debug_name}")
            
            detected_texts = []
            full_text_combined = ""

            # 1. Preprocess Image
            img = self.preprocess_image(image_bytes)
            if img is None:
                 return {"full_text": "Image Error", "batch_number": "UNKNOWN"}

            # --- ENGINE A: TESSERACT ---
            print("--- Running Engine A: Tesseract ---")
            try:
                import pytesseract
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Use a reliable config
                text_tess = pytesseract.image_to_string(gray, config=r'--oem 3 --psm 6')
                if text_tess.strip():
                    print(f"  → Tesseract found: {len(text_tess)} chars")
                    full_text_combined += text_tess + "\n"
                    detected_texts.extend([l.strip() for l in text_tess.split('\n') if len(l.strip())>2])
            except Exception as e:
                print(f"  → Tesseract failed: {e}")

            # --- ENGINE B: PADDLEOCR (Safe Mode) ---
            print("--- Running Engine B: PaddleOCR (Safe Mode) ---")
            try:
                # Paddle expect BGR image
                result_paddle = ocr_engine.ocr(img)
                texts_paddle = self._parse_ocr_result(result_paddle)
                if texts_paddle:
                     print(f"  → Paddle found: {len(texts_paddle)} segments")
                     detected_texts.extend(texts_paddle)
                     full_text_combined += " ".join(texts_paddle) + "\n"
            except Exception as e:
                print(f"  → Paddle failed: {e}")

            # 3. Deduplicate and Extract
            detected_texts = list(set(detected_texts))
            print(f"  → Total combined unique lines: {len(detected_texts)}")
            
            if detected_texts:
                result = self._extract_batch_from_texts(detected_texts, known_batches)
                result["full_text"] = full_text_combined
                return result

            return {"full_text": full_text_combined or "OCR Error", "batch_number": "UNKNOWN"}

        except Exception as e:
            err_msg = f"[ERROR] Unhandled exception: {e}\n{traceback.format_exc()}"
            print(err_msg)
            try:
                with open("debug_ocr.log", "a", encoding="utf-8") as log:
                    log.write(f"\nCRITICAL OCR ERROR:\n{err_msg}\n")
            except:
                pass
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
            combined_match = re.search(r'\b(?:B|B\.|B\.?NO|BNO|BATCH)[\.\s:\/\-]*([^\s]+)', text, flags=re.I)
            if combined_match:
                tail = combined_match.group(1)
                
                # STOP at common keywords if they were glued (e.g. EA25049MFD)
                stop_pattern = re.search(r'(MFD|MFG|EXP|MRP|DAT|DT|\*)', tail, flags=re.I)
                if stop_pattern:
                    tail = tail[:stop_pattern.start()]

                # sanitize: keep alnum and hyphen only
                candidate = re.sub(r'[^A-Z0-9\-]', '', tail.upper() if isinstance(tail, str) else str(tail).upper())
                
                # STRIP leading "NO", "N0", "0" from the candidate itself
                # This handles cases where regex captured "NOEA25..."
                candidate = re.sub(r'^(NO|N0|0|O)+', '', candidate)

                # ignore pure words and tokens w/out digits
                if candidate and re.search(r'\d', candidate) and not any(tok in candidate for tok in IGNORE_TOKENS):
                    # if trailing month letters were glued, strip trailing alpha sequences after digits
                    candidate = re.sub(r'([0-9]+)[A-Z]+$', r'\1', candidate)
                    candidate = candidate.strip().strip(".-")
                    
                    # Sanity check length (Batches usually 4-12 chars)
                    if 3 <= len(candidate) <= 15:
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

        # STRATEGY 3: Pattern matching (Stricter)
        patterns = [
            # Require at least 2 letters, 4+ digits, and avoid common words
            # e.g. AB123456, but not "TABLET" (which matches [A-Z]{2,}[0-9] in some loose regex)
            r'(?:^|\s)([A-Z]{2,4}[0-9]{4,8})(?:\s|$)', 
            # Pure numeric batch (rare but possible, usually long)
            r'(?:^|\s)([0-9]{7,10})(?:\s|$)',
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

        # STRATEGY 4: Serial Number Extraction
        # Look for SN, S/N, Serial, etc.
        match_sn = re.search(
            r'(?:SN|S\.?N|SERIAL|UID)[\s\.:\-]*([A-Z0-9]{3,20})',
            clean_text,
            flags=re.I
        )
        extracted_sn = None
        if match_sn:
            extracted_sn = match_sn.group(1).strip().strip(".-")
            print(f"[Strategy 4] Serial match: {extracted_sn}")

        print(f"[No match] Returning UNKNOWN")
        return {
            "full_text": clean_text if clean_text else "NO_TEXT_DETECTED",
            "batch_number": "UNKNOWN",
        }

    def compare_visuals(self, image_bytes, reference_vector_str):
        return 0.95
