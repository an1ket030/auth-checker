import sys
import os
import cv2
import numpy as np

# Ensure backend can be imported
sys.path.append(os.getcwd())

try:
    print("Importing inference engine...")
    from ml.inference.engine import InferenceEngine
    
    print("Initializing engine...")
    engine = InferenceEngine()
    
    print("Creating dummy image...")
    # Create a white image with text "BATCH EA25049"
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "BATCH EA25049", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    success, encoded_img = cv2.imencode('.jpg', img)
    image_bytes = encoded_img.tobytes()
    
    print("Running OCR...")
    result = engine.extract_text(image_bytes, known_batches=["EA25049"])
    print("Result:", result)

except Exception as e:
    print("CRITICAL ERROR:", e)
    import traceback
    traceback.print_exc()
