# hf_space/app.py — FastAPI ML Inference for HuggingFace Spaces
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference_engine import MLInferenceEngine

app = FastAPI(title="AuthChecker ML Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML engine (persists while the Space is awake)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        print("[HF Space] Loading ML Engine...")
        start = time.time()
        _engine = MLInferenceEngine()
        print(f"[HF Space] ML Engine loaded in {time.time() - start:.2f}s")
    return _engine


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Keep-alive endpoint for UptimeRobot / cron pings."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    filename = file.filename.lower() if file.filename else ""
    if not filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only JPG, PNG, WEBP allowed."
        )

    # Read image bytes
    image_bytes = await file.read()

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    # Run inference
    engine = get_engine()
    start = time.time()
    result = engine.predict(image_bytes)
    inference_time = time.time() - start
    print(f"[HF Space] Inference took {inference_time:.3f}s -> {result}")

    if result.get("label") == "ERROR":
        raise HTTPException(
            status_code=500, detail=result.get("reason", "Inference failed")
        )

    return result
