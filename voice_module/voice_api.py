"""
AstroMind — Voice Diagnostic FastAPI Router
File: voice_module/voice_api.py

Mounted in app.py via:
    from voice_module.voice_api import router as voice_router
    app.include_router(voice_router, prefix="/voice", tags=["Voice Diagnostics"])
"""
### Use this to activate -> "uvicorn voice_module.voice_api:app --reload --port 8000"


import io
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from functools import lru_cache

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

router = APIRouter()

# ── Model path — update before deploying ────────────────────────
MODEL_PATH = Path(__file__).parent / "saved_wav2vec2_model"

TARGET_SR    = 16000
CHUNK_DUR    = 10           # seconds per inference chunk
MAX_CHUNKS   = 3           # ensemble over N chunks

# ── Lazy-loaded singleton (loaded once on first request) ─────────
@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run the notebook first.")
    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_PATH))
    model     = Wav2Vec2ForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.eval()
    return processor, model


# ── Response schema (frontend contract) ──────────────────────────
class VoiceDiagnosticResponse(BaseModel):
    prediction:       str    # "healthy" | "depressed"
    label_int:        int    # 0 | 1
    confidence:       float  # 0.0 – 1.0
    healthy_prob:     float
    depressed_prob:   float
    chunks_analyzed:  int
    risk_flag:        bool   # True if confidence > 0.70 and depressed
    model_version:    str


def _run_inference(waveform: np.ndarray) -> VoiceDiagnosticResponse:
    processor, model = get_model()
    chunk_len = TARGET_SR * CHUNK_DUR
    all_probs = []

    for i in range(MAX_CHUNKS):
        start = i * chunk_len
        if start >= len(waveform):
            break
        chunk = waveform[start : start + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)), "constant")

        inputs = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        all_probs.append(torch.softmax(logits, dim=-1).squeeze().tolist())

    if not all_probs:
        raise HTTPException(status_code=422, detail="Audio too short (< 1 s).")

    mean_probs  = np.mean(all_probs, axis=0)
    
    # Instead of picking the absolute maximum, we prioritize catching depression if confidence >= 35%.
    DEPRESSION_THRESHOLD = 0.35
    if mean_probs[1] >= DEPRESSION_THRESHOLD:
        pred_label = 1
    else:
        pred_label = 0
        
    confidence  = float(mean_probs[pred_label])

    return VoiceDiagnosticResponse(
        prediction      = "depressed" if pred_label == 1 else "healthy",
        label_int       = pred_label,
        confidence      = round(confidence, 4),
        healthy_prob    = round(float(mean_probs[0]), 4),
        depressed_prob  = round(float(mean_probs[1]), 4),
        chunks_analyzed = len(all_probs),
        risk_flag       = confidence > 0.70 and pred_label == 1,
        model_version   = "wav2vec2-base-astromind-v1",
    )


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/diagnose", response_model=VoiceDiagnosticResponse)
async def diagnose_voice(file: UploadFile = File(...)):
    """
    Upload a .wav audio clip. Returns a depression risk assessment.
    Accepts: audio/wav, audio/x-wav, audio/mpeg
    """
    allowed = {"audio/wav", "audio/x-wav", "audio/mpeg", "application/octet-stream"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    raw = await file.read()
    try:
        audio_buf = io.BytesIO(raw)
        waveform, sr = sf.read(audio_buf)
        if waveform.ndim > 1:              # stereo → mono
            waveform = waveform.mean(axis=1)
        if sr != TARGET_SR:
            waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read audio file: {e}")

    return _run_inference(waveform.astype(np.float32))


@router.get("/health")
async def health_check():
    """Liveness probe — returns model load status."""
    try:
        get_model()
        return {"status": "ok", "model": "loaded"}
    except RuntimeError as e:
        return {"status": "error", "detail": str(e)}

from fastapi import FastAPI
app = FastAPI(title="Voice Module Standalone Server")
app.include_router(router)