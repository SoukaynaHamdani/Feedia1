"""Local MoulSot (atlasia/moulsot.v0.3) ASR server for Moroccan Darija.

Run from the hackai-main folder inside asr-venv:
    uvicorn backend.asr_server:app --port 8001
"""
import os
import tempfile

import torch
from fastapi import FastAPI, File, UploadFile
from qwen_asr import Qwen3ASRModel

app = FastAPI(title="MoulSot ASR")

print("Loading atlasia/moulsot.v0.3 (first run downloads ~4 GB)...")
asr = Qwen3ASRModel.from_pretrained(
    "atlasia/moulsot.v0.3",
    dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    max_inference_batch_size=4,
)
print("MoulSot ready.")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        path = tmp.name
    try:
        results = asr.transcribe(audio=path, language=None)
        return {"text": getattr(results[0], "text", "") or ""}
    finally:
        os.remove(path)


@app.get("/health")
def health():
    return {"status": "ok"}
