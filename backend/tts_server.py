"""Local Darija TTS server (KandirResearch/DarijaTTS-v0.1-500M, OuteTTS GGUF).

Run from the hackai-main folder inside tts-venv:
    uvicorn backend.tts_server:app --port 8002
"""
import os
import re
import tempfile
import threading

import numpy as np
import outetts
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from outetts.models.config import GenerationConfig
from pydantic import BaseModel
from starlette.background import BackgroundTask

app = FastAPI(title="Darija TTS")

print("Loading DarijaTTS (first run downloads ~0.5 GB)...")
model_path = hf_hub_download(
    repo_id="KandirResearch/DarijaTTS-v0.1-500M",
    filename="unsloth.Q8_0.gguf",
)
interface = outetts.InterfaceGGUF(
    model_version="0.3",
    cfg=outetts.GGUFModelConfig_v2(
        model_path=model_path,
        tokenizer_path="KandirResearch/DarijaTTS-v0.1-500M",
    ),
)
lock = threading.Lock()  # llama.cpp context is not thread-safe
print("DarijaTTS ready.")


class TTSRequest(BaseModel):
    text: str
    temperature: float = 0.3
    repetition_penalty: float = 1.1


MAX_CHUNK_CHARS = 90
MIN_SEC_PER_CHAR = 0.04   # shorter than this = the model stopped early
MAX_SEC_PER_CHAR = 0.25   # longer than this = runaway / babbling
MAX_ATTEMPTS = 5


def split_text(text: str) -> list[str]:
    """Split into short sentence-like chunks; DarijaTTS is unreliable on long inputs."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?؟،,;:؛\n])\s*", text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > MAX_CHUNK_CHARS:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def synth_chunk(text: str, temperature: float, rep_pen: float):
    """Generate one chunk, retrying until its duration is plausible for its length."""
    best, best_score = None, None
    for attempt in range(MAX_ATTEMPTS):
        # raise temperature slightly on retries to escape a bad sample
        temp = temperature + 0.1 * attempt
        output = interface.generate(
            config=GenerationConfig(
                text=text,
                temperature=temp,
                repetition_penalty=rep_pen,
                max_length=4096,
                speaker=None,
            )
        )
        audio = output.audio.squeeze().cpu().numpy()
        sr = output.sr
        per_char = (len(audio) / sr) / max(len(text), 1)
        if MIN_SEC_PER_CHAR <= per_char <= MAX_SEC_PER_CHAR:
            return audio, sr
        # keep the closest-to-plausible attempt as a fallback
        score = abs(per_char - 0.08)
        if best is None or score < best_score:
            best, best_score = (audio, sr), score
        print(f"[tts] retry {attempt + 1}: {per_char:.3f}s/char for {text!r}")
    return best


@app.post("/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.close()
    try:
        pieces, sr = [], None
        with lock:
            for chunk in split_text(text):
                audio, sr = synth_chunk(chunk, req.temperature, req.repetition_penalty)
                pieces.append(audio)
                pieces.append(np.zeros(int(sr * 0.18), dtype=audio.dtype))
        # torchaudio>=2.9 needs torchcodec to save; write with soundfile instead
        sf.write(out.name, np.concatenate(pieces), sr)
    except Exception as e:
        os.remove(out.name)
        raise HTTPException(status_code=500, detail=str(e))
    return FileResponse(
        out.name, media_type="audio/wav",
        background=BackgroundTask(os.remove, out.name),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
