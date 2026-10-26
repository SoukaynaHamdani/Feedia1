"""Voice helpers: language detection, multilingual ASR routing and TTS (Darija / French / English)."""
import os
import re
import tempfile
from pathlib import Path

import edge_tts
import requests

# Neural voices (Microsoft Edge TTS, no GPU, works on any host). ar-MA is Moroccan Arabic.
VOICES = {
    "darija": "ar-MA-MounaNeural",
    "tamazight": "ar-MA-JamalNeural",
    "fr": "fr-FR-DeniseNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-PT-RaquelNeural",
}

WHISPER_LANGS = {
    "english": "en", "french": "fr", "arabic": "ar", "spanish": "es",
    "german": "de", "italian": "it", "portuguese": "pt",
}

FR_WORDS = {"le", "la", "les", "des", "une", "un", "est", "pour", "avec", "mon", "mes", "je", "j", "comment",
            "quoi", "que", "qui", "dans", "sur", "mouton", "moutons", "vache", "vaches", "chèvre", "nourrir",
            "combien", "bonjour", "merci", "et", "du", "au", "aux", "ce", "cette", "ont", "ai", "nous", "vous"}
EN_WORDS = {"the", "is", "are", "for", "with", "my", "i", "how", "what", "which", "in", "on", "sheep", "cow",
            "cows", "goat", "feed", "much", "many", "hello", "thanks", "and", "to", "of", "do", "have", "we", "you"}


def normalize(lang: str | None) -> str:
    """Map UI/ASR language names to: auto, darija, tamazight, fr, en, es, de, it, pt."""
    lang = (lang or "auto").lower().strip()
    if lang in ("ar", "darija", "ary", "arabic"):
        return "darija"
    if lang in ("tz", "tamazight", "amazigh"):
        return "tamazight"
    if lang in ("fr", "french", "francais", "français"):
        return "fr"
    return lang


def detect_text_lang(text: str) -> str:
    """Cheap script/stopword detector for typed text and as an ASR fallback."""
    if re.search(r"[ⴰ-⵿]", text):
        return "tamazight"
    if re.search(r"[؀-ۿ]", text):
        return "darija"
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    fr = sum(w in FR_WORDS for w in words) + sum(ch in "àâçéèêëîïôùûœ" for ch in text.lower())
    en = sum(w in EN_WORDS for w in words)
    return "fr" if fr > en else "en"


def reply_prompt(lang: str) -> str:
    """Instruction telling the LLM which language to answer in (lang is a normalized code)."""
    if lang == "darija":
        return ("IMPORTANT: You MUST answer in Moroccan Darija (الدارجة المغربية) using Arabic script. "
                "DO NOT use formal Classical Arabic. Use common Moroccan vocabulary (e.g. 'علف', 'ما', 'خروف').")
    if lang == "tamazight":
        return ("IMPORTANT: You MUST answer in Tamazight (ⵜⴰⵎⴰⵣⵉⵖⵜ) using Latin transliteration or Tifinagh. "
                "Keep your sentences short and natural to avoid repetition.")
    if lang == "fr":
        return "IMPORTANT: You MUST answer in French (français), in simple everyday words."
    if lang == "en":
        return "IMPORTANT: You MUST answer in English, in simple everyday words."
    return ("IMPORTANT: Answer in the SAME language the user wrote or spoke in. If it is Moroccan Darija, "
            "answer in Darija using Arabic script (not Classical Arabic).")


# ---------------------------------------------------------------- ASR

def _moulsot(audio_bytes: bytes, filename: str, ctype: str) -> str:
    """Best Darija ASR: local MoulSot server if configured, else the hosted AtlasIA Space."""
    url = os.environ.get("MOULSOT_URL")
    if url:
        try:
            r = requests.post(url, files={"audio": (filename, audio_bytes, ctype)}, timeout=180)
            r.raise_for_status()
            text = r.json().get("text", "").strip()
            if text:
                return text
        except Exception as e:
            print(f"Local MoulSot ASR error: {e}")
    tmp_path = None
    try:
        from gradio_client import Client, handle_file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix or ".webm") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = Client("atlasia/MoulSot.v0.3", verbose=False).predict(handle_file(tmp_path), api_name="/transcribe")
        if isinstance(result, (list, tuple)):
            result = next((r for r in result if isinstance(r, str)), "")
        return str(result or "").strip()
    except Exception as e:
        print(f"MoulSot ASR error: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


DARIJA_PROMPT = "هضرة بالدارجة المغربية على الدجاج، الغنم، البقر، العلف، الماء، الشعير، الفصة، شحال ف اليوم."


def _whisper(audio_bytes: bytes, filename: str, ctype: str, language: str | None) -> tuple[str, str | None, float]:
    """Groq Whisper; language=None lets Whisper auto-detect.
    Returns (text, detected_code, avg_logprob); the code is None for languages we don't route."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return "", None, -9.0
    data = {"model": "whisper-large-v3", "response_format": "verbose_json", "temperature": "0"}
    if language:
        data["language"] = language
    if language == "ar":
        data["prompt"] = DARIJA_PROMPT
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (filename, audio_bytes, ctype)},
            data=data, timeout=60,
        )
        r.raise_for_status()
        j = r.json()
        segs = j.get("segments") or []
        logprob = sum(s.get("avg_logprob", -9.0) for s in segs) / len(segs) if segs else -9.0
        return j.get("text", "").strip(), WHISPER_LANGS.get((j.get("language") or "").lower()), logprob
    except Exception as e:
        print(f"Groq ASR error: {e} | {getattr(getattr(e, 'response', None), 'text', '')}")
        return "", None, -9.0


def _hf_whisper(audio_bytes: bytes, ctype: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return ""
    try:
        r = requests.post(
            "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3",
            headers={"Authorization": f"Bearer {token}", "Content-Type": ctype},
            data=audio_bytes, timeout=60,
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()
    except Exception as e:
        print(f"HF ASR error: {e}")
        return ""


def transcribe(audio_bytes: bytes, filename: str, ctype: str, language: str) -> tuple[str, str]:
    """Return (text, language_code). language='auto' detects what the user spoke."""
    language = normalize(language)

    if language in ("darija", "tamazight"):
        text = _moulsot(audio_bytes, filename, ctype)
        if not text:
            text, _, _ = _whisper(audio_bytes, filename, ctype, "ar")
        return text or _hf_whisper(audio_bytes, ctype), language

    if language != "auto":
        text, _, _ = _whisper(audio_bytes, filename, ctype, language)
        return text or _hf_whisper(audio_bytes, ctype), language

    text, detected, logprob = _whisper(audio_bytes, filename, ctype, None)
    if not text:
        text = _hf_whisper(audio_bytes, ctype)
    # Whisper often mislabels Darija as English/French/other and hallucinates a translation-like text.
    # Trust a non-Arabic result only when it is a routed language AND the model is confident.
    suspicious = (
        detected in (None, "ar")
        or bool(re.search(r"[؀-ۿ]", text))
        or logprob < -0.45
    )
    if suspicious or not text:
        darija = _moulsot(audio_bytes, filename, ctype)
        if darija and re.search(r"[؀-ۿ]", darija):
            return darija, "darija"
        if not text:
            text, _, _ = _whisper(audio_bytes, filename, ctype, "ar")
            return text, "darija"
    return text, detected or detect_text_lang(text)


# ---------------------------------------------------------------- TTS

async def synthesize(text: str, language: str) -> tuple[bytes, str]:
    """Return (audio_bytes, media_type). Darija can use a local DarijaTTS server via DARIJA_TTS_URL."""
    language = normalize(language)
    if language == "darija" and os.environ.get("DARIJA_TTS_URL"):
        try:
            r = requests.post(os.environ["DARIJA_TTS_URL"], json={"text": text.replace("\n", ". ")}, timeout=300)
            r.raise_for_status()
            return r.content, "audio/wav"
        except requests.RequestException as e:
            print(f"Local DarijaTTS unavailable, using Edge voice: {e}")
    voice = VOICES.get(language, VOICES["en"])
    audio = b""
    async for chunk in edge_tts.Communicate(text.replace("\n", ". "), voice).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    if not audio:
        raise RuntimeError("TTS returned no audio")
    return audio, "audio/mpeg"
