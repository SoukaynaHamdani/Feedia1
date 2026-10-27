"""Voice helpers: language detection, multilingual ASR routing and TTS (Darija / French / English)."""
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
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


# Darija -> English retrieval keywords (the PDF index is English). Lets us skip an LLM round trip (~2s).
DARIJA_KEYWORDS = {
    "sheep": ["غنم", "لحولي", "حولي", "خروف", "نعجة", "كبش", "غنمة"],
    "cow": ["بقر", "فرد", "عجل"],
    "goat": ["معز", "ماعز", "جدي"],
    "chickens poultry": ["دجاج", "فروج", "طيور", "الطير", "دجاجة"],
    "camel": ["جمل", "إبل", "ابل"],
    "horse": ["عود", "خيل", "حصان"],
    "rabbit": ["ارنب", "أرنب"],
    "feed ration nutrition": ["علف", "ماكلة", "اكل", "أكل", "تغذية", "طعام", "تعلف", "نعلف", "شعير", "فصة", "تبن", "نخالة"],
    "water requirement": ["ماء", "الما", "شرب", "تشرب", "كتشرب", "سقي"],
    "grazing pasture": ["مرعى", "رعي", "سرح", "كلأ"],
    "disease vaccination health": ["مرض", "امراض", "أمراض", "تلقيح", "لقاح", "دواء", "مريض", "صحة"],
    "milk production": ["حليب", "لبن", "حلب"],
    "pregnant lactating": ["حامل", "حوامل", "ترضع", "رضيع"],
    "lambs kids calves young": ["صغار", "ربيع", "تلميذ", "صغير", "فطام"],
    "winter": ["شتاء", "برد", "الشتا"],
    "summer heat": ["صيف", "حرارة", "الحر", "سخون"],
    "cost budget price": ["ثمن", "تكلفة", "فلوس", "ميزانية", "غالي"],
    "daily amount per head quantity": ["شحال", "كمية", "كم "],
    "plan schedule program": ["تخطط", "نخطط", "برنامج", "جدول", "خطة"],
}


def darija_keywords(text: str) -> str | None:
    """English keyword query for a Darija/Arabic question, or None if we can't confidently build one."""
    hits = [en for en, words in DARIJA_KEYWORDS.items() if any(w in text for w in words)]
    animals = {"sheep", "cow", "goat", "chickens poultry", "camel", "horse", "rabbit"}
    if not any(h in animals for h in hits):
        return None
    return "livestock " + " ".join(hits)


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
    data = {"model": os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo"), "response_format": "verbose_json", "temperature": "0"}
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


MOULSOT_GRACE = float(os.environ.get("MOULSOT_GRACE", "6"))  # extra seconds we wait for MoulSot after Whisper
_pool = ThreadPoolExecutor(max_workers=4)


def _darija_asr(audio_bytes: bytes, filename: str, ctype: str) -> str:
    """Race Groq Whisper (~1s) against MoulSot (more accurate on Darija but can be slow / cold).
    MoulSot wins if it answers within MOULSOT_GRACE seconds of Whisper finishing, else Whisper's text is used."""
    moulsot = _pool.submit(_moulsot, audio_bytes, filename, ctype)
    whisper_text, _, _ = _whisper(audio_bytes, filename, ctype, "ar")
    try:
        better = moulsot.result(timeout=MOULSOT_GRACE if whisper_text else None)
        if better:
            return better
    except Exception:
        pass
    return whisper_text or _hf_whisper(audio_bytes, ctype)


def transcribe(audio_bytes: bytes, filename: str, ctype: str, language: str) -> tuple[str, str]:
    """Return (text, language_code). language='auto' detects what the user spoke."""
    language = normalize(language)

    if language in ("darija", "tamazight"):
        return _darija_asr(audio_bytes, filename, ctype), language

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
        darija = _darija_asr(audio_bytes, filename, ctype)
        if darija and (re.search(r"[؀-ۿ]", darija) or not text):
            return darija, "darija"
    return text, detected or detect_text_lang(text)


# ---------------------------------------------------------------- TTS

_tts_cache: dict[tuple[str, str], tuple[bytes, str]] = {}


async def synthesize(text: str, language: str) -> tuple[bytes, str]:
    """Return (audio_bytes, media_type), cached so replaying a message is instant."""
    language = normalize(language)
    key = (language, text)
    if key not in _tts_cache:
        if len(_tts_cache) > 100:
            _tts_cache.pop(next(iter(_tts_cache)))
        _tts_cache[key] = await _synthesize(text, language)
    return _tts_cache[key]


async def _synthesize(text: str, language: str) -> tuple[bytes, str]:
    """Darija can use a local DarijaTTS server via DARIJA_TTS_URL."""
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
