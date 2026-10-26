import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import requests
import base64
from dotenv import load_dotenv

try:
    from backend import voice  # uvicorn backend.api:app
except ImportError:
    import voice  # python backend/api.py

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="LivestockAI Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components globally
print("Loading embeddings and vector DB...")
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=str(BASE_DIR / "chroma_db_pdf"),
    embedding_function=embeddings,
)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Prefer Groq if available, fallback to Gemini
if GROQ_API_KEY:
    print("Using Groq (Llama-3.3) for AI generation...")
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, groq_api_key=GROQ_API_KEY)
elif GOOGLE_API_KEY:
    print("Using Google (Gemini) for AI generation...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
else:
    print("Warning: No LLM API Key found!")
    llm = None


def clean_reply(text: str) -> str:
    """Strip markdown so replies display and are read aloud cleanly."""
    import re
    text = re.sub(r"[*_`#>]+", "", text or "")
    text = re.sub(r"\s+(\d+)[.)]\s+", r"\n\1. ", text)
    text = re.sub(r"\s+-\s+", "\n- ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def retrieval_query(text: str, history: list | None = None) -> str:
    """The PDF index is English (all-MiniLM-L6-v2): turn the question (any language, possibly a
    follow-up like "and how much water?") into a standalone English search query."""
    if not llm:
        return text
    convo = "\n".join(f"{m.get('role', 'user')}: {m.get('text', '')}" for m in (history or [])[-6:])
    try:
        out = llm.invoke(
            "Rewrite the user's last message as ONE short, standalone English search query about "
            "livestock/farming for a document search. Resolve references (it, them, and, also) using the "
            "conversation. The message may be Moroccan Darija, French or English. Output only the query.\n\n"
            f"Conversation:\n{convo}\n\nLast message: {text}"
        )
        return (out.content or "").strip() or text
    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return text


class ReportRequest(BaseModel):
    animalType: str
    animalBreed: str = ""
    herdSize: str
    age: str = ""
    location: str = ""
    season: str = ""
    budget: str = ""
    language: str = "en"

class ChatRequest(BaseModel):
    query: str
    language: str = "en"
    history: list[dict] = []

@app.post("/api/generate-report")
async def generate_report(request: ReportRequest):
    if not llm:
        raise HTTPException(
            status_code=503,
            detail="No LLM API key configured. Set GROQ_API_KEY or GOOGLE_API_KEY.",
        )

    # 1. Use the vector database to find context about this animal
    query_str = f"operational livestock management feeding tables daily schedules water intake grazing schedules for {request.animalType} {request.animalBreed} {request.age} {request.season} {request.location}"
    docs = retriever.invoke(query_str)
    context = "\n\n".join(d.page_content for d in docs)
    
    # 2. Build the System Prompt for JSON generation based on language
    system_prompt = """You are an expert agricultural AI assistant and livestock nutritionist. 
You must always reply with valid JSON only.
CRITICAL RULES:
1. NEVER hallucinate or invent data. Use ONLY information explicitly supported by the provided context.
2. If exact feeding quantities or prices are unavailable in the context, explicitly state that in the JSON string (e.g. "Data unavailable in veterinary database").
3. Your plan must be extremely concrete and scientifically precise based on the context."""
    
    language_instruction = ""
    if request.language == 'darija':
        system_prompt += "\nYou speak Moroccan Darija (الدارجة المغربية). All text values in the JSON MUST be written in Moroccan Darija using Arabic script. DO NOT use formal Classical Arabic. Use common Moroccan vocabulary (e.g. 'علف', 'ما', 'خروف')."
        language_instruction = "\nIMPORTANT: Translate all explanations, tips, and summaries into local Moroccan Darija. Keep the JSON keys in English."
    elif request.language == 'tamazight':
        system_prompt += "\nYou speak Tamazight (ⵜⴰⵎⴰⵣⵉⵖⵜ). All text values in the JSON MUST be written in Tamazight using Latin transliteration or Tifinagh. Keep the sentences short and clear to avoid repetition."
        language_instruction = "\nIMPORTANT: Translate all explanations, tips, and summaries into Tamazight. Keep the JSON keys in English."

    prompt_content = f"""
    Generate a highly structured, concrete, and scientifically grounded daily nutrition and hydration plan.
    Animal Type: {request.animalType}
    Breed: {request.animalBreed}
    Herd Size: {request.herdSize}
    Age/Stage: {request.age}
    Location: {request.location}
    Season: {request.season}
    Budget: {request.budget}
    {language_instruction}
    
    Here is the authoritative context retrieved from our specialized veterinary database:
    ---
    {context}
    ---
    
    Please provide the response ONLY as a valid JSON object with the exact following structure. 
    Ensure all text fields contain highly concrete, schedule-based, and scientific details derived strictly from the context.
    {{
    "overview": {{
        "animal": "animal name",
        "size": "{request.herdSize}",
        "summary": "1-2 sentence extremely precise summary factoring in age, location, and season"
      }},
      "dailyNeeds": {{
        "water": "Exact total liters string (or 'Data unavailable')",
        "food": "Exact total kg string (or 'Data unavailable')",
        "waterPerHead": "L/head string",
        "foodPerHead": "kg/head string"
      }},
      "nutritionalBreakdown": [
        {{ "name": "Alfalfa/Roughage", "value": 50 }},
        {{ "name": "Grains/Concentrates", "value": 30 }},
        {{ "name": "Proteins", "value": 15 }},
        {{ "name": "Vitamins/Minerals", "value": 5 }}
      ],
      "schedule": [
        {{ "time": "Morning", "action": "Highly specific, concrete morning feeding and watering action based on context" }},
        {{ "time": "Midday", "action": "Highly specific midday management action" }},
        {{ "time": "Evening", "action": "Highly specific evening action" }}
      ],
      "recommendations": {{
        "localFood": "Concrete recommendation about local food sourcing based on location {request.location} and context",
        "healthWarnings": "Important health warning for {request.season} based on context",
        "costSaving": "Cost saving strategy considering budget {request.budget} based on context",
        "productivity": "Limitations note if data is missing, OR a concrete productivity tip"
      }}
    }}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content)
    ]
    
    try:
        response = llm.invoke(messages)
        
        # Robust JSON extraction
        raw_content = response.content.strip()
        
        import re
        # Find the first { and the last }
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_json = json.loads(json_str)
            return parsed_json
        else:
            raise ValueError("No valid JSON object found in response.")
            
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        print(f"Raw response was: {response.content if 'response' in locals() else 'None'}")
        raise HTTPException(status_code=500, detail=str(e))

def answer(question: str, lang: str, from_audio: bool = False, history: list | None = None) -> str:
    docs = retriever.invoke(retrieval_query(question, history))
    context = "\n\n".join(d.page_content for d in docs)
    convo = "\n".join(f"{m.get('role', 'user')}: {m.get('text', '')}" for m in (history or [])[-6:])
    label = "User question (transcribed from audio, may contain small speech errors)" if from_audio else "Question"
    prompt = f"""You are a helpful agricultural AI assistant for Moroccan livestock farmers. Answer in at most 3 short sentences, in plain conversational text. Do NOT use markdown, asterisks, bullet points or numbered lists.
Answer exactly what the user asked, using the conversation so far to understand follow-ups. Use the Context only where it is relevant to the question; ignore parts that are not. If the context does not contain the answer, say so briefly instead of answering about something else. If the question is not about farming or livestock, say you can only help with livestock.
{voice.reply_prompt(lang)}

Conversation so far:
{convo or "(none)"}

Context:
{context}

{label}: {question}
Answer:"""
    return clean_reply(llm.invoke(prompt).content)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not llm:
        raise HTTPException(
            status_code=503,
            detail="No LLM API key configured. Set GROQ_API_KEY or GOOGLE_API_KEY.",
        )
    lang = voice.normalize(request.language)
    if lang == "auto":
        lang = voice.detect_text_lang(request.query)
    try:
        return {"result": answer(request.query, lang, history=request.history), "language": lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice-chat")
async def voice_chat(audio: UploadFile = File(...), language: str = Form("auto"), history: str = Form("[]")):
    if not (os.environ.get("GROQ_API_KEY") or os.environ.get("HF_TOKEN")):
        raise HTTPException(status_code=503, detail="GROQ_API_KEY or HF_TOKEN is required for voice processing.")
    if not llm:
        raise HTTPException(
            status_code=503,
            detail="No LLM API key configured. Set GROQ_API_KEY or GOOGLE_API_KEY.",
        )
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")

    # 1. ASR (auto-detects the spoken language unless one was picked)
    text, lang = voice.transcribe(
        audio_bytes, audio.filename or "audio.webm", audio.content_type or "audio/webm", language
    )
    if not text:
        raise HTTPException(status_code=502, detail="Speech transcription failed for the configured models.")

    # 2. RAG answer in the language the user spoke
    try:
        hist = json.loads(history)
    except ValueError:
        hist = []
    try:
        ai_text = answer(text, lang, from_audio=True, history=hist)
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="The AI service failed to process the transcribed request.",
        ) from error

    # 3. TTS is generated on demand by the frontend via /api/tts
    return {"transcription": text, "response": ai_text, "language": lang, "audio_base64": None}


class TTSRequest(BaseModel):
    text: str
    language: str = "darija"


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Speech in Darija / French / English (see backend/voice.py)."""
    text = clean_reply(request.text)
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    try:
        audio, media_type = await voice.synthesize(text, request.language)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS unavailable: {e}")
    return Response(content=audio, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
