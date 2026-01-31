from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid

from rag import load_portfolio, search_context
from detector import detect_profile
from memory import get_memory, save_memory

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró la API KEY. Revisa tu archivo .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_portfolio()


@app.get("/")
async def home():
    return FileResponse("chat.html")



SYSTEM_BASE = """
Eres el asistente oficial del portafolio de Juan Echeverri.

Objetivo:
Impresionar a quien visite el portafolio.

Reglas:
- Nunca inventes experiencia.
- Sé seguro y profesional.
- Destaca impacto real.
- Guía la conversación inteligentemente.
"""


@app.post("/chat")
async def chat(data: dict):

    try:

        message = data["message"]
        session_id = data.get("session_id") or str(uuid.uuid4())

        memory = get_memory(session_id)

        profile = detect_profile(client, message)

        context = search_context(message)

        system_prompt = f"""
{SYSTEM_BASE}

El usuario parece ser: {profile}

Si es DEV:
Profundiza en arquitectura, decisiones técnicas y escalabilidad.

Si es RECLUTADOR:
Habla en términos de valor, resultados y profesionalismo.
"""

        messages = [
            {"role":"system","content":system_prompt},
            {"role":"system","content":f"Contexto:\n{context}"}
        ]

        messages.extend(memory)

        messages.append({"role":"user","content":message})

        # 🔥 LLAMADA AL MODELO (LO QUE TE FALTABA)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            messages=messages
        )

        response = completion.choices[0].message.content

        # 🔥 Guardar memoria
        save_memory(session_id, message, response)

        return {
            "response": response,
            "session_id": session_id
        }

    except Exception as e:
        print("ERROR EN /chat:", e)
        return {
            "response": "Hubo un error interno.",
            "session_id": None
        }