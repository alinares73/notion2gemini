import os
import re
import traceback
import concurrent.futures
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

ai_client = genai.Client(api_key=GEMINI_API_KEY)

class UserQuery(BaseModel):
    prompt: str

def procesar_pagina(item, headers):
    item_id = item.get("id")
    item_url = item.get("url", "")
    
    titulo = "Página sin título"
    for p_name, p_val in item.get("properties", {}).items():
        if p_val.get("type") == "title" and p_val.get("title"):
            titulo = p_val['title'][0].get('plain_text', 'Página sin título')
            break
    
    blocks_url = f"https://api.notion.com/v1/blocks/{item_id}/children"
    try:
        # Timeout optimizado para respuestas rápidas
        b_res = requests.get(blocks_url, headers=headers, timeout=4)
        page_text = ""
        if b_res.status_code == 200:
            textos = []
            for b in b_res.json().get("results", []):
                b_type = b.get("type")
                if b_type in b and "rich_text" in b[b_type]:
                    for segment in b[b_type]["rich_text"]:
                        textos.append(segment.get("plain_text", ""))
            page_text = "\n".join(textos)

        if page_text.strip():
            return {
                "titulo": titulo,
                "url": item_url,
                "contenido": page_text
            }
    except Exception:
        pass
    return None

def obtener_corpus_notion(token: str, prompt: str) -> list:
    url = "https://api.notion.com/v1/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    try:
        # Búsqueda avanzada: delegamos en Notion para que nos dé solo el TOP 7 máximo
        # Si encuentra menos (ej. 2), solo devolverá 2.
        payload = {
            "query": prompt, 
            "page_size": 7, 
            "filter": {"value": "page", "property": "object"}
        }
        res = requests.post(url, headers=headers, json=payload, timeout=5)
        if res.status_code != 200:
            return []

        results = res.json().get("results", [])
        font_sources = []

        # Limitamos los hilos a 3 para respetar estrictamente el límite de la API de Notion
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(procesar_pagina, item, headers) for item in results]
            
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if data:
                    font_sources.append(data)

        return font_sources
    except Exception:
        return []

@app.post("/api/chat")
async def chat_gemini_notion(query: UserQuery):
    user_prompt = query.prompt.strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="El prompt está vacío.")

    # Obtenemos un contexto preciso y ligero
    fuentes = obtener_corpus_notion(NOTION_TOKEN, user_prompt)
    
    corpus_texto = []
    for idx, f in enumerate(fuentes):
        corpus_texto.append(
            f"--- FUENTE [{idx + 1}] ---\n"
            f"Título de la Nota: {f['titulo']}\n"
            f"URL de Notion: {f['url']}\n"
            f"Contenido:\n{f['contenido']}"
        )

    texto_contexto = "\n\n".join(corpus_texto) if corpus_texto else "No se encontraron notas relevantes."

    system_prompt = (
        "Eres un asistente conectado al espacio de Notion del usuario.\n"
        "INSTRUCCIONES OBLIGATORIAS:\n"
        "1. Incluye abundantes citas textuales y directas basadas en el contenido de las fuentes. "
        "Escribe las citas simplemente entre comillas dobles normales (\"ejemplo\"), sin usar bloques de código, cursivas especiales ni formatos que cambien el color del texto.\n"
        "2. NO escribas nombres largos de notas dentro del texto redactado.\n"
        "3. Detrás de cada cita o afirmación importante, coloca un número de referencia correlativo entre paréntesis que sea un enlace Markdown apuntando a la fuente correspondiente, con este formato exacto: `([1](URL_DE_NOTION))`, `([2](URL_DE_NOTION))`, etc.\n"
        "No inventes URLs; utiliza estrictamente las proporcionadas en cada fuente."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"{texto_contexto}\n\n"
        f"--- PETICIÓN DEL USUARIO ---\n{user_prompt}"
    )

    try:
        chat = ai_client.chats.create(model="gemini-3.6-flash")
        response = chat.send_message(full_prompt)
        
        return {
            "response": response.text,
            "sources": fuentes
        }
    except Exception as e:
        error_str = str(e)
        print("--- ERROR DETALLADO EN /api/chat ---")
        traceback.print_exc()
        
        if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
            match = re.search(r'retry in ([\d\.]+)s', error_str)
            wait_seconds = int(float(match.group(1))) + 1 if match else 60
            raise HTTPException(
                status_code=429, 
                detail={"type": "rate_limit", "wait_seconds": wait_seconds}
            )
            
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/manifest.json")
async def get_manifest():
    return FileResponse("manifest.json")

@app.get("/sw.js")
async def get_sw():
    return FileResponse("sw.js", media_type="application/javascript")