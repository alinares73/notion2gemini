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
    advanced_search: bool = False
    databases: list[str] = []

def obtener_ids_bases_de_datos(token: str, nombres_objetivo: list) -> set:
    """Busca en Notion los IDs de las bases de datos cuyos nombres coincidan con los seleccionados."""
    if not nombres_objetivo:
        return set()
    
    url = "https://api.notion.com/v1/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    try:
        payload = {
            "filter": {"value": "database", "property": "object"},
            "page_size": 50
        }
        res = requests.post(url, headers=headers, json=payload, timeout=5)
        if res.status_code != 200:
            return set()
        
        db_ids = set()
        for item in res.json().get("results", []):
            titulo = "Sin título"
            for title_obj in item.get("title", []):
                titulo = title_obj.get("plain_text", "")
                break
            if titulo in nombres_objetivo:
                db_ids.add(item.get("id"))
        return db_ids
    except Exception:
        return set()

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

def buscar_en_notion(token: str, prompt: str, limit: int, target_dbs: list) -> tuple[list, bool]:
    """Realiza la búsqueda en Notion aplicando filtros de bases de datos y límites."""
    url = "https://api.notion.com/v1/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    try:
        # Obtenemos los IDs permitidos si el usuario filtró bases de datos
        allowed_db_ids = obtener_ids_bases_de_datos(token, target_dbs) if target_dbs else set()
        
        # Solicitamos un poco más para comprobar si hay más de 20 en búsqueda avanzada
        payload = {
            "query": prompt, 
            "page_size": limit + 1 if limit == 20 else limit, 
            "filter": {"value": "page", "property": "object"}
        }
        res = requests.post(url, headers=headers, json=payload, timeout=5)
        if res.status_code != 200:
            return [], False

        data = res.json()
        results = data.get("results", [])
        has_more = data.get("has_more", False)

        # Filtrar por bases de datos seleccionadas si aplica
        if allowed_db_ids:
            filtered_results = []
            for item in results:
                parent = item.get("parent", {})
                if parent.get("type") == "database_id" and parent.get("database_id") in allowed_db_ids:
                    filtered_results.append(item)
            results = filtered_results

        # Comprobar si hay más del límite establecido
        is_truncated = False
        if len(results) > limit:
            is_truncated = True
            results = results[:limit]
        elif has_more and len(results) == limit:
            is_truncated = True

        return results, is_truncated
    except Exception:
        return [], False

@app.post("/api/chat")
async def chat_gemini_notion(query: UserQuery):
    user_prompt = query.prompt.strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="El prompt está vacío.")

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    # MODO BÚSQUEDA AVANZADA (Hasta 20 notas, solo títulos y enlaces, sin Gemini)
    if query.advanced_search:
        results, is_truncated = buscar_en_notion(NOTION_TOKEN, user_prompt, limit=20, target_dbs=query.databases)
        
        if not results:
            return {"response": "No se encontraron notas que coincidan con la búsqueda avanzada.", "sources": []}

        response_lines = ["### Resultados de Búsqueda Avanzada:\n"]
        sources = []
        
        for idx, item in enumerate(results):
            item_url = item.get("url", "")
            titulo = "Página sin título"
            for p_name, p_val in item.get("properties", {}).items():
                if p_val.get("type") == "title" and p_val.get("title"):
                    titulo = p_val['title'][0].get('plain_text', 'Página sin título')
                    break
            
            response_lines.append(f"{idx + 1}. [{titulo}]({item_url})")
            sources.append({"titulo": titulo, "url": item_url})

        if is_truncated:
            response_lines.append("\n⚠️ **Refina más tu búsqueda para acotar los resultados...**")

        return {
            "response": "\n".join(response_lines),
            "sources": sources
        }

    # MODO NORMAL (Hasta 7 notas con contenido completo procesado por Gemini)
    results, _ = buscar_en_notion(NOTION_TOKEN, user_prompt, limit=7, target_dbs=query.databases)
    
    fuentes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(procesar_pagina, item, headers) for item in results]
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            if data:
                fuentes.append(data)

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
        f"--- PETICIÓN DEL USUARIO --.~(\n{user_prompt})" if False else f"{system_prompt}\n\n{texto_contexto}\n\n--- PETICIÓN DEL USUARIO ---\n{user_prompt}"
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

@app.get("/icon-192.png")
async def get_icon():
    return FileResponse("icon-192.png")