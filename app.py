import os
import re
import traceback
import unicodedata
import concurrent.futures
from datetime import datetime, timezone
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CANDIDATOS_ETIQUETAS = ["Tags", "Etiquetas", "Etiqueta"]
MAX_BLOCK_DEPTH = 3

ai_client = genai.Client(api_key=GEMINI_API_KEY)

def normalizar(texto: str) -> str:
    if not texto: return ""
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))

def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

def listar_bases_de_datos(headers) -> dict:
    resultado = {}
    cursor = None
    while True:
        payload = {"filter": {"value": "database", "property": "object"}, "page_size": 100}
        if cursor: payload["start_cursor"] = cursor
        res = requests.post("https://api.notion.com/v1/search", headers=headers, json=payload, timeout=10)
        if res.status_code != 200: break
        data = res.json()
        for item in data.get("results", []):
            titulo = "".join(t.get("plain_text", "") for t in item.get("title", []))
            resultado[item["id"]] = titulo
        if not data.get("has_more"): break
        cursor = data.get("next_cursor")
    return resultado

def extraer_tags_de_propiedades(properties: dict) -> list:
    for candidato in CANDIDATOS_ETIQUETAS:
        prop = properties.get(candidato)
        if not prop: continue
        if prop.get("type") == "multi_select":
            return [o.get("name", "") for o in prop.get("multi_select", [])]
        if prop.get("type") == "select" and prop.get("select"):
            return [prop["select"].get("name", "")]
    return []

def extraer_titulo_de_propiedades(properties: dict) -> str:
    for p_val in properties.values():
        if p_val.get("type") == "title" and p_val.get("title"):
            return p_val["title"][0].get("plain_text", "Sin título")
    return "Sin título"

def obtener_texto_bloques(block_id: str, headers: dict, profundidad: int = 0) -> str:
    if profundidad > MAX_BLOCK_DEPTH: return ""
    textos = []
    cursor = None
    while True:
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        params = {"page_size": 100}
        if cursor: params["start_cursor"] = cursor
        try:
            res = requests.get(url, headers=headers, params=params, timeout=8)
        except Exception: break
        if res.status_code != 200: break
        data = res.json()
        for b in data.get("results", []):
            b_type = b.get("type")
            contenido_bloque = b.get(b_type, {})
            if isinstance(contenido_bloque, dict) and "rich_text" in contenido_bloque:
                for segmento in contenido_bloque["rich_text"]:
                    textos.append(segmento.get("plain_text", ""))
            if b.get("has_children"):
                textos.append(obtener_texto_bloques(b["id"], headers, profundidad + 1))
        if not data.get("has_more"): break
        cursor = data.get("next_cursor")
    return "\n".join(t for t in textos if t)

def procesar_pagina_sync(item, headers, db_nombre_por_id):
    item_id = item["id"]
    parent = item.get("parent", {})
    db_id = parent.get("database_id")
    properties = item.get("properties", {})

    titulo = extraer_titulo_de_propiedades(properties) if properties else "Página sin título"
    tags = extraer_tags_de_propiedades(properties) if db_id else []
    contenido = obtener_texto_bloques(item_id, headers)

    return {
        "id": item_id,
        "titulo": titulo,
        "titulo_norm": normalizar(titulo),
        "url": item.get("url", ""),
        "tags": tags,
        "tags_norm": [normalizar(t) for t in tags],
        "db_id": db_id,
        "db_nombre": db_nombre_por_id.get(db_id, ""),
        "contenido": contenido,
        "contenido_norm": normalizar(contenido),
        "last_edited_time": item.get("last_edited_time")
    }

class SyncRequest(BaseModel):
    last_sync: str = None

@app.post("/api/sync")
async def sync_notion(req: SyncRequest):
    """Devuelve las páginas actualizadas desde last_sync y una lista de todos los IDs activos (para borrar los eliminados)."""
    headers = notion_headers()
    try:
        last_sync_dt = datetime.fromisoformat(req.last_sync.replace('Z', '+00:00')) if req.last_sync else None
    except:
        last_sync_dt = None

    db_nombre_por_id = listar_bases_de_datos(headers)
    all_search_items = []
    cursor = None
    
    # 1. Obtener metadatos de todas las páginas
    while True:
        payload = {"filter": {"value": "page", "property": "object"}, "page_size": 100}
        if cursor: payload["start_cursor"] = cursor
        res = requests.post("https://api.notion.com/v1/search", headers=headers, json=payload, timeout=10)
        if res.status_code != 200: break
        data = res.json()
        all_search_items.extend(data.get("results", []))
        if not data.get("has_more"): break
        cursor = data.get("next_cursor")

    active_ids = [item["id"] for item in all_search_items]
    pages_to_fetch = []
    
    # 2. Filtrar solo las páginas que han sido editadas después de la última sincronización
    for item in all_search_items:
        edited_str = item.get("last_edited_time")
        if edited_str:
            edited_dt = datetime.fromisoformat(edited_str.replace('Z', '+00:00'))
            if not last_sync_dt or edited_dt > last_sync_dt:
                pages_to_fetch.append(item)

    # 3. Descargar el contenido profundo solo de las modificadas
    updated_pages = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futuros = [executor.submit(procesar_pagina_sync, item, headers, db_nombre_por_id) for item in pages_to_fetch]
        for f in concurrent.futures.as_completed(futuros):
            try:
                updated_pages.append(f.result())
            except Exception:
                pass

    return {
        "updated_pages": updated_pages,
        "active_ids": active_ids,
        "sync_time": datetime.now(timezone.utc).isoformat()
    }


class ChatRequest(BaseModel):
    prompt: str
    advanced_search: bool = False
    context_pages: list = []  # El frontend nos envía las páginas ya filtradas y elegidas

@app.post("/api/chat")
async def chat_gemini(req: ChatRequest):
    user_prompt = req.prompt.strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="El prompt está vacío.")

    # =========================================================
    # MODO BÚSQUEDA AVANZADA
    # =========================================================
    if req.advanced_search:
        if not req.context_pages:
            return {"response": "No se encontraron notas en tu caché que cumplan con los filtros.", "sources": []}

        gemini_user = f"INTENCIÓN DEL USUARIO: {user_prompt}\n\nNOTAS DISPONIBLES:\n"
        for i, nota in enumerate(req.context_pages):
            gemini_user += f"[{i + 1}] {nota.get('titulo', '')} - {nota.get('url', '')}\n"

        gemini_sys = (
            f"Eres el cerebro del Buscador Avanzado. Tienes hasta {len(req.context_pages)} notas preseleccionadas localmente.\n"
            f"Tu tarea: selecciona un MÁXIMO DE 20 títulos que mejor respondan a la intención.\n"
            "INSTRUCCIÓN ESTRICTA: tu respuesta debe ser EXCLUSIVAMENTE una lista en formato Markdown numerado, con "
            "el título de cada nota como enlace clickeable. No añadas introducciones."
        )

        try:
            chat = ai_client.chats.create(model=GEMINI_MODEL)
            gem_res = chat.send_message(f"{gemini_sys}\n\n{gemini_user}")
            
            response_text = "### 🔍 Índices de Búsqueda Avanzada:\n\n" + gem_res.text
            if len(req.context_pages) > 20:
                response_text += "\n\n⚠️ **Nota:** Hay más resultados. Afina tu búsqueda."
            return {"response": response_text, "sources": []}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str: raise HTTPException(status_code=429, detail={"type": "rate_limit", "wait_seconds": 60})
            if "503" in error_str: raise HTTPException(status_code=503, detail={"type": "server_busy"})
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================
    # MODO BÚSQUEDA NORMAL
    # =========================================================
    try:
        corpus_texto = []
        fuentes = []
        for idx, p in enumerate(req.context_pages):
            fuentes.append({"titulo": p.get('titulo'), "url": p.get('url')})
            corpus_texto.append(
                f"--- FUENTE [{idx + 1}] ---\n"
                f"Título de la Nota: {p.get('titulo')}\n"
                f"URL de Notion: {p.get('url')}\n"
                f"Contenido:\n{p.get('contenido', '')[:6000]}"
            )
        texto_contexto = "\n\n".join(corpus_texto) if corpus_texto else "No se encontraron notas relevantes."

        system_prompt = (
            "Eres un asistente conectado al espacio de Notion del usuario.\n"
            "INSTRUCCIONES OBLIGATORIAS:\n"
            "1. Incluye abundantes citas textuales y directas.\n"
            "2. NO escribas nombres largos de notas dentro del texto redactado.\n"
            "3. Tras cada cita o afirmación, coloca un número de referencia correlativo entre "
            "paréntesis como enlace Markdown: `([1](URL_DE_NOTION))`. Utiliza estrictamente las URLs de las fuentes."
        )

        full_prompt = f"{system_prompt}\n\n{texto_contexto}\n\n--- PETICIÓN DEL USUARIO ---\n{user_prompt}"

        chat = ai_client.chats.create(model=GEMINI_MODEL)
        response = chat.send_message(full_prompt)

        return {"response": response.text, "sources": fuentes}

    except Exception as e:
        error_str = str(e)
        traceback.print_exc()
        if "429" in error_str: raise HTTPException(status_code=429, detail={"type": "rate_limit", "wait_seconds": 60})
        if "503" in error_str: raise HTTPException(status_code=503, detail={"type": "server_busy"})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root(): return FileResponse("index.html")
@app.get("/manifest.json")
async def get_manifest(): return FileResponse("manifest.json")
@app.get("/sw.js")
async def get_sw(): return FileResponse("sw.js", media_type="application/javascript")
@app.get("/icon-192.png")
async def get_icon(): return FileResponse("icon-192.png")