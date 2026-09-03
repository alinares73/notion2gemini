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

COLUMNA_ETIQUETAS = "Tags"

ai_client = genai.Client(api_key=GEMINI_API_KEY)

class UserQuery(BaseModel):
    prompt: str
    advanced_search: bool = False
    databases: list[str] = []

def obtener_ids_bases_de_datos(token: str, nombres_objetivo: list) -> set:
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
            titulo_db = ""
            for title_obj in item.get("title", []):
                titulo_db += title_obj.get("plain_text", "")
            
            titulo_lower = titulo_db.lower()
            for n in nombres_objetivo:
                palabras = n.lower().split()
                if all(p in titulo_lower for p in palabras):
                    db_ids.add(item.get("id"))
                    break
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

def extraer_tags_y_logica(prompt: str):
    tags = re.findall(r'#(\w+)', prompt)
    prompt_semantico = re.sub(r'#\w+', '', prompt).strip()
    
    is_or = re.search(r'\b(o)\b', prompt.lower())
    logica = "or" if is_or else "and"
    
    return tags, logica, prompt_semantico

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

    # =========================================================
    # MODO BÚSQUEDA AVANZADA (Filtro por etiquetas nativo)
    # =========================================================
    if query.advanced_search:
        tags, logica, prompt_semantico = extraer_tags_y_logica(user_prompt)
        if not prompt_semantico:
            prompt_semantico = "Busca las notas más relevantes"

        allowed_db_ids = obtener_ids_bases_de_datos(NOTION_TOKEN, query.databases) if query.databases else set()
        all_items = []

        if tags and allowed_db_ids:
            # FILTRO PLANO Y ESTRICTO COMO LO PIDE LA API DE NOTION
            condiciones_tags = [{"property": COLUMNA_ETIQUETAS, "multi_select": {"contains": tag}} for tag in tags]
            filtro_notion = {logica: condiciones_tags} if len(condiciones_tags) > 1 else condiciones_tags[0]
            
            for db_id in allowed_db_ids:
                payload = {"page_size": 100, "filter": filtro_notion}
                try:
                    res = requests.post(f"https://api.notion.com/v1/databases/{db_id}/query", headers=headers, json=payload, timeout=8)
                    if res.status_code == 200:
                        all_items.extend(res.json().get("results", []))
                    else:
                        print(f"Error Notion API en BD {db_id}: {res.text}")
                except Exception:
                    pass

        # Si no se usan etiquetas (#), usa la búsqueda global
        elif not tags:
            query_texto = user_prompt.replace('#', '')
            payload = {"query": query_texto, "page_size": 100, "filter": {"value": "page", "property": "object"}}
            try:
                res = requests.post("https://api.notion.com/v1/search", headers=headers, json=payload, timeout=8)
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    if allowed_db_ids:
                        all_items = [item for item in results if item.get("parent", {}).get("database_id") in allowed_db_ids]
                    else:
                        all_items = results
            except Exception:
                pass

        if not all_items:
            mensaje = f"No se encontraron notas con las etiquetas ({', '.join(tags)})." if tags else "No se encontraron notas en Notion."
            return {"response": mensaje, "sources": []}

        # Extraemos solo títulos y URLs
        lista_notas = []
        for item in all_items:
            item_url = item.get("url", "")
            titulo = "Sin título"
            for p_val in item.get("properties", {}).values():
                if p_val.get("type") == "title" and p_val.get("title"):
                    titulo = p_val["title"][0].get("plain_text", "Sin título")
                    break
            lista_notas.append({"titulo": titulo, "url": item_url})
            
        lista_notas = lista_notas[:100]

        gemini_sys = (
            "Eres el cerebro del Buscador Avanzado. Se ha realizado un filtrado por base de datos y etiquetas. "
            "A continuación tienes una lista de títulos de notas encontradas en Notion.\n"
            "Tu tarea: Analiza la intención semántica del usuario y SELECCIONA los MÁXIMO 20 TÍTULOS que mejor respondan a lo que busca.\n"
            "INSTRUCCIÓN ESTRICTA: Tu respuesta DEBE SER EXCLUSIVAMENTE una lista en formato Markdown numerado, donde "
            "el título de la nota sea un enlace clickeable hacia su URL. No añadas introducciones."
        )
        
        gemini_user = f"INTENCIÓN DEL USUARIO: {prompt_semantico}\n\nNOTAS DISPONIBLES:\n"
        for i, nota in enumerate(lista_notas):
             gemini_user += f"[{i+1}] {nota['titulo']} - {nota['url']}\n"

        try:
            chat = ai_client.chats.create(model="gemini-3.8-flash")
            gem_res = chat.send_message(f"{gemini_sys}\n\n{gemini_user}")
            
            response_text = "### 🔍 Índices de Búsqueda Avanzada:\n\n" + gem_res.text
            if len(lista_notas) > 20:
                 response_text += "\n\n⚠️ **Nota:** Se han ocultado resultados adicionales. Refina tu búsqueda."
                 
            return {"response": response_text, "sources": []}
        
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
                raise HTTPException(status_code=429, detail={"type": "rate_limit", "wait_seconds": 60})
            if "503" in error_str and "UNAVAILABLE" in error_str:
                raise HTTPException(status_code=503, detail={"type": "server_busy"})
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================
    # MODO BÚSQUEDA NORMAL
    # =========================================================
    try:
        url = "https://api.notion.com/v1/search"
        allowed_db_ids = obtener_ids_bases_de_datos(NOTION_TOKEN, query.databases) if query.databases else set()
        
        payload = {"query": user_prompt, "page_size": 7, "filter": {"value": "page", "property": "object"}}
        res = requests.post(url, headers=headers, json=payload, timeout=5)
        results = []
        if res.status_code == 200:
            data = res.json().get("results", [])
            if allowed_db_ids:
                for item in data:
                    parent = item.get("parent", {})
                    if parent.get("type") == "database_id" and parent.get("database_id") in allowed_db_ids:
                        results.append(item)
            else:
                results = data
                
        results = results[:7]

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
            "1. Incluye abundantes citas textuales y directas basadas en el contenido de las fuentes.\n"
            "2. NO escribas nombres largos de notas dentro del texto redactado.\n"
            "3. Detrás de cada cita o afirmación importante, coloca un número de referencia correlativo entre paréntesis que sea un enlace Markdown apuntando a la fuente correspondiente: `([1](URL_DE_NOTION))`, etc."
        )

        full_prompt = f"{system_prompt}\n\n{texto_contexto}\n\n--- PETICIÓN DEL USUARIO ---\n{user_prompt}"

        chat = ai_client.chats.create(model="gemini-3.8-flash")
        response = chat.send_message(full_prompt)
        
        return {"response": response.text, "sources": fuentes}
        
    except Exception as e:
        error_str = str(e)
        print("--- ERROR DETALLADO EN /api/chat ---")
        traceback.print_exc()
        if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
            match = re.search(r'retry in ([\d\.]+)s', error_str)
            wait_seconds = int(float(match.group(1))) + 1 if match else 60
            raise HTTPException(status_code=429, detail={"type": "rate_limit", "wait_seconds": wait_seconds})
        if "503" in error_str and "UNAVAILABLE" in error_str:
            raise HTTPException(status_code=503, detail={"type": "server_busy"})
            
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