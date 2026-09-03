import os
import re
import json
import asyncio
import traceback
import unicodedata
import concurrent.futures
from datetime import datetime, timezone
from threading import Lock

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

# Nombres de propiedad candidatos para las etiquetas (por si difieren entre bases de datos)
CANDIDATOS_ETIQUETAS = ["Tags", "Etiquetas", "Etiqueta"]

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notion_cache.json")
REINDEX_INTERVAL_MINUTES = int(os.getenv("REINDEX_INTERVAL_MINUTES", "30"))
MAX_BLOCK_DEPTH = 3

NUM_FUENTES_NORMAL = int(os.getenv("NUM_FUENTES_NORMAL", "15"))
NUM_CANDIDATOS_AVANZADO = int(os.getenv("NUM_CANDIDATOS_AVANZADO", "60"))
MAX_RESULTADOS_AVANZADO = 20

ai_client = genai.Client(api_key=GEMINI_API_KEY)


# ------------------------------------------------------------------
# CACHÉ EN MEMORIA DEL WORKSPACE DE NOTION
# ------------------------------------------------------------------
class NotionCache:
    def __init__(self):
        self.pages = []       # lista de dicts: id, titulo, titulo_norm, url, tags, tags_norm, db_id, db_nombre, contenido, contenido_norm
        self.databases = {}   # db_id -> nombre
        self.last_updated = None
        self.building = False
        self.lock = Lock()

    def snapshot(self):
        with self.lock:
            return list(self.pages), dict(self.databases), self.last_updated

    def replace(self, pages, databases):
        with self.lock:
            self.pages = pages
            self.databases = databases
            self.last_updated = datetime.now(timezone.utc)


cache = NotionCache()


def normalizar(texto: str) -> str:
    """minúsculas + sin acentos, para comparar sin depender de mayúsculas/tildes"""
    if not texto:
        return ""
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }


def listar_bases_de_datos() -> dict:
    """{db_id: nombre} de todas las bases de datos compartidas con la integración."""
    headers = notion_headers()
    resultado = {}
    cursor = None
    while True:
        payload = {"filter": {"value": "database", "property": "object"}, "page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        res = requests.post("https://api.notion.com/v1/search", headers=headers, json=payload, timeout=10)
        if res.status_code != 200:
            break
        data = res.json()
        for item in data.get("results", []):
            titulo = "".join(t.get("plain_text", "") for t in item.get("title", []))
            resultado[item["id"]] = titulo
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return resultado


def extraer_tags_de_propiedades(properties: dict) -> list:
    for candidato in CANDIDATOS_ETIQUETAS:
        prop = properties.get(candidato)
        if not prop:
            continue
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
    """Recorre recursivamente los bloques hijos y concatena su texto (incluye toggles, listas anidadas, etc.)."""
    if profundidad > MAX_BLOCK_DEPTH:
        return ""
    textos = []
    cursor = None
    while True:
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        params = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        try:
            res = requests.get(url, headers=headers, params=params, timeout=8)
        except Exception:
            break
        if res.status_code != 200:
            break
        data = res.json()
        for b in data.get("results", []):
            b_type = b.get("type")
            contenido_bloque = b.get(b_type, {})
            if isinstance(contenido_bloque, dict) and "rich_text" in contenido_bloque:
                for segmento in contenido_bloque["rich_text"]:
                    textos.append(segmento.get("plain_text", ""))
            if b.get("has_children"):
                textos.append(obtener_texto_bloques(b["id"], headers, profundidad + 1))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return "\n".join(t for t in textos if t)


def indexar_pagina(item, headers, db_nombre_por_id):
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
    }


def construir_indice_sync():
    """Recorre TODO el workspace compartido con la integración y reconstruye el índice local."""
    if not NOTION_TOKEN:
        return
    cache.building = True
    try:
        headers = notion_headers()
        db_nombre_por_id = listar_bases_de_datos()

        items = []
        cursor = None
        while True:
            payload = {"filter": {"value": "page", "property": "object"}, "page_size": 100}
            if cursor:
                payload["start_cursor"] = cursor
            res = requests.post("https://api.notion.com/v1/search", headers=headers, json=payload, timeout=10)
            if res.status_code != 200:
                break
            data = res.json()
            items.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

        pages = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futuros = [executor.submit(indexar_pagina, item, headers, db_nombre_por_id) for item in items]
            for f in concurrent.futures.as_completed(futuros):
                try:
                    pages.append(f.result())
                except Exception:
                    pass

        cache.replace(pages, db_nombre_por_id)
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "pages": pages,
                        "databases": db_nombre_por_id,
                        "last_updated": cache.last_updated.isoformat(),
                    },
                    fh,
                )
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
    finally:
        cache.building = False


def cargar_cache_de_disco() -> bool:
    if not os.path.exists(CACHE_PATH):
        return False
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cache.pages = data.get("pages", [])
        cache.databases = data.get("databases", {})
        lu = data.get("last_updated")
        cache.last_updated = datetime.fromisoformat(lu) if lu else None
        return bool(cache.pages)
    except Exception:
        return False


async def reindexar_periodicamente():
    while True:
        await asyncio.sleep(REINDEX_INTERVAL_MINUTES * 60)
        await asyncio.get_event_loop().run_in_executor(None, construir_indice_sync)


@app.on_event("startup")
async def al_arrancar():
    cargado = cargar_cache_de_disco()
    if not cargado:
        asyncio.get_event_loop().run_in_executor(None, construir_indice_sync)
    asyncio.create_task(reindexar_periodicamente())


@app.post("/api/reindex")
async def reindex_manual():
    if cache.building:
        return {"status": "ya_en_progreso"}
    asyncio.get_event_loop().run_in_executor(None, construir_indice_sync)
    return {"status": "iniciado"}


@app.get("/api/index-status")
async def index_status():
    pages, _, last_updated = cache.snapshot()
    return {
        "paginas_indexadas": len(pages),
        "ultima_actualizacion": last_updated.isoformat() if last_updated else None,
        "indexando": cache.building,
    }


# ------------------------------------------------------------------
# BÚSQUEDA LOCAL (sobre el índice, sin llamar a Notion)
# ------------------------------------------------------------------
def filtrar_por_bases_de_datos(pages, nombres_objetivo):
    if not nombres_objetivo:
        return pages
    objetivos_norm = [normalizar(n) for n in nombres_objetivo]
    resultado = []
    for p in pages:
        nombre_db_norm = normalizar(p.get("db_nombre", ""))
        if any(all(palabra in nombre_db_norm for palabra in obj.split()) for obj in objetivos_norm):
            resultado.append(p)
    return resultado


def puntuar_texto(pagina, palabras_clave):
    puntuacion = 0
    for palabra in palabras_clave:
        puntuacion += pagina["titulo_norm"].count(palabra) * 5
        puntuacion += pagina["contenido_norm"].count(palabra)
    return puntuacion


def buscar_por_texto(pages, texto_busqueda, limite):
    palabras_clave = [normalizar(w) for w in texto_busqueda.split() if len(w) > 2]
    if not palabras_clave:
        return pages[:limite]
    puntuadas = [(puntuar_texto(p, palabras_clave), p) for p in pages]
    puntuadas = [(s, p) for s, p in puntuadas if s > 0]
    puntuadas.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in puntuadas[:limite]]


def extraer_tags_y_logica(prompt: str):
    tags = re.findall(r"#(\w+)", prompt)
    prompt_semantico = re.sub(r"#\w+", "", prompt).strip()
    is_or = re.search(r"#\w+\s+o\s+#\w+", prompt.lower())
    logica = "or" if is_or else "and"
    return tags, logica, prompt_semantico


def filtrar_por_tags(pages, tags, logica):
    tags_norm = [normalizar(t) for t in tags]
    resultado = []
    for p in pages:
        coincidencias = [t in p["tags_norm"] for t in tags_norm]
        ok = any(coincidencias) if logica == "or" else all(coincidencias)
        if ok:
            resultado.append(p)
    return resultado


class UserQuery(BaseModel):
    prompt: str
    advanced_search: bool = False
    databases: list[str] = []


@app.post("/api/chat")
async def chat_gemini_notion(query: UserQuery):
    user_prompt = query.prompt.strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="El prompt está vacío.")

    pages_disponibles, _, _ = cache.snapshot()

    if not pages_disponibles:
        if cache.building:
            raise HTTPException(status_code=503, detail={"type": "index_building"})
        # No hay índice y no se está construyendo: intenta construirlo ahora (primer uso)
        await asyncio.get_event_loop().run_in_executor(None, construir_indice_sync)
        pages_disponibles, _, _ = cache.snapshot()
        if not pages_disponibles:
            raise HTTPException(status_code=503, detail={"type": "index_building"})

    pages_disponibles = filtrar_por_bases_de_datos(pages_disponibles, query.databases)

    # =========================================================
    # MODO BÚSQUEDA AVANZADA
    # =========================================================
    if query.advanced_search:
        tags, logica, prompt_semantico = extraer_tags_y_logica(user_prompt)

        candidatos = pages_disponibles
        if tags:
            candidatos = filtrar_por_tags(candidatos, tags, logica)

        if prompt_semantico:
            candidatos = buscar_por_texto(candidatos, prompt_semantico, NUM_CANDIDATOS_AVANZADO)
        else:
            candidatos = candidatos[:NUM_CANDIDATOS_AVANZADO]

        if not candidatos:
            return {
                "response": "No se encontraron notas en Notion que cumplan con esos filtros/etiquetas.",
                "sources": [],
            }

        gemini_user = f"INTENCIÓN DEL USUARIO: {user_prompt}\n\nNOTAS DISPONIBLES:\n"
        for i, nota in enumerate(candidatos):
            gemini_user += f"[{i + 1}] {nota['titulo']} - {nota['url']}\n"

        gemini_sys = (
            f"Eres el cerebro del Buscador Avanzado. A continuación tienes hasta {len(candidatos)} notas de Notion "
            "ya preseleccionadas por etiquetas y relevancia textual.\n"
            f"Tu tarea: analiza la intención del usuario y selecciona un MÁXIMO DE {MAX_RESULTADOS_AVANZADO} títulos "
            "que mejor respondan a lo que busca.\n"
            "INSTRUCCIÓN ESTRICTA: tu respuesta debe ser EXCLUSIVAMENTE una lista en formato Markdown numerado, con "
            "el título de cada nota como enlace clickeable hacia su URL. No añadas introducciones ni conclusiones."
        )

        try:
            chat = ai_client.chats.create(model=GEMINI_MODEL)
            gem_res = chat.send_message(f"{gemini_sys}\n\n{gemini_user}")

            response_text = "### 🔍 Índices de Búsqueda Avanzada:\n\n" + gem_res.text
            if len(candidatos) > MAX_RESULTADOS_AVANZADO:
                response_text += (
                    "\n\n⚠️ **Nota:** Hay más resultados de los mostrados. Afina tu búsqueda con más "
                    "etiquetas u otras palabras para acotar."
                )
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
        candidatos = buscar_por_texto(pages_disponibles, user_prompt, NUM_FUENTES_NORMAL)
        fuentes = [
            {"titulo": p["titulo"], "url": p["url"], "contenido": p["contenido"]}
            for p in candidatos
            if p["contenido"].strip()
        ]

        corpus_texto = []
        for idx, f in enumerate(fuentes):
            corpus_texto.append(
                f"--- FUENTE [{idx + 1}] ---\n"
                f"Título de la Nota: {f['titulo']}\n"
                f"URL de Notion: {f['url']}\n"
                f"Contenido:\n{f['contenido'][:6000]}"
            )
        texto_contexto = "\n\n".join(corpus_texto) if corpus_texto else "No se encontraron notas relevantes."

        system_prompt = (
            "Eres un asistente conectado al espacio de Notion del usuario.\n"
            "INSTRUCCIONES OBLIGATORIAS:\n"
            "1. Incluye abundantes citas textuales y directas basadas en el contenido de las fuentes.\n"
            "2. NO escribas nombres largos de notas dentro del texto redactado.\n"
            "3. Detrás de cada cita o afirmación importante, coloca un número de referencia correlativo entre "
            "paréntesis que sea un enlace Markdown apuntando a la fuente correspondiente: `([1](URL_DE_NOTION))`, etc.\n"
            "No inventes URLs; utiliza estrictamente las proporcionadas en cada fuente."
        )

        full_prompt = f"{system_prompt}\n\n{texto_contexto}\n\n--- PETICIÓN DEL USUARIO ---\n{user_prompt}"

        chat = ai_client.chats.create(model=GEMINI_MODEL)
        response = chat.send_message(full_prompt)

        return {"response": response.text, "sources": fuentes}

    except Exception as e:
        error_str = str(e)
        print("--- ERROR DETALLADO EN /api/chat ---")
        traceback.print_exc()
        if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
            match = re.search(r"retry in ([\d\.]+)s", error_str)
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
