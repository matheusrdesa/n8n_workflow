# FastAPI que extrai texto do PDF, aplica REGEX e usa Groq (fallback).
# pip install fastapi uvicorn pdfminer.six pydantic requests

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pdfminer.high_level import extract_text
import io, re, os, json, requests

app = FastAPI(title="PDF Extractor")

RE_OS_NUM    = re.compile(r'(?:OS|Ordem de Servi[cç]o)\s*(?:n[ºo]\s*)?(\d{3,})', re.I)
RE_DATA      = re.compile(r'\b(\d{2}/\d{2}/\d{4})\b')
RE_CNPJ      = re.compile(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b')
RE_VALOR     = re.compile(r'R\$\s*([\d\.\,]+)')
RE_SOLIC     = re.compile(r'(?:Solicitante|Respons[aá]vel)\s*[:\-]\s*(.+)', re.I)
RE_DESC      = re.compile(r'(?:Descri[cç][aã]o|Objeto)\s*[:\-]\s*(.+)', re.I)

def regex_extract(txt: str) -> dict:
    g = lambda m: m.group(1) if m else None
    return {
        "os_num":      g(RE_OS_NUM.search(txt)),
        "data":        g(RE_DATA.search(txt)),
        "cnpj":        g(RE_CNPJ.search(txt)),
        "valor_total": g(RE_VALOR.search(txt)),
        "solicitante": g(RE_SOLIC.search(txt)),
        "descricao":   g(RE_DESC.search(txt)),
    }

PROMPT = """Você é um extrator. A partir do texto abaixo de uma OS, retorne APENAS JSON:
{
  "os_num": string|null,
  "data": string|null,
  "cnpj": string|null,
  "valor_total": string|null,
  "solicitante": string|null,
  "descricao": string|null
}
Sem explicações. Use null quando não souber.

Texto:
<<<
{texto}
>>>"""

def groq_llm(texto: str, model="llama-3.1-8b-instruct", api_key=None):
    headers = {"Authorization": f"Bearer {api_key or os.getenv('GROQ_API_KEY','')}",
               "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role":"system","content":"Você extrai campos e responde somente em JSON válido."},
            {"role":"user","content": PROMPT.format(texto=texto[:15000])}
        ]
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

class ExtractResponse(BaseModel):
    os_num: str|None; data: str|None; cnpj: str|None; valor_total: str|None
    solicitante: str|None; descricao: str|None; engine: str

@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...), model: str = Form("llama-3.1-8b-instruct")):
    pdf_bytes = await file.read()
    text = extract_text(io.BytesIO(pdf_bytes)) or ""
    rx = regex_extract(text)
    need_llm = not rx.get("os_num") or not rx.get("descricao") or not rx.get("valor_total")
    if need_llm:
        try:
            llm_out = json.loads(groq_llm(text, model=model))
        except Exception:
            llm_out = {}
        for k in ["os_num","data","cnpj","valor_total","solicitante","descricao"]:
            rx[k] = rx.get(k) or llm_out.get(k)
        engine = "regex+groq"
    else:
        engine = "regex"
    return {**rx, "engine": engine}
