"""FastAPI app: ingest, query-with-citations, and the eval dashboard.

    uvicorn main:app --reload   (run from backend/)

Production concerns handled here: lifespan startup/shutdown, CORS, optional
API-key auth on write endpoints, request-size limits, structured logging,
and a /healthz probe.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import evals
import rag
from settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ACTIVE CONFIG → store=%s | embeddings=%s | llm=%s",
             "memory" if rag.use_memory() else "postgres",
             settings.resolved_embed, settings.resolved_llm)
    if settings.resolved_embed == "hashing":
        log.warning("embeddings=hashing is DEMO-ONLY (poor retrieval). "
                    "Install fastembed or set OPENAI_API_KEY for real quality.")
    if settings.resolved_embed == "fastembed":
        # warm up at startup — first load downloads the model (~30s); without
        # this the first /query request hangs on "Thinking…" for that long
        rag.fastembed_model()
    if not rag.use_memory():
        from db import init_schema
        if init_schema():
            n = rag.reembed_all()
            log.info("re-embedded %d chunks after embedding-dimension change", n)
        log.info("postgres schema ready")
    # seed the sample corpus if the store is empty, so the app (and its evals)
    # are useful immediately — applies to both memory and a fresh Postgres
    if settings.seed_sample_docs and not rag.list_documents():
        docs = Path(__file__).parent.parent / "sample_docs"
        for p in sorted(docs.glob("*.md")):
            rag.ingest(source=p.name, text=p.read_text(), title=p.stem)
        log.info("seeded sample corpus [%s]",
                 "memory" if rag.use_memory() else "postgres")
    # make sure the dashboard has eval numbers on first visit (last_run.json
    # does not survive container rebuilds)
    if evals.latest() is None:
        try:
            out = evals.run()
            if out.get("metrics"):
                log.info("startup eval run: %s", out["metrics"])
        except Exception as e:                                    # noqa: BLE001
            log.warning("startup eval run failed: %s", e)
    yield
    if not rag.use_memory():
        from db import close_pool
        close_pool()


app = FastAPI(title="RAG-as-a-Service + Eval Dashboard", version="1.0.0",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_list,
    allow_methods=["*"], allow_headers=["*"])


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op if settings.api_key is unset; otherwise enforce it on writes."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class IngestBody(BaseModel):
    source: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=512)


class QueryBody(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=settings.default_k, ge=1, le=50)


@app.get("/healthz")
def healthz():
    ok = True
    if not rag.use_memory():
        try:
            from db import healthcheck
            ok = healthcheck()
        except Exception as e:                                    # noqa: BLE001
            raise HTTPException(503, f"db unavailable: {e}")
    return {"status": "ok", "store": "memory" if rag.use_memory() else "postgres",
            "embeddings": settings.resolved_embed,
            "llm": settings.resolved_llm}


@app.post("/ingest", dependencies=[Depends(require_api_key)])
def ingest(body: IngestBody):
    try:
        return rag.ingest(body.source, body.text, body.title)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/ingest-file", dependencies=[Depends(require_api_key)])
async def ingest_file(file: UploadFile = File(...)):
    """Upload a PDF / .md / .txt, extract its text, and index it."""
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"file exceeds {settings.max_upload_bytes} bytes")
    from loaders import extract_text
    try:
        text = extract_text(data, file.filename)
    except Exception as e:                                        # noqa: BLE001
        raise HTTPException(422, f"could not read file: {e}")
    if not text.strip():
        raise HTTPException(422, "no extractable text (a scanned/image PDF needs OCR)")
    try:
        return rag.ingest(source=file.filename, text=text, title=file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/query")
def query(body: QueryBody):
    return rag.answer(body.query, body.k)


@app.get("/api/documents")
def documents():
    return rag.list_documents()


@app.get("/api/evals")
def get_evals():
    return evals.latest() or {"message": "no eval run yet — POST /api/evals/run"}


@app.post("/api/evals/run", dependencies=[Depends(require_api_key)])
def run_evals():
    return evals.run()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>RAG Eval Dashboard</title>
<style>
 body{font:15px/1.5 -apple-system,Segoe UI,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#111}
 h1{font-size:1.4rem} .cards{display:flex;gap:1rem;margin:1.5rem 0;flex-wrap:wrap}
 .card{flex:1;min-width:150px;border:1px solid #eee;border-radius:10px;padding:1rem}
 .card .n{font-size:2rem;font-weight:700} .card .l{color:#888;font-size:.85rem;text-transform:uppercase}
 table{width:100%;border-collapse:collapse;margin-top:1rem} th,td{text-align:left;padding:.5rem;border-bottom:1px solid #eee}
 .ok{color:#0a7d33;font-weight:700} .bad{color:#c0392b;font-weight:700}
 button{font:inherit;padding:.5rem 1rem;border:1px solid #0b62d6;background:#0b62d6;color:#fff;border-radius:8px;cursor:pointer}
</style></head><body>
<h1>RAG — Ask &amp; Evals</h1>

<section style="border:1px solid #eee;border-radius:10px;padding:1rem 1.25rem;margin-bottom:1.25rem">
 <h2 style="font-size:1.1rem;margin:.2rem 0 .8rem">Add a document</h2>
 <div style="display:flex;gap:.5rem;align-items:center">
   <input type="file" id="file" accept=".pdf,.txt,.md" style="flex:1">
   <button onclick="upload()">Upload &amp; index</button>
 </div>
 <div id="uploadmsg" style="margin-top:.6rem;color:#555;font-size:.9rem"></div>
 <div id="doclist" style="margin-top:.8rem"></div>
</section>

<section style="border:1px solid #eee;border-radius:10px;padding:1rem 1.25rem;margin-bottom:2rem">
 <h2 style="font-size:1.1rem;margin:.2rem 0 .8rem">Ask a question</h2>
 <div style="display:flex;gap:.5rem">
   <input id="q" placeholder="e.g. what is the return window?"
          style="flex:1;padding:.6rem;border:1px solid #ccc;border-radius:8px;font:inherit"
          onkeydown="if(event.key==='Enter')ask()">
   <button onclick="ask()">Ask</button>
 </div>
 <div id="answer" style="margin-top:1rem"></div>
</section>

<h2 style="font-size:1.1rem">Retrieval evals</h2>
<p style="color:#666;margin-top:.2rem">Precision@k / Recall@k / MRR over a labelled test set.
 Most RAG demos ship with zero evals — this one is measured.</p>
<button onclick="runEvals()">Run evals now</button>
<div class="cards" id="cards"></div>
<table id="detail"><thead><tr><th>Question</th><th>Hit?</th><th>Rank</th><th>Top score</th></tr></thead><tbody></tbody></table>
<script>
async function loadDocs(){
 const r=await fetch('/api/documents'); const docs=await r.json();
 const el=document.getElementById('doclist');
 if(!docs.length){el.innerHTML='<span style="color:#999;font-size:.85rem">No documents indexed yet.</span>';return;}
 el.innerHTML='<b style="font-size:.9rem">Indexed documents:</b><ul style="margin:.4rem 0 0;padding-left:1.1rem">'+
   docs.map(d=>`<li style="font-size:.9rem">${d.source} <span style="color:#999">· ${d.chunks} chunks</span></li>`).join('')+'</ul>';
}
async function upload(){
 const f=document.getElementById('file').files[0];
 const msg=document.getElementById('uploadmsg');
 if(!f){msg.textContent='Pick a file first.';return;}
 msg.textContent='Indexing '+f.name+'…';
 const fd=new FormData(); fd.append('file', f);
 const r=await fetch('/ingest-file',{method:'POST',body:fd});
 const d=await r.json();
 msg.textContent = r.ok
   ? `Indexed “${f.name}” — ${d.chunks} chunks. Ask a question about it below.`
   : ('Error: '+(d.detail||'upload failed'));
 if(r.ok) loadDocs();
}
async function ask(){
 const q=document.getElementById('q').value.trim(); if(!q)return;
 const box=document.getElementById('answer'); box.innerHTML='<em>Thinking…</em>';
 try{
   const r=await fetch('/query',{method:'POST',headers:{'content-type':'application/json'},
     body:JSON.stringify({query:q,k:3})});
   const d=await r.json();
   if(!r.ok){
     box.innerHTML='<span class="bad">Error: '+(d.detail||('HTTP '+r.status))+'</span>';
     return;
   }
   const cites=(d.citations||[]).map(c=>
     `<span style="display:inline-block;background:#eef4ff;border-radius:6px;padding:.15rem .5rem;margin:.15rem .25rem .15rem 0;font-size:.85rem">[${c.n}] ${c.source} · ${c.score}</span>`).join('');
   box.innerHTML=`<div style="color:#666;font-size:.9rem;margin-bottom:.4rem">Q: ${q}</div>
     <div style="white-space:pre-wrap;background:#fafafa;border-radius:8px;padding:.8rem">${d.answer}</div>
     <div style="margin-top:.5rem"><b>Sources:</b><br>${cites||'—'}</div>
     ${d.mode?`<div style="color:#999;font-size:.8rem;margin-top:.4rem">mode: ${d.mode}</div>`:''}`;
 }catch(e){box.innerHTML='<span class="bad">Request failed: '+e+'</span>';}
}
function renderEvals(d){
 const cards=document.getElementById('cards');
 const tb=document.querySelector('#detail tbody'); tb.innerHTML='';
 if(!d.metrics){
   cards.innerHTML='<p>No eval run yet on this server. Click “Run evals now”.</p>';
   tb.innerHTML='<tr><td colspan="4" style="color:#999">No eval data yet — run evals to populate this table.</td></tr>';
   return;
 }
 const m=d.metrics;
 cards.innerHTML=`
   <div class="card"><div class="n">${(m.precision_at_k*100).toFixed(0)}%</div><div class="l">Precision@${m.k}</div></div>
   <div class="card"><div class="n">${(m.recall_at_k*100).toFixed(0)}%</div><div class="l">Recall@${m.k}</div></div>
   <div class="card"><div class="n">${m.mrr.toFixed(2)}</div><div class="l">MRR</div></div>
   <div class="card"><div class="n">${m.n}</div><div class="l">Test questions</div></div>`;
 d.details.forEach(x=>tb.insertAdjacentHTML('beforeend',
   `<tr><td>${x.question??'—'}</td>
     <td class="${x.hit?'ok':'bad'}">${x.hit?'yes':'no'}</td>
     <td>${x.rank||'-'}</td><td>${x.top_score??'-'}</td></tr>`));
}
async function load(){
 try{
   const r=await fetch('/api/evals'); renderEvals(await r.json());
 }catch(e){document.getElementById('cards').innerHTML='<p class="bad">Could not load evals: '+e+'</p>';}
}
async function runEvals(){
 const cards=document.getElementById('cards');
 cards.innerHTML='<p>Running…</p>';
 try{
   const r=await fetch('/api/evals/run',{method:'POST'});
   const d=await r.json();
   if(!r.ok||d.error){
     cards.innerHTML='<p class="bad">Eval run failed: '+(d.detail||d.error||('HTTP '+r.status))+'</p>';
     return;
   }
   renderEvals(d);
 }catch(e){cards.innerHTML='<p class="bad">Eval run failed: '+e+'</p>';}
}
load(); loadDocs();
</script></body></html>"""
