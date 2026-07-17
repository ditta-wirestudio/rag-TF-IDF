"""Core RAG: chunk, embed, store, retrieve-with-citations, answer.

Two interchangeable backends, chosen by config — so the repo runs anywhere:

  Embeddings:  OpenAI (OPENAI_API_KEY set)  ||  offline hashing embeddings
  Vector store: Postgres+pgvector (STORE=postgres)  ||  in-memory (STORE=memory)

Default STORE=auto uses Postgres when DATABASE_URL + psycopg are available, else
memory. The offline path lets anyone run with zero external services; production
uses OpenAI + pgvector.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time

from settings import settings

log = logging.getLogger("rag")

_embed_client = None            # OpenAI client for embeddings (OpenAI only)
_chat_client = None             # OpenAI-compatible client for chat (OpenAI or Groq)
_fastembed = None               # local fastembed model (lazy, downloads once)
_mem: list[dict] = []           # in-memory store (offline/demo backend)
_doc_seq = 0
_chunk_seq = 0


# ---------------------------------------------------------------- backend selection
_backend: str | None = None     # cached "memory" | "postgres" (resolved once)


def _resolve_backend() -> str:
    if settings.store == "memory":
        return "memory"
    if settings.store == "postgres":
        return "postgres"                       # explicit: fail loudly if DB is down
    # auto: use Postgres only if the driver is present AND the DB is reachable,
    # otherwise fall back to the in-memory store so the app still boots.
    if not settings.database_url:
        return "memory"
    try:
        import psycopg
        import psycopg_pool  # noqa: F401
    except Exception:
        return "memory"
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return "postgres"
    except Exception as e:                       # noqa: BLE001
        log.warning("auto: Postgres unreachable (%s); using in-memory store. "
                    "Set STORE=postgres to require the DB, or start it.", e)
        return "memory"


def use_memory() -> bool:
    global _backend
    if _backend is None:
        _backend = _resolve_backend()
    return _backend == "memory"


def embed_client():
    """OpenAI client for embeddings (Groq has no embeddings endpoint)."""
    global _embed_client
    if _embed_client is None:
        from openai import OpenAI
        _embed_client = OpenAI(api_key=settings.openai_api_key, max_retries=0)
    return _embed_client


def chat_client():
    """OpenAI-compatible client for the active chat provider (OpenAI or Groq)."""
    global _chat_client
    if _chat_client is None:
        from openai import OpenAI
        cfg = settings.chat_config
        _chat_client = OpenAI(api_key=cfg["api_key"],
                              base_url=cfg["base_url"], max_retries=0)
    return _chat_client


# ---------------------------------------------------------------- chunking
def chunk_text(text: str) -> list[str]:
    """Paragraph-aware sliding window. Cheap, deterministic."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    cc, ov = settings.chunk_chars, settings.chunk_overlap
    chunks, start = [], 0
    while start < len(text):
        end = start + cc
        window = text[start:end]
        brk = max(window.rfind("\n\n"), window.rfind(". "))
        if brk > cc // 2 and end < len(text):
            end = start + brk + 1
        chunks.append(text[start:end].strip())
        start = max(end - ov, end)
    return [c for c in chunks if c]


# ---------------------------------------------------------------- embeddings
def _offline_embed(text: str) -> list[float]:
    """Deterministic hashing embedding (tf-weighted bag-of-words, L2-normed).
    Cosine then reflects lexical overlap — no API key needed."""
    dim = settings.offline_dim
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _with_retry(fn, *args):
    """Exponential backoff around a flaky network call."""
    last = None
    for attempt in range(settings.max_retries):
        try:
            return fn(*args)
        except Exception as e:                       # noqa: BLE001 - retry any transient error
            last = e
            delay = settings.retry_base_delay * (2 ** attempt)
            log.warning("embedding call failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1, settings.max_retries, e, delay)
            time.sleep(delay)
    raise RuntimeError(f"embedding failed after {settings.max_retries} retries") from last


def fastembed_model():
    """Lazy-load the local fastembed model (downloads ~50MB on first use)."""
    global _fastembed
    if _fastembed is None:
        from fastembed import TextEmbedding
        log.info("loading fastembed model %s (first run downloads it)…",
                 settings.fastembed_model)
        _fastembed = TextEmbedding(model_name=settings.fastembed_model)
    return _fastembed


def embed(texts: list[str]) -> list[list[float]]:
    provider = settings.resolved_embed

    if provider == "openai":
        out: list[list[float]] = []
        bs = settings.embed_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            resp = _with_retry(
                lambda b=batch: embed_client().embeddings.create(
                    model=settings.embed_model, input=b))
            out.extend(d.embedding for d in resp.data)
        return out

    if provider == "fastembed":
        # local on-device model; returns numpy arrays -> plain lists
        return [list(map(float, v)) for v in fastembed_model().embed(texts)]

    return [_offline_embed(t) for t in texts]      # hashing fallback


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))          # both L2-normalized offline


# ---------------------------------------------------------------- ingest
def ingest(source: str, text: str, title: str | None = None) -> dict:
    if not source or not text.strip():
        raise ValueError("source and text are required")
    if len(text) > settings.max_ingest_chars:
        raise ValueError(f"document exceeds max_ingest_chars ({settings.max_ingest_chars})")

    chunks = chunk_text(text)
    vectors = embed(chunks)

    if use_memory():
        global _doc_seq, _chunk_seq
        _doc_seq += 1
        doc_id = _doc_seq
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            _chunk_seq += 1
            _mem.append({"chunk_id": _chunk_seq, "document_id": doc_id,
                         "source": source, "title": title or source,
                         "content": chunk, "chunk_index": i, "vec": vec})
        log.info("ingested %s: %d chunks [memory]", source, len(chunks))
        return {"document_id": doc_id, "chunks": len(chunks), "store": "memory"}

    from db import connection
    with connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO documents (source, title) VALUES (%s, %s) RETURNING id",
                    (source, title or source))
        doc_id = cur.fetchone()["id"]
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s)",
            [(doc_id, i, c, str(v)) for i, (c, v) in enumerate(zip(chunks, vectors))])
        conn.commit()
    log.info("ingested %s: %d chunks [postgres]", source, len(chunks))
    return {"document_id": doc_id, "chunks": len(chunks), "store": "postgres"}


# ---------------------------------------------------------------- retrieval
def retrieve(query: str, k: int | None = None) -> list[dict]:
    k = k or settings.default_k
    qvec = embed([query])[0]
    if use_memory():
        scored = sorted(({**c, "score": _cosine(qvec, c["vec"])} for c in _mem),
                        key=lambda x: x["score"], reverse=True)
        return [{"chunk_id": c["chunk_id"], "document_id": c["document_id"],
                 "content": c["content"], "source": c["source"],
                 "title": c["title"], "score": c["score"]} for c in scored[:k]]

    from db import connection
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT c.id AS chunk_id, c.document_id, c.content, d.source, d.title,
                      1 - (c.embedding <=> %s::vector) AS score
               FROM chunks c JOIN documents d ON d.id = c.document_id
               ORDER BY c.embedding <=> %s::vector LIMIT %s""",
            (str(qvec), str(qvec), k))
        return cur.fetchall()


# ---------------------------------------------------------------- listing
def list_documents() -> list[dict]:
    """Distinct indexed sources with their chunk counts (newest first)."""
    if use_memory():
        agg: dict[str, int] = {}
        for c in _mem:
            agg[c["source"]] = agg.get(c["source"], 0) + 1
        return [{"source": s, "chunks": n} for s, n in agg.items()]

    from db import connection
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT d.source, count(c.id) AS chunks
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               GROUP BY d.source ORDER BY max(d.created_at) DESC""")
        return cur.fetchall()


# ---------------------------------------------------------------- answer
SYSTEM = (
    "You answer strictly from the provided context. Cite sources inline as [n] "
    "using the given numbers. If the context does not contain the answer, say so. "
    "Never invent facts not in the context."
)


def answer(query: str, k: int | None = None) -> dict:
    k = k or settings.default_k
    hits = retrieve(query, k)
    if not hits:
        return {"answer": "No documents indexed yet.", "citations": []}
    citations = [
        {"n": i + 1, "source": h["source"], "title": h["title"],
         "chunk_id": h["chunk_id"], "score": round(h["score"], 3)}
        for i, h in enumerate(hits)
    ]
    if not settings.chat_enabled:
        return {"answer": f"{hits[0]['content']} [1]",
                "citations": citations, "mode": "offline-extractive"}

    context = "\n\n".join(
        f"[{i+1}] (source: {h['source']}) {h['content']}" for i, h in enumerate(hits))
    resp = _with_retry(lambda: chat_client().chat.completions.create(
        model=settings.chat_config["model"],
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
        temperature=0))
    return {"answer": resp.choices[0].message.content,
            "citations": citations, "mode": settings.resolved_llm}
