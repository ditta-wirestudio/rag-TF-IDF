# RAG-as-a-Service — with retrieval evals

A production-shaped RAG service: ingest docs, answer questions **with inline
citations**, and — the part most demos skip — a **retrieval eval dashboard** that
scores Precision@k, Recall@k, and MRR on a labelled test set.

> Anyone can wire an LLM to a vector DB. The differentiator is proving the
> retrieval is *accurate*. This repo measures it.

---

## Understanding RAG (and how this project implements it)

### The problem RAG solves

A large language model (LLM) like GPT-4 or Llama only knows what was in its
training data. That creates three problems for real applications:

1. **Knowledge cutoff** — it doesn't know anything after its training date.
2. **No private data** — it has never seen *your* docs, wiki, tickets, or code.
3. **Hallucination** — asked about something it doesn't know, it confidently
   makes up a plausible-sounding but wrong answer.

You could re-train or fine-tune the model on your data, but that's expensive,
slow, and goes stale the moment your docs change. **RAG (Retrieval-Augmented
Generation)** is the cheaper, faster answer: instead of putting your knowledge
*into* the model, you fetch the relevant pieces *at question time* and hand them
to the model as context. The model then answers from what you gave it — and can
cite exactly which source it used.

Analogy: a closed-book exam (plain LLM, answers from memory, sometimes bluffs)
vs. an open-book exam (RAG — look up the relevant page first, then answer, and
point to the page).

### The RAG pipeline, step by step

```
INGEST (once per document)
  document ──chunk──► small passages ──embed──► vectors ──store──► vector DB

QUERY (per question)
  question ──embed──► query vector
                        │
                        ▼
             similarity search over stored vectors
                        │
                        ▼
             top-k most relevant chunks  ──►  LLM  ──►  answer + [citations]
```

**1. Chunking.** Documents are too big to feed whole, and you want to retrieve
*precise* passages, not entire files. So each document is split into small
overlapping windows (~900 chars here, 150-char overlap so a sentence spanning a
boundary isn't lost). → `rag.py::chunk_text`

**2. Embedding.** Each chunk is converted into an **embedding**: a list of
numbers (a *vector*) that captures its meaning. The crucial property is that
texts with *similar meaning* get *similar vectors* — even if they use different
words ("ship to" ≈ "delivery regions"). This is what makes search *semantic*
rather than keyword-based. → `rag.py::embed`

**3. Storing.** The chunks + their vectors go into a vector store. In production
that's **Postgres with the `pgvector` extension**, which can index and search
millions of vectors fast. In offline/demo mode it's a plain Python list. → `db.py`, `rag.py::ingest`

**4. Retrieval.** When a question comes in, it's embedded the same way, then
compared against every stored vector using **cosine similarity** (how close two
vectors point in the same direction; 1.0 = identical meaning). The top *k*
closest chunks are returned. → `rag.py::retrieve`

**5. Generation.** Those chunks are stitched into a prompt with the question and
sent to the LLM, with a system instruction: *answer only from this context, cite
sources as [n], and if the answer isn't here, say so.* That instruction is what
suppresses hallucination — the model is grounded in retrieved facts. → `rag.py::answer`

**6. Citations.** Because we know exactly which chunks we sent, every answer can
point back to its sources. That traceability is the anti-hallucination proof and
the thing enterprise buyers care about most.

### Why retrieval evals are the whole point

Here's the insight most tutorials miss: **RAG quality is mostly retrieval
quality.** If step 4 fetches the wrong chunks, even the best LLM produces a wrong
or "I don't know" answer — garbage in, garbage out. Yet almost every RAG demo
ships with *zero measurement* of whether retrieval actually works.

This project measures it. `eval_data/testset.json` is a labelled set: each
question is tagged with the source file(s) that *should* be retrieved. The eval
harness (`evals.py`) runs retrieval for every question and computes:

- **Precision@k** — of the *k* chunks retrieved, what fraction were relevant.
  (Are we pulling in junk alongside the good stuff?)
- **Recall@k** — of all the relevant sources, what fraction we actually retrieved.
  (Are we missing the docs that hold the answer?)
- **MRR (Mean Reciprocal Rank)** — how *high* the first correct source ranked
  (1.0 = always #1, 0.5 = typically #2, …). (Is the right answer at the top?)

These numbers turn "the RAG feels okay" into "retrieval scores 0.9 recall on our
test set" — the difference between a toy and something you'd ship. Change your
chunk size, embedding model, or add reranking, re-run the evals, and *see* whether
it helped. That's engineering, not vibes.

### How this specific project is built

| Concept | Where it lives | Notes |
|---|---|---|
| Chunking | `rag.py::chunk_text` | paragraph-aware sliding window |
| Embeddings | `rag.py::embed` | OpenAI, or offline hashing (no key) |
| Vector store | `db.py` + `rag.py` | Postgres+pgvector, or in-memory |
| Retrieval | `rag.py::retrieve` | cosine similarity, top-k |
| Generation | `rag.py::answer` | OpenAI or Groq, grounded + cited |
| Evals | `evals.py` | Precision@k / Recall@k / MRR |
| API + dashboard | `main.py` | FastAPI endpoints + eval UI |
| Config | `settings.py` | all env-driven |

**Two design decisions worth calling out:**

1. **Dual backend (offline ↔ production).** With no API key and no database, it
   runs on a local hashing embedding + in-memory store, so *anyone* can boot it
   instantly. Set `OPENAI_API_KEY` + `STORE=postgres` and the exact same code path
   uses real embeddings + pgvector. The offline mode is honest about its limits
   (crude embeddings → modest eval scores) but makes the repo runnable in 20s.

2. **Chat and embeddings are decoupled providers.** Answer generation (chat) and
   retrieval (embeddings) are separate concerns, so you can mix providers — e.g.
   **Groq for fast/cheap answers** (it's chat-only, no embeddings endpoint) while
   embeddings run offline or on OpenAI. See the provider table below.

---

## Stack

FastAPI · Postgres + pgvector · OpenAI embeddings/chat (swappable) · vanilla-JS dashboard

## Instant demo — no Postgres, no API key

The service runs in **offline mode** (in-memory store + local hashing embeddings)
so a reviewer can boot it in 20 seconds with nothing installed but Python:

```bash
pip install fastapi uvicorn
cd backend
STORE=memory uvicorn main:app --reload      # sample docs auto-load
```

Open http://127.0.0.1:8000/ and click **Run evals**. Try a query:

```bash
curl -XPOST localhost:8000/query -H 'content-type: application/json' \
     -d '{"query":"which regions do you ship to?","k":3}'
# -> answer + citations back to shipping.md
```

Set `OPENAI_API_KEY` and `STORE=postgres` (below) for the real embeddings + pgvector path.

## Production start (OpenAI + pgvector)

**The only extra dependency is Postgres with the `pgvector` extension.** Two ways:

### Option A — Docker (recommended, one command)

Brings up Postgres+pgvector **and** the API. You only need Docker installed.

```bash
cp .env.example .env            # add OPENAI_API_KEY (optional; blank = offline embeddings)
docker compose up --build
```

- API + dashboard: http://127.0.0.1:8000/
- Postgres (pgvector) on `localhost:5432` (user/pass/db = `rag`/`rag`/`ragdemo`)

### Option B — native macOS

```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
createdb ragdemo
psql -d ragdemo -c "CREATE EXTENSION IF NOT EXISTS vector;"

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set STORE=postgres, DATABASE_URL, OPENAI_API_KEY

cd backend
python db.py                    # create tables + index
python ingest_folder.py ../sample_docs
uvicorn main:app --reload
```

Query it:
```bash
curl -XPOST localhost:8000/query -H 'content-type: application/json' \
     -d '{"query":"what is the return window?"}'
```

## LLM provider (answers) — OpenAI or Groq

Chat and embeddings are **decoupled**. Groq is OpenAI-compatible for *chat only*
(it has no embeddings endpoint), so it generates the cited answers while
embeddings come from OpenAI (if you set `OPENAI_API_KEY`) or the offline model.

Use your **Groq key** for fast, cheap answer generation:

```bash
# in .env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
# (no OPENAI_API_KEY -> retrieval uses offline embeddings, answers use Groq)
```

`GET /healthz` reports both: `{"llm":"groq","embeddings":"offline"}`.

| `LLM_PROVIDER` | Answers by | Embeddings by |
|---|---|---|
| `groq` | Groq (your key) | OpenAI if key set, else offline |
| `openai` | OpenAI | OpenAI |
| `none` | offline extractive (top chunk) | offline |
| `auto` (default) | Groq if its key set, else OpenAI, else none | OpenAI if key set, else offline |

## Embeddings provider (retrieval) — OpenAI, local, or offline

Retrieval quality depends on the embedding model. Three options, auto-selected:

| `EMBED_PROVIDER` | Model | Cost | Quality |
|---|---|---|---|
| `openai` | `text-embedding-3-small` (1536-d) | API (cheap) | best |
| `fastembed` | `BAAI/bge-small-en-v1.5` (384-d), **runs locally** | **free** | strong |
| `hashing` | deterministic bag-of-words (256-d) | free | crude (demo only) |
| `auto` (default) | openai if key → else fastembed if installed → else hashing | — | — |

**Best pairing for a Groq-only setup:** `LLM_PROVIDER=groq` + `EMBED_PROVIDER=fastembed`
→ great answers from Groq, strong retrieval from a free on-device model, **zero
embedding API cost**. `fastembed` downloads a ~50 MB model on first run.

> **pgvector note:** the vector column dimension must match the embedder. The
> schema uses it automatically (`settings.active_embed_dim`): openai=1536,
> fastembed=384, hashing=256. If you switch embedders on an existing Postgres DB,
> drop/recreate the `chunks` table so the dimension matches.

## Answering questions from your own PDF (or docs)

RAG doesn't "train" — you **ingest** a document (chunk → embed → store) and it's
instantly queryable. Send a PDF, ask about it seconds later. Three ways:

**1. Upload in the browser.** Open `/`, use the **Add a document** box, pick a
`.pdf` / `.md` / `.txt`, click *Upload & index*, then ask in the *Ask* box.

**2. API (multipart upload):**
```bash
curl -F "file=@/path/to/handbook.pdf" localhost:8000/ingest-file
curl -XPOST localhost:8000/query -H 'content-type: application/json' \
     -d '{"query":"what does the handbook say about refunds?"}'
```

**3. Bulk-ingest a folder of PDFs:**
```bash
cd backend && python ingest_folder.py ~/Documents/my_pdfs
```

Notes:
- Text-based PDFs work out of the box (via `pypdf`). **Scanned/image-only PDFs**
  have no embedded text and would need OCR (not included).
- Big PDF? It's chunked automatically; retrieval returns only the relevant
  passages, and answers cite the page-region they came from.
- For good answers from real PDFs, run with **fastembed** (retrieval) + **Groq or
  OpenAI** (generation) — hashing/offline mode is demo-only.

## Production features

- **Config** — all tunables in `settings.py` (pydantic-settings), read from env/`.env`.
- **Connection pooling** — `psycopg_pool`, opened on startup, closed on shutdown (lifespan).
- **Resilient embeddings** — batched (100/call) with exponential-backoff retries.
- **Auth** — set `API_KEY` to require `X-API-Key` on `/ingest` and eval-run (writes).
- **CORS**, **request-size limits**, **input validation** (Pydantic field constraints).
- **`/healthz`** — liveness + DB probe (used by the Docker healthcheck).
- **Structured logging**, graceful 422/401/503 errors.
- **Tests** — `cd backend && pytest` (runs fully offline, no DB/key needed).

## Testing

```bash
cd backend && pytest -q      # 7 tests, offline, ~0.2s
```

## What to show in the demo / teardown post

1. Open `/`, click **Run evals** → Precision@k / Recall@k / MRR light up.
2. Hit `/query` and show the answer **with `[1] [2]` citations** back to sources.
3. The teardown line: *"Most RAG demos have no evals. Here's mine, measured."*

## How the eval works

`eval_data/testset.json` maps each question to the source file(s) that *should*
be retrieved. `evals.py` runs retrieval for each, checks whether the right sources
came back in the top-k, and averages:

- **Precision@k** — of the k retrieved, how many were relevant.
- **Recall@k** — of the relevant sources, how many were retrieved.
- **MRR** — how high the first relevant hit ranked.

Add your own questions to the test set to benchmark on real data.

## Layout

```
backend/
  settings.py       pydantic-settings config (env-driven)
  db.py             pgvector schema + pooled connections + healthcheck
  rag.py            chunk, embed (openai|offline), retrieve, answer — dual backend
  evals.py          Precision@k / Recall@k / MRR harness
  ingest_folder.py  bulk-ingest .md/.txt
  main.py           FastAPI: /ingest /query /api/evals /healthz + dashboard
  tests/            offline pytest suite
eval_data/testset.json   labelled questions -> expected sources
sample_docs/             demo corpus (returns, faq, shipping, api)
Dockerfile, docker-compose.yml   Postgres+pgvector + API, one command
```

## Config (env)

| Var | Default | Purpose |
|-----|---------|---------|
| `STORE` | `auto` | `auto` / `postgres` / `memory` |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/ragdemo` | Postgres DSN |
| `LLM_PROVIDER` | `auto` | `auto` / `openai` / `groq` / `none` |
| `GROQ_API_KEY` | *(empty)* | Groq key → answers generated by Groq |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `OPENAI_API_KEY` | *(empty)* | OpenAI chat and/or embeddings |
| `EMBED_PROVIDER` | `auto` | `auto` / `openai` / `fastembed` / `hashing` |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | free local embedding model |
| `EMBED_DIM` | `1536` | openai dim; fastembed=384 auto-applied |
| `API_KEY` | *(empty)* | if set, required on write endpoints |