"""Offline test suite — no Postgres, no API key. `pytest` from backend/.

Forces STORE=memory so the whole pipeline (chunk -> embed -> retrieve -> eval)
is exercised deterministically in-process.
"""
import os

os.environ["STORE"] = "memory"
os.environ["LLM_PROVIDER"] = "none"
os.environ["EMBED_PROVIDER"] = "hashing"     # keep tests offline + fast (no model download)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)

import evals  # noqa: E402
import rag  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402


def setup_function(_):
    rag._mem.clear()
    rag._doc_seq = 0
    rag._chunk_seq = 0


def test_chunking_overlaps_and_splits():
    text = ("alpha. " * 200) + "\n\n" + ("beta. " * 200)
    chunks = rag.chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= rag.settings.chunk_chars + 5 for c in chunks)


def test_offline_embed_is_deterministic_and_normalized():
    v1 = rag._offline_embed("hello world")
    v2 = rag._offline_embed("hello world")
    assert v1 == v2
    assert abs(sum(x * x for x in v1) ** 0.5 - 1.0) < 1e-6


def test_ingest_and_retrieve_ranks_relevant_doc():
    rag.ingest("shipping.md", "We ship to the US, Canada, the UK and the EU.")
    rag.ingest("returns.md", "Returns accepted within 30 days for a full refund.")
    hits = rag.retrieve("which countries do you ship to?", k=2)
    assert hits[0]["source"] == "shipping.md"
    assert hits[0]["score"] >= hits[1]["score"]


def test_answer_returns_citations():
    rag.ingest("api.md", "The API is rate limited to 100 requests per minute.")
    out = rag.answer("what is the rate limit?", k=1)
    assert out["citations"]
    assert out["citations"][0]["source"] == "api.md"


def test_ingest_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        rag.ingest("x.md", "   ")


def test_eval_metrics_shape():
    for src, txt in [("returns-policy.md", "Returns within 30 days."),
                     ("account-faq.md", "Reset your password from the login screen."),
                     ("shipping.md", "We ship to the US and EU."),
                     ("api-docs.md", "Rate limited to 100 requests per minute.")]:
        rag.ingest(src, txt)
    res = evals.run()
    m = res["metrics"]
    assert set(m) >= {"k", "n", "precision_at_k", "recall_at_k", "mrr"}
    assert 0.0 <= m["recall_at_k"] <= 1.0
    assert 0.0 <= m["mrr"] <= 1.0


def test_http_endpoints():
    with TestClient(app) as c:
        assert c.get("/healthz").json()["status"] == "ok"
        r = c.post("/query", json={"query": "shipping regions", "k": 3})
        assert r.status_code == 200
        assert "citations" in r.json()
        assert c.post("/api/evals/run").status_code == 200


def test_ingest_file_upload_and_query():
    """Upload a document via multipart, then answer a question from it."""
    with TestClient(app) as c:
        files = {"file": ("policy.md",
                          b"Refunds are issued within 14 business days of approval.",
                          "text/markdown")}
        r = c.post("/ingest-file", files=files)
        assert r.status_code == 200, r.text
        assert r.json()["chunks"] >= 1
        a = c.post("/query", json={"query": "how long do refunds take?", "k": 2}).json()
        assert any(cit["source"] == "policy.md" for cit in a["citations"])


def test_loaders_reads_markdown_bytes():
    from loaders import extract_text
    assert "hello" in extract_text(b"# Title\nhello world", "note.md")


def test_groq_provider_config_resolves():
    """Setting a Groq key routes chat to Groq's OpenAI-compatible endpoint,
    without touching embeddings (which stay offline)."""
    from settings import Settings
    s = Settings(llm_provider="auto", groq_api_key="gsk_test", openai_api_key="")
    assert s.resolved_llm == "groq"
    assert s.chat_enabled is True
    assert s.chat_config["base_url"] == "https://api.groq.com/openai/v1"
    assert s.chat_config["model"] == s.groq_model
    assert s.resolved_embed != "openai"        # embeddings never go to Groq/OpenAI here
