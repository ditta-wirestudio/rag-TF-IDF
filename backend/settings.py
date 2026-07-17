"""Centralised, validated configuration (pydantic-settings).

All tunables live here and are read from the environment / .env once at startup.
"""
from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("*", mode="before")
    @classmethod
    def _strip_inline_env_comments(cls, v):
        """Guard against .env inline comments leaking into values.

        Some parsers (notably docker compose) pass 'KEY=   # note' through as
        the literal value '# note', which then looks like a real API key /
        setting. Follow dotenv semantics: '#' at the start of the value or
        preceded by whitespace begins a comment.
        """
        if not isinstance(v, str):
            return v
        s = v.strip()
        if s.startswith("#"):
            return ""
        for i in range(1, len(s)):
            if s[i] == "#" and s[i - 1] in " \t":
                return s[:i].strip()
        return s

    # --- storage / embeddings backend -------------------------------------
    store: str = "auto"                     # auto | postgres | memory
    database_url: str = "postgresql://rag:rag@localhost:5432/ragdemo"

    # embeddings provider: auto picks openai (if key) -> fastembed (if installed)
    # -> hashing (always works). fastembed is a free, local, on-device model.
    embed_provider: str = "auto"            # auto | openai | fastembed | hashing
    embed_model: str = "text-embedding-3-small"   # openai
    fastembed_model: str = "BAAI/bge-small-en-v1.5"  # local, 384-dim
    embed_dim: int = 1536                   # openai text-embedding-3-small
    fastembed_dim: int = 384                # bge-small-en-v1.5
    offline_dim: int = 256                  # hashing fallback

    # --- LLM (answer generation) ------------------------------------------
    # Chat and embeddings are decoupled. Groq is OpenAI-compatible for CHAT
    # only (no embeddings endpoint), so it drives answers; embeddings use
    # OpenAI if OPENAI_API_KEY is set, else the offline hashing model.
    llm_provider: str = "auto"              # auto | openai | groq | none
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"         # used when provider = openai
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # --- retrieval / chunking ---------------------------------------------
    chunk_chars: int = 900
    chunk_overlap: int = 150
    default_k: int = 5
    eval_k: int = 5

    # --- robustness -------------------------------------------------------
    embed_batch_size: int = 100             # OpenAI accepts up to 2048; keep modest
    max_retries: int = 4
    retry_base_delay: float = 0.5           # seconds, exponential backoff
    pool_min_size: int = 1
    pool_max_size: int = 10
    max_ingest_chars: int = 2_000_000       # ~2 MB of text per document, reject larger
    max_upload_bytes: int = 25_000_000      # 25 MB uploaded file cap

    # --- api --------------------------------------------------------------
    api_key: str = ""                       # if set, required on write endpoints
    cors_origins: str = "*"                 # comma-separated
    seed_sample_docs: bool = True           # auto-ingest sample_docs/ when store is empty

    @property
    def resolved_llm(self) -> str:
        """Which chat provider actually runs, given keys + explicit choice."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.groq_api_key:
            return "groq"
        if self.openai_api_key:
            return "openai"
        return "none"

    @property
    def chat_enabled(self) -> bool:
        return self.resolved_llm in ("openai", "groq")

    @property
    def chat_config(self) -> dict:
        """base_url / api_key / model for the active chat provider."""
        if self.resolved_llm == "groq":
            return {"base_url": self.groq_base_url,
                    "api_key": self.groq_api_key, "model": self.groq_model}
        if self.resolved_llm == "openai":
            return {"base_url": None,
                    "api_key": self.openai_api_key, "model": self.chat_model}
        return {}

    @property
    def resolved_embed(self) -> str:
        """Which embedding backend actually runs."""
        if self.embed_provider != "auto":
            return self.embed_provider
        if self.openai_api_key:
            return "openai"                 # Groq has no embeddings endpoint
        try:
            import fastembed  # noqa: F401
            return "fastembed"
        except Exception:
            return "hashing"

    @property
    def active_embed_dim(self) -> int:
        """Vector dimension for the active embedder (used by the pgvector schema)."""
        return {"openai": self.embed_dim,
                "fastembed": self.fastembed_dim,
                "hashing": self.offline_dim}[self.resolved_embed]

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
