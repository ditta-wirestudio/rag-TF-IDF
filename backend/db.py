"""Postgres + pgvector: pooled connections and schema bootstrap.

A module-level connection pool is opened lazily on first use and reused across
requests (opening a fresh connection per request does not scale).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from settings import settings

log = logging.getLogger("db")

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.pool_min_size,
            max_size=settings.pool_max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def connection():
    with pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({settings.active_embed_dim}),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


def _existing_embed_dim(cur) -> int | None:
    """Dimension of chunks.embedding if the table already exists, else None.
    (pgvector stores the dimension as the column's atttypmod.)"""
    cur.execute(
        """SELECT atttypmod AS dim FROM pg_attribute
           WHERE attrelid = to_regclass('chunks')
             AND attname = 'embedding' AND NOT attisdropped""")
    row = cur.fetchone()
    return row["dim"] if row and row["dim"] and row["dim"] > 0 else None


def init_schema() -> bool:
    """Create the schema. Returns True if an existing embedding column had a
    different dimension and was migrated — callers must then re-embed chunks
    (their text content is preserved; only the vectors are reset)."""
    target = settings.active_embed_dim
    migrated = False
    with connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # drop the legacy ivfflat index if present: built on an empty table its
        # centroids are degenerate and nearest-neighbour queries can return
        # ZERO rows. Replaced by the HNSW index in SCHEMA, which handles
        # incremental inserts correctly.
        cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
        current = _existing_embed_dim(cur)
        if current is not None and current != target:
            log.warning(
                "pgvector dimension mismatch: chunks.embedding is vector(%d) but the "
                "active embedder (%s) produces %d dims — migrating column, existing "
                "chunks will be re-embedded",
                current, settings.resolved_embed, target)
            cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
            cur.execute(
                f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({target}) "
                f"USING NULL::vector({target})")
            migrated = True
        cur.execute(SCHEMA)
        conn.commit()
    return migrated


def healthcheck() -> bool:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        return cur.fetchone() is not None


if __name__ == "__main__":
    init_schema()
    print("schema ready")
