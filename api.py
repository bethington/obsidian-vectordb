#!/usr/bin/env python3
"""
Obsidian Vector Search API
FastAPI service for semantic search over Obsidian notes.
"""

import os
import logging
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://obsidian:changeme_in_env@postgres:5432/obsidian")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Obsidian Vector Search",
    description="Semantic search API for Obsidian vault notes",
    version="1.0.0",
)

# Global model and connection
model = None
conn = None


def get_model():
    global model
    if model is None:
        logger.info(f"Loading model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def get_conn():
    global conn
    if conn is None or conn.closed:
        conn = psycopg2.connect(DATABASE_URL)
    return conn


class SearchResult(BaseModel):
    note_id: int
    file_path: str
    title: str
    chunk_content: str
    similarity: float


class NoteResult(BaseModel):
    id: int
    file_path: str
    title: str
    content: Optional[str] = None
    frontmatter: Optional[dict] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int


class StatsResponse(BaseModel):
    total_notes: int
    total_chunks: int
    vault_path: str


@app.on_event("startup")
async def startup():
    """Pre-load model on startup."""
    get_model()
    logger.info("API ready")


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        c = get_conn()
        cur = c.cursor()
        cur.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {e}")


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get database statistics."""
    c = get_conn()
    cur = c.cursor()
    cur.execute("SELECT COUNT(*) FROM notes")
    total_notes = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cur.fetchone()[0]
    return StatsResponse(
        total_notes=total_notes,
        total_chunks=total_chunks,
        vault_path=os.environ.get("VAULT_PATH", "/vault"),
    )


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Max results"),
    threshold: float = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity"),
    hybrid: bool = Query(False, description="Use hybrid search (semantic + fulltext)"),
):
    """
    Semantic search over Obsidian notes.
    Returns the most relevant note chunks ranked by cosine similarity.
    """
    m = get_model()
    embedding = m.encode(q).tolist()
    c = get_conn()
    cur = c.cursor(cursor_factory=RealDictCursor)

    if hybrid:
        cur.execute("""
            SELECT * FROM hybrid_search(%s::vector, %s, %s)
        """, (str(embedding), q, limit))
        rows = cur.fetchall()
        results = [
            SearchResult(
                note_id=row["note_id"],
                file_path=row["file_path"],
                title=row["title"],
                chunk_content=row["chunk_content"],
                similarity=float(row["combined_score"]),
            )
            for row in rows
        ]
    else:
        cur.execute("""
            SELECT * FROM semantic_search(%s::vector, %s, %s)
        """, (str(embedding), limit, threshold))
        rows = cur.fetchall()
        results = [
            SearchResult(
                note_id=row["note_id"],
                file_path=row["file_path"],
                title=row["title"],
                chunk_content=row["chunk_content"],
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]

    return SearchResponse(query=q, results=results, count=len(results))


@app.get("/notes", response_model=List[NoteResult])
async def list_notes(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
):
    """List all indexed notes, optionally filtered by title/path."""
    c = get_conn()
    cur = c.cursor(cursor_factory=RealDictCursor)

    if search:
        cur.execute("""
            SELECT id, file_path, title, frontmatter
            FROM notes
            WHERE file_path ILIKE %s OR title ILIKE %s
            ORDER BY title
            LIMIT %s OFFSET %s
        """, (f"%{search}%", f"%{search}%", limit, offset))
    else:
        cur.execute("""
            SELECT id, file_path, title, frontmatter
            FROM notes ORDER BY title LIMIT %s OFFSET %s
        """, (limit, offset))

    return [NoteResult(**row) for row in cur.fetchall()]


@app.get("/notes/{note_id}", response_model=NoteResult)
async def get_note(note_id: int):
    """Get a specific note by ID, including full content."""
    c = get_conn()
    cur = c.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM notes WHERE id = %s", (note_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteResult(**row)


@app.get("/similar/{note_id}", response_model=SearchResponse)
async def find_similar(
    note_id: int,
    limit: int = Query(10, ge=1, le=50),
):
    """Find notes similar to a given note."""
    c = get_conn()
    cur = c.cursor(cursor_factory=RealDictCursor)

    # Get the note's content
    cur.execute("SELECT title, content FROM notes WHERE id = %s", (note_id,))
    note = cur.fetchone()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Use the note's content as the search query
    m = get_model()
    embedding = m.encode(note["content"][:1000]).tolist()

    cur.execute("""
        SELECT * FROM semantic_search(%s::vector, %s, 0.3)
    """, (str(embedding), limit + 1))  # +1 to exclude self

    rows = cur.fetchall()
    results = [
        SearchResult(
            note_id=row["note_id"],
            file_path=row["file_path"],
            title=row["title"],
            chunk_content=row["chunk_content"],
            similarity=float(row["similarity"]),
        )
        for row in rows
        if row["note_id"] != note_id  # Exclude the source note
    ][:limit]

    return SearchResponse(query=f"Similar to: {note['title']}", results=results, count=len(results))


# ── OpenAI-compatible embeddings endpoint ──────────────────────────────
# This allows Clawdbot (or any OpenAI-compatible client) to generate
# embeddings using the same sentence-transformers model.

class EmbeddingRequest(BaseModel):
    input: object  # str or List[str]
    model: str = "all-MiniLM-L6-v2"
    encoding_format: Optional[str] = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(req: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.
    Accepts single string or list of strings, returns embeddings
    in the same format as OpenAI's API.
    """
    m = get_model()

    # Normalize input to list
    if isinstance(req.input, str):
        texts = [req.input]
    elif isinstance(req.input, list):
        texts = [str(t) for t in req.input]
    else:
        raise HTTPException(status_code=400, detail="input must be a string or list of strings")

    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="input must not be empty")

    # Generate embeddings
    embeddings = m.encode(texts).tolist()

    data = [
        EmbeddingData(embedding=emb, index=i)
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=req.model,
        usage=EmbeddingUsage(prompt_tokens=sum(len(t.split()) for t in texts), total_tokens=sum(len(t.split()) for t in texts)),
    )
