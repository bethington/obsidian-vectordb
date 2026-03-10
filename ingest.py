#!/usr/bin/env python3
"""
Obsidian Vault Ingestion Service
Reads markdown notes, chunks them, generates embeddings, and stores in pgvector.
Watches for file changes and re-indexes automatically.
"""

import os
import sys
import time
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Tuple

import frontmatter
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://obsidian:changeme_in_env@postgres:5432/obsidian")
VAULT_PATH = os.environ.get("VAULT_PATH", "/vault")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
MAX_CHUNKS_PER_NOTE = int(os.environ.get("MAX_CHUNKS_PER_NOTE", "200"))
WATCH_INTERVAL = int(os.environ.get("WATCH_INTERVAL", "30"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection with retry logic."""
    for attempt in range(10):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"DB connection attempt {attempt + 1}/10 failed: {e}")
            time.sleep(3)
    raise RuntimeError("Could not connect to database after 10 attempts")


def file_hash(filepath: str) -> str:
    """Compute MD5 hash of file content."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def parse_markdown(filepath: str) -> Tuple[str, str, dict]:
    """Parse a markdown file, extracting frontmatter, title, and content."""
    try:
        post = frontmatter.load(filepath)
        content = post.content
        metadata = dict(post.metadata) if post.metadata else {}
    except Exception:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        metadata = {}

    # Extract title from frontmatter, first H1, or filename
    title = metadata.get("title", "")
    if not title:
        h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
        else:
            title = Path(filepath).stem

    # Clean content: remove obsidian-specific syntax for better embedding
    clean = content
    clean = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', clean)
    clean = re.sub(r'\[\[([^\]]+)\]\]', r'\1', clean)
    clean = re.sub(r'!\[\[([^\]]+)\]\]', '', clean)
    clean = re.sub(r'(?<!\w)#(\w+)', r'\1', clean)
    clean = re.sub(r'\n{3,}', '\n\n', clean)

    return title, clean.strip(), metadata


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries."""
    if not text or len(text) < 100:
        return [text] if text else []

    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    words = current_chunk.split()
                    overlap_words = words[-overlap:] if len(words) > overlap else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = (current_chunk + " " + sentence).strip()
        elif len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            words = current_chunk.split()
            overlap_words = words[-overlap:] if len(words) > overlap else words
            current_chunk = " ".join(overlap_words) + "\n\n" + para
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_single_file(rel_path: str, abs_path: str, existing_hash: str,
                        model: SentenceTransformer, conn) -> dict:
    """Process a single file: parse, chunk, embed, store. Commits per file."""
    result = {"status": "unchanged", "chunks": 0}

    try:
        current_hash = file_hash(abs_path)

        if existing_hash and existing_hash == current_hash:
            return result

        title, content, metadata = parse_markdown(abs_path)

        if not content or len(content) < 10:
            return result

        chunks = chunk_text(content)
        if not chunks:
            return result

        # Cap chunks to avoid massive books eating all resources
        original_count = len(chunks)
        if len(chunks) > MAX_CHUNKS_PER_NOTE:
            # Keep first and last chunks, sample evenly from the rest
            step = len(chunks) / MAX_CHUNKS_PER_NOTE
            indices = [int(i * step) for i in range(MAX_CHUNKS_PER_NOTE)]
            chunks = [chunks[i] for i in indices]
            logger.info(f"  Capped {rel_path}: {original_count} -> {len(chunks)} chunks")

        # Generate embeddings in batches
        embeddings = model.encode(chunks, show_progress_bar=False, batch_size=64)

        cur = conn.cursor()

        # Upsert note
        cur.execute("""
            INSERT INTO notes (file_path, title, content, frontmatter, file_hash, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (file_path) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                frontmatter = EXCLUDED.frontmatter,
                file_hash = EXCLUDED.file_hash,
                updated_at = NOW()
            RETURNING id
        """, (rel_path, title, content, Json(metadata), current_hash))

        note_id = cur.fetchone()[0]

        # Delete old chunks
        cur.execute("DELETE FROM chunks WHERE note_id = %s", (note_id,))

        # Insert new chunks
        chunk_data = [
            (note_id, i, chunk, embedding.tolist(), Json({}))
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        execute_values(
            cur,
            "INSERT INTO chunks (note_id, chunk_index, content, embedding, metadata) VALUES %s",
            chunk_data,
            template="(%s, %s, %s, %s::vector, %s)"
        )

        # COMMIT after each file!
        conn.commit()

        result["status"] = "updated" if existing_hash else "new"
        result["chunks"] = len(chunks)

    except Exception as e:
        conn.rollback()
        result["status"] = "error"
        logger.error(f"Error processing {rel_path}: {e}")

    return result


def ingest_vault(model: SentenceTransformer, conn) -> dict:
    """Full vault ingestion with change detection and per-file commits."""
    vault_path = Path(VAULT_PATH)
    stats = {"new": 0, "updated": 0, "unchanged": 0, "deleted": 0, "errors": 0, "chunks": 0}

    cur = conn.cursor()
    cur.execute("SELECT file_path, file_hash FROM notes")
    existing = {row[0]: row[1] for row in cur.fetchall()}

    # Find all markdown files
    md_files = {}
    for md_file in vault_path.rglob("*.md"):
        rel_path = str(md_file.relative_to(vault_path))
        if any(part.startswith('.') for part in md_file.parts):
            continue
        md_files[rel_path] = str(md_file)

    total = len(md_files)
    logger.info(f"Found {total} markdown files to process")

    for idx, (rel_path, abs_path) in enumerate(sorted(md_files.items()), 1):
        result = process_single_file(
            rel_path, abs_path,
            existing.get(rel_path, ""),
            model, conn
        )

        if result["status"] == "new":
            stats["new"] += 1
            stats["chunks"] += result["chunks"]
            logger.info(f"[{idx}/{total}] Indexed: {rel_path} ({result['chunks']} chunks)")
        elif result["status"] == "updated":
            stats["updated"] += 1
            stats["chunks"] += result["chunks"]
            logger.info(f"[{idx}/{total}] Updated: {rel_path} ({result['chunks']} chunks)")
        elif result["status"] == "error":
            stats["errors"] += 1
        else:
            stats["unchanged"] += 1

        # Progress log every 50 files
        if idx % 50 == 0:
            logger.info(f"Progress: {idx}/{total} files ({stats['new']} new, {stats['unchanged']} unchanged, {stats['errors']} errors)")

    # Handle deleted files
    deleted_paths = set(existing.keys()) - set(md_files.keys())
    for del_path in deleted_paths:
        cur.execute("DELETE FROM notes WHERE file_path = %s", (del_path,))
        conn.commit()
        stats["deleted"] += 1
        logger.info(f"Removed: {del_path}")

    return stats


class VaultChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.changed = False

    def on_modified(self, event):
        if event.src_path.endswith(".md"):
            self.changed = True

    def on_created(self, event):
        if event.src_path.endswith(".md"):
            self.changed = True

    def on_deleted(self, event):
        if event.src_path.endswith(".md"):
            self.changed = True


def main():
    logger.info("=" * 60)
    logger.info("Obsidian Vector DB - Ingestion Service")
    logger.info(f"Vault: {VAULT_PATH}")
    logger.info(f"Model: {EMBEDDING_MODEL}")
    logger.info(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    logger.info(f"Max chunks per note: {MAX_CHUNKS_PER_NOTE}")
    logger.info("=" * 60)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded successfully")

    conn = get_db_connection()
    logger.info("Connected to database")

    logger.info("Starting initial vault ingestion...")
    start = time.time()
    stats = ingest_vault(model, conn)
    elapsed = time.time() - start
    logger.info(f"Initial ingestion complete in {elapsed:.1f}s: {stats}")

    # Set up file watcher
    handler = VaultChangeHandler()
    observer = Observer()
    observer.schedule(handler, VAULT_PATH, recursive=True)
    observer.start()
    logger.info(f"Watching vault for changes (check interval: {WATCH_INTERVAL}s)...")

    try:
        while True:
            time.sleep(WATCH_INTERVAL)
            if handler.changed:
                logger.info("Changes detected, re-indexing...")
                handler.changed = False
                try:
                    stats = ingest_vault(model, conn)
                    logger.info(f"Re-index complete: {stats}")
                except Exception as e:
                    logger.error(f"Re-index failed: {e}")
                    try:
                        conn.close()
                    except:
                        pass
                    conn = get_db_connection()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    conn.close()


if __name__ == "__main__":
    main()
