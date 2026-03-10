# Obsidian Vector DB

Semantic search for Obsidian vault notes using PostgreSQL + pgvector.

## Architecture

- **PostgreSQL 16 + pgvector**: Stores notes, chunks, and 384-dimensional embeddings
- **Embedder Service**: Ingests markdown notes, chunks them, generates embeddings using `all-MiniLM-L6-v2`, watches for changes
- **Search API**: FastAPI service for semantic search, hybrid search, and similar note discovery

## Quick Start

```bash
# Start all services
docker compose up -d

# Check ingestion progress
docker logs -f obsidian-embedder

# Test search
curl "http://localhost:8420/search?q=stoicism+and+faith"
curl "http://localhost:8420/stats"
curl "http://localhost:8420/health"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Database statistics (note/chunk counts) |
| `/search?q=...` | GET | Semantic search (add `&hybrid=true` for combined) |
| `/notes` | GET | List indexed notes |
| `/notes/{id}` | GET | Get full note content |
| `/similar/{id}` | GET | Find notes similar to a given note |

### Search Parameters

- `q` (required): Search query text
- `limit` (default 10): Max results (1-50)
- `threshold` (default 0.3): Minimum similarity score (0.0-1.0)
- `hybrid` (default false): Use hybrid semantic + fulltext search

## Configuration

Environment variables (set in `.env` or `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | `changeme_in_env` | Database password |
| `VAULT_PATH` | `/vault` | Path to Obsidian vault (in container) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Word overlap between chunks |
| `WATCH_INTERVAL` | `30` | Seconds between change checks |

## GPU Acceleration

The embedder and API services are configured to use NVIDIA GPU via Docker's GPU runtime.
Requires `nvidia-container-toolkit` on the host.

## Vault Location

Notes are mounted read-only from: `/home/ben/obsidian/config/Notes`
