# Agent Integration Guide

Documentation for AI agents to connect to and use this vector search service.

## Quick Start

**Base URL:** `http://10.0.10.30:8420` (internal network)

### Search for Information

```bash
curl "http://10.0.10.30:8420/search?q=your+query&limit=5"
```

### Generate Embeddings (OpenAI-compatible)

```bash
curl -X POST http://10.0.10.30:8420/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "your text here", "model": "all-MiniLM-L6-v2"}'
```

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/stats` | GET | Database statistics |
| `/search?q=...` | GET | Semantic search |
| `/notes` | GET | List indexed notes |
| `/notes/{id}` | GET | Get note with full content |
| `/similar/{id}` | GET | Find similar notes |
| `/v1/embeddings` | POST | OpenAI-compatible embeddings |

## Search Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | required | Natural language query |
| `limit` | int | 10 | Max results (1-50) |
| `threshold` | float | 0.3 | Min similarity (0.0-1.0) |
| `hybrid` | bool | false | Semantic + fulltext |

## Response Format

```json
{
  "query": "your search",
  "count": 3,
  "results": [
    {
      "note_id": 42,
      "file_path": "path/to/note.md",
      "title": "Note Title",
      "chunk_content": "Relevant excerpt...",
      "similarity": 0.847
    }
  ]
}
```

## OpenClaw Integration

Add to your `openclaw.json`:

```json
{
  "agents": {
    "defaults": {
      "memorySearch": {
        "provider": "openai",
        "remote": {
          "baseUrl": "http://10.0.10.30:8420/v1/",
          "apiKey": "not-needed"
        },
        "model": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

## Python Example

```python
import requests

def search_knowledge(query, limit=5):
    resp = requests.get(
        "http://10.0.10.30:8420/search",
        params={"q": query, "limit": limit}
    )
    return resp.json()["results"]

# Usage
results = search_knowledge("packet encryption")
for r in results:
    print(f"{r['similarity']:.2f} - {r['title']}")
```

## Technical Details

- **Model:** all-MiniLM-L6-v2 (384 dimensions)
- **Database:** PostgreSQL 16 + pgvector
- **Chunk size:** ~512 tokens
