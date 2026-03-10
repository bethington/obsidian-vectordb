-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Notes table: stores each note and its metadata
CREATE TABLE IF NOT EXISTS notes (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT,
    frontmatter JSONB DEFAULT '{}',
    file_hash TEXT,  -- MD5 of file content for change detection
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Chunks table: stores chunked text with embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    note_id INTEGER REFERENCES notes(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast similarity search (IVFFlat for good performance at this scale)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- Index for note lookups
CREATE INDEX IF NOT EXISTS notes_file_path_idx ON notes (file_path);
CREATE INDEX IF NOT EXISTS notes_file_hash_idx ON notes (file_hash);
CREATE INDEX IF NOT EXISTS chunks_note_id_idx ON chunks (note_id);

-- Full text search index on content
CREATE INDEX IF NOT EXISTS notes_content_fts_idx ON notes USING gin(to_tsvector('english', content));

-- Function to search by semantic similarity
CREATE OR REPLACE FUNCTION semantic_search(
    query_embedding vector(384),
    match_count INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id INTEGER,
    note_id INTEGER,
    file_path TEXT,
    title TEXT,
    chunk_content TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS chunk_id,
        n.id AS note_id,
        n.file_path,
        n.title,
        c.content AS chunk_content,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    JOIN notes n ON c.note_id = n.id
    WHERE 1 - (c.embedding <=> query_embedding) > similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function for hybrid search (semantic + full text)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(384),
    search_text TEXT,
    match_count INTEGER DEFAULT 10,
    semantic_weight FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    note_id INTEGER,
    file_path TEXT,
    title TEXT,
    chunk_content TEXT,
    semantic_score FLOAT,
    text_score FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH semantic AS (
        SELECT
            c.note_id,
            c.content AS chunk_content,
            1 - (c.embedding <=> query_embedding) AS score
        FROM chunks c
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count * 3
    ),
    fulltext AS (
        SELECT
            n.id AS note_id,
            ts_rank(to_tsvector('english', n.content), plainto_tsquery('english', search_text)) AS score
        FROM notes n
        WHERE to_tsvector('english', n.content) @@ plainto_tsquery('english', search_text)
    )
    SELECT
        n.id AS note_id,
        n.file_path,
        n.title,
        s.chunk_content,
        s.score AS semantic_score,
        COALESCE(f.score, 0) AS text_score,
        (semantic_weight * s.score + (1 - semantic_weight) * COALESCE(f.score, 0)) AS combined_score
    FROM semantic s
    JOIN notes n ON s.note_id = n.id
    LEFT JOIN fulltext f ON s.note_id = f.note_id
    ORDER BY (semantic_weight * s.score + (1 - semantic_weight) * COALESCE(f.score, 0)) DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
