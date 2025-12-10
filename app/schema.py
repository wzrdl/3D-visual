"""
Database Schema Definition
Defines the structure of the SQLite database for 3D model metadata storage.
"""

# Table: models
# Columns:
#   - id: TEXT PRIMARY KEY 
#   - filename: TEXT NOT NULL
#   - display_name: TEXT NOT NULL
#   - tags: TEXT NOT NULL
#   - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#   - modified_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

# Indexes:
#   - idx_models_display_name
#   - idx_models_tags

CREATE_TABLE_MODELS = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    display_name TEXT NOT NULL,
    tags TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEX_DISPLAY_NAME = """
CREATE INDEX IF NOT EXISTS idx_models_display_name ON models(display_name);
"""

CREATE_INDEX_TAGS = """
CREATE INDEX IF NOT EXISTS idx_models_tags ON models(tags);
"""

SCHEMA_VERSION = 1

