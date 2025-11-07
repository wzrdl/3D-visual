"""
Database Schema Definition
Defines the structure of the SQLite database for 3D model metadata storage.
"""

# Table: models
# Columns:
#   - id: TEXT PRIMARY KEY - Unique identifier for each model
#   - filename: TEXT NOT NULL - Name of the model file in assets/models/
#   - display_name: TEXT NOT NULL - User-friendly display name
#   - tags: TEXT NOT NULL - JSON array of tags (stored as TEXT, parsed as JSON)
#   - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP - Creation timestamp
#   - modified_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP - Last modification timestamp

# Indexes:
#   - idx_models_display_name: Index on display_name for type-ahead search
#   - idx_models_tags: Index on tags for tag-based search

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

# Schema version for migrations
SCHEMA_VERSION = 1

