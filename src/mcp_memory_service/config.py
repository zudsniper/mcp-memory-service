# config.py
import os
from pathlib import Path

# Base paths
HOME = str(Path.home())
ICLOUD_BASE = os.path.join(HOME, "Library", "Mobile Documents", "com~apple~CloudDocs", "ai", "claude-memory")
CHROMA_PATH = os.path.join(ICLOUD_BASE, "chroma_db")
BACKUPS_PATH = os.path.join(ICLOUD_BASE, "backups")

# Server settings
SERVER_NAME = "memory"
SERVER_VERSION = "0.2.0"

# Model settings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# ChromaDB settings
CHROMA_SETTINGS = {
    "anonymized_telemetry": False,
    "allow_reset": True,
    "is_persistent": True
}

# Memory collection settings
COLLECTION_NAME = "memory_collection"
COLLECTION_METADATA = {"hnsw:space": "cosine"}