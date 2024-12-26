import os

CHROMA_PATH = os.environ.get(
    "MCP_MEMORY_CHROMA_PATH",
    os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/AI/claude-memory/chroma_db")
)

BACKUPS_PATH = os.environ.get(
    "MCP_MEMORY_BACKUPS_PATH", 
    os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/AI/claude-memory/backups")
)