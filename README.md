# MCP Memory Service

An MCP server providing semantic memory and persistent storage capabilities for Claude using ChromaDB and sentence transformers.

<a href="https://glama.ai/mcp/servers/bzvl3lz34o"><img width="380" height="200" src="https://glama.ai/mcp/servers/bzvl3lz34o/badge" alt="Memory Service MCP server" /></a>

## Features

- Semantic search using sentence transformers
- Tag-based memory retrieval
- Persistent storage using ChromaDB
- Automatic backups
- Memory optimization tools

## Installation

1. Create Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python src/memory_server.py
```

2. Connect to websocket at `ws://localhost:8765`

## Available Tools

1. `store_memory`: Store new information with tags
2. `retrieve_memory`: Semantic search for relevant memories
3. `search_by_tag`: Find memories by tags
4. `create_backup`: Create database backup
5. `get_stats`: Get memory statistics
6. `optimize_db`: Optimize database performance

## Storage Structure
```
/users/hkr/library/mobile documents/com~apple~clouddocs/ai/claude-memory/
├── chroma_db/    # Main vector database
└── backups/      # Automatic backups
```

## Required Dependencies
```
chromadb==0.5.23
sentence-transformers>=2.2.2
tokenizers==0.20.3
websockets>=11.0.3
```

## Important Notes
- Always ensure iCloud sync is complete before accessing from another device
- Regular backups are crucial when testing new features
- Monitor ChromaDB storage size in iCloud
