# MCP Memory Service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![smithery badge](https://smithery.ai/badge/@doobidoo/mcp-memory-service)](https://smithery.ai/server/@doobidoo/mcp-memory-service)

An MCP server providing semantic memory and persistent storage capabilities for Claude Desktop using ChromaDB and sentence transformers. This service enables long-term memory storage with semantic search capabilities, making it ideal for maintaining context across conversations and instances. For personal use only. No user management is provided.

<img width="240" alt="grafik" src="https://github.com/user-attachments/assets/eab1f341-ca54-445c-905e-273cd9e89555" />
<a href="https://glama.ai/mcp/servers/bzvl3lz34o"><img width="380" height="200" src="https://glama.ai/mcp/servers/bzvl3lz34o/badge" alt="Memory Service MCP server" /></a>

## Features

- Semantic search using sentence transformers
- Tag-based memory retrieval system
- Persistent storage using ChromaDB
- Automatic database backups
- Memory optimization tools
- Exact match retrieval
- Debug mode for similarity analysis
- Database health monitoring
- Duplicate detection and cleanup
- Customizable embedding model

## Available Tools and Operations

The list covers all the core functionalities exposed through the MCP server's tools, organized by their primary purposes and use cases. Each category represents a different aspect of memory management and interaction available through the system.

### Core Memory Operations

1. `store_memory`
   - Store new information with optional tags
   - Parameters:
     - content: String (required)
     - metadata: Object (optional)
       - tags: Array or list of strings
       - type: String

2. `retrieve_memory`
   - Perform semantic search for relevant memories
   - Parameters:
     - query: String (required)
     - n_results: Number (optional, default: 5)

3. `search_by_tag`
   - Find memories using specific tags
   - Parameters:
     - tags: Array of strings (required)

### Advanced Operations

4. `exact_match_retrieve`
   - Find memories with exact content match
   - Parameters:
     - content: String (required)

5. `debug_retrieve`
   - Retrieve memories with similarity scores
   - Parameters:
     - query: String (required)
     - n_results: Number (optional)
     - similarity_threshold: Number (optional)

### Database Management

6. `create_backup`
   - Create database backup
   - Parameters: None

7. `get_stats`
   - Get memory statistics
   - Returns: Database size, memory count

8. `optimize_db`
   - Optimize database performance
   - Parameters: None

9. `check_database_health`
   - Get database health metrics
   - Returns: Health status and statistics

10. `check_embedding_model`
    - Verify model status
    - Parameters: None

### Memory Management

11. `delete_memory`
    - Delete specific memory by hash
    - Parameters:
      - content_hash: String (required)

12. `delete_by_tag`
    - Delete all memories with specific tag
    - Parameters:
      - tag: String (required)

13. `cleanup_duplicates`
    - Remove duplicate entries
    - Parameters: None

## Performance and Maintenance

- Default similarity threshold: 0.7
- Maximum recommended memories per query: 10
- Automatic optimization at 10,000 memories
- Daily automatic backups with 7-day retention
- Regular database health monitoring recommended
- Cloud storage sync must complete before access
- Debug mode available for troubleshooting


## Installation

### Installing via Smithery

To install Memory Service for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@doobidoo/mcp-memory-service):

```bash
npx -y @smithery/cli install @doobidoo/mcp-memory-service --client claude
```

### Manual Installation
1. Create Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
uv add mcp
pip install -e .
```

## Usage

1. Start the server:(for testing purposes) 
```bash
python src/test_management.py 
```
Isaolated test for methods
```bash
python src/chroma_test_isolated.py
```

## Claude MCP configuration

Add the following to your `claude_desktop_config.json` file:

```json
{
  "memory": {
    "command": "uv",
    "args": [
      "--directory",
      "your_mcp_memory_service_directory",  # e.g., "C:\\REPOSITORIES\\mcp-memory-service",
      "run",
      "memory"
    ],
    "env": {
      "MCP_MEMORY_CHROMA_PATH": "your_chroma_db_path",  # e.g., "C:\\Users\\John.Doe \\AppData\\Local\\mcp-memory\\chroma_db", 
      "MCP_MEMORY_BACKUPS_PATH": "your_backups_path"  # e.g., "C:\\Users\\John.Doe \\AppData\\Local\\mcp-memory\\backups"
    }
  }
}
```

### Storage Structure and Settings
```
../your_mcp_memory_service_directory/mcp-memory/
├── chroma_db/    # Main vector database
└── backups/      # Automatic backups
```

Configure through environment variables:
```
CHROMA_DB_PATH: Path to ChromaDB storage
BACKUP_PATH: Path for backups
AUTO_BACKUP_INTERVAL: Backup interval in hours (default: 24)
MAX_MEMORIES_BEFORE_OPTIMIZE: Threshold for auto-optimization (default: 10000)
SIMILARITY_THRESHOLD: Default similarity threshold (default: 0.7)
MAX_RESULTS_PER_QUERY: Maximum results per query (default: 10)
BACKUP_RETENTION_DAYS: Number of days to keep backups (default: 7)
LOG_LEVEL: Logging level (default: INFO)
```

## Sample Use Cases
Semantic requests:

<img width="750" alt="grafik" src="https://github.com/user-attachments/assets/4bc854c6-721a-4abe-bcc5-7ef274628db7" />
<img width="1112" alt="grafik" src="https://github.com/user-attachments/assets/502477d2-ade6-4a5e-a756-b6302d9d6931" />

Call by tool name:
<img width="1112" alt="grafik" src="https://github.com/user-attachments/assets/23d161a8-f62c-41c6-bcd8-e9b16f369c95" />

## Performance and Maintenance

- Default similarity threshold: 0.7
- Maximum recommended memories per query: 10
- Automatic optimization at 10,000 memories
- Daily automatic backups with 7-day retention
- Regular database health monitoring recommended
- Cloud storage sync must complete before access
- Debug mode available for troubleshooting

## Testing

The project includes test suites for verifying the core functionality:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_memory_ops.py
pytest tests/test_semantic_search.py
pytest tests/test_database.py
```

Test scripts are available in the `tests/` directory:
- `test_memory_ops.py`: Tests core memory operations (store, retrieve, delete)
- `test_semantic_search.py`: Tests semantic search functionality and similarity scoring
- `test_database.py`: Tests database operations (backup, health checks, optimization)

Each test file includes:
- Proper test fixtures for server setup and teardown
- Async test support using pytest-asyncio
- Comprehensive test cases for the related functionality
- Error case handling and validation


## Project Structure
```
../your_mcp_memory_service_directory/src/mcp_memory_service/
├── __init__.py
├── config.py
├── models/
│   ├── __init__.py
│   └── memory.py      # Memory data models
├── storage/
│   ├── __init__.py
│   ├── base.py        # Abstract base storage class
│   └── chroma.py      # ChromaDB implementation
├── utils/
│   ├── __init__.py
│   ├── db_utils.py    # Database utility functions
│   ├── debug.py       # Debugging utilities
│   └── hashing.py     # Hashing utilities
├── config.py     # Configuration utilities
└──server.py     # Main MCP server
```

##  Additional Stuff for Development
```
../your_mcp_memory_service_directory
├── scripts/
│   ├── migrate_tags.py    # Tag migration script
│   ├── repair_memories.py # Memory repair script
│   └── validate_memories.py # Memory validation script
└── tests/
    ├── __init__.py
    ├── test_memory_ops.py
    ├── test_semantic_search.py
    └── test_database.py
```

## Required Dependencies
```
chromadb==0.5.23
sentence-transformers>=2.2.2
tokenizers==0.20.3
websockets>=11.0.3
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

## Important Notes
- When storing in cloud, always ensure iCloud or other Cloud Drives sync is complete before accessing from another device
- Regular backups are crucial when testing new features
- Monitor ChromaDB storage size and optimize as needed
- The service includes automatic backup functionality that runs every 24 hours(tbd)
- Debug mode is available for troubleshooting semantic search results
- Memory optimization runs automatically when database size exceeds configured thresholds

## Performance Considerations
- Default similarity threshold for semantic search: 0.7
- Maximum recommended memories per query: 10
- Automatic optimization triggers at 10,000 memories
- Backup retention policy: 7 days

## Troubleshooting
- Check logs in `..\Claude\logs\mcp-server-memory.log`
- Use `debug_retrieve` for search issues
- Monitor database health with `check_database_health`
- Use `exact_match_retrieve` for precise matching

## Development Guidelines
- Follow PEP 8
- Use type hints
- Include docstrings for all functions and classes
- Add tests for new features

### Pull Request Process
1. Create a feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit PR with description of changes

## License
MIT License - See LICENSE file for details

## Acknowledgments
- ChromaDB team for the vector database
- Sentence Transformers project for embedding models
- MCP project for the protocol specification

## Contact

[telegram](t.me/doobeedoo)

# Statement of Gratitude
*A special thanks to God, my ultimate source of strength and guidance, and to my wife for her unwavering patience and support throughout this project. I'd also like to express my gratitude to Claude from Antrophic for his invaluable contributions and expertise. This project wouldn't have been possible without your collective support.*
