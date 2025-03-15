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
- **Cross-platform compatibility** (Apple Silicon, Intel, Windows, Linux)
- **Hardware-aware optimizations** for different environments
- **Graceful fallbacks** for limited hardware resources

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
- Hardware-aware resource allocation
- Adaptive batch sizes based on available memory

## Hardware Compatibility

The MCP Memory Service now includes enhanced cross-platform compatibility with the following features:

- **Dynamic hardware detection**: Automatically detects the system architecture and available hardware accelerators
- **Adaptive resource usage**: Adjusts memory usage and batch sizes based on available system resources
- **Flexible model selection**: Chooses the optimal embedding model for the detected hardware
- **Graceful fallbacks**: Falls back to lighter models or CPU-only mode when necessary

### Supported Platforms

| Platform | Architecture | Accelerator | Status |
|----------|--------------|-------------|--------|
| macOS | Apple Silicon (M1/M2/M3) | MPS | ✅ Fully supported |
| macOS | Apple Silicon under Rosetta 2 | CPU | ✅ Supported with fallbacks |
| macOS | Intel | CPU | ✅ Fully supported |
| Windows | x86_64 | CUDA | ✅ Fully supported |
| Windows | x86_64 | DirectML | ✅ Supported |
| Windows | x86_64 | CPU | ✅ Supported with fallbacks |
| Linux | x86_64 | CUDA | ✅ Fully supported |
| Linux | x86_64 | ROCm | ✅ Supported |
| Linux | x86_64 | CPU | ✅ Supported with fallbacks |
| Linux | ARM64 | CPU | ✅ Supported with fallbacks |

## Installation

### Enhanced Installation Script

The project now includes an enhanced installation script that automatically detects your system and installs the appropriate dependencies:

```bash
# Clone the repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the installation script
python install.py
```

The installation script will:
1. Detect your system architecture and available hardware accelerators
2. Install the appropriate dependencies for your platform
3. Configure the optimal settings for your environment
4. Verify the installation and provide diagnostics if needed

### Platform-Specific Installation Notes

#### Windows

On Windows, PyTorch installation requires special handling due to platform-specific wheel availability. We provide a Windows-specific installation script that handles all the complexities:

```bash
# Run the Windows-specific installation script
python scripts/install_windows.py
```

This script will:
1. Detect CUDA availability and version
2. Install the appropriate PyTorch version from the correct index URL
3. Install other dependencies without conflicting with PyTorch
4. Install the MCP Memory Service package
5. Verify the installation

For manual installation, you can use:

```bash
# First, install PyTorch with the appropriate index URL
# For NVIDIA GPU (CUDA) support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR for CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies (without PyTorch)
pip install -r requirements.txt --no-deps
pip install "mcp>=1.0.0,<2.0.0"

# Finally install the package
pip install -e .
```

The standard installation script (`install.py`) will also attempt to handle this for you, but if you encounter issues, use the Windows-specific script above.

#### macOS

On macOS (especially Apple Silicon M1/M2/M3), the installation should work smoothly:

```bash
# The installation script will automatically detect Apple Silicon and use MPS acceleration
python install.py
```

#### Linux

On Linux, the installation depends on your GPU:

```bash
# For NVIDIA GPU (CUDA) support, ensure CUDA toolkit is installed
# For AMD GPU (ROCm) support, ensure ROCm is installed

# The installation script will detect your hardware and install appropriate dependencies
python install.py
```

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

2. Install dependencies (platform-specific):

For macOS/Linux:
```bash
pip install -r requirements.txt
uv add mcp
pip install -e .
```

For Windows:
```bash
# Option 1: Use the Windows-specific installation script (recommended)
python scripts/install_windows.py

# Option 2: Manual installation
# First install PyTorch with the appropriate index URL
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Then install other dependencies (without PyTorch)
pip install -r requirements.txt --no-deps
pip install "mcp>=1.0.0,<2.0.0"
# Finally install the package
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

### Standard Configuration

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
      "MCP_MEMORY_CHROMA_PATH": "your_chroma_db_path",  # e.g., "C:\\Users\\John.Doe\\AppData\\Local\\mcp-memory\\chroma_db",
      "MCP_MEMORY_BACKUPS_PATH": "your_backups_path"  # e.g., "C:\\Users\\John.Doe\\AppData\\Local\\mcp-memory\\backups"
    }
  }
}
```

### Windows-Specific Configuration (Recommended)

For Windows users, we recommend using the wrapper script to ensure PyTorch is properly installed:

```json
{
  "memory": {
    "command": "python",
    "args": [
      "C:\\path\\to\\mcp-memory-service\\memory_wrapper.py"
    ],
    "env": {
      "MCP_MEMORY_CHROMA_PATH": "C:\\Users\\YourUsername\\AppData\\Local\\mcp-memory\\chroma_db",
      "MCP_MEMORY_BACKUPS_PATH": "C:\\Users\\YourUsername\\AppData\\Local\\mcp-memory\\backups"
    }
  }
}
```

The wrapper script will:
1. Check if PyTorch is installed and properly configured
2. Install PyTorch with the correct index URL if needed
3. Run the memory server with the appropriate configuration

This ensures that the memory service works correctly on Windows platforms.

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

# Hardware-specific environment variables
PYTORCH_ENABLE_MPS_FALLBACK: Enable MPS fallback for Apple Silicon (default: 1)
MCP_MEMORY_USE_ONNX: Use ONNX Runtime for CPU-only deployments (default: 0)
MCP_MEMORY_USE_DIRECTML: Use DirectML for Windows acceleration (default: 0)
MCP_MEMORY_MODEL_NAME: Override the default embedding model
MCP_MEMORY_BATCH_SIZE: Override the default batch size
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
- Hardware-aware resource allocation
- Adaptive batch sizes based on available memory

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

# Verify environment compatibility
python scripts/verify_environment_enhanced.py

# Verify PyTorch installation on Windows
python scripts/verify_pytorch_windows.py
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
│   ├── hashing.py     # Hashing utilities
│   └── system_detection.py  # Hardware detection utilities
├── config.py     # Configuration utilities
└──server.py     # Main MCP server
```

##  Additional Stuff for Development
```
../your_mcp_memory_service_directory
├── scripts/
│   ├── migrate_tags.py    # Tag migration script
│   ├── repair_memories.py # Memory repair script
│   ├── validate_memories.py # Memory validation script
│   ├── verify_environment_enhanced.py # Enhanced environment verification
│   └── verify_pytorch_windows.py # Windows-specific PyTorch verification
├── memory_wrapper.py      # Windows wrapper script for Claude Desktop
├── setup.py    # Setup script with platform-specific dependencies
├── install.py  # Enhanced installation script
└── tests/
    ├── __init__.py
    ├── test_memory_ops.py
    ├── test_semantic_search.py
    └── test_database.py
```

## Required Dependencies
```
# Core dependencies
chromadb==0.5.23
sentence-transformers>=2.2.2
tokenizers>=0.13.3,<0.21.0
websockets>=11.0.3
mcp>=1.0.0,<2.0.0

# PyTorch - platform-specific versions installed automatically
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Optional dependencies for specific platforms
# onnxruntime>=1.15.0  # For CPU-only deployments
# torch-directml>=0.2.0  # For Windows DirectML acceleration
```

## Troubleshooting

### Enhanced Diagnostics

The project now includes enhanced diagnostics tools to help troubleshoot issues:

```bash
# Run the enhanced environment verification script
python scripts/verify_environment_enhanced.py

# Export detailed diagnostics to a JSON file
python scripts/verify_environment_enhanced.py --export
```

The verification script will:
1. Check your system architecture and hardware capabilities
2. Verify that all required dependencies are installed correctly
3. Test the embedding model and ChromaDB functionality
4. Provide detailed diagnostics and recommendations for any issues found

### Common Issues

#### PyTorch Installation Issues on Windows

If you encounter errors like:
```
error: Distribution `torch==2.5.1 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform

hint: You're on Windows (`win_amd64`), but `torch` (v2.5.1) only has wheels for the following platform: `manylinux1_x86_64`
```

This means PyTorch doesn't have a wheel available for your Windows platform. To fix this:

1. **Use the Windows-specific installation script** (recommended):
   ```bash
   python scripts/install_windows.py
   ```
   This script handles all the complexities of installing PyTorch on Windows, including:
   - Detecting CUDA availability and version
   - Installing the appropriate PyTorch version from the correct index URL
   - Installing other dependencies without conflicting with PyTorch
   - Verifying the installation

2. **Manual installation**: Install PyTorch manually with the appropriate index URL:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Then install other dependencies without PyTorch:
   ```bash
   pip install -r requirements.txt --no-deps
   pip install "mcp>=1.0.0,<2.0.0"
   ```
   Finally, install the package:
   ```bash
   pip install -e .
   ```

3. **Version specification**: If you need a specific version, use:
   ```bash
   pip install torch==2.1.0 torchvision==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Check compatibility**: Visit [PyTorch's website](https://pytorch.org/get-started/locally/) to find the latest version with Windows wheels available.

5. **Verify installation**: Run the Windows-specific verification script:
   ```bash
   python scripts/verify_pytorch_windows.py
   ```
   This script will check if PyTorch is properly installed and configured for Windows.

#### Other Common Issues

- **Installation fails on Apple Silicon**: Make sure you're using Python 3.10+ built for ARM64
- **CUDA not detected on Windows/Linux**: Ensure CUDA toolkit is installed and in PATH
- **Out of memory errors**: Try setting a smaller batch size with `MCP_MEMORY_BATCH_SIZE=4`
- **Slow performance on CPU**: Consider using a smaller model with `MCP_MEMORY_MODEL_NAME=paraphrase-MiniLM-L3-v2`

Additional troubleshooting:
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
