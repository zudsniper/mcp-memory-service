# MCP Memory Service

[Previous content remains the same until the Testing section]

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

[Rest of the README.md remains the same]