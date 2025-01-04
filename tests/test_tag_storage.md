{
  `content`: `Tag Storage Implementation Guide

1. File Structure
Create three new files in your project:

```
mcp_memory_service/
├── tests/
│   └── test_tag_storage.py    # Integration tests
├── scripts/
│   ├── validate_memories.py   # Validation script
│   └── migrate_tags.py        # Migration script
```

2. Implementation Steps:

Step 1: Create test_tag_storage.py
```python
# test_tag_storage.py

import asyncio
from mcp_memory_service.storage.chroma import ChromaMemoryStorage
from mcp_memory_service.models.memory import Memory

async def run_tag_integration_tests():
    # Copy integration test code from first memory entry
    # Reference ID: 729fc332d1589801d64ea412f536e71778b20fb08f82ddb11bd2e1fb1a38e8cb

async def main():
    storage = ChromaMemoryStorage(\"path/to/test/db\")
    results = await run_tag_integration_tests()
    print(\"Test Results:\", results)

if __name__ == \"__main__\":
    asyncio.run(main())
```

Step 2: Create validate_memories.py
```python
# validate_memories.py

import asyncio
from mcp_memory_service.storage.chroma import ChromaMemoryStorage

# Copy validation script from third memory entry
# Reference ID: 10d5693235183a38308900a2bb0d56808f089f60a6472c9b2fa2b2edecd33a5b

async def main():
    storage = ChromaMemoryStorage(\"path/to/your/db\")
    report = await run_validation_report()
    print(report)
    
    # Optionally save report to file
    with open('validation_report.txt', 'w') as f:
        f.write(report)

if __name__ == \"__main__\":
    asyncio.run(main())
```

Step 3: Create migrate_tags.py
```python
# migrate_tags.py

import asyncio
from mcp_memory_service.storage.chroma import ChromaMemoryStorage

# Copy migration code from second memory entry
# Reference ID: 702aa2b31293a15f10623fcf28f28cc950e9960a26b3e891b0e4a938b7f9de53

async def main():
    storage = ChromaMemoryStorage(\"path/to/your/db\")
    
    # 1. Create backup
    print(\"Creating backup...\")
    await backup_memories()
    
    # 2. Run validation
    print(\"Validating current state...\")
    current_state = await validate_current_state()
    
    # 3. Run migration if user confirms
    print(\"\
Current state:\", current_state)
    proceed = input(\"Proceed with migration? (yes/no): \")
    if proceed.lower() == 'yes':
        print(\"Running migration...\")
        await migrate_tags()
        
        # 4. Verify migration
        print(\"Verifying migration...\")
        verification = await verify_migration()
        print(\"\
Migration verification:\", verification)
    else:
        print(\"Migration cancelled\")

if __name__ == \"__main__\":
    asyncio.run(main())
```

3. Implementation Order:

1. First run validation:
```bash
python scripts/validate_memories.py
```

2. Run integration tests:
```bash
python tests/test_tag_storage.py
```

3. If both look good, run migration:
```bash
python scripts/migrate_tags.py
```

4. After migration, run validation again:
```bash
python scripts/validate_memories.py
```

4. Monitoring:

- Keep the backup file for at least 7 days
- Monitor logs for any tag-related errors
- Run validation script daily for the first week
- Check search functionality with various tag formats

5. Rollback if needed:

If issues are detected, you can rollback using:
```bash
python scripts/migrate_tags.py --rollback
````,
  `metadata`: {
    `tags`: [
      `implementation`,
      `guide`,
      `steps`
    ],
    `type`: `documentation`
  }
}