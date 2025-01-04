# tests/test_tag_storage.py

import asyncio
import pytest
from mcp_memory_service.storage.chroma import ChromaMemoryStorage
from mcp_memory_service.models.memory import Memory
import argparse

def verify_test_results(tests, expectations):
    """Helper function to verify test results against expectations"""
    results = []
    for i, (test, expectation) in enumerate(zip(tests, expectations)):
        passed = expectation(test)
        results.append({
            "test_number": i + 1,
            "passed": passed,
            "result": test
        })
    return results

async def run_tag_integration_tests(storage):
    """Comprehensive test suite for tag handling"""
    
    # Test Case 1: Array Format Tags
    memory1 = await storage.store(Memory(
        content="Array format test",
        tags=["test1", "test2"]
    ))
    
    # Test Case 2: String Format Tags
    memory2 = await storage.store(Memory(
        content="String format test",
        tags="test2,test3"
    ))
    
    # Test Case 3: Mixed Content Tags
    memory3 = await storage.store(Memory(
        content="Mixed format test",
        tags=["test3", "test4,test5"]
    ))
    
    # Test Case 4: Special Characters
    memory4 = await storage.store(Memory(
        content="Special chars test",
        tags=["test#1", "test@2"]
    ))
    
    # Test Case 5: Empty Tags
    memory5 = await storage.store(Memory(
        content="Empty tags test",
        tags=[]
    ))
    
    # Verification Tests
    tests = [
        # Single tag search
        await storage.search_by_tag(["test1"]),
        
        # Multiple tag search
        await storage.search_by_tag(["test2", "test3"]),
        
        # Special character search
        await storage.search_by_tag(["test#1"]),
        
        # Partial tag search (should not match)
        await storage.search_by_tag(["test"]),
        
        # Case sensitivity test
        await storage.search_by_tag(["TEST1"])
    ]
    
    # Expected results
    expectations = [
        lambda r: len(r) == 1,  # Single tag
        lambda r: len(r) == 2,  # Multiple tags
        lambda r: len(r) == 1,  # Special chars
        lambda r: len(r) == 0,  # Partial match
        lambda r: len(r) == 0   # Case sensitivity
    ]
    
    return verify_test_results(tests, expectations)

@pytest.mark.asyncio
async def test_tag_storage():
    """Main test function that runs all tag storage tests"""
    storage = ChromaMemoryStorage("tests/test_db")

    # storage = ChromaMemoryStorage("path/to/your/db")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate memory data tags')
    parser.add_argument('--db-path', required=True, help='Path to ChromaDB database')
    args = parser.parse_args()
    
    # Initialize storage with provided path
    logger.info(f"Connecting to database at: {args.db_path}")
    storage = ChromaMemoryStorage(args.db_path)
    

    results = await run_tag_integration_tests(storage)
    
    # Check if all tests passed
    for result in results:
        assert result["passed"], f"Test {result['test_number']} failed"

if __name__ == "__main__":
    asyncio.run(test_tag_storage())