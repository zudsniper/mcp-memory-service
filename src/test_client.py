# test_client.py
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_memory_operations():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # 1. List available tools
        logger.info("1. Listing available tools...")
        await websocket.send(json.dumps({
            "method": "list_tools",
            "params": {},
            "id": "list_test"
        }))
        response = await websocket.recv()
        logger.info(f"Tools response: {json.dumps(json.loads(response), indent=2)}")

        # 2. Store some memories
        test_memories = [
            {
                "content": "Remember to update documentation for API changes",
                "metadata": {"type": "todo"}
            },
            {
                "content": "Team meeting notes: Discussed new feature rollout plan",
                "metadata": {"type": "notes"}
            }
        ]

        logger.info("\n2. Storing test memories...")
        for memory in test_memories:
            await websocket.send(json.dumps({
                "method": "store_memory",
                "params": memory,
                "id": str(hash(memory["content"]))
            }))
            response = await websocket.recv()
            logger.info(f"Store response: {json.dumps(json.loads(response), indent=2)}")

        # 3. Test semantic search
        logger.info("\n3. Testing semantic search...")
        await websocket.send(json.dumps({
            "method": "retrieve_memory",
            "params": {
                "query": "What needs to be done for the API?",
                "n_results": 2
            },
            "id": "search_test"
        }))
        response = await websocket.recv()
        logger.info(f"Search response: {json.dumps(json.loads(response), indent=2)}")

        # 4. Get statistics
        logger.info("\n4. Getting memory statistics...")
        await websocket.send(json.dumps({
            "method": "get_stats",
            "params": {},
            "id": "stats_test"
        }))
        response = await websocket.recv()
        logger.info(f"Stats response: {json.dumps(json.loads(response), indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_memory_operations())
