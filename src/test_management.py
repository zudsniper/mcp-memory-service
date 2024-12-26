import asyncio
import websockets
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_management_features():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # 1. Store some test data first
        test_memories = [
            {
                "content": "Important meeting with team tomorrow at 10 AM",
                "metadata": {
                    "tags": ["meeting", "important"],
                    "type": "calendar"
                }
            },
            {
                "content": "Need to review the ML model performance metrics",
                "metadata": {
                    "tags": ["ml", "task"],
                    "type": "todo"
                }
            }
        ]

        logger.info("1. Storing test memories...")
        for memory in test_memories:
            await websocket.send(json.dumps({
                "method": "store_memory",
                "params": memory,
                "id": str(hash(memory["content"]))
            }))
            response = await websocket.recv()
            logger.info(f"Store response: {json.dumps(json.loads(response), indent=2)}")

        # 2. Test backup creation
        logger.info("\n2. Testing backup creation...")
        await websocket.send(json.dumps({
            "method": "create_backup",
            "params": {},
            "id": "backup_test"
        }))
        response = await websocket.recv()
        logger.info(f"Backup response: {json.dumps(json.loads(response), indent=2)}")

        # 3. Get database statistics
        logger.info("\n3. Getting database statistics...")
        await websocket.send(json.dumps({
            "method": "get_stats",
            "params": {},
            "id": "stats_test"
        }))
        response = await websocket.recv()
        logger.info(f"Stats response: {json.dumps(json.loads(response), indent=2)}")

        # 4. Test database optimization
        logger.info("\n4. Testing database optimization...")
        await websocket.send(json.dumps({
            "method": "optimize_db",
            "params": {},
            "id": "optimize_test"
        }))
        response = await websocket.recv()
        logger.info(f"Optimization response: {json.dumps(json.loads(response), indent=2)}")

        # 5. Verify data after optimization
        logger.info("\n5. Verifying data retrieval after optimization...")
        await websocket.send(json.dumps({
            "method": "search_by_tag",
            "params": {"tags": ["important"]},
            "id": "verify_test"
        }))
        response = await websocket.recv()
        logger.info(f"Verification response: {json.dumps(json.loads(response), indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_management_features())