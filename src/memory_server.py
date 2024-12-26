import asyncio
import websockets
import json
import chromadb
from sentence_transformers import SentenceTransformer
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import os
from utils.db_manager import DatabaseManager, CHROMA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Tool:
    name: str
    description: str
    params: Dict[str, Any]

class MemoryServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info(f"Initializing ChromaDB at {CHROMA_PATH}...")
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            try:
                self.collection = self.chroma_client.get_collection(name="memory_collection")
                logger.info("Found existing collection")
            except:
                logger.info("Creating new collection")
                self.collection = self.chroma_client.create_collection(
                    name="memory_collection",
                    metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {str(e)}")
            raise
        
        # Initialize database manager
        self.db_manager = DatabaseManager(self.chroma_client)
        # Ensure db_manager has the same collection reference
        self.db_manager.collection = self.collection

        # Define available tools
        self.tools = {
            "store_memory": Tool(
                name="store_memory",
                description="Store new information with optional tags",
                params={
                    "content": "string",
                    "metadata": {
                        "tags": ["string"],
                        "type": "string"
                    }
                }
            ),
            "retrieve_memory": Tool(
                name="retrieve_memory",
                description="Find relevant memories based on query",
                params={
                    "query": "string",
                    "n_results": "number"
                }
            ),
            "search_by_tag": Tool(
                name="search_by_tag",
                description="Search memories by tags",
                params={
                    "tags": ["string"]
                }
            ),
            "create_backup": Tool(
                name="create_backup",
                description="Create a backup of the memory database",
                params={}
            ),
            "get_stats": Tool(
                name="get_stats",
                description="Get detailed statistics about stored memories",
                params={}
            ),
            "optimize_db": Tool(
                name="optimize_db",
                description="Optimize the database for better performance",
                params={}
            )
        }

    def _reinitialize_collection(self):
        """Helper method to reinitialize the collection reference"""
        try:
            self.collection = self.chroma_client.get_collection("memory_collection")
            # Also update the db_manager's collection reference
            self.db_manager.collection = self.collection
            logger.info("Successfully reinitialized collection reference")
            return True
        except Exception as e:
            logger.error(f"Failed to reinitialize collection: {str(e)}")
            return False

    async def store_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = params.get("content")
            metadata = params.get("metadata", {})
            
            if not content:
                raise ValueError("Content is required")
            
            # Create a new metadata dictionary with processed values
            processed_metadata = {}
            
            # Convert tags list to string and process other metadata
            if metadata:
                if "tags" in metadata:
                    processed_metadata["tags_str"] = ",".join(metadata["tags"])
                if "type" in metadata:
                    processed_metadata["type"] = metadata["type"]
            
            # Add timestamp
            processed_metadata["timestamp"] = str(time.time())
                
            # Generate embedding
            embedding = self.model.encode(content).tolist()
            
            # Generate unique ID
            memory_id = str(int(time.time() * 1000))
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[processed_metadata],
                ids=[memory_id]
            )
            
            return {"status": "success", "memory_id": memory_id}
            
        except Exception as e:
            logger.error(f"Store memory failed: {str(e)}")
            raise

    async def retrieve_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = params.get("query")
            n_results = params.get("n_results", 5)
            
            if not query:
                raise ValueError("Query is required")
                
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            memories = []
            for i in range(len(results["ids"][0])):
                memories.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            return {"memories": memories}
            
        except Exception as e:
            logger.error(f"Retrieve memory failed: {str(e)}")
            raise

    async def search_by_tag(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            tags = params.get("tags", [])
            
            if not tags:
                raise ValueError("Tags are required")
            
            # Ensure we have a valid collection reference
            if not self._reinitialize_collection():
                raise Exception("Failed to access memory collection")
            
            # Get all memories (we'll filter them manually for more flexibility)
            results = self.collection.get()
            
            if not results or not results.get("ids"):
                return {"memories": []}
            
            # Filter results for memories that have any of the requested tags
            filtered_ids = []
            filtered_documents = []
            filtered_metadatas = []
            
            for i, metadata in enumerate(results["metadatas"]):
                if metadata and "tags_str" in metadata:
                    memory_tags = set(metadata["tags_str"].split(","))
                    if any(tag in memory_tags for tag in tags):
                        filtered_ids.append(results["ids"][i])
                        filtered_documents.append(results["documents"][i])
                        filtered_metadatas.append(metadata)
            
            # Format results
            memories = []
            for i in range(len(filtered_ids)):
                memories.append({
                    "id": filtered_ids[i],
                    "content": filtered_documents[i],
                    "metadata": filtered_metadatas[i]
                })
            
            return {"memories": memories}
            
        except Exception as e:
            logger.error(f"Search by tag failed: {str(e)}")
            raise

    async def create_backup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            backup_path = self.db_manager.create_backup()
            return {"status": "backup_created", "path": backup_path}
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise

    async def get_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            stats = self.db_manager.get_memory_stats()
            return {"stats": stats}
        except Exception as e:
            logger.error(f"Stats generation failed: {str(e)}")
            raise

    async def optimize_db(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            success = self.db_manager.optimize_database()
            if success:
                # Update our collection reference after optimization
                self._reinitialize_collection()
                return {"status": "optimization_complete"}
            return {"status": "no_data_to_optimize"}
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

    async def handle_connection(self, websocket):
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    method = data.get("method")
                    params = data.get("params", {})
                    request_id = data.get("id")

                    if method == "list_tools":
                        response = {
                            "tools": [
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.params
                                }
                                for tool in self.tools.values()
                            ]
                        }
                    else:
                        if method not in self.tools:
                            raise ValueError(f"Unknown method: {method}")
                        
                        handler = getattr(self, method)
                        response = await handler(params)

                    await websocket.send(json.dumps({
                        "result": response,
                        "id": request_id,
                        "error": None
                    }))

                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}")
                    await websocket.send(json.dumps({
                        "result": {},
                        "id": request_id if 'request_id' in locals() else None,
                        "error": {"message": str(e)}
                    }))

        except Exception as e:
            logger.error(f"Connection error: {str(e)}")

    async def start(self):
        server = await websockets.serve(self.handle_connection, self.host, self.port)
        logger.info(f"Memory server started on ws://{self.host}:{self.port}")
        await server.wait_closed()

if __name__ == "__main__":
    server = MemoryServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        logger.info("Server shutting down")