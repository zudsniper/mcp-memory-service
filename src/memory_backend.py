from mcp.server import MCPServer, Tool, Request, Response
import chromadb
from sentence_transformers import SentenceTransformer
import time
import json
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMCPServer(MCPServer):
    def __init__(self):
        super().__init__("memory-server")
        
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="memory_collection",
            metadata={"hnsw:space": "cosine"}
        )

        # Register tools
        self.register_tool(
            Tool(
                "store_memory",
                self.store_memory,
                "Store new information with optional tags",
                {
                    "content": "string",
                    "metadata?": {
                        "tags?": ["string"],
                        "type?": "string"
                    }
                }
            )
        )
        
        self.register_tool(
            Tool(
                "retrieve_memory",
                self.retrieve_memory,
                "Find relevant memories based on query",
                {
                    "query": "string",
                    "n_results?": "number"
                }
            )
        )
        
        self.register_tool(
            Tool(
                "search_by_tag",
                self.search_by_tag,
                "Search memories by tags",
                {
                    "tags": ["string"]
                }
            )
        )

    async def store_memory(self, request: Request) -> Response:
        try:
            content = request.params.get("content")
            if not content:
                return Response.error("No content provided")
            
            metadata = request.params.get("metadata", {})
            if "tags" in metadata and isinstance(metadata["tags"], list):
                metadata["tags_str"] = ",".join(metadata["tags"])
                del metadata["tags"]
                
            metadata["timestamp"] = str(int(time.time()))
            
            # Generate embedding
            embedding = self.model.encode(content).tolist()
            memory_id = f"mem_{int(time.time())}_{hash(content)}"
            
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[memory_id]
            )
            
            return Response.success({"status": "stored", "id": memory_id})
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return Response.error(str(e))

    async def retrieve_memory(self, request: Request) -> Response:
        try:
            query = request.params.get("query")
            if not query:
                return Response.error("No query provided")
            
            n_results = request.params.get("n_results", 3)
            query_embedding = self.model.encode(query).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            
            memories = []
            if results["documents"] and len(results["documents"]) > 0:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    if meta and "tags_str" in meta:
                        meta["tags"] = meta["tags_str"].split(",")
                        del meta["tags_str"]
                    memories.append({
                        "content": doc,
                        "metadata": meta or {}
                    })
            
            return Response.success({"memories": memories})
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return Response.error(str(e))

    async def search_by_tag(self, request: Request) -> Response:
        try:
            tags = request.params.get("tags", [])
            if not tags:
                return Response.error("No tags provided")
            
            results = self.collection.get()
            memories = []
            
            if results["documents"]:
                for doc, meta, id in zip(results["documents"], results["metadatas"], results["ids"]):
                    if meta and "tags_str" in meta:
                        memory_tags = meta["tags_str"].split(",")
                        if any(tag in memory_tags for tag in tags):
                            meta["tags"] = memory_tags
                            del meta["tags_str"]
                            memories.append({
                                "content": doc,
                                "metadata": meta,
                                "id": id
                            })
            
            return Response.success({"memories": memories})
            
        except Exception as e:
            logger.error(f"Error searching by tags: {str(e)}")
            return Response.error(str(e))

if __name__ == "__main__":
    server = MemoryMCPServer()
    server.start()