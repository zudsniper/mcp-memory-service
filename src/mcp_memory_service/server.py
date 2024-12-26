from typing import Any, List
import asyncio
import os
import time
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from mcp.server.models import InitializationOptions, Tool
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for PyTorch MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Constants
CHROMA_PATH = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/ai/claude-memory/chroma_db")
BACKUPS_PATH = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/ai/claude-memory/backups")

class MemoryServer:
    def __init__(self):
        self.server = Server("memory")
        
        # Initialize paths
        os.makedirs(CHROMA_PATH, exist_ok=True)
        os.makedirs(BACKUPS_PATH, exist_ok=True)
        
        # Initialize components
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info(f"Initializing ChromaDB at {CHROMA_PATH}...")
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
                persist_directory=CHROMA_PATH
            )
        )
        
        # Initialize or get collection
        try:
            self.collection = self.chroma_client.get_collection(name="memory_collection")
            logger.info("Found existing collection")
        except Exception:
            logger.info("Creating new collection")
            self.collection = self.chroma_client.create_collection(
                name="memory_collection",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Register handlers
        self.register_handlers()

    def register_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="store_memory",
                    description="Store new information with optional tags",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "metadata": {
                                "type": "object",
                                "properties": {
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                    "type": {"type": "string"}
                                }
                            }
                        },
                        "required": ["content"]
                    }
                ),
                types.Tool(
                    name="retrieve_memory",
                    description="Find relevant memories based on query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "n_results": {"type": "number", "default": 5}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="search_by_tag",
                    description="Search memories by tags",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tags": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["tags"]
                    }
                ),
                types.Tool(
                    name="create_backup",
                    description="Create a backup of the memory database",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_stats",
                    description="Get memory statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> List[types.TextContent]:
            try:
                if arguments is None:
                    arguments = {}
                
                if name == "store_memory":
                    return await self.handle_store_memory(arguments)
                elif name == "retrieve_memory":
                    return await self.handle_retrieve_memory(arguments)
                elif name == "search_by_tag":
                    return await self.handle_search_by_tag(arguments)
                elif name == "create_backup":
                    return await self.handle_create_backup()
                elif name == "get_stats":
                    return await self.handle_get_stats()
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def handle_store_memory(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        metadata = arguments.get("metadata", {})
        
        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]
        
        # Process metadata
        processed_metadata = {}
        if metadata:
            if "tags" in metadata:
                processed_metadata["tags_str"] = ",".join(metadata["tags"])
            if "type" in metadata:
                processed_metadata["type"] = metadata["type"]
        
        processed_metadata["timestamp"] = str(time.time())
        
        # Generate embedding and store
        embedding = self.model.encode(content).tolist()
        memory_id = str(int(time.time() * 1000))
        
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[processed_metadata],
            ids=[memory_id]
        )
        
        return [types.TextContent(
            type="text",
            text=f"Successfully stored memory with ID: {memory_id}"
        )]

    async def handle_retrieve_memory(self, arguments: dict) -> List[types.TextContent]:
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)
        
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        results = self.collection.query(
            query_embeddings=[self.model.encode(query).tolist()],
            n_results=n_results
        )
        
        if not results["documents"][0]:
            return [types.TextContent(type="text", text="No matching memories found")]
        
        formatted_results = []
        for i in range(len(results["ids"][0])):
            memory = (
                f"Memory {i+1}:\n"
                f"Content: {results['documents'][0][i]}\n"
                f"Relevance Score: {1 - results['distances'][0][i]:.2f}\n"
                "---"
            )
            formatted_results.append(memory)
            
        return [types.TextContent(
            type="text",
            text="Found the following memories:\n\n" + "\n".join(formatted_results)
        )]

    async def handle_search_by_tag(self, arguments: dict) -> List[types.TextContent]:
        tags = arguments.get("tags", [])
        
        if not tags:
            return [types.TextContent(type="text", text="Error: Tags are required")]
        
        results = self.collection.get()
        
        if not results or not results.get("ids"):
            return [types.TextContent(type="text", text="No memories found")]
        
        matching_memories = []
        for i, metadata in enumerate(results["metadatas"]):
            if metadata and "tags_str" in metadata:
                memory_tags = set(metadata["tags_str"].split(","))
                if any(tag in memory_tags for tag in tags):
                    memory = (
                        f"Memory {len(matching_memories)+1}:\n"
                        f"Content: {results['documents'][i]}\n"
                        f"Tags: {metadata['tags_str']}\n"
                        "---"
                    )
                    matching_memories.append(memory)
        
        if not matching_memories:
            return [types.TextContent(type="text", text=f"No memories found with tags: {', '.join(tags)}")]
        
        return [types.TextContent(
            type="text",
            text="Found the following memories:\n\n" + "\n".join(matching_memories)
        )]

    async def handle_create_backup(self) -> List[types.TextContent]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUPS_PATH, f"memory_backup_{timestamp}")
        
        try:
            # Persist current state
            self.chroma_client.persist()
            
            # Create backup directory
            os.makedirs(backup_path)
            
            # Copy database files
            import shutil
            shutil.copytree(CHROMA_PATH, os.path.join(backup_path, "chroma_db"))
            
            return [types.TextContent(
                type="text",
                text=f"Successfully created backup at: {backup_path}"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error creating backup: {str(e)}"
            )]

    async def handle_get_stats(self) -> List[types.TextContent]:
        try:
            results = self.collection.get()
            
            num_memories = len(results["ids"]) if results.get("ids") else 0
            unique_tags = set()
            types_count = {}
            
            for metadata in results.get("metadatas", []):
                if metadata:
                    if "tags_str" in metadata:
                        unique_tags.update(metadata["tags_str"].split(","))
                    if "type" in metadata:
                        types_count[metadata["type"]] = types_count.get(metadata["type"], 0) + 1
            
            stats = (
                f"Total Memories: {num_memories}\n"
                f"Unique Tags: {len(unique_tags)}\n"
                f"Tags: {', '.join(sorted(unique_tags))}\n"
                f"Memory Types: {dict(sorted(types_count.items()))}"
            )
            
            return [types.TextContent(type="text", text=stats)]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error getting stats: {str(e)}"
            )]

async def main():
    memory_server = MemoryServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await memory_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="memory",
                server_version="0.1.0",
                capabilities=memory_server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )