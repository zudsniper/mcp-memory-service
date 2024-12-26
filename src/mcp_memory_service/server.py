import asyncio
import os
import time
import logging
import chromadb
import traceback
from sentence_transformers import SentenceTransformer
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variable for PyTorch MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Constants to be set in Claude Desktop Config
# CHROMA_PATH = os.path.expanduser("path_to/chroma_db")
# BACKUPS_PATH = os.path.expanduser("path_to/backups")

class MemoryServer:
    def __init__(self):
        self.server = Server("memory")
        
        # Initialize paths
        logger.info(f"Creating directories if they don't exist...")
        os.makedirs(CHROMA_PATH, exist_ok=True)
        os.makedirs(BACKUPS_PATH, exist_ok=True)
        
        try:
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
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
            
            try:
                self.collection = self.chroma_client.get_collection(name="memory_collection")
                logger.info("Found existing collection")
            except Exception as e:
                logger.info(f"Creating new collection (reason: {str(e)})")
                self.collection = self.chroma_client.create_collection(
                    name="memory_collection",
                    metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise
        
        # Register handlers
        self.register_handlers()
        logger.info("Server initialization complete")

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
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> List[types.TextContent]:
            try:
                logger.debug(f"Tool call received: {name} with arguments {arguments}")
                if arguments is None:
                    arguments = {}
                
                if name == "store_memory":
                    return await self.handle_store_memory(arguments)
                elif name == "retrieve_memory":
                    return await self.handle_retrieve_memory(arguments)
                elif name == "search_by_tag":
                    return await self.handle_search_by_tag(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}\
{traceback.format_exc()}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def handle_store_memory(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        metadata = arguments.get("metadata", {})
        
        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]
        
        try:
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
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}\
{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error storing memory: {str(e)}")]

    async def handle_retrieve_memory(self, arguments: dict) -> List[types.TextContent]:
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)
        
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        try:
            results = self.collection.query(
                query_embeddings=[self.model.encode(query).tolist()],
                n_results=n_results
            )
            
            if not results["documents"][0]:
                return [types.TextContent(type="text", text="No matching memories found")]
            
            formatted_results = []
            for i in range(len(results["ids"][0])):
                memory = (
                    f"Memory {i+1}:\
"
                    f"Content: {results['documents'][0][i]}\
"
                    f"Relevance Score: {1 - results['distances'][0][i]:.2f}\
"
                    "---"
                )
                formatted_results.append(memory)
                
            return [types.TextContent(
                type="text",
                text="Found the following memories:\
\
" + "\
".join(formatted_results)
            )]
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}\
{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error retrieving memories: {str(e)}")]

    async def handle_search_by_tag(self, arguments: dict) -> List[types.TextContent]:
        tags = arguments.get("tags", [])
        
        if not tags:
            return [types.TextContent(type="text", text="Error: Tags are required")]
        
        try:
            results = self.collection.get()
            
            if not results or not results.get("ids"):
                return [types.TextContent(type="text", text="No memories found")]
            
            matching_memories = []
            for i, metadata in enumerate(results["metadatas"]):
                if metadata and "tags_str" in metadata:
                    memory_tags = set(metadata["tags_str"].split(","))
                    if any(tag in memory_tags for tag in tags):
                        memory = (
                            f"Memory {len(matching_memories)+1}:\
"
                            f"Content: {results['documents'][i]}\
"
                            f"Tags: {metadata['tags_str']}\
"
                            "---"
                        )
                        matching_memories.append(memory)
            
            if not matching_memories:
                return [types.TextContent(type="text", text=f"No memories found with tags: {', '.join(tags)}")]
            
            return [types.TextContent(
                type="text",
                text="Found the following memories:\
\
" + "\
".join(matching_memories)
            )]
        except Exception as e:
            logger.error(f"Error searching by tags: {str(e)}\
{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error searching by tags: {str(e)}")]

def parse_args():
    parser = argparse.ArgumentParser(
        description="MCP Memory Service - A semantic memory service using the Model Context Protocol"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=CHROMA_PATH,
        help="Path to ChromaDB storage"
    )
    return parser.parse_args()

async def async_main():
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    global CHROMA_PATH
    CHROMA_PATH = args.chroma_path
    
    logger.info(f"Starting MCP Memory Service with ChromaDB path: {CHROMA_PATH}")
    
    try:
        memory_server = MemoryServer()
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Server started and ready to handle requests")
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
    except Exception as e:
        logger.error(f"Server error: {str(e)}\
{traceback.format_exc()}")
        raise

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}\
{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()`
