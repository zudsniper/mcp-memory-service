import asyncio
import os
import logging
import traceback
import argparse
import sys
from typing import List

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

from .config import (
    CHROMA_PATH, 
    BACKUPS_PATH, 
    SERVER_NAME, 
    SERVER_VERSION
)
from .storage.chroma import ChromaMemoryStorage
from .models.memory import Memory
from .utils.hashing import generate_content_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variable for PyTorch MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class MemoryServer:
    def __init__(self):
        self.server = Server(SERVER_NAME)
        
        # Initialize paths
        logger.info(f"Creating directories if they don't exist...")
        os.makedirs(CHROMA_PATH, exist_ok=True)
        os.makedirs(BACKUPS_PATH, exist_ok=True)
        
        try:
            # Initialize storage
            self.storage = ChromaMemoryStorage(CHROMA_PATH)
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
                ),
                types.Tool(
                    name="delete_memory",
                    description="Delete a specific memory by its hash",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content_hash": {"type": "string"}
                        },
                        "required": ["content_hash"]
                    }
                ),
                types.Tool(
                    name="delete_by_tag",
                    description="Delete all memories with a specific tag",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tag": {"type": "string"}
                        },
                        "required": ["tag"]
                    }
                ),
                types.Tool(
                    name="cleanup_duplicates",
                    description="Find and remove duplicate entries",
                    inputSchema={
                        "type": "object",
                        "properties": {}
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
                elif name == "delete_memory":
                    return await self.handle_delete_memory(arguments)
                elif name == "delete_by_tag":
                    return await self.handle_delete_by_tag(arguments)
                elif name == "cleanup_duplicates":
                    return await self.handle_cleanup_duplicates(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}\n{traceback.format_exc()}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def handle_store_memory(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        metadata = arguments.get("metadata", {})
        
        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]
        
        try:
            # Create memory object
            content_hash = generate_content_hash(content, metadata)
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=metadata.get("tags", []),
                memory_type=metadata.get("type")
            )
            
            # Store memory
            success, message = await self.storage.store(memory)
            return [types.TextContent(type="text", text=message)]
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error storing memory: {str(e)}")]

    async def handle_retrieve_memory(self, arguments: dict) -> List[types.TextContent]:
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)
        
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        try:
            results = await self.storage.retrieve(query, n_results)
            
            if not results:
                return [types.TextContent(type="text", text="No matching memories found")]
            
            formatted_results = []
            for i, result in enumerate(results):
                memory_info = [
                    f"Memory {i+1}:",
                    f"Content: {result.memory.content}",
                    f"Hash: {result.memory.content_hash}",
                    f"Relevance Score: {result.relevance_score:.2f}"
                ]
                if result.memory.tags:
                    memory_info.append(f"Tags: {', '.join(result.memory.tags)}")
                memory_info.append("---")
                formatted_results.append("\n".join(memory_info))
            
            return [types.TextContent(
                type="text",
                text="Found the following memories:\n\n" + "\n".join(formatted_results)
            )]
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error retrieving memories: {str(e)}")]

    async def handle_search_by_tag(self, arguments: dict) -> List[types.TextContent]:
        tags = arguments.get("tags", [])
        
        if not tags:
            return [types.TextContent(type="text", text="Error: Tags are required")]
        
        try:
            memories = await self.storage.search_by_tag(tags)
            
            if not memories:
                return [types.TextContent(
                    type="text",
                    text=f"No memories found with tags: {', '.join(tags)}"
                )]
            
            formatted_results = []
            for i, memory in enumerate(memories):
                memory_info = [
                    f"Memory {i+1}:",
                    f"Content: {memory.content}",
                    f"Hash: {memory.content_hash}",
                    f"Tags: {', '.join(memory.tags)}"
                ]
                if memory.memory_type:
                    memory_info.append(f"Type: {memory.memory_type}")
                memory_info.append("---")
                formatted_results.append("\n".join(memory_info))
            
            return [types.TextContent(
                type="text",
                text="Found the following memories:\n\n" + "\n".join(formatted_results)
            )]
        except Exception as e:
            logger.error(f"Error searching by tags: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error searching by tags: {str(e)}")]

    async def handle_delete_memory(self, arguments: dict) -> List[types.TextContent]:
        content_hash = arguments.get("content_hash")
        success, message = await self.storage.delete(content_hash)
        return [types.TextContent(type="text", text=message)]

    async def handle_delete_by_tag(self, arguments: dict) -> List[types.TextContent]:
        tag = arguments.get("tag")
        count, message = await self.storage.delete_by_tag(tag)
        return [types.TextContent(type="text", text=message)]

    async def handle_cleanup_duplicates(self, arguments: dict) -> List[types.TextContent]:
        count, message = await self.storage.cleanup_duplicates()
        return [types.TextContent(type="text", text=message)]

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
                    server_name=SERVER_NAME,
                    server_version=SERVER_VERSION,
                    capabilities=memory_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        raise

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()