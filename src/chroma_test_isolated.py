import chromadb
import json
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple, Dict
import asyncio
from dataclasses import dataclass

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Memory:
    content: str
    content_hash: str
    tags: List[str]
    memory_type: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def generate_content_hash(content, metadata):
     return str(hash(content + json.dumps(metadata)))
     

class ChromaMemoryStorage:
    def __init__(self, path: str = "chroma_test_db"):
        """Initialize ChromaDB storage with proper embedding function."""
        self.path = path
        
        # Initialize sentence transformer first
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embedding function for ChromaDB
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        
        # Initialize ChromaDB with new client format
        self.client = chromadb.PersistentClient(
            path=path
        )
        
        # Get or create collection with proper embedding function
        try:
            self.collection = self.client.get_or_create_collection(
                name="memory_collection",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            logger.info("Collection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    def sanitized(self, tags):
        if tags is None:
            return json.dumps("")
        
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            raise ValueError("Tags must be a string or list of strings")
            
        # Ensure all elements are strings and remove any invalid ones    
        sanitized = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            tag = tag.strip()
            if tag:
                sanitized.append(tag)
                
        logger.error(f"**************Sanitized: {json.dumps(sanitized)}")
        # Convert to JSON string
        return json.dumps(sanitized)

    def _format_metadata_for_chroma(self, memory: Memory) -> dict:
        """Formats metadata for ChromaDB, ensuring no list types are included."""
        metadata = {
            "type": memory.memory_type,
            "content_hash": memory.content_hash,
        }
        return metadata

    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """Store a memory with proper embedding handling."""
        try:
            # Check for duplicates
            existing = self.collection.get(
                where={"content_hash": memory.content_hash}
            )
            if existing["ids"]:
                return False, "Duplicate content detected"
            
            # Format metadata properly
            metadata = self._format_metadata_for_chroma(memory)
            
            # Add additional metadata
            metadata.update(memory.metadata)

            # Generate ID based on content hash
            memory_id = memory.content_hash
            
            # Add to collection - embedding will be automatically generated
            self.collection.add(
                documents=[memory.content],
                metadatas=[metadata],
                ids=[memory_id]
            )
            
            return True, f"Successfully stored memory with ID: {memory_id}"
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return False
        
    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
       """Retrieves memories that match any of the specified tags."""
       try:
           results = self.collection.get(
               include=["metadatas", "documents"]
           )

           memories = []
           if results["ids"]:
               for i, doc in enumerate(results["documents"]):
                   memory_meta = results["metadatas"][i]
                    
                    # Deserialize tags from the string
                   try:
                       retrieved_tags_string = memory_meta.get("tags", "[]")
                       retrieved_tags = json.loads(retrieved_tags_string)
                   except json.JSONDecodeError:
                        retrieved_tags = [] # Handle the case where the stored tags are not valid JSON
                    
                   # Check if any of the searched tags are in the retrieved tags list
                   if any(tag in retrieved_tags for tag in tags):
                       memory = Memory(
                           content=doc,
                           content_hash=memory_meta["content_hash"],
                           tags=retrieved_tags,
                           memory_type=memory_meta.get("type")
                        )
                       memories.append(memory)
            
           return memories
        
       except Exception as e:
           logger.error(f"Error searching by tags: {e}")
           return []
    
    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Deletes memories that match the specified tag."""
        try:
            # Get all the documents from ChromaDB
            results = self.collection.get(
                include=["metadatas"]
            )

            ids_to_delete = []
            if results["ids"]:
                for i, meta in enumerate(results["metadatas"]):
                    try:
                        retrieved_tags_string = meta.get("tags", "[]")
                        retrieved_tags = json.loads(retrieved_tags_string)
                    except json.JSONDecodeError:
                        retrieved_tags = []

                    if tag in retrieved_tags:
                        ids_to_delete.append(results["ids"][i])

            if not ids_to_delete:
                return 0, f"No memories found with tag: {tag}"

            # Delete memories
            self.collection.delete(ids=ids_to_delete)

            return len(ids_to_delete), f"Successfully deleted {len(ids_to_delete)} memories with tag: {tag}"

        except Exception as e:
            logger.error(f"Error deleting memories by tag: {e}")
            return 0, f"Error deleting memories by tag: {e}"

    async def cleanup(self):
      """Cleans the database"""
      self.client.delete_collection(name="memory_collection")
      logger.info("Collection deleted")
      
    
async def main():
    storage = ChromaMemoryStorage()

    # Sample data
    memories_data = [
        {
            "content": "Meeting with team tomorrow at 10 AM",
            "memory_type": "calendar",
            "tags_str": "meeting,important"
        },
        {
            "content": "Review ML model performance metrics",
            "memory_type": "todo",
            "tags_str": "ml,review"
        },
        {
            "content": "Backup the database weekly",
            "memory_type": "reminder",
            "tags_str": "backup,database"
        },
        {
            "content": "Another meeting",
            "memory_type": "calendar",
            "tags_str": "meeting,test"
        }
    ]

    stored_memories = []
    for data in memories_data:
       tags = [tag.strip() for tag in data.get("tags_str", "").split(",") if tag.strip()]
       memory = Memory(
            content=data["content"],
            content_hash=generate_content_hash(data["content"], data),
            tags=tags,
            memory_type=data["memory_type"],
           metadata = {"tags":storage.sanitized(tags), **data}
        )

       success, message = await storage.store(memory)
       logger.info(f"Store response: [{success}, {message}]")
       if success:
            stored_memories.append(memory)


    # Search by tag
    search_tag = "meeting"
    logger.info(f"Searching for memories with tag: {search_tag}")
    tag_results = await storage.search_by_tag([search_tag])
    logger.info(f"Found {len(tag_results)} memories with tag '{search_tag}'")
    for memory in tag_results:
        logger.info(f"  Content: {memory.content}")
        logger.info(f"  Tags: {memory.tags}")
    
    # Search for "ml"
    search_tag = "ml"
    logger.info(f"Searching for memories with tag: {search_tag}")
    tag_results = await storage.search_by_tag([search_tag])
    logger.info(f"Found {len(tag_results)} memories with tag '{search_tag}'")
    for memory in tag_results:
        logger.info(f"  Content: {memory.content}")
        logger.info(f"  Tags: {memory.tags}")

    # Delete by tag
    delete_tag = "meeting"
    logger.info(f"Deleting memories with tag: {delete_tag}")
    count, message = await storage.delete_by_tag(delete_tag)
    logger.info(f"Delete by tag result: {message}, count:{count}")


    # verify deletion
    search_tag = "meeting"
    logger.info(f"Searching for memories with tag: {search_tag}")
    tag_results = await storage.search_by_tag([search_tag])
    logger.info(f"Found {len(tag_results)} memories with tag '{search_tag}'")

    for memory in tag_results:
        logger.info(f"  Content: {memory.content}")
        logger.info(f"  Tags: {memory.tags}")
    
    # Clean up database
    await storage.cleanup()


if __name__ == "__main__":
    asyncio.run(main())