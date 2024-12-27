import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Tuple, Set
import time

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult
from ..utils.hashing import generate_content_hash

logger = logging.getLogger(__name__)

class ChromaMemoryStorage(MemoryStorage):
    def __init__(self, path: str):
        """Initialize ChromaDB storage."""
        self.path = path
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name="memory_collection")
            logger.info("Found existing collection")
        except Exception as e:
            logger.info(f"Creating new collection (reason: {str(e)})")
            self.collection = self.client.create_collection(
                name="memory_collection",
                metadata={"hnsw:space": "cosine"}
            )

    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """Store a memory with duplicate checking."""
        try:
            # Check for duplicates using content hash
            existing = self.collection.get(
                where={"content_hash": memory.content_hash}
            )
            if existing["ids"]:
                return False, "Duplicate content detected, skipping storage"
            
            # Generate embedding if not provided
            if not memory.embedding:
                memory.embedding = self.model.encode(memory.content).tolist()
            
            # Generate ID and store
            memory_id = str(int(time.time() * 1000))
            
            self.collection.add(
                embeddings=[memory.embedding],
                documents=[memory.content],
                metadatas=[memory.to_dict()],
                ids=[memory_id]
            )
            
            return True, f"Successfully stored memory with ID: {memory_id}"
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return False, f"Error storing memory: {str(e)}"

    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Retrieve memories using semantic search."""
        try:
            query_embedding = self.model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            memory_results = []
            for i in range(len(results["ids"][0])):
                memory = Memory.from_dict(
                    {
                        "content": results["documents"][0][i],
                        **results["metadatas"][0][i]
                    },
                    embedding=results["embeddings"][0][i] if "embeddings" in results else None
                )
                relevance = 1 - results["distances"][0][i]
                memory_results.append(MemoryQueryResult(memory, relevance))
            
            return memory_results
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return []

    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
        """Search memories by tags using exact matching."""
        try:
            # Get all memories - we'll filter by tags in Python
            # since ChromaDB doesn't support array contains operations
            results = self.collection.get()
            
            memories = []
            for i, metadata in enumerate(results["metadatas"]):
                if not metadata.get("tags_str"):
                    continue
                    
                memory_tags = set(metadata["tags_str"].split(","))
                if any(tag in memory_tags for tag in tags):
                    memory = Memory.from_dict(
                        {
                            "content": results["documents"][i],
                            **metadata
                        },
                        embedding=results["embeddings"][i] if "embeddings" in results else None
                    )
                    memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Error searching by tags: {str(e)}")
            return []

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Delete a memory by its hash."""
        try:
            # Get existing memory to confirm deletion
            existing = self.collection.get(
                where={"content_hash": content_hash}
            )
            if not existing["ids"]:
                return False, f"No memory found with hash {content_hash}"
            
            # Delete the memory
            self.collection.delete(
                where={"content_hash": content_hash}
            )
            return True, f"Memory with hash {content_hash} deleted successfully"
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            return False, f"Error deleting memory: {str(e)}"

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Delete memories by tag."""
        try:
            results = self.collection.get()
            to_delete = []
            
            for i, metadata in enumerate(results["metadatas"]):
                if metadata.get("tags_str"):
                    memory_tags = set(metadata["tags_str"].split(","))
                    if tag in memory_tags:
                        to_delete.append(results["ids"][i])
            
            if to_delete:
                self.collection.delete(
                    ids=to_delete
                )
                return len(to_delete), f"Deleted {len(to_delete)} memories with tag '{tag}'"
            return 0, f"No memories found with tag '{tag}'"
        except Exception as e:
            logger.error(f"Error deleting memories by tag: {str(e)}")
            return 0, f"Error deleting memories by tag: {str(e)}"

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Remove duplicate memories based on content hash."""
        try:
            results = self.collection.get()
            seen_hashes: Set[str] = set()
            duplicates = []
            
            for i, metadata in enumerate(results["metadatas"]):
                content_hash = metadata.get("content_hash")
                if not content_hash:
                    # Generate hash if missing
                    content_hash = generate_content_hash(
                        results["documents"][i],
                        metadata
                    )
                
                if content_hash in seen_hashes:
                    duplicates.append(results["ids"][i])
                else:
                    seen_hashes.add(content_hash)
            
            if duplicates:
                self.collection.delete(ids=duplicates)
                return len(duplicates), f"Removed {len(duplicates)} duplicate memories"
            return 0, "No duplicates found"
        except Exception as e:
            logger.error(f"Error cleaning up duplicates: {str(e)}")
            return 0, f"Error cleaning up duplicates: {str(e)}"