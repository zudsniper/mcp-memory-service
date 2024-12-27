import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
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

    def _format_metadata_for_chroma(self, memory: Memory) -> Dict[str, Any]:
        """Format metadata to be compatible with ChromaDB requirements."""
        # ChromaDB expects simple types (str, int, float, bool)
        metadata = {
            "content_hash": memory.content_hash,
            "tags": ",".join(memory.tags) if memory.tags else "",
            "memory_type": memory.memory_type if memory.memory_type else "",
            "timestamp": str(memory.timestamp.timestamp())
        }
        
        # Add any additional metadata that's of simple types
        for key, value in memory.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        
        return metadata

    def _generate_embedding(self, content: str) -> List[float]:
        """Generate embedding for content."""
        try:
            return self.model.encode(content).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

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
                memory.embedding = self._generate_embedding(memory.content)
            
            # Format metadata for ChromaDB
            metadata = self._format_metadata_for_chroma(memory)
            
            # Generate ID and store
            memory_id = str(int(time.time() * 1000))
            
            self.collection.add(
                embeddings=[memory.embedding],
                documents=[memory.content],
                metadatas=[metadata],
                ids=[memory_id]
            )
            
            return True, f"Successfully stored memory with ID: {memory_id}"
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return False, f"Error storing memory: {str(e)}"

    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Retrieve memories using semantic search."""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                return []
            
            memory_results = []
            for i in range(len(results["ids"][0])):
                # Convert metadata back to Memory format
                metadata = results["metadatas"][0][i]
                tags = metadata.get("tags", "").split(",") if metadata.get("tags") else []
                memory_type = metadata.get("memory_type", "")
                
                memory = Memory(
                    content=results["documents"][0][i],
                    content_hash=metadata["content_hash"],
                    tags=[tag for tag in tags if tag],
                    memory_type=memory_type if memory_type else None
                )
                
                # Calculate cosine similarity from distance
                relevance = 1 - results["distances"][0][i]
                memory_results.append(MemoryQueryResult(memory, relevance))
            
            return memory_results
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return []

    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
        """Search memories by tags."""
        try:
            # Convert tags list to comma-separated string for search
            tag_str = ",".join(tags)
            results = self.collection.get(
                where={"tags": {"$contains": tag_str}}
            )
            
            if not results["ids"]:
                return []
            
            memories = []
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i]
                memory_tags = metadata.get("tags", "").split(",")
                memory = Memory(
                    content=results["documents"][i],
                    content_hash=metadata["content_hash"],
                    tags=[tag for tag in memory_tags if tag],
                    memory_type=metadata.get("memory_type")
                )
                memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Error searching by tags: {str(e)}")
            return []

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Delete a memory by its hash."""
        try:
            existing = self.collection.get(
                where={"content_hash": content_hash}
            )
            if not existing["ids"]:
                return False, f"No memory found with hash {content_hash}"
            
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
            results = self.collection.get(
                where={"tags": {"$contains": tag}}
            )
            
            if not results["ids"]:
                return 0, f"No memories found with tag '{tag}'"
            
            self.collection.delete(
                ids=results["ids"]
            )
            return len(results["ids"]), f"Deleted {len(results['ids'])} memories with tag '{tag}'"
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