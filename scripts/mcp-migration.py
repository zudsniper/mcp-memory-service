#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

# Import our environment verifier
from verify_environment import EnvironmentVerifier

def verify_environment():
    """Verify the environment before proceeding with migration"""
    verifier = EnvironmentVerifier()
    verifier.run_verifications()
    if not verifier.print_results():
        print("\n⚠️  Environment verification failed! Migration cannot proceed.")
        sys.exit(1)
    print("\n✓ Environment verification passed! Proceeding with migration.")

# Load environment variables
load_dotenv()

import chromadb
from chromadb import HttpClient, Settings
import json
import time
from chromadb.utils import embedding_functions

def get_claude_desktop_chroma_path():
    """Get ChromaDB path from Claude Desktop config"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, 'claude_config', 'mcp-memory', 'chroma_db')
    print(f"Using ChromaDB path: {config_path}")
    return config_path

def migrate_memories(aws_ip):
    print(f"Starting migration to AWS ChromaDB at {aws_ip}")
    
    try:
        # Set up embedding function
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        
        # Connect to AWS ChromaDB
        aws_client = HttpClient(host=aws_ip, port=8000)
        print("Connected to AWS ChromaDB")
        
        # Get or create collection for imported memories
        try:
            aws_collection = aws_client.get_collection(
                name="mcp_imported_memories",
                embedding_function=embedding_function
            )
            print("Found existing collection 'mcp_imported_memories' on AWS")
        except Exception:
            aws_collection = aws_client.create_collection(
                name="mcp_imported_memories",
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function
            )
            print("Created new collection 'mcp_imported_memories' on AWS")
        
        # Connect to local ChromaDB directly with settings
        local_path = get_claude_desktop_chroma_path()
        print(f"Connecting to local ChromaDB at: {local_path}")
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
            persist_directory=local_path
        )
        local_client = chromadb.Client(settings)
        print("Connected to local ChromaDB")
        
        # List collections
        collections = local_client.list_collections()
        print(f"Found {len(collections)} collections locally")
        for coll in collections:
            print(f"- {coll.name}")
        
        # Try to get the memory collection
        try:
            local_collection = local_client.get_collection(
                name="memory_collection",
                embedding_function=embedding_function
            )
            print("Found local memory collection")
        except ValueError as e:
            print(f"Error accessing local collection: {str(e)}")
            return
            
        # Get all memories from local
        print("Fetching local memories...")
        results = local_collection.get()
        
        if not results["ids"]:
            print("No memories found in local collection")
            return
            
        print(f"Found {len(results['ids'])} memories to migrate")
        
        # Check for existing memories in AWS to avoid duplicates
        aws_existing = aws_collection.get()
        existing_ids = set(aws_existing["ids"])
        
        # Filter out already migrated memories
        new_memories = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }
        
        for i, memory_id in enumerate(results["ids"]):
            if memory_id not in existing_ids:
                new_memories["ids"].append(memory_id)
                new_memories["documents"].append(results["documents"][i])
                new_memories["metadatas"].append(results["metadatas"][i])
        
        if not new_memories["ids"]:
            print("All memories are already migrated!")
            return
            
        print(f"Found {len(new_memories['ids'])} new memories to migrate")
        
        # Import in batches of 10
        batch_size = 10
        for i in range(0, len(new_memories['ids']), batch_size):
            batch_end = min(i + batch_size, len(new_memories['ids']))
            
            batch_ids = new_memories['ids'][i:batch_end]
            batch_documents = new_memories['documents'][i:batch_end]
            batch_metadatas = new_memories['metadatas'][i:batch_end]
            
            print(f"Migrating batch {i//batch_size + 1} ({len(batch_ids)} memories)...")
            
            aws_collection.add(
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            # Small delay between batches
            time.sleep(1)
        
        print("\nMigration complete!")
        
        # Verify migration
        aws_results = aws_collection.get()
        print(f"Verification: {len(aws_results['ids'])} total memories in AWS collection")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        print("Please ensure both ChromaDB instances are running and accessible")

if __name__ == "__main__":
    # First verify the environment
    verify_environment()
    
    AWS_IP = "16.171.169.46"  # Your EC2 public IP
    migrate_memories(AWS_IP)
