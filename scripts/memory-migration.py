from chromadb import HttpClient
import json
import time

def migrate_memories(aws_ip):
    # Connect to AWS ChromaDB
    print(f"Connecting to AWS ChromaDB at {aws_ip}...")
    aws_client = HttpClient(host=aws_ip, port=8000)
    
    try:
        # Create a collection for imported memories
        aws_collection = aws_client.create_collection(name="imported_memories")
        print("Created collection 'imported_memories' on AWS")
        
        # Connect to local ChromaDB
        local_client = HttpClient(host='localhost', port=8000)
        
        # Get local collections
        # Note: You'll need to specify your collection name if different
        local_collection = local_client.get_collection("memories")
        
        # Get all memories from local
        print("Fetching local memories...")
        results = local_collection.get()
        
        if not results["ids"]:
            print("No memories found in local collection")
            return
            
        print(f"Found {len(results['ids'])} memories to migrate")
        
        # Batch size for import
        batch_size = 100
        
        # Import in batches
        for i in range(0, len(results['ids']), batch_size):
            batch_end = min(i + batch_size, len(results['ids']))
            
            batch_ids = results['ids'][i:batch_end]
            batch_documents = results['documents'][i:batch_end]
            batch_metadatas = results['metadatas'][i:batch_end]
            
            print(f"Migrating batch {i//batch_size + 1} ({len(batch_ids)} memories)...")
            
            aws_collection.add(
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            # Small delay between batches
            time.sleep(1)
        
        print(f"\nMigration complete! {len(results['ids'])} memories transferred to AWS ChromaDB")
        
        # Verify migration
        aws_results = aws_collection.get()
        print(f"Verification: {len(aws_results['ids'])} memories found in AWS collection")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")

if __name__ == "__main__":
    # Replace with your EC2 public IP
    AWS_IP = "your-ec2-public-ip"
    migrate_memories(AWS_IP)
