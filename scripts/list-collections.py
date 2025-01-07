from chromadb import HttpClient

def list_collections():
    try:
        # Connect to local ChromaDB
        client = HttpClient(host='localhost', port=8000)
        
        # List all collections
        collections = client.list_collections()
        
        print("\nFound Collections:")
        print("------------------")
        for collection in collections:
            print(f"Name: {collection.name}")
            print(f"Metadata: {collection.metadata}")
            print(f"Count: {collection.count()}")
            print("------------------")
            
    except Exception as e:
        print(f"Error connecting to local ChromaDB: {str(e)}")

if __name__ == "__main__":
    list_collections()
