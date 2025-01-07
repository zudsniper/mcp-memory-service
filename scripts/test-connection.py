from chromadb import HttpClient

def test_connection(port=8000):
    try:
        # Try to connect to local ChromaDB
        client = HttpClient(host='localhost', port=port)
        # Try a simple operation
        heartbeat = client.heartbeat()
        print(f"Successfully connected to ChromaDB on port {port}")
        print(f"Heartbeat: {heartbeat}")
        
        # List collections
        collections = client.list_collections()
        print("\nFound collections:")
        for collection in collections:
            print(f"- {collection.name} (count: {collection.count()})")
        
    except Exception as e:
        print(f"Error connecting to ChromaDB on port {port}: {str(e)}")

if __name__ == "__main__":
    # Try default port
    test_connection()
    
    # If the above fails, you might want to try other common ports:
    # test_connection(8080)
    # test_connection(9000)
