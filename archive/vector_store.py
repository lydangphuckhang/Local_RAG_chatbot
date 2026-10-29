import chromadb
import uuid

class ChromaVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db",collection_name: str = "rag_documents"):
        """
        Initialize the connection to ChromaDB and prepare the storage space.
        """
        print(f"Connecting to ChromaDB at {persist_directory}")

        self.client = chromadb.PersistentClient(path= persist_directory)
        self.collection= self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list[str], metadata: list[dict] = None):
        """
        Recieve the split text chunks and save them entirely to disk.
        """
        if not chunks:
            print("No chunk data to add to the database.")
            return 
        ids=[]
        for i in range(len(chunks)):
            ids.append(str(uuid.uuid4()))
        self.collection.add(
            documents= chunks,
            metadatas= metadata,
            ids= ids
        )
        print(f"Successfully saved! Total chunks in DB: {self.collection.count()}")

#Testing 
if __name__ == "__main__":
    db= ChromaVectorStore()

    sample_chunks= [
        "How to change password: go to settings, select security and enter new password.",
        "Company policy: Employees must bes present at 8:00 AM every day."
    ]
    sample_metadata = [
        {"source": "IT_guide.pdf", "page": 1},
        {"source": "employee_handbook.txt", "page": 5}
    ]
    db.add_chunks(chunks= sample_chunks, metadata=sample_metadata)
