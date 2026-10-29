import chromadb

class ContextRetriever:
    """
    Retrieve revelant document chinks from a persistent ChromaDB collection.
    """
    def __init__(self, persistent_directory: str = "./chroma_db", collection_name: str = "rag_documents" ):
        self.client= chromadb.PersistentClient(path=persistent_directory)
        self.connection= self.client.get_or_create_collection(name= collection_name)

    def search_context(self, query: str, top_k: int = 3):
        results= self.connection.query(
            query_texts=[query],
            n_results= top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(documents)):
            print(f"---Result {i+1} (Distance: {distances[i]:.4f})---")
            print(
                f"Source: {metadatas[i]['source']}"
                f"(Page: {metadatas[i].get('page','N/A')})"
            )
            print(f"Content: {documents[i]}")
        return documents, metadatas
if __name__ == "__main__":
    retriever= ContextRetriever()

    question = "How can i change my password?"

    retriever.search_context(query= question, top_k= 1)
