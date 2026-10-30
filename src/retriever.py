from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_store(chunks: List[Document], embedding_model: HuggingFaceEmbeddings, persist_directory: str = "D:/Project/Local_RAG_chatbot/chroma_db" ) -> Chroma:
    """
    Build and persist a Chroma vector database from text chunks and an embedding model.

    Args:
        chunks(List[Document]): The list of split text chunks.
        embedding_model (HuggingFaceEmbeddings): The embedding model to vectorize the text. 
        persist_directory (str): The directory to save the ChromaDB files.
    
    Returns: 
        Chroma: The initialized Langchain Chrom vector sotre object.
    
    Raises:
        ValueError: If the 'chunks' list is empty.
        RuntimeError: If vector sotre creation fails.
    """

    if not chunks:
        raise ValueError("The Chunk list is empty.Can not build vector store.")
    try:
        vector_store= Chroma.from_documents(
            documents= chunks,
            embedding= embedding_model,
            persist_directory= persist_directory
        )
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to build Chroma vector store: {str(e)}")

def get_retriever(vector_store: Chroma, k:int =3):
    """
    Covert a Chroma vector into a Langchian retriever.

    Args: 
        vector_store (Chroma): The loaded Chroma vector database.
        k(int): The number of top documents to retrieve.Default is 3.
    
    Returns: 
        VectorStoreRetriever: The configured retriever objet.
    """
    try:
        retriever= vector_store.as_retriever(
            search_type= "mmr",
            search_kwargs= {
            'k':k ,
            "fetch_k": 20,   
            "lambda_mult": 0.5}
        )
        return retriever
    except Exception as e:
        raise RuntimeError(f"Failed to configure retriever: {str(e)}")

if __name__ == "__main__":
    import os
    from document_loader import load_document
    from text_splitter import get_text_chunks
    from embedding_service import get_embedding_model

    test_file = "D:/Project/Local_RAG_chatbot/data/raw/sample.pdf"
    if os.path.exists(test_file):
        try: 
            raw_docs= load_document(test_file)
            chunks= get_text_chunks(raw_docs)
            embeddings= get_embedding_model()
            vector_store= build_vector_store(chunks=chunks, embedding_model= embeddings)
            retriever= get_retriever(vector_store=vector_store,k=2)
            test_question= "What is this document about?"
            reponse= retriever.invoke(test_question)
            print(f"Numbers of retrieved chunks: {len(reponse)}")
            for i in range(len(reponse)):
                print(f"The {i+1} chunk retrieved content: {reponse[i].page_content}")
                print(f"(Metadata: {reponse[i].metadata}) ")
        
        except Exception as e:
            print(f"Test failed: {str(e)}")
    else:
        raise FileNotFoundError("Test file not found")