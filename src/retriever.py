from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


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

