from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize and return a HuggingFace embedding model wrapped for Langchain.
    The model is forced to run on CPU to conserve GPU resources for the LLM.

    Args:
        model_name(Str): The name of the model from HuggingFace Hub.
                        Default is 'sentence-transformers/all-MiniLM-L6-v2'
    Returns:
        HuggingFaceEmbeddings: The Langchain-compatible embedding object.
    Raises:
        RuntimeError: If the embedding model fails to initialize.
    """
    try:
        model_kwargs= {'device':'cpu'}
        encode_kwargs= {'normalize_embeddings':'True'}

        embeddings= HuggingFaceEmbeddings(
            model_name= model_name,
            model_kwargs= model_kwargs,
            encode_kwargs= encode_kwargs
        )

        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the HuggingFace embedding model: {str(e)}")

if __name__ == "__main__":
    try:
        test_text= "Who are you?"
        embedder= get_embedding_model()
        vector_embedded= embedder.embed_query(test_text)
        print(f"The legnht of the vector: {len(vector_embedded)}")
        print(f"THe first 5 elements in the vector: {vector_embedded[:5]}")
    except Exception as e:
        print("Test failed: {error}")