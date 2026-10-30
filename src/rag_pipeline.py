from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llm_service import QwenLocal
from retriever import build_vector_store,get_retriever
def create_rag_chain(retriever, llm):
    """
    Construct the RAG pipeline using LangChain Expression Language (LCEL).
    
    Args:
        retriever: The configured LangChain retriever (Chroma).
        llm: The wrapped HuggingFacePipeline LLM.
        
    Returns:
        Runnable: The compiled RAG chain ready to be invoked.
    """
    template ="""<|im_start|>system
    You are a helpful and precise AI assistant. Use the following retrieved context to answer the user's question. 
    If you don't know the answer based on the context, just say that you don't know. Do not make up information.
    Context:
    {context}<|im_end|>
    <|im_start|>user
    Question: {question}<|im_end|>
    <|im_start|>assistant
    """
    
    prompt= PromptTemplate.from_template(template)
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    
    return rag_chain


if __name__ == "__main__":
    from langchain_chroma import Chroma
    from embedding_service import get_embedding_model
    import os
    from document_loader import load_document
    from text_splitter import get_text_chunks

    test_file = "D:/Project/Local_RAG_chatbot/data/raw/sample.pdf"
    raw_docs= load_document(test_file)
    chunks= get_text_chunks(raw_docs)
    embeddings= get_embedding_model()
    vector_store= build_vector_store(chunks=chunks, embedding_model= embeddings)
    
    db_path = "../chroma_db"
    
    if not os.path.exists(db_path):
        print(f"Error: Vector store not found at {db_path}. Please run vector_store.py first.")
    else:
        try:
            embedder = get_embedding_model()
            
            vector_store = Chroma(persist_directory=db_path, embedding_function=embedder)
            
            retriever = get_retriever(vector_store, k=2)
            
            qwen_service = QwenLocal()
            llm = qwen_service.get_llm()
            
            chain= create_rag_chain(retriever=retriever, llm= llm)
            test_question= "What is INSA Centre vel de Loire?"
            print(f"\nUser: {test_question}")
            answer= chain.invoke(test_question) 
            print(f"RAG: {answer}")
            
        except Exception as error:
            print(f"\nPipeline execution failed: {error}")