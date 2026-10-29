import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain_core.documents import Document

def load_document(file_path: str) -> List[Document]:
    """
    Load a document (PDF or TXT) from the given file path and return a list of Langchain DOcuments.

    Args:
        file_path (str): the absolute path to the file.
    
    Returns:
        List[Document]: A list of Langchain Document objects containing page_content and metadata
    
    Raises: 
        FileNotFoundError: If the provided file_path does not exist.
        ValueError: If the file extension is not supported (.pdf or .txt).
        RuntimeError: If an error occurs during the doxument loading process.
    """

    if not os.path.exists(file_path):
        raise FileExistsError(f"File nit found at the specified path: {file_path}")

    extension_file = os.path.splitext(file_path)[1].lower()
    if extension_file == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension_file == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: '{extension_file}'. Only .pdf and .txt are supported.")

    try:
        documents= loader.load()
        return documents
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the file '{file_path}' : {str(e)}")
    
if __name__ == "__main__":
    test_file = "D:/Project/Local_RAG_chatbot/data/raw/sample.pdf"

    try:
        docs = load_document(test_file)
        print(f"Sucessfully loaded {len(docs)} pages")
        if docs:
            print(f"Metadata of the first element (oject Documents): {docs[0].metadata}")
            print(f"Content preview (firsts 100 characters): {docs[0].page_content[:100]}...")
    except Exception as error:
        print(f"Error: {error}")