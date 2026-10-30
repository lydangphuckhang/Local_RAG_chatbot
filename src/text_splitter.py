from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

def get_text_chunks(documents: List[Document], chunk_size : int = 300, chunk_overlap :int =50 ) -> List[Document]:
    """
    Split a list of Langchain DOcuments into smaller chunks using RecursiveCharacterTextSplitter. 

    Args:
        documents (List[Document]): The list of loaded documents to be split.
        chunk_size (int): The maximum number of token per chunk. Default is 300.
        chunk_overlap (int): The number of overlapping token betwween conservutive chunks. Default is 50.
    
    Returns:
        List[Document]: A new list of Documents containing the smaller text chunks.
    
    Raises:
        ValueError: If the input documents list is empty.
        RuntimeError: IF an error occurs during the splitting process.
    """

    if not documents:
        raise ValueError("The input documents list is empty. Can not split")

    try: 
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer= AutoTokenizer.from_pretrained(model_name)
        splitter= RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer= tokenizer,
            chunk_size= chunk_size,
            chunk_overlap= chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks= splitter.split_documents(documents)
        return chunks
    except Exception as e:
        raise RuntimeError(f"An error occurred during text splitting : {str(e)}")

if __name__ == "__main__":
    from document_loader import load_document
    import os

    test_file = "D:/Project/Local_RAG_chatbot/data/raw/sample.pdf"

    if os.path.exists(test_file):
        try:
            raw_doc= load_document(test_file)
            chunks= get_text_chunks(raw_doc)
            print(f"The numbers of chunks splitted: {len(chunks)}")
            print(f"Content of the first chunk: {chunks[0].page_content}")
            print(f"Metadata of first chunks: {chunks[0].metadata}")

        except Exception as error:
            print(f"Error: {error}")
    else:
        raise FileExistsError(f"Test file not found at: {test_file}")
    