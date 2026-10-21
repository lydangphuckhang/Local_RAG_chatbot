def manual_text_splitter( text: str,chunk_size: int,chunk_overlap: int, source_name: str) -> list:
    if chunk_size <=0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap <0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    chunks = []
    start = 0
    chunk_id= 0
    step= chunk_size - chunk_overlap 
    while (start < len(text)):
        end =start + chunk_size
        chunk_text= text[start:end]
        chunk= {
            "chunk_id": chunk_id,
            "source": source_name,
            "text": chunk_text
        }
        chunks.append(chunk)
        if end >= len(text):
            break
        start +=step
        chunk_id +=1
    return chunks

#Test manual_split
from document_loader import load_document

reel_text = load_document("data/raw/sample.pdf")
manual_chunks= manual_text_splitter(
    text=reel_text,
    chunk_size=1000,
    chunk_overlap=200,
    source_name="sample.pdf"
    )
# Print first chunks

for i in range(len(manual_chunks)):
    print(f"--CHUNK {i+1}---")
    print(f"Longeur: {len(manual_chunks[i]['text'])}")
    print(f"Content of Chunk {i}: {manual_chunks[i]['text'][:150]}...\n")

# Test with langchain:
from langchain_text_splitters import RecursiveCharacterTextSplitter

langchain_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap=200,
    length_function= len
)
metadata= {
    "source": "sample.pdf",
    "type": "pdf" 
}
chunks= langchain_splitter.create_documents(texts= [reel_text], metadatas=[metadata] )
i=0
print(f"---CHUNK {i+1}---")
print(f"Longeur: {len(chunks[i].page_content)}")
print(f"Content of chunk {i}: {chunks[i].page_content[:150]}")
print(chunks[i].metadata)