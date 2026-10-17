import os
from pypdf import PdfReader

def load_txt(file_path: str)-> str:
    """
    Read a TXT file and return its content as a string
    """
    with open(file_path, "r", encoding= "utf-8") as f:
        content = f.read()
    clean_content= content.strip()
    return clean_content

def load_pdf(file_path: str) -> str:
    """
    Read a PDF file and extract text from all of its pages
    """
    list_content = []
    reader = PdfReader(file_path)
    for i in range(len(reader.pages)):
        page_i = reader.pages[i]
        content_page_i = page_i.extract_text()
        
        if content_page_i:
            content_i_clean = " ".join(content_page_i.split())
            list_content.append(content_i_clean)
            
    content = "\n".join(list_content)
    return content

def load_document(file_path: str) -> str:
    """
    Load a document and extract its text based on the file extension.
    Raises ValueError for unsupported file types and FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith(".txt"):
        return load_txt(file_path)
    elif file_path.endswith(".pdf"):
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

if __name__ == "__main__":
    # Test TXT
    txt_path = "data/raw/sample.txt"
    print("--- TEST TXT ---")
    print(load_document(txt_path))
    
    # Test PDF
    pdf_path = "data/raw/sample.pdf"
    print("\n--- TEST PDF ---")
    print(load_document(pdf_path))