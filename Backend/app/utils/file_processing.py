from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
import re

def process_pdf(file_bytes: bytes) -> str:
    """Extract and process text from PDF bytes"""
    # Check PDF magic number
    if file_bytes[:4] != b'%PDF':
        raise ValueError("Invalid PDF file format - missing PDF header")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()
        full_text = "\n\n".join([p.page_content for p in pages])
        
        # Clean up text
        full_text = re.sub(r'\s+', ' ', full_text)  # Replace multiple spaces
        full_text = re.sub(r'-\n', '', full_text)  # Remove hyphenated line breaks
        
        # Validate we extracted text
        if len(full_text.strip()) < 100:
            raise ValueError("PDF text extraction failed - document may be scanned or encrypted")
            
        return full_text
    finally:
        os.unlink(tmp_path)

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "! ", "? ", "; ", ", ", " "]
    )
    return splitter.split_text(text)