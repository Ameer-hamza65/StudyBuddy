from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
import re

import logging
logger = logging.getLogger(__name__)

import os
import re
import tempfile
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def process_pdf(file_bytes: bytes) -> str:
    """Extract and process text from PDF bytes"""
    # Create temporary file outside try/except blocks
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        logger.info("Trying PyPDFLoader...")
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
    except Exception as e:
        logger.error(f"PyPDFLoader failed: {str(e)}. Trying fallback...")
        try:
            # Try alternative PDF parser
            import fitz  # PyMuPDF
            logger.info("Trying PyMuPDF...")
            text = ""
            with fitz.open(tmp_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except ImportError:
            logger.error("PyMuPDF not installed. Please install with 'pip install pymupdf'")
            raise
        except Exception as fallback_e:
            logger.error(f"PyMuPDF failed: {str(fallback_e)}")
            raise
    finally:
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Error deleting temp file: {str(e)}")

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "! ", "? ", "; ", ", ", " "]
    )
    return splitter.split_text(text)