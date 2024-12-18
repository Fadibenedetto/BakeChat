import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('index_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_index(documents: List[Document]) -> Optional[FAISS]:
    """
    Create FAISS index from documents
    """
    if not documents:
        logger.warning("No documents provided for indexing")
        return None
    
    try:
        embeddings = OpenAIEmbeddings()
        index = FAISS.from_documents(documents, embeddings)
        logger.info(f"Index created with {len(documents)} documents")
        return index
    
    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        return None

def update_index(
    existing_index: Optional[FAISS], 
    new_documents: List[Document]
) -> Optional[FAISS]:
    """
    Update existing index with new documents
    """
    if not new_documents:
        return existing_index
    
    if existing_index is None:
        return create_index(new_documents)
    
    try:
        existing_index.add_documents(new_documents)
        logger.info(f"Index updated with {len(new_documents)} new documents")
        return existing_index
    
    except Exception as e:
        logger.error(f"Index update failed: {e}")
        return existing_index

def save_index(index: FAISS, path: str = "faiss_index") -> bool:
    """
    Save index to disk
    """
    if index is None:
        logger.warning("Cannot save None index")
        return False
    
    try:
        index.save_local(path)
        logger.info(f"Index saved to {path}")
        return True
    
    except Exception as e:
        logger.error(f"Index save failed: {e}")
        return False

def load_index(path: str = "faiss_index") -> Optional[FAISS]:
    """
    Load index from disk
    """
    try:
        embeddings = OpenAIEmbeddings()
        index = FAISS.load_local(path, embeddings)
        logger.info(f"Index loaded from {path}")
        return index
    
    except Exception as e:
        logger.error(f"Index load failed: {e}")
        return None