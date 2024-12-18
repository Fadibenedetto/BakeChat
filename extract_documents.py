import os
import re
import logging
from typing import List, Dict, Any
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('document_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_article_with_context(text: str) -> List[Dict[str, str]]:
    """
    Extract articles while preserving their complete content and context
    """
    # Patrón mejorado para detectar artículos
    article_pattern = r'(?:Artículo|Art\.) *(\d+)[.\s]+(.*?)(?=(?:Artículo|Art\.) *\d+[.\s]|$)'
    articles = []
    
    for match in re.finditer(article_pattern, text, re.DOTALL):
        article_num = match.group(1)
        content = match.group(2).strip()
        articles.append({
            'number': article_num,
            'content': f"Artículo {article_num}. {content}"
        })
    
    return articles

def clean_text(text: str) -> str:
    """
    Clean text while preserving important structure and information
    """
    if not text:
        return ""
    
    # Preservar estructura
    text = text.replace('\f', '\n\n')
    
    # Normalizar espacios pero mantener estructura
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Limpiar caracteres especiales pero mantener puntuación importante
    text = re.sub(r'[^\w\s\.,;:()¿?¡!-]', '', text)
    
    return text.strip()

def log_extraction_details(filename: str, text: str, articles: List[Dict[str, str]]):
    """
    Log extraction details for verification
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing file: {filename}")
    logger.info(f"Total text length: {len(text)}")
    logger.info(f"Number of articles found: {len(articles)}")
    
    # Log artículos específicos para verificación
    for article in articles:
        if article['number'] in ['21', '20', '22']:  # Artículos importantes
            logger.info(f"\nArticle {article['number']} content:")
            logger.info(f"{article['content'][:200]}...")  # Primeros 200 caracteres

def extract_text_from_folder(folder_path: str) -> List[Document]:
    """
    Process all PDFs with enhanced article preservation
    """
    all_documents = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        logger.info(f"\nProcessing: {filename}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                page_texts = {}
                
                # Extract text with page numbers
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        page_texts[i] = page_text
                        full_text += f"\n[PÁGINA {i}]\n{page_text}"
                
                # Clean the text
                cleaned_text = clean_text(full_text)
                
                # Extract articles
                articles = extract_article_with_context(cleaned_text)
                
                # Log extraction details
                log_extraction_details(filename, cleaned_text, articles)
                
                # Create base metadata
                base_metadata = {
                    "source": filename,
                    "file_path": file_path
                }
                
                # Process articles first (priority content)
                for article in articles:
                    # Find the page number for this article
                    page_num = None
                    article_content = article['content']
                    for page_num, page_text in page_texts.items():
                        if article_content in clean_text(page_text):
                            break
                    
                    metadata = base_metadata.copy()
                    metadata.update({
                        'content_type': 'article',
                        'article_number': article['number'],
                        'page': page_num
                    })
                    
                    # Create document for the complete article
                    all_documents.append(Document(
                        page_content=article['content'],
                        metadata=metadata
                    ))
                
                # Process remaining text (non-article content)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,  # Increased chunk size
                    chunk_overlap=300,  # Increased overlap
                    separators=["\n\n", "\n", ". "]
                )
                
                # Remove article content from the text
                for article in articles:
                    cleaned_text = cleaned_text.replace(article['content'], '')
                
                # Split remaining text
                chunks = splitter.split_text(cleaned_text)
                
                # Create documents for non-article content
                for chunk in chunks:
                    if len(chunk.strip()) < 100:  # Skip very small chunks
                        continue
                    
                    # Find page number for this chunk
                    page_num = None
                    for pnum, ptext in page_texts.items():
                        if chunk in clean_text(ptext):
                            page_num = pnum
                            break
                    
                    metadata = base_metadata.copy()
                    metadata.update({
                        'content_type': 'general',
                        'page': page_num
                    })
                    
                    all_documents.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    logger.info(f"\nTotal documents created: {len(all_documents)}")
    return all_documents
