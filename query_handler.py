import os
import logging
import re
from typing import List, Dict, Optional
from openai import OpenAI
from langchain_community.vectorstores import FAISS

def preprocess_query(query: str) -> str:
    """
    Simple and generic query preprocessing
    """
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    
    # Generic expansions for common terms
    expansions = {
        'fecha': ['plazo', 'cuando'],
        'plazo': ['fecha', 'periodo'],
        'requisito': ['condicion', 'requerimiento'],
        'documento': ['documentacion', 'papel'],
        'ayuda': ['subvencion', 'financiacion']
    }
    
    # Simple query expansion
    expanded_terms = set([query])
    words = query.split()
    
    # Add expansions for individual words if they exist
    for word in words:
        if word in expansions:
            expanded_terms.update(expansions[word])
    
    expanded_query = ' '.join(expanded_terms)
    return expanded_query

def answer_query_with_context(
    query: str, 
    index: FAISS, 
    chat_history: List[Dict[str, str]] = []
) -> str:
    """
    Enhanced response generation with better article handling
    """
    try:
        if not index:
            return "La base de conocimiento no está disponible."

        processed_query = preprocess_query(query)
        
        # Realizar dos búsquedas: una para artículos y otra general
        docs = index.similarity_search_with_relevance_scores(processed_query, k=20)
        
        if not docs:
            return ("No encontré información relevante para tu pregunta. "
                   "¿Podrías reformularla?")
        
        # Separar documentos por tipo y relevancia
        article_docs = []
        general_docs = []
        
        for doc, score in docs:
            if doc.metadata.get('content_type') == 'article':
                article_docs.append((doc, score))
            else:
                general_docs.append((doc, score))
        
        # Ordenar por relevancia
        relevant_docs = []
        
        # Primero añadir artículos relevantes
        for doc, score in article_docs:
            if score > 0.03:  # Umbral más bajo para artículos
                relevant_docs.append(doc)
        
        # Luego añadir documentos generales relevantes
        for doc, score in general_docs:
            if score > 0.05:  # Umbral normal para contenido general
                relevant_docs.append(doc)
        
        if not relevant_docs:
            return ("No encontré información suficientemente relevante. "
                   "¿Podrías reformular tu pregunta?")

        # Procesar contexto
        context_parts = []
        for doc in relevant_docs[:10]:
            source = doc.metadata.get('source', 'Documento sin nombre')
            page_match = re.search(r'\[Página (\d+)\]', doc.page_content)
            page = page_match.group(1) if page_match else 'N/A'
            
            # Limpiar el contenido
            content = doc.page_content
            content = re.sub(r'\[Página \d+\]', '', content)
            content = content.strip()
            
            context_parts.append(f"[Fuente: {source} - Página {page}]:\n{content}")

        docs_context = "\n\n".join(context_parts)

        system_prompt = """
        Eres un asistente especializado en normativa institucional. Al responder:

        1. Si encuentras información en artículos específicos:
           - Cita el número de artículo y la página
           - Proporciona el texto exacto relevante
           - Explica el contexto si es necesario

        2. Si la información involucra plazos o fechas:
           - Especifica si el plazo depende de algún evento o resolución
           - Menciona todas las condiciones relevantes
           - Indica si hay excepciones o casos especiales

        3. Para cualquier tipo de información:
           - Cita la fuente y página exacta
           - Proporciona contexto cuando sea necesario
           - Si hay ambigüedad, menciona todas las interpretaciones posibles
        """

        client = OpenAI()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexto:\n{docs_context}\n\nPregunta: {query}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}"

def format_chat_history(
    chat_history: List[Dict[str, str]], 
    max_history: int = 5
) -> str:
    """
    Format recent chat history
    """
    recent_history = chat_history[-max_history*2:]
    formatted = []
    
    for msg in recent_history:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)