import os
import sys
import streamlit as st
from typing import List
from langchain.docstore.document import Document
from extract_documents import extract_text_from_folder
from index_manager import create_index, load_index, save_index
from query_handler import answer_query_with_context

# Configure page settings
st.set_page_config(
    page_title="BakeChat",
    page_icon="🤖",
    layout="wide"
)

# Constants
FOLDER_PATH = "convocatorias"
INDEX_PATH = "faiss_index"

# Initialize session state
def init_session_state():
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False

def load_documents() -> List[Document]:
    """Load and process documents from the convocatorias folder"""
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
    return extract_text_from_folder(FOLDER_PATH)

def initialize_index():
    """Initialize or load the FAISS index"""
    if os.path.exists(INDEX_PATH):
        index = load_index(INDEX_PATH)
        if index:
            return index
    
    documents = load_documents()
    if documents:
        index = create_index(documents)
        if index:
            save_index(index, INDEX_PATH)
            return index
    return None

def main():
    # Initialize session state
    init_session_state()
    
    # App title and description
    st.title("BakeChat 🤖")
    st.write("Hola! He sido entrenada para emular a una consultora experta. Pregúntame lo que quieras sobre normativa de convocatorias asociadas a nuestra institución.")
    
    # Initialize or load index if not already done
    if not st.session_state.documents_loaded:
        with st.spinner("Cargando documentos..."):
            st.session_state.index = initialize_index()
            if st.session_state.index:
                st.session_state.documents_loaded = True
                st.success("Base de conocimiento cargada exitosamente!")
            else:
                st.warning("No se pudo cargar la base de conocimiento. Por favor, verifica los documentos.")
    
    # File uploader
    with st.expander("Subir nuevos documentos"):
        uploaded_files = st.file_uploader(
            "Sube tus PDFs (opcional)",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if uploaded_files:
            with st.spinner("Procesando nuevos documentos..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file
                    save_path = os.path.join(FOLDER_PATH, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.read())
                
                # Reload documents and recreate index
                documents = load_documents()
                if documents:
                    st.session_state.index = create_index(documents)
                    if st.session_state.index:
                        save_index(st.session_state.index, INDEX_PATH)
                        st.success("Documentos procesados exitosamente!")
    
    # Show available documents
    with st.expander("Ver documentos disponibles"):
        docs_list = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.pdf')]
        if docs_list:
            for doc in docs_list:
                st.write(f"📄 {doc}")
        else:
            st.write("No hay documentos disponibles")
    
    # Chat interface
    st.write("### Chat")
    
    # Clear chat history button
    if st.button("Limpiar historial"):
        st.session_state.chat_history = []
        st.success("Historial limpiado!")
        st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        role = "Usuario" if message["role"] == "user" else "Bake"
        with st.chat_message(role.lower()):
            st.write(message["content"])
    
    # Chat input
    if query := st.chat_input("Escribe tu pregunta..."):
        # Display user message
        with st.chat_message("usuario"):
            st.write(query)
        
        # Generate and display response
        with st.chat_message("bake"):
            if st.session_state.index is None:
                response = "Lo siento, no hay documentos cargados en la base de conocimiento. Por favor, verifica que los documentos estén disponibles."
            else:
                with st.spinner("Pensando..."):
                    response = answer_query_with_context(
                        query,
                        st.session_state.index,
                        st.session_state.chat_history
                    )
            st.write(response)
        
        # Update chat history
        st.session_state.chat_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])

if __name__ == "__main__":
    main()