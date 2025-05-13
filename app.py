import streamlit as st
import os
import sys
from agent import QAAgent
from ingest import ingest_docs
import fitz  # PyMuPDF for PDF extraction

# Set page configuration
st.set_page_config(
    page_title="RAG-Powered Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = QAAgent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main app title
st.title("🤖 RAG-Powered Multi-Agent Q&A Assistant")

# Function to convert PDF to text and save as .txt
def convert_pdf_to_txt(pdf_file):
    """
    Converts a PDF file to a .txt file and saves it in the 'data' directory.
    """
    try:
        # Create a temporary path to save the uploaded PDF file
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Save the uploaded PDF file
        pdf_file_path = os.path.join(data_dir, pdf_file.name)
        with open(pdf_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Open the saved PDF with fitz
        doc = fitz.open(pdf_file_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        # Save the extracted text as a .txt file
        txt_file_path = pdf_file_path.replace(".pdf", ".txt")
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        st.success(f"✅ Converted {pdf_file.name} to text")
        return txt_file_path
    except Exception as e:
        st.error(f"❌ Error converting {pdf_file.name}: {e}")
        return None

# Sidebar for document ingestion
with st.sidebar:
    st.header("Document Management")
    
    # Check if vector store exists
    vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
    vector_store_exists = os.path.exists(vector_store_path)
    
    if vector_store_exists:
        st.success("✅ Document database is ready")
    else:
        st.warning("⚠️ Document database not found. Please ingest documents.")
    
    # Upload documents (accept txt and pdf files)
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf"])
    
    if uploaded_files:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        for file in uploaded_files:
            if file.type == "application/pdf":
                # If the file is a PDF, convert it to txt
                txt_file_path = convert_pdf_to_txt(file)
                if txt_file_path:
                    st.success(f"✅ PDF file {file.name} converted and saved.")
                else:
                    st.error(f"❌ PDF conversion failed for {file.name}")
            else:
                # For TXT files, just save them
                file_path = os.path.join(data_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                st.success(f"✅ {file.name} uploaded successfully")
        
    # Ingest documents button
    if st.button("Ingest Documents"):
        with st.spinner("Ingesting documents..."):
            ingest_docs()
            st.success("✅ Documents ingested successfully")
            st.rerun()

# Main chat interface
st.subheader("Ask a question")
user_query = st.text_input("Enter your question:")

if user_query:
    # Process the query
    with st.spinner("Processing your question..."):
        result = st.session_state.agent.process_query(user_query)
    
    # Add to chat history
    st.session_state.chat_history.append({"query": user_query, "result": result})
    
    # Display the result
    st.subheader("Answer")
    st.write(result["answer"])
    
    # Display the tool used
    st.subheader("Processing Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Tool Used**: {result['tool_used']}")
    
    # Display the context if available
    if result["context"]:
        with col2:
            with st.expander("Retrieved Context"):
                for i, context in enumerate(result["context"]):
                    st.markdown(f"**Document {i+1}**")
                    st.text(context)

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {chat['query']}"):
            st.write(f"**Answer**: {chat['result']['answer']}")
            st.write(f"**Tool Used**: {chat['result']['tool_used']}")
            st.write(f"**Context**: {chat['result']['context']}")

# Footer
st.markdown(
    """
    ---
    Made with ❤️ by Arkaprabha Banerjee
    """
)
