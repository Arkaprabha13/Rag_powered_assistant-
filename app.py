import streamlit as st
import os
import csv
import json
import pandas as pd
import xml.etree.ElementTree as ET
from agent import QAAgent
from ingest import ingest_docs
import fitz  # PyMuPDF for PDF extraction
from fpdf import FPDF

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

# Function to convert CSV to text and save as .txt
def convert_csv_to_txt(csv_file):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Save the uploaded CSV file
        csv_file_path = os.path.join(data_dir, csv_file.name)
        with open(csv_file_path, "wb") as f:
            f.write(csv_file.getbuffer())
        
        # Read the CSV file
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Save the CSV data as .txt
        txt_file_path = csv_file_path.replace(".csv", ".txt")
        with open(txt_file_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write("\t".join(row) + "\n")
        
        st.success(f"✅ Converted {csv_file.name} to text")
        return txt_file_path
    except Exception as e:
        st.error(f"❌ Error converting {csv_file.name}: {e}")
        return None

# Function to convert JSON to text and save as .txt
def convert_json_to_txt(json_file):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Save the uploaded JSON file
        json_file_path = os.path.join(data_dir, json_file.name)
        with open(json_file_path, "wb") as f:
            f.write(json_file.getbuffer())
        
        # Read the JSON file
        with open(json_file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        # Save the JSON data as .txt
        txt_file_path = json_file_path.replace(".json", ".txt")
        with open(txt_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        
        st.success(f"✅ Converted {json_file.name} to text")
        return txt_file_path
    except Exception as e:
        st.error(f"❌ Error converting {json_file.name}: {e}")
        return None

# Function to convert XLSX to text and save as .txt
def convert_xlsx_to_txt(xlsx_file):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Save the uploaded XLSX file
        xlsx_file_path = os.path.join(data_dir, xlsx_file.name)
        with open(xlsx_file_path, "wb") as f:
            f.write(xlsx_file.getbuffer())
        
        # Read the XLSX file using pandas
        df = pd.read_excel(xlsx_file_path)
        
        # Save the XLSX data as .txt
        txt_file_path = xlsx_file_path.replace(".xlsx", ".txt")
        df.to_csv(txt_file_path, sep='\t', index=False)
        
        st.success(f"✅ Converted {xlsx_file.name} to text")
        return txt_file_path
    except Exception as e:
        st.error(f"❌ Error converting {xlsx_file.name}: {e}")
        return None

# Function to convert XML to text and save as .txt
def convert_xml_to_txt(xml_file):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Save the uploaded XML file
        xml_file_path = os.path.join(data_dir, xml_file.name)
        with open(xml_file_path, "wb") as f:
            f.write(xml_file.getbuffer())
        
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Save the XML data as .txt
        txt_file_path = xml_file_path.replace(".xml", ".txt")
        with open(txt_file_path, "w", encoding="utf-8") as f:
            for elem in root.iter():
                f.write(f"{elem.tag}: {elem.text}\n")
        
        st.success(f"✅ Converted {xml_file.name} to text")
        return txt_file_path
    except Exception as e:
        st.error(f"❌ Error converting {xml_file.name}: {e}")
        return None

# Function to save chat history to a PDF
def save_chat_history_to_pdf(chat_history):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        for chat in chat_history:
            pdf.cell(200, 10, txt=f"Q: {chat['query']}", ln=True)
            pdf.multi_cell(200, 10, txt=f"A: {chat['result']['answer']}\n")
            pdf.ln(10)
        
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        pdf_file_path = os.path.join(data_dir, "chat_history.pdf")
        pdf.output(pdf_file_path)
        return pdf_file_path
    except Exception as e:
        st.error(f"❌ Error saving PDF: {e}")
        return None

# Function to save chat history to an Excel file
def save_chat_history_to_xlsx(chat_history):
    try:
        data = [{"Query": chat["query"], "Answer": chat["result"]["answer"]} for chat in chat_history]
        df = pd.DataFrame(data)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        xlsx_file_path = os.path.join(data_dir, "chat_history.xlsx")
        df.to_excel(xlsx_file_path, index=False)
        return xlsx_file_path
    except Exception as e:
        st.error(f"❌ Error saving Excel: {e}")
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
    
    # Upload documents (accept txt, pdf, csv, json, xlsx, xml files)
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf", "csv", "json", "xlsx", "xml"])
    
    if uploaded_files:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        for file in uploaded_files:
            if file.type == "application/pdf":
                # If the file is a PDF, convert it to txt
                txt_file_path = convert_pdf_to_txt(file)
            elif file.type == "application/json":
                txt_file_path = convert_json_to_txt(file)
            elif file.type == "text/csv":
                txt_file_path = convert_csv_to_txt(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                txt_file_path = convert_xlsx_to_txt(file)
            elif file.type == "application/xml":
                txt_file_path = convert_xml_to_txt(file)
            else:
                # For TXT files, just save them
                file_path = os.path.join(data_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                txt_file_path = file_path
                st.success(f"✅ {file.name} uploaded successfully")
            
            if txt_file_path:
                st.success(f"✅ {file.name} converted and saved as .txt")
            else:
                st.error(f"❌ Failed to convert {file.name}")
        
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

# Download options
st.download_button(
    label="Download Chat History as PDF",
    data=save_chat_history_to_pdf(st.session_state.chat_history),
    file_name="chat_history.pdf",
    mime="application/pdf"
)

st.download_button(
    label="Download Chat History as Excel",
    data=save_chat_history_to_xlsx(st.session_state.chat_history),
    file_name="chat_history.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Footer
st.markdown(
    """
    ---
    Made with ❤️ by Arkaprabha Banerjee
    """
)
