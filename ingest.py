import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF extraction

load_dotenv()

def convert_pdf_to_txt(pdf_path):
    """
    Converts a PDF to a .txt file.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        # Save the extracted text to a .txt file
        txt_path = pdf_path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Converted {pdf_path} to {txt_path}")
        return txt_path
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        return None

def ingest_docs():
    """
    Load documents from the data directory, split them into chunks,
    and create a vector store for retrieval.
    """
    # Load documents from the data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    documents = []
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}. Please add your documents there.")
        return
    
    # Convert any PDF files to text and load them
    for file_path in glob.glob(os.path.join(data_dir, "*.pdf")):
        txt_path = convert_pdf_to_txt(file_path)
        if txt_path:
            loader = TextLoader(txt_path, encoding='utf-8')
            documents.extend(loader.load())
    
    # Load all text files from the data directory
    for file_path in glob.glob(os.path.join(data_dir, "*.txt")):
        try:
            # Detect file encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = 'utf-8'  # Default to utf-8
                try:
                    raw_data.decode('utf-8')
                except UnicodeDecodeError:
                    encoding = 'latin-1'  # Fallback encoding
            
            # Use detected encoding with TextLoader
            loader = TextLoader(file_path, encoding=encoding)
            documents.extend(loader.load())
            print(f"Loaded document: {file_path} with encoding {encoding}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not documents:
        print("No documents found in the data directory. Please add some .txt or .pdf files.")
        return
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks")
    
    # Create embeddings and vector store using HuggingFace embeddings
    try:
        # Use a sentence transformer model for embeddings
        model_name = "intfloat/e5-base-v2"  # A good alternative to OpenAI embeddings
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Save the vector store
        vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
        vector_store.save_local(vector_store_path)
        print(f"Vector store created and saved at {vector_store_path}")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Please check if the HuggingFace model is accessible")

if __name__ == "__main__":
    ingest_docs()
