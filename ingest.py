import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

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
        print("No documents found in the data directory. Please add some .txt files.")
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
