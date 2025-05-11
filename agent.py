import os
import re
import math
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("QAAgent")

# Load environment variables
load_dotenv()

class QAAgent:
    def __init__(self):
        """Initialize the QA Agent with tools and Groq client."""
        logger.info("Initializing QA Agent...")
        
        # Set up vector store path
        self.vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
        logger.info(f"Vector store path: {self.vector_store_path}")
        
        # Initialize Groq client
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        
        try:
            # Initialize Groq client directly without additional parameters
            self.groq_client = Groq(api_key=groq_api_key)
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
        
        # Set model name
        self.model_name = "llama3-8b-8192"  # You can also use "llama3-70b-8192" for better results
        logger.info(f"Using model: {self.model_name}")
        
        # Create tools
        self.tools = self._create_tools()
        logger.info("QA Agent initialized successfully")
    
    def _create_tools(self):
        """Create the tools for the agent."""
        logger.info("Creating agent tools...")
        
        # RAG tool
        def rag_tool(query: str) -> str:
            """Search for information in the document collection."""
            logger.info(f"RAG tool processing query: {query}")
            
            try:
                # Check if vector store exists
                if not os.path.exists(self.vector_store_path):
                    logger.warning("Vector store not found")
                    return "Vector store not found. Please run ingest.py first."
                
                # Load embeddings model
                try:
                    model_name = "intfloat/e5-base-v2"
                    logger.info(f"Loading embedding model: {model_name}")
                    embeddings = HuggingFaceEmbeddings(model_name=model_name)
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    return f"Error loading embedding model: {e}. Please ensure sentence-transformers is installed with 'pip install sentence-transformers'."
                
                # Load vector store
                try:
                    logger.info("Loading vector store from disk")
                    vector_store = FAISS.load_local(self.vector_store_path, embeddings)
                except Exception as e:
                    logger.error(f"Failed to load vector store: {e}")
                    return f"Error loading vector store: {e}"
                
                # Search for relevant documents
                try:
                    logger.info(f"Performing similarity search with k=3")
                    docs = vector_store.similarity_search(query, k=3)
                    logger.info(f"Retrieved {len(docs)} documents")
                except Exception as e:
                    logger.error(f"Failed to perform similarity search: {e}")
                    return f"Error during similarity search: {e}"
                
                # Format the results
                results = []
                for i, doc in enumerate(docs):
                    logger.debug(f"Document {i+1} content: {doc.page_content[:100]}...")
                    results.append(f"Document {i+1}:\n{doc.page_content}\n")
                
                return "\n".join(results)
            except Exception as e:
                logger.error(f"Unexpected error in RAG tool: {e}")
                return f"Error searching documents: {e}"
        
        # Calculator tool
        def calculator_tool(query: str) -> str:
            """Calculate mathematical expressions."""
            logger.info(f"Calculator tool processing query: {query}")
            
            # Extract the mathematical expression
            expression = re.sub(r'[^0-9+\-*/().\s]', '', query)
            expression = expression.strip()
            logger.info(f"Extracted expression: {expression}")
            
            if not expression:
                return "No valid mathematical expression found in the query."
            
            try:
                # Safe evaluation of mathematical expressions
                result = eval(expression, {"__builtins__": None}, 
                              {"abs": abs, "round": round, "max": max, "min": min, 
                               "pow": pow, "math": math, "sin": math.sin, "cos": math.cos, 
                               "tan": math.tan, "sqrt": math.sqrt, "log": math.log, 
                               "log10": math.log10, "exp": math.exp})
                logger.info(f"Calculation result: {result}")
                return f"The result of {expression} is {result}"
            except Exception as e:
                logger.error(f"Error calculating expression: {e}")
                return f"Error calculating {expression}: {e}"
        
        # Dictionary tool
        def dictionary_tool(word: str) -> str:
            """Define a word using the RAG system."""
            logger.info(f"Dictionary tool processing word: {word}")
            
            try:
                # Use RAG to find definitions
                context_text = rag_tool(f"define {word}")
                logger.info(f"Retrieved context for definition")
                
                # Generate answer using Groq with streaming
                prompt = f"Based on the following information, provide a clear and concise definition of '{word}':\n\n{context_text}"
                
                try:
                    logger.info("Sending definition request to Groq")
                    answer = self._generate_with_groq(prompt, system_message="You are a helpful assistant that provides clear definitions based on the provided context.")
                    logger.info("Successfully generated definition")
                    return answer
                except Exception as e:
                    logger.error(f"Error with Groq LLM: {e}")
                    return f"Error generating definition: {e}. Using retrieved information directly: {context_text}"
            except Exception as e:
                logger.error(f"Unexpected error in Dictionary tool: {e}")
                return f"Error defining '{word}': {e}"
        
        logger.info("All tools created successfully")
        return {
            "RAG": rag_tool,
            "Calculator": calculator_tool,
            "Dictionary": dictionary_tool
        }
    
    def _generate_with_groq(self, prompt: str, system_message: str = "You are a helpful assistant that answers questions based on the provided context. If the information is not in the context, say so.") -> str:
        """Generate a response using Groq with streaming."""
        try:
            stream = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                stream=True
            )
            
            # Collect the streaming response
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
            
            return full_response
        except Exception as e:
            logger.error(f"Error generating with Groq: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the result."""
        logger.info(f"Processing query: {query}")
        
        # Check if vector store exists
        if not os.path.exists(self.vector_store_path):
            logger.warning("Vector store not found")
            return {
                "answer": "The document database has not been created yet. Please run ingest.py first.",
                "tool_used": None,
                "context": None
            }
        
        # Determine which tool to use based on keywords
        tool_used = self._determine_tool(query)
        logger.info(f"Selected tool: {tool_used}")
        
        # Execute the appropriate tool
        context = None
        try:
            if tool_used == "Calculator":
                # Extract the mathematical expression
                answer = self.tools["Calculator"](query)
            
            elif tool_used == "Dictionary":
                # Extract the word to define
                words = query.lower().replace("define", "").replace("what is", "").replace("?", "").strip()
                logger.info(f"Extracted word for definition: {words}")
                
                # Use RAG to find definitions
                context_text = self.tools["RAG"](words)
                context = context_text.split("Document")
                context = [doc for doc in context if doc.strip()]
                
                # Generate answer using Groq
                prompt = f"Define the term: {words}\n\nContext: {context_text}"
                
                try:
                    logger.info("Sending definition request to Groq")
                    answer = self._generate_with_groq(
                        prompt, 
                        system_message="You are a helpful assistant that provides clear definitions based on the provided context."
                    )
                    logger.info("Successfully generated definition")
                except Exception as e:
                    logger.error(f"Error with Groq LLM: {e}")
                    answer = f"Error generating definition: {e}. Using retrieved information directly: {context_text}"
            
            else:  # Default to RAG
                context_text = self.tools["RAG"](query)
                context = context_text.split("Document")
                context = [doc for doc in context if doc.strip()]
                
                # Generate answer using Groq
                prompt = f"Based on the following information, please answer the question: {query}\n\nContext: {context_text}"
                
                try:
                    logger.info("Sending query to Groq")
                    answer = self._generate_with_groq(prompt)
                    logger.info("Successfully generated answer")
                except Exception as e:
                    logger.error(f"Error with Groq LLM: {e}")
                    answer = f"Error generating answer: {e}. Using retrieved information directly: {context_text}"
        
        except Exception as e:
            logger.error(f"Unexpected error processing query: {e}")
            answer = f"Error processing query: {e}"
        
        logger.info("Query processing completed")
        return {
            "answer": answer,
            "tool_used": tool_used,
            "context": context
        }
    
    def _determine_tool(self, query: str) -> str:
        """Determine which tool to use based on the query."""
        query = query.lower()
        logger.info(f"Determining tool for query: {query}")
        
        # Check for calculator keywords
        calculator_keywords = ["calculate", "compute", "solve", "math", "arithmetic"]
        
        # More specific regex for calculator detection - looks for actual calculations
        # rather than just any number in the text
        if any(keyword in query for keyword in calculator_keywords) or re.search(r'(\d+\s*[\+\-\*/\(\)\^]\s*\d+)', query):
            logger.info("Selected Calculator tool based on keywords/pattern")
            return "Calculator"
        
        # Check for dictionary keywords
        dictionary_keywords = ["define", "definition", "meaning", "what is", "what are"]
        if any(keyword in query for keyword in dictionary_keywords):
            logger.info("Selected Dictionary tool based on keywords")
            return "Dictionary"
        
        # Default to RAG
        logger.info("Defaulting to RAG tool")
        return "RAG"
