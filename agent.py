import os
import re
import math
from dotenv import load_dotenv
from groq import Groq
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

class QAAgent:
    def __init__(self):
        """Initialize the QA Agent with tools and Groq client."""
        self.vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
        
        # Initialize Groq client directly
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        else:
            print("GROQ_API_KEY found in environment variables.")
        
        # self.groq_client = Groq(api_key=groq_api_key)
        try:
            self.groq_client = Groq(api_key=groq_api_key)
        except TypeError as e:
            if "unexpected keyword argument 'proxies'" in str(e):
                # Handle the specific error by creating a client without proxies
                import httpx
                http_client = httpx.Client(base_url="https://api.groq.com")
                self.groq_client = Groq(api_key=groq_api_key, http_client=http_client)
            else:
                raise e

        self.model_name = "llama3-8b-8192"  # You can also use "llama3-70b-8192" for better results
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create tools
        self.tools = self._create_tools()
        
        print("QA Agent initialized successfully with Groq.")
    
    def _create_tools(self):
        """Create the tools for the agent."""
        # RAG tool
        def rag_tool(query):
            """Search for information in the document collection."""
            try:
                model_name = "intfloat/e5-base-v2"
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                vector_store = FAISS.load_local(self.vector_store_path, embeddings)
                
                # Search for relevant documents
                docs = vector_store.similarity_search(query, k=3)
                
                # Format the results
                results = []
                for i, doc in enumerate(docs):
                    results.append(f"Document {i+1}:\n{doc.page_content}\n")
                
                return "\n".join(results)
            except Exception as e:
                return f"Error searching documents: {e}"
        
        # Calculator tool
        def calculator_tool(query):
            """Calculate mathematical expressions."""
            expression = query.strip()
            try:
                # Safe evaluation of mathematical expressions
                result = eval(expression, {"__builtins__": None}, 
                              {"abs": abs, "round": round, "max": max, "min": min, 
                               "pow": pow, "math": math, "sin": math.sin, "cos": math.cos, 
                               "tan": math.tan, "sqrt": math.sqrt, "log": math.log, 
                               "log10": math.log10, "exp": math.exp})
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error calculating {expression}: {e}"
        
        # Dictionary tool
        def dictionary_tool(word):
            """Define a word using the RAG system."""
            try:
                # Use RAG to find definitions
                context_text = rag_tool(f"define {word}")
                
                # Use Groq to generate a definition based on the context
                prompt = f"Based on the following information, provide a clear and concise definition of '{word}':\n\n{context_text}"
                
                completion = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides clear definitions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                
                return completion.choices[0].message.content
            except Exception as e:
                return f"Error defining '{word}': {e}. Using retrieved information directly: {context_text}"
        
        return [
            Tool(
                name="RAG",
                func=rag_tool,
                description="Useful for answering questions about information in the document collection."
            ),
            Tool(
                name="Calculator",
                func=calculator_tool,
                description="Useful for performing mathematical calculations."
            ),
            Tool(
                name="Dictionary",
                func=dictionary_tool,
                description="Useful for defining words and terms."
            )
        ]
    
    def process_query(self, query):
        """Process a user query and return the result."""
        # Check if vector store exists
        if not os.path.exists(self.vector_store_path):
            return {
                "answer": "The document database has not been created yet. Please run ingest.py first.",
                "tool_used": None,
                "context": None
            }
        
        # Determine which tool to use based on keywords
        tool_used = self._determine_tool(query)
        
        # Execute the appropriate tool
        context = None
        try:
            if tool_used == "Calculator":
                # Extract the mathematical expression
                expression = re.sub(r'[^0-9+\-*/().\s]', '', query)
                expression = expression.strip()
                answer = self.tools[1].func(expression)
                
            elif tool_used == "Dictionary":
                # Extract the word to define
                words = query.lower().replace("define", "").replace("what is", "").replace("?", "").strip()
                
                # Use RAG to find definitions
                context_text = self.tools[0].func(words)
                context = context_text.split("Document")
                context = [doc for doc in context if doc.strip()]
                
                # Generate answer using Groq
                prompt = f"Define the term: {words}\n\nContext: {context_text}"
                
                completion = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides clear definitions based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                
                answer = completion.choices[0].message.content
                
            else:  # Default to RAG
                context_text = self.tools[0].func(query)
                context = context_text.split("Document")
                context = [doc for doc in context if doc.strip()]
                
                # Generate answer using Groq
                prompt = f"Based on the following information, please answer the question: {query}\n\nContext: {context_text}"
                
                completion = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the information is not in the context, say so."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                answer = completion.choices[0].message.content
                
        except Exception as e:
            answer = f"Error processing query: {e}. Using context directly: {context_text if 'context_text' in locals() else 'No context available'}"
        
        return {
            "answer": answer,
            "tool_used": tool_used,
            "context": context
        }
    
    def _determine_tool(self, query):
        """Determine which tool to use based on the query."""
        query = query.lower()
        
        # Check for calculator keywords
        calculator_keywords = ["calculate", "compute", "solve", "math", "arithmetic"]
        if any(keyword in query for keyword in calculator_keywords) or re.search(r'[0-9+\-*/()^]', query):
            return "Calculator"
        
        # Check for dictionary keywords
        dictionary_keywords = ["define", "definition", "meaning", "what is", "what are"]
        if any(keyword in query for keyword in dictionary_keywords):
            return "Dictionary"
        
        # Default to RAG
        return "RAG"
