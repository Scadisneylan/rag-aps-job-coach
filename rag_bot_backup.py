import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import tiktoken
import json
from flask import Flask, request, jsonify
import logging
from dotenv import load_dotenv
import signal
import functools
import threading
import time

# --- DEBUG PRINT: Start of script ---
print("DEBUG: Script started. Setting up logging.")

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DEBUG PRINT: Before loading dotenv ---
print("DEBUG: Attempting to load .env file.")
# Load environment variables from .env file (if present)
load_dotenv()
# --- DEBUG PRINT: After loading dotenv ---
print(f"DEBUG: .env loaded. OPENAI_API_KEY from env: {os.getenv('OPENAI_API_KEY')[:5] if os.getenv('OPENAI_API_KEY') else 'NotSet'}")


# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
structured_data_file = "my_structured_document.csv"
# Updated path: CSV file is now in the main directory
structured_data_path = os.path.join(current_dir, structured_data_file)

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

logger.info(f"Constructed data path: {structured_data_path}")
print(f"DEBUG: Constructed data path check: {structured_data_path}")


# --- Global variables for RAG components ---
chunks = []
embeddings = None
vectorstore = None
retriever = None
llm = None
qa_chain = None

# --- Timeout wrapper function ---
def run_with_timeout(func, timeout_seconds=30):
    """
    Run a function with a timeout using threading.
    This is more reliable across different platforms than signal-based timeouts.
    """
    def wrapper(*args, **kwargs):
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    return wrapper

# --- Function to initialize RAG components ---
def initialize_rag_system():
    global chunks, embeddings, vectorstore, retriever, llm, qa_chain
    print("DEBUG: Inside initialize_rag_system function.")

    # 1. Load Manually Structured Data from CSV
    print(f"DEBUG: Checking for CSV existence at: {structured_data_path}")
    if not os.path.exists(structured_data_path):
        logger.error(f"Error: The structured data file does NOT exist at '{structured_data_path}'.")
        print(f"DEBUG: CSV file NOT found at: {structured_data_path}") # Explicit print for debug
        return False
    else:
        print(f"DEBUG: CSV file found at: {structured_data_path}. Attempting to load.") # Explicit print for debug
        try:
            logger.info(f"\nLoading data from {structured_data_file}...")
            df = pd.read_csv(structured_data_path)

            for index, row in df.iterrows():
                page_content = str(row['chunk_content'])
                metadata = {col: str(row[col]) for col in df.columns if col != 'chunk_content'}
                metadata['source_file'] = structured_data_file
                metadata['row_number'] = index
                doc = Document(page_content=page_content, metadata=metadata)
                chunks.append(doc)

            logger.info(f"Successfully loaded {len(chunks)} chunks from {structured_data_file}.")

            if not chunks:
                logger.warning("No chunks were created from the CSV. Please check your CSV content and column names.")
                return False

        except KeyError as e:
            logger.error(f"Error: Missing expected column in CSV. Please ensure your CSV has a 'chunk_content' column and any other columns you try to access. Missing: {e}")
            logger.error("Double-check the exact spelling of your column headers in Google Sheets.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during CSV loading: {e}", exc_info=True) # Added exc_info
            return False

    logger.info("Chunks populated with custom-structured documents!")
    print("DEBUG: Chunks populated.")

    # 2. Initialize Embeddings Model
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set. Please set it before running.")
        print("DEBUG: OPENAI_API_KEY NOT SET.")
        return False

    logger.info("\nInitializing OpenAI Embeddings...")
    print("DEBUG: Initializing OpenAI Embeddings.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    logger.info("OpenAI Embeddings initialized.")
    print("DEBUG: OpenAI Embeddings initialized.")

    # 3. Create/Load Chroma Vector Store
    persist_directory = "./chroma_db"
    print(f"DEBUG: Checking for chroma_db existence at: {persist_directory}")
    
    # Ensure chroma_db directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    
    try:
        chroma_db_exists = os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0

        if chroma_db_exists:
            logger.info(f"Loading existing ChromaDB from {persist_directory}...")
            print("DEBUG: Loading existing ChromaDB.")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            logger.info("ChromaDB loaded.")
        else:
            logger.info(f"Creating new ChromaDB and populating with {len(chunks)} chunks...")
            print("DEBUG: Creating new ChromaDB.")
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
            logger.info("New ChromaDB created and populated.")

        vectorstore.persist()
        logger.info("Vectorstore persisted to disk.")
        print("DEBUG: Vectorstore persisted.")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}", exc_info=True)
        print(f"DEBUG: ChromaDB initialization failed: {e}")
        return False


    # 4. Set up the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Reduced from 3 to 2
    logger.info("Retriever configured.")
    print("DEBUG: Retriever configured.")

    # 5. Set up the RAG Chain
    logger.info("\nInitializing ChatOpenAI LLM...")
    print("DEBUG: Initializing ChatOpenAI LLM.")
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, max_tokens=500)  # Added max_tokens limit
    logger.info("ChatOpenAI LLM initialized.")
    print("DEBUG: ChatOpenAI LLM initialized.")

    logger.info("Setting up RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    logger.info("RetrievalQA chain ready.")
    print("DEBUG: RetrievalQA chain ready.")

    print("DEBUG: initialize_rag_system completed.")
    return True

# --- RAG System Initialization ---
print("DEBUG: Calling initialize_rag_system.")
if not initialize_rag_system():
    logger.critical("CRITICAL ERROR: RAG system failed to initialize on startup. API will not function.")
    print("DEBUG: RAG system failed to initialize.")
    # Consider sys.exit(1) here if you want to prevent Flask from starting
    # if the RAG system is critical for function. For now, we allow it.


# --- Flask API Setup ---
app = Flask(__name__)
print("DEBUG: Flask app created.")

# --- Health Check Endpoint ---
@app.route('/')
def health_check():
    print("DEBUG: Health check endpoint hit.")
    status = {
        "status": "OK",
        "rag_initialized": qa_chain is not None,
        "chunks_loaded": len(chunks) if chunks else 0,
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))
    }
    return jsonify(status), 200

# --- Test endpoint for debugging ---
@app.route('/test_timeout', methods=['POST'])
def test_timeout():
    """Test endpoint to verify timeout functionality"""
    print("DEBUG: Test timeout endpoint hit.")
    data = request.get_json()
    test_duration = data.get('duration', 30)  # Changed to 5 seconds
    
    def long_running_task():
        time.sleep(test_duration)
        return f"Task completed after {test_duration} seconds"
    
    try:
        logger.info(f"Testing timeout with {test_duration} second task...")
        start_time = time.time()
        
        timeout_task = run_with_timeout(long_running_task, timeout_seconds=10)
        result = timeout_task()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Test task completed in {elapsed_time:.2f} seconds")
        return jsonify({"result": result, "elapsed_time": elapsed_time}), 200
    except TimeoutError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Test task timed out after {elapsed_time:.2f} seconds: {e}")
        return jsonify({"error": "Test task timed out", "elapsed_time": elapsed_time}), 408
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Test task error after {elapsed_time:.2f} seconds: {e}")
        return jsonify({"error": str(e), "elapsed_time": elapsed_time}), 500

# --- Test embedding endpoint ---
@app.route('/test_embedding', methods=['POST'])
def test_embedding():
    """Test endpoint to verify embedding functionality"""
    print("DEBUG: Test embedding endpoint hit.")
    data = request.get_json()
    test_text = data.get('text', 'test query')
    
    try:
        logger.info(f"Testing embedding generation for: {test_text}")
        start_time = time.time()
        
        if embeddings is None:
            return jsonify({"error": "Embeddings not initialized"}), 500
            
        embedding = embeddings.embed_query(test_text)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Embedding generated in {elapsed_time:.2f} seconds, length: {len(embedding)}")
        return jsonify({
            "success": True, 
            "embedding_length": len(embedding),
            "elapsed_time": elapsed_time
        }), 200
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Embedding generation failed after {elapsed_time:.2f} seconds: {e}")
        return jsonify({"error": f"Embedding generation failed: {str(e)}", "elapsed_time": elapsed_time}), 500

# --- Main Query RAG API Endpoint ---
@app.route('/query_rag', methods=['POST'])
def query_rag_api():
    print("DEBUG: Query RAG API endpoint hit.")
    if qa_chain is None:
        logger.error("Attempted to query RAG, but qa_chain is not initialized.")
        return jsonify({"error": "RAG system not initialized."}), 500

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No 'query' provided in the request body."}), 400

    logger.info(f"Received query: {user_query}")
    try:
        # Add debug logging
        logger.info("Starting RAG query with timeout...")
        start_time = time.time()
        
        # Let's break down the qa_chain.run() into its components to see where it hangs
        logger.info("Step 1: Retrieving relevant documents...")
        
        # First, let's test if the retriever itself is working
        logger.info("Testing retriever initialization...")
        if retriever is None:
            logger.error("Retriever is None!")
            return jsonify({"error": "Retriever not initialized"}), 500
            
        logger.info("Retriever is initialized, attempting retrieval...")
        
        # Test embedding generation first
        logger.info("Testing embedding generation...")
        try:
            test_embedding = embeddings.embed_query("test query")
            logger.info(f"Embedding generation successful, vector length: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return jsonify({"error": f"Embedding generation failed: {str(e)}"}), 500
        
        timeout_retrieval = run_with_timeout(retriever.get_relevant_documents, timeout_seconds=10)
        docs = timeout_retrieval(user_query)
        logger.info(f"Retrieved {len(docs)} documents")
        
        logger.info("Step 2: Running LLM generation...")
        # Create a simple prompt for the LLM
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Based on the following context, answer the question. If you cannot answer from the context, say so.\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
        logger.debug(f"Prompt: {prompt}")
        # Use the LLM directly with timeout
        timeout_llm_call = run_with_timeout(llm.invoke, timeout_seconds=15)
        llm_response = timeout_llm_call(prompt)
        logger.debug(f"LLM response: {llm_response}")
        elapsed_time = time.time() - start_time
        logger.info(f"Query processed successfully in {elapsed_time:.2f} seconds.")
        return jsonify({"response": llm_response.content}), 200
    except TimeoutError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Query timed out after {elapsed_time:.2f} seconds: {e}")
        return jsonify({"error": "Query timed out. Please try again with a simpler question."}), 408
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error processing query after {elapsed_time:.2f} seconds: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# --- Main entry point for local development ---
if __name__ == '__main__':
    print("DEBUG: Running Flask app locally.")
    port = int(os.environ.get('PORT', 5001))
    print(f"DEBUG: Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)