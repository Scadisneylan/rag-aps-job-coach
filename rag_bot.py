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
# IMPORTANT CHANGE HERE: Add 'data' to the path
structured_data_path = os.path.join(current_dir, 'data', structured_data_file)

# Ensure data directory exists
data_dir = os.path.join(current_dir, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever configured.")
    print("DEBUG: Retriever configured.")

    # 5. Set up the RAG Chain
    logger.info("\nInitializing ChatOpenAI LLM...")
    print("DEBUG: Initializing ChatOpenAI LLM.")
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
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
        rag_response = qa_chain.run(user_query)
        logger.info("Query processed successfully.")
        return jsonify({"response": rag_response}), 200
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing the query."}), 500

# --- Main entry point for local development ---
if __name__ == '__main__':
    print("DEBUG: Running Flask app locally.")
    port = int(os.environ.get('PORT', 5000))
    print(f"DEBUG: Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)