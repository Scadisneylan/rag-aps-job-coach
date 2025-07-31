import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import tiktoken
import json # Added for JSON handling
from flask import Flask, request, jsonify # Added Flask imports
import logging # Added for logging
from dotenv import load_dotenv # For loading environment variables from .env

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (if present)
load_dotenv()

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
structured_data_file = "my_structured_document.csv" # Ensure this filename is correct
structured_data_path = os.path.join(current_dir, 'data', structured_data_file)

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

logger.info(f"Constructed data path: {structured_data_path}")

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

    # 1. Load Manually Structured Data from CSV
    if not os.path.exists(structured_data_path):
        logger.error(f"Error: The structured data file does NOT exist at '{structured_data_path}'.")
        return False
    else:
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
            logger.error(f"An unexpected error occurred during CSV loading: {e}")
            return False

    logger.info("Chunks populated with custom-structured documents!")

    # 2. Initialize Embeddings Model
    # Ensure OPENAI_API_KEY is set as an environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set. Please set it before running.")
        return False
    
    logger.info("\nInitializing OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    logger.info("OpenAI Embeddings initialized.")

    # 3. Create/Load Chroma Vector Store
    persist_directory = "./chroma_db"
    
    # Check if chroma_db directory exists and contains files
    chroma_db_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)
    
    if chroma_db_exists:
        logger.info(f"Loading existing ChromaDB from {persist_directory}...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        logger.info("ChromaDB loaded.")
    else:
        logger.info(f"Creating new ChromaDB and populating with {len(chunks)} chunks...")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
        logger.info("New ChromaDB created and populated.")
        
    # Ensure the vector store is persisted
    vectorstore.persist()
    logger.info("Vectorstore persisted to disk.")


    # 4. Set up the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever configured.")

    # 5. Set up the RAG Chain
    logger.info("\nInitializing ChatOpenAI LLM...")
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    logger.info("ChatOpenAI LLM initialized.")

    logger.info("Setting up RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    logger.info("RetrievalQA chain ready.")
    
    return True

# --- Flask API Setup ---
app = Flask(__name__)

# Initialize RAG system once when the Flask app starts
# This is crucial for performance, so embeddings and vector store aren't reloaded on every API call.
if not initialize_rag_system():
    logger.error("Failed to initialize RAG system. Exiting.")
    # In a production app, you might want to raise an exception or handle this more gracefully
    # For now, we'll let Flask try to run, but API calls will fail.

@app.route('/query_rag', methods=['POST'])
def query_rag_api():
    if qa_chain is None:
        return jsonify({"error": "RAG system not initialized."}), 500

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No 'query' provided in the request body."}), 400

    logger.info(f"Received query: {user_query}")
    try:
        # LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
        # For this example, we'll stick to .run() for simplicity, but .invoke is the future.
        rag_response = qa_chain.run(user_query)
        logger.info("Query processed successfully.")
        return jsonify({"response": rag_response})
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": "An internal error occurred while processing the query."}), 500

# Main entry point for local development
if __name__ == '__main__':
    # You'll need to install Flask: pip install Flask
    # For production, Gunicorn will run this app
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production