import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import tiktoken
import json
from flask import Flask, request, jsonify # Ensure jsonify is imported
import logging
from dotenv import load_dotenv

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (if present)
load_dotenv()

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
structured_data_file = "my_structured_document.csv"
# IMPORTANT CHANGE HERE: Add 'data' to the path
structured_data_path = os.path.join(current_dir, 'data', structured_data_file)

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

logger.info(f"Constructed data path: {structured_data_path}")

# --- Global variables for RAG components ---
# These will be initialized by initialize_rag_system
chunks = []
embeddings = None
vectorstore = None
retriever = None
llm = None
qa_chain = None # This will hold your initialized RAG chain

# --- Function to initialize RAG components ---
def initialize_rag_system():
    # Declare global to modify the variables defined at the top
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
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set. Please set it before running.")
        return False

    logger.info("\nInitializing OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    logger.info("OpenAI Embeddings initialized.")

    # 3. Create/Load Chroma Vector Store
    persist_directory