import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings # For generating embeddings
from langchain_community.vectorstores import Chroma # For storing embeddings and chunks
from langchain.chains import RetrievalQA # For building the RAG chain
from langchain_community.chat_models import ChatOpenAI # For the LLM
import tiktoken # For token counting (optional, but good for diagnostics)

# --- Configuration ---
# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the name of your structured data file (your CSV from Google Sheets)
# IMPORTANT: Ensure this filename exactly matches your uploaded CSV
structured_data_file = "my_structured_document.csv"

# Construct the full path to your CSV file
# Assuming your CSV is in a 'data' subfolder within your 'rag-bot' directory
structured_data_path = os.path.join(current_dir, 'data', structured_data_file)

# If your CSV is directly in the 'rag-bot' directory (not in 'data' subfolder), use this instead:
# structured_data_path = os.path.join(current_dir, structured_data_file)

# --- (Optional) Tokenizer for Chunk Size Estimation ---
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

print(f"Constructed data path: {structured_data_path}")

# --- 1. Load Manually Structured Data from CSV ---
chunks = [] # This list will hold your LangChain Document objects

if not os.path.exists(structured_data_path):
    print(f"Error: The structured data file does NOT exist at '{structured_data_path}'.")
    print("Please ensure your CSV file is saved at the specified path relative to your script.")
else:
    try:
        print(f"\nLoading data from {structured_data_file}...")
        df = pd.read_csv(structured_data_path)

        for index, row in df.iterrows():
            page_content = str(row['chunk_content'])
            metadata = {col: str(row[col]) for col in df.columns if col != 'chunk_content'}
            metadata['source_file'] = structured_data_file
            metadata['row_number'] = index
            doc = Document(page_content=page_content, metadata=metadata)
            chunks.append(doc)

        print(f"Successfully loaded {len(chunks)} chunks from {structured_data_file}.")

        if chunks:
            print("\n--- First 5 Chunks (from your CSV) ---")
            for i, chunk in enumerate(chunks[:5]):
                token_count = len(tokenizer.encode(chunk.page_content)) if 'tokenizer' in locals() else len(chunk.page_content) // 4
                print(f"Chunk {i+1} (Tokens: {token_count}, Chars: {len(chunk.page_content)}):")
                print(chunk.page_content[:200] + "...")
                print(f"Metadata: {chunk.metadata}")
                print("-" * 50)
        else:
            print("No chunks were created from the CSV. Please check your CSV content and column names.")

    except KeyError as e:
        print(f"Error: Missing expected column in CSV. Please ensure your CSV has a 'chunk_content' column and any other columns you try to access. Missing: {e}")
        print("Double-check the exact spelling of your column headers in Google Sheets.")
    except Exception as e:
        print(f"An unexpected error occurred during CSV loading: {e}")

print("\nYour 'chunks' list is now populated with custom-structured documents!")

# --- 2. Initialize Embeddings Model ---
# Ensure your OPENAI_API_KEY environment variable is set
# e.g., export OPENAI_API_KEY="your_key_here" in your terminal before running the script
print("\nInitializing OpenAI Embeddings...")
embeddings = OpenAIEmbeddings()
print("OpenAI Embeddings initialized.")

# --- 3. Create/Load Chroma Vector Store ---
# This will create a local directory named 'chroma_db' to store your embeddings
# If the directory already exists, it will load the existing vector store
persist_directory = "./chroma_db"

if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print(f"Loading existing ChromaDB from {persist_directory}...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("ChromaDB loaded.")
else:
    print(f"Creating new ChromaDB and populating with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    print("ChromaDB created and populated.")

# Ensure the vector store is persisted
vectorstore.persist()
print("Vectorstore persisted to disk.")

# --- 4. Set up the Retriever ---
# The retriever fetches relevant documents from the vector store based on a query
# k=3 means it will retrieve the top 3 most similar chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever configured.")

# --- 5. Set up the RAG Chain ---
# Initialize the Large Language Model (LLM) for generation
# You can adjust 'temperature' for creativity (0.0 for factual, higher for more creative)
print("\nInitializing ChatOpenAI LLM...")
llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo") # You can specify a different model if desired
print("ChatOpenAI LLM initialized.")

# Create the RetrievalQA chain
# chain_type="stuff" means it will put all retrieved documents into the prompt
print("Setting up RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
print("RetrievalQA chain ready.")

# --- 6. Test your RAG System ---
print("\n--- Testing RAG System ---")
query = "What are the core responsibilities of an APS6, particularly regarding knowledge and accountability?"
print(f"Query: {query}")
response = qa_chain.run(query)
print(f"\nResponse:\n{response}")

# You can try another query
query_2 = "Describe the 'Independence and Decision-Making' aspects of the APS6 role."
print(f"\nQuery: {query_2}")
response_2 = qa_chain.run(query_2)
print(f"\nResponse:\n{response_2}")

print("\nRAG system setup and tested successfully!")