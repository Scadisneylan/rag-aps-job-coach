import os
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
structured_data_path = os.path.join(current_dir, "my_structured_document.csv")

# Global variables
chunks = []
embeddings = None
vectorstore = None
retriever = None
llm = None
qa_chain = None

def initialize_rag_system():
    global chunks, embeddings, vectorstore, retriever, llm, qa_chain
    print("🔧 Initializing RAG system...")
    
    # Check if CSV file exists
    if not os.path.exists(structured_data_path):
        print(f"⚠️ CSV file not found: {structured_data_path}")
        return False
    
    # Load CSV data
    df = pd.read_csv(structured_data_path)
    print(f"📊 Loaded {len(df)} rows from CSV")
    
    if df.empty:
        print("⚠️ CSV file is empty")
        return False
    
    # Create documents
    for index, row in df.iterrows():
        page_content = str(row['chunk_content'])
        metadata = {col: str(row[col]) for col in df.columns if col != 'chunk_content'}
        metadata['source_file'] = "my_structured_document.csv"
        metadata['row_number'] = index
        doc = Document(page_content=page_content, metadata=metadata)
        chunks.append(doc)

    # Initialize embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️ OPENAI_API_KEY not found in environment")
        return False
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create vector store
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    
    chroma_db_exists = os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0

    try:
        if chroma_db_exists:
            print("🔄 Loading existing Chroma database...")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        else:
            print("🆕 Creating new Chroma database...")
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
            # Only persist once after creation
            vectorstore.persist()
            print("💾 Database persisted successfully")

        # Pre-warm the vectorstore with a test query
        print("🔥 Pre-warming vectorstore...")
        test_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        test_docs = test_retriever.get_relevant_documents("test")
        print(f"✅ Vectorstore warmed up with {len(test_docs)} test documents")

        # Set up retriever and chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, max_tokens=500, request_timeout=20)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        # Reinitialize qa_chain to ensure fresh LLM object with proper timeout
        print("🔄 Reinitializing QA chain with fresh LLM object...")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        print("✅ RAG system initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return False

# Initialize RAG system
rag_initialized = initialize_rag_system()
if not rag_initialized:
    print("❌ Failed to initialize RAG system. Check the logs above for errors.")

# Flask app
app = Flask(__name__)

@app.route('/')
def health_check():
    status = {
        "status": "OK",
        "rag_initialized": qa_chain is not None,
        "chunks_loaded": len(chunks) if chunks else 0,
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))
    }
    return jsonify(status), 200

@app.route('/query_rag', methods=['POST'])
def query_rag_api():
    try:
        data = request.get_json()
        user_query = data.get('query')

        print("📥 Query received:", user_query)

        if not user_query:
            print("⚠️ No query provided")
            return jsonify({"error": "No 'query' provided in the request body."}), 400

        print("🧠 About to run QA chain...")

        rag_response = qa_chain.invoke(user_query)

        print("✅ RAG chain completed:", rag_response)
        return jsonify({"response": rag_response}), 200

    except Exception as e:
        print("❌ Error during query:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)