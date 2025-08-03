import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
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
    
    # Load CSV data
    df = pd.read_csv(structured_data_path)
    
    for index, row in df.iterrows():
        page_content = str(row['chunk_content'])
        metadata = {col: str(row[col]) for col in df.columns if col != 'chunk_content'}
        metadata['source_file'] = "my_structured_document.csv"
        metadata['row_number'] = index
        doc = Document(page_content=page_content, metadata=metadata)
        chunks.append(doc)

    # Initialize embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create vector store
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    
    chroma_db_exists = os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0

    if chroma_db_exists:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

    vectorstore.persist()

    # Set up retriever and chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, max_tokens=500)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    return True

# Initialize RAG system
initialize_rag_system()

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

        rag_response = qa_chain.run(user_query)

        print("✅ RAG chain completed:", rag_response)
        return jsonify({"response": rag_response}), 200

    except Exception as e:
        print("❌ Error during query:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)