import os
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
structured_data_path = os.path.join(current_dir, "my_structured_document.csv")

chunks = []
embeddings = None
vectorstore = None
retriever = None
llm = None
qa_chain = None

def initialize_rag_system():
    global chunks, embeddings, vectorstore, retriever, llm, qa_chain
    
    df = pd.read_csv(structured_data_path)
    
    for index, row in df.iterrows():
        page_content = str(row['chunk_content'])
        metadata = {col: str(row[col]) for col in df.columns if col != 'chunk_content'}
        metadata['source_file'] = "my_structured_document.csv"
        metadata['row_number'] = index
        doc = Document(page_content=page_content, metadata=metadata)
        chunks.append(doc)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    
    chroma_db_exists = os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0

    if chroma_db_exists:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
        vectorstore.persist()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, max_tokens=500)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

initialize_rag_system()

app = Flask(__name__)

@app.route('/')
def health_check():
    return jsonify({"status": "OK"}), 200

@app.route('/query_rag', methods=['POST'])
def query_rag_api():
    data = request.get_json()
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({"error": "No 'query' provided in the request body."}), 400

    rag_response = qa_chain.invoke(user_query)
    return jsonify({"response": rag_response}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)