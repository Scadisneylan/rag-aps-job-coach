#!/usr/bin/env python3
"""
Test script to debug embedding and timeout issues
"""
import os
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_timeout_wrapper():
    """Test the timeout wrapper functionality"""
    print("=== Testing Timeout Wrapper ===")
    
    # Import the timeout wrapper from rag_bot
    import sys
    sys.path.append('.')
    
    # Import the timeout function
    from rag_bot import run_with_timeout
    
    def long_task(duration):
        time.sleep(duration)
        return f"Task completed after {duration} seconds"
    
    # Test 1: Short task (should succeed)
    print("Test 1: Short task (3 seconds)")
    try:
        timeout_task = run_with_timeout(long_task, timeout_seconds=5)
        result = timeout_task(3)
        print(f"✅ SUCCESS: {result}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test 2: Long task (should timeout)
    print("Test 2: Long task (10 seconds)")
    try:
        timeout_task = run_with_timeout(long_task, timeout_seconds=5)
        result = timeout_task(10)
        print(f"❌ FAILED: Should have timed out but got {result}")
    except Exception as e:
        print(f"✅ SUCCESS: Timed out as expected - {e}")

def test_embeddings():
    """Test embedding functionality"""
    print("\n=== Testing Embeddings ===")
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ FAILED: OPENAI_API_KEY not set")
        return
    
    print(f"✅ OpenAI API key found: {api_key[:10]}...")
    
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
        
        print("Initializing OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        print("Testing embedding generation...")
        start_time = time.time()
        
        test_text = "This is a test query for embedding generation"
        embedding = embeddings.embed_query(test_text)
        
        elapsed_time = time.time() - start_time
        print(f"✅ SUCCESS: Embedding generated in {elapsed_time:.2f} seconds")
        print(f"   Vector length: {len(embedding)}")
        print(f"   First few values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

def test_rag_components():
    """Test RAG components initialization"""
    print("\n=== Testing RAG Components ===")
    
    try:
        # Import the initialization function
        from rag_bot import initialize_rag_system
        
        print("Testing RAG system initialization...")
        start_time = time.time()
        
        success = initialize_rag_system()
        
        elapsed_time = time.time() - start_time
        if success:
            print(f"✅ SUCCESS: RAG system initialized in {elapsed_time:.2f} seconds")
        else:
            print(f"❌ FAILED: RAG system initialization failed after {elapsed_time:.2f} seconds")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")

def test_retrieval():
    """Test document retrieval"""
    print("\n=== Testing Document Retrieval ===")
    
    try:
        from rag_bot import retriever
        
        if retriever is None:
            print("❌ FAILED: Retriever is None")
            return
        
        print("Testing document retrieval...")
        start_time = time.time()
        
        test_query = "What is this document about?"
        docs = retriever.get_relevant_documents(test_query)
        
        elapsed_time = time.time() - start_time
        print(f"✅ SUCCESS: Retrieved {len(docs)} documents in {elapsed_time:.2f} seconds")
        
        if docs:
            print(f"   First document preview: {docs[0].page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    print("Starting debug tests...\n")
    
    # Run tests
    test_timeout_wrapper()
    test_embeddings()
    test_rag_components()
    test_retrieval()
    
    print("\n=== Debug tests completed ===") 