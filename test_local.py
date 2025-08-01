#!/usr/bin/env python3
"""
Local test script for RAG bot before deployment
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test environment setup"""
    print("=== Environment Test ===")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OPENAI_API_KEY is set (first 5 chars: {api_key[:5]}...)")
    else:
        print("❌ OPENAI_API_KEY is not set")
        return False
    
    # Check data file
    data_file = "data/my_structured_document.csv"
    if os.path.exists(data_file):
        print(f"✅ Data file exists: {data_file}")
    else:
        print(f"❌ Data file missing: {data_file}")
        return False
    
    return True

def test_imports():
    """Test all required imports"""
    print("\n=== Import Test ===")
    
    try:
        import pandas as pd
        print("✅ pandas imported")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        from langchain_core.documents import Document
        print("✅ langchain_core imported")
    except ImportError as e:
        print(f"❌ langchain_core import failed: {e}")
        return False
    
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
        print("✅ OpenAIEmbeddings imported")
    except ImportError as e:
        print(f"❌ OpenAIEmbeddings import failed: {e}")
        return False
    
    try:
        from langchain_community.vectorstores import Chroma
        print("✅ Chroma imported")
    except ImportError as e:
        print(f"❌ Chroma import failed: {e}")
        return False
    
    try:
        from langchain.chains import RetrievalQA
        print("✅ RetrievalQA imported")
    except ImportError as e:
        print(f"❌ RetrievalQA import failed: {e}")
        return False
    
    try:
        from langchain_community.chat_models import ChatOpenAI
        print("✅ ChatOpenAI imported")
    except ImportError as e:
        print(f"❌ ChatOpenAI import failed: {e}")
        return False
    
    try:
        import tiktoken
        print("✅ tiktoken imported")
    except ImportError as e:
        print(f"❌ tiktoken import failed: {e}")
        return False
    
    try:
        from flask import Flask
        print("✅ Flask imported")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    return True

def test_rag_initialization():
    """Test RAG system initialization"""
    print("\n=== RAG Initialization Test ===")
    
    try:
        # Import the initialization function
        from rag_bot import initialize_rag_system
        
        # Test initialization
        success = initialize_rag_system()
        if success:
            print("✅ RAG system initialized successfully")
            return True
        else:
            print("❌ RAG system initialization failed")
            return False
    except Exception as e:
        print(f"❌ RAG initialization error: {e}")
        return False

if __name__ == "__main__":
    print("Starting local deployment tests...\n")
    
    # Run all tests
    env_ok = test_environment()
    imports_ok = test_imports()
    
    if env_ok and imports_ok:
        print("\n✅ All basic tests passed! Ready for deployment.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY in Render environment variables")
        print("2. Deploy to Render using the render.yaml configuration")
        print("3. Monitor the deployment logs for any issues")
    else:
        print("\n❌ Some tests failed. Please fix issues before deploying.")
        sys.exit(1) 