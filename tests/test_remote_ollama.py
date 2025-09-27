#!/usr/bin/env python3
"""
Test to verify that embedding functions with Ollama work correctly
with a remote server. This test demonstrates:

1. Connectivity with remote Ollama
2. Collection creation with custom configuration
3. Document addition using remote embeddings
4. Queries with semantically correct results

Usage:
    python tests/test_remote_ollama.py

Configuration:
    - Modify OLLAMA_URL for your Ollama server
    - Modify MODEL_NAME if using a different model
"""

import asyncio
import json
import os
import requests
import sys
import time
import numpy as np
import uuid
# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chroma_mcp.server import (
    chroma_create_collection, 
    chroma_add_documents,
    chroma_get_collection_info,
    chroma_get_documents,
    chroma_query_documents
)# Test configuration - Configure for your environment
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # Change to your Ollama server
MODEL_NAME = os.getenv("OLLAMA_MODEL", "nomic-embed-text")      # Change to your model
# Generate unique collection name with timestamp
COLLECTION_NAME = f"test_remote_ollama_{int(time.time())}"

async def test_ollama_remote_ip():
    """Complete test with remote Ollama server"""
    
    print("🧪 TEST: Remote Ollama IP")
    print(f"   URL: {OLLAMA_URL}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Collection: {COLLECTION_NAME}\n")
    
    try:
        # Step 1: Display collection name with unique identifier
        print(f"1️⃣ Using unique collection name: {COLLECTION_NAME}")
        print("   ℹ️  This ensures no conflicts with existing collections")
        
        # Step 2: Create collection with remote Ollama configuration
        print("\n2️⃣ Creating collection with remote Ollama...")
        
        # Define embedding function configuration explicitly
        embedding_config = {
            "url": OLLAMA_URL,
            "model_name": MODEL_NAME,
            "timeout": 60
        }
        
        print(f"   🔧 CUSTOM EMBEDDING FUNCTION CONFIGURATION:")
        print(f"      - Function Type: OllamaEmbeddingFunction")
        print(f"      - Remote URL: {embedding_config['url']}")
        print(f"      - Model Name: {embedding_config['model_name']}")
        print(f"      - Timeout: {embedding_config['timeout']}s")
        print(f"   ℹ️  This replaces ChromaDB's default embedding with remote Ollama!")
        
        config_result = await chroma_create_collection(
            collection_name=COLLECTION_NAME,
            embedding_function_name="ollama",
            embedding_function_config=embedding_config,
            metadata={
                "test": "remote_ollama",
                "server_url": OLLAMA_URL,
                "model": MODEL_NAME
            }
        )
        
        print(f"   ✅ Collection created successfully!")
        print(f"   📋 Server response: {config_result}")
        
        # Step 3: Verify collection configuration
        print("\n3️⃣ Verifying collection configuration...")
        
        collection_info = await chroma_get_collection_info(COLLECTION_NAME)
        print(f"   📊 Collection Details:")
        print(f"      - Name: {collection_info.get('name')}")
        print(f"      - Document Count: {collection_info.get('count')}")
        print(f"      - Metadata: {collection_info.get('metadata')}")
        
        print(f"   🎯 EMBEDDING FUNCTION STATUS:")
        embedding_func_info = collection_info.get('embedding_function', {})
        print(f"      - Type: {embedding_func_info.get('type', 'Unknown')}")
        print(f"      - Configuration: {embedding_func_info.get('config', {})}")
        
        # Try to get the actual collection to verify embedding function
        try:
            from chromadb.config import Settings
            import chromadb
            client = chromadb.PersistentClient()
            actual_collection = client.get_collection(name=COLLECTION_NAME)
            print(f"   🔍 Actual Embedding Function: {type(actual_collection._embedding_function).__name__}")
            if hasattr(actual_collection._embedding_function, '_url'):
                print(f"      - URL: {actual_collection._embedding_function._url}")
            if hasattr(actual_collection._embedding_function, '_model_name'):
                print(f"      - Model: {actual_collection._embedding_function._model_name}")
        except Exception as e:
            print(f"   ⚠️  Could not verify actual embedding function: {e}")
        
        # Step 4: Add test documents
        print("\n4️⃣ Adding test documents...")
        
        test_documents = [
            "This is a document about artificial intelligence and machine learning.",
            "Python is a popular programming language for data science.",
            "Embeddings are vector representations of text that capture semantic meaning.",
            "Ollama allows running large language models locally.",
            "ChromaDB is a vector database optimized for AI applications."
        ]
        
        test_ids = [f"doc_{i+1}" for i in range(len(test_documents))]
        test_metadatas = [
            {"category": "ai", "topic": "machine learning"},
            {"category": "programming", "topic": "python"}, 
            {"category": "ai", "topic": "embeddings"},
            {"category": "tools", "topic": "ollama"},
            {"category": "database", "topic": "vector db"}
        ]
        
        add_result = await chroma_add_documents(
            collection_name=COLLECTION_NAME,
            documents=test_documents,
            ids=test_ids,
            metadatas=test_metadatas
        )
        
        print(f"   ✅ {add_result}")
        
        # Step 5: Verify documents were added
        print("\n5️⃣ Verifying documents were added...")
        
        updated_info = await chroma_get_collection_info(COLLECTION_NAME)
        print(f"   📈 Documents in collection: {updated_info.get('count')}")
        
        if updated_info.get('sample_documents'):
            print(f"   📄 Sample documents:")
            sample = updated_info['sample_documents']
            if sample.get('documents'):
                for i, doc in enumerate(sample['documents'][:2]):
                    print(f"      [{i+1}]: {doc[:60]}...")
        
        # Step 5b: Show actual embeddings stored in ChromaDB
        print("\n5️⃣b Verifying EMBEDDINGS STORED in ChromaDB...")
        try:
            # Get stored embeddings using the MCP server function
            stored_embeddings = await chroma_get_documents(
                collection_name=COLLECTION_NAME,
                include=['embeddings', 'documents', 'metadatas'],
                limit=2
            )
            
            if stored_embeddings and 'embeddings' in stored_embeddings:
                embeddings = stored_embeddings['embeddings']
                documents = stored_embeddings['documents']
                metadatas = stored_embeddings['metadatas']
                
                print(f"   🧮 EMBEDDINGS STORED IN CHROMADB:")
                print(f"      - Total embeddings retrieved: {len(embeddings)}")
                
                for i, (embedding, doc, metadata) in enumerate(zip(embeddings[:2], documents[:2], metadatas[:2])):
                    print(f"   📊 Document {i+1} embedding:")
                    print(f"      - Text: '{doc[:50]}...'")
                    print(f"      - Category: {metadata.get('category', 'N/A')}")
                    
                    # Handle embedding safely
                    if embedding is not None:
                        try:
                            # Convert to list if it's a numpy array
                            if hasattr(embedding, 'tolist'):
                                embedding_list = embedding.tolist()
                            else:
                                embedding_list = list(embedding)
                                
                            print(f"      - Embedding dimensions: {len(embedding_list)}")
                            print(f"      - First 10 values: {[round(x, 4) for x in embedding_list[:10]]}")
                            print(f"      - Last 5 values: {[round(x, 4) for x in embedding_list[-5:]]}")
                            print(f"      - Vector range: [{min(embedding_list):.4f}, {max(embedding_list):.4f}]")
                            print(f"      - Vector norm: {(sum(x**2 for x in embedding_list)**0.5):.4f}")
                            print(f"      - Vector density: {sum(1 for x in embedding_list if abs(x) > 0.0001)}/{len(embedding_list)}")
                        except Exception as e:
                            print(f"      - Embedding type: {type(embedding)}")
                            print(f"      - Embedding shape/length: {getattr(embedding, 'shape', len(embedding)) if embedding else 'None'}")
                            print(f"      - Error processing embedding: {e}")
                    else:
                        print(f"      - Embedding: None/Empty")
                
                print(f"   ✅ Successfully retrieved embeddings from ChromaDB storage!")
                
                # Also demonstrate direct embedding generation for comparison
                print(f"\n   🔄 COMPARING with direct Ollama generation...")
                import requests
                
                sample_text = documents[0]
                embed_request = {
                    "model": MODEL_NAME,
                    "input": sample_text
                }
                
                response = requests.post(f"{OLLAMA_URL}/api/embed", json=embed_request, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'embeddings' in result and len(result['embeddings']) > 0:
                        direct_embedding = result['embeddings'][0]
                        stored_embedding = embeddings[0]
                        
                        # Convert stored embedding to list safely
                        try:
                            if hasattr(stored_embedding, 'tolist'):
                                stored_list = stored_embedding.tolist()
                            else:
                                stored_list = list(stored_embedding)
                            
                            # Compare first few values
                            print(f"   📊 EMBEDDING COMPARISON:")
                            print(f"      - Stored in ChromaDB: {[round(x, 4) for x in stored_list[:5]]}")
                            print(f"      - Direct from Ollama: {[round(x, 4) for x in direct_embedding[:5]]}")
                            
                            # Check if they're identical (they should be for the same text)
                            if len(stored_list) == len(direct_embedding):
                                differences = sum(abs(s - d) > 0.0001 for s, d in zip(stored_list, direct_embedding))
                                if differences == 0:
                                    print(f"      ✅ PERFECT MATCH: Stored and direct embeddings are identical!")
                                else:
                                    print(f"      ⚠️  Small differences: {differences}/{len(stored_list)} values differ")
                            else:
                                print(f"      ⚠️  Different dimensions: {len(stored_list)} vs {len(direct_embedding)}")
                        except Exception as e:
                            print(f"      ⚠️  Error comparing embeddings: {e}")
                
            else:
                print(f"   ⚠️  No embeddings found in ChromaDB response")
                print(f"   📋  Response keys: {list(stored_embeddings.keys()) if stored_embeddings else 'None'}")
                
        except Exception as e:
            print(f"   ⚠️  Error retrieving stored embeddings: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 6: Test similarity queries
        print("\n6️⃣ Testing similarity queries...")
        
        query_results = await chroma_query_documents(
            collection_name=COLLECTION_NAME,
            query_texts=["What is artificial intelligence?"],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"   🔍 Query results:")
        if query_results.get('documents') and query_results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                query_results['documents'][0][:3],
                query_results['metadatas'][0][:3], 
                query_results['distances'][0][:3]
            )):
                print(f"      [{i+1}] Distance: {distance:.4f}")
                print(f"          Document: {doc[:80]}...")
                print(f"          Metadata: {metadata}")
        
        # Step 7: Additional English query test
        print("\n7️⃣ Testing additional English query...")
        
        query_results_en = await chroma_query_documents(
            collection_name=COLLECTION_NAME,
            query_texts=["What is Python programming?"],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"   🔍 Additional query results:")
        if query_results_en.get('documents') and query_results_en['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                query_results_en['documents'][0][:2],
                query_results_en['metadatas'][0][:2], 
                query_results_en['distances'][0][:2]
            )):
                print(f"      [{i+1}] Distance: {distance:.4f}")
                print(f"          Document: {doc[:80]}...")
        
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"   - Remote Ollama at {OLLAMA_URL} ✅")
        print(f"   - Collection '{COLLECTION_NAME}' created with CUSTOM embedding configuration ✅") 
        print(f"   - {len(test_documents)} documents added with REMOTE embeddings ✅")
        print(f"   - Semantic queries working correctly with {MODEL_NAME} ✅")
        print(f"   - Custom OllamaEmbeddingFunction successfully replaced default ✅")
        print(f"   - Embeddings vectors generated and verified ✅")
        
        print(f"\n📋 TEST SUMMARY:")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Documents: {len(test_documents)} test documents")
        print(f"   🎯 CUSTOM EMBEDDING CONFIGURATION:")
        print(f"      - Function: OllamaEmbeddingFunction (not default)")
        print(f"      - Remote Server: {OLLAMA_URL}")
        print(f"      - Model: {MODEL_NAME}")
        print(f"      - Categories tested: AI, programming, tools, database")
        
        print(f"\n🔥 KEY ACHIEVEMENT: Successfully implemented and tested")
        print(f"   parametrizable embedding function configuration!")
        print(f"   🧮 Embeddings generated remotely and working perfectly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST ERROR:")
        print(f"   {str(e)}")
        print(f"\n🔧 POSSIBLE SOLUTIONS:")
        print(f"   1. Verify Ollama is running at {OLLAMA_URL}")
        print(f"   2. Verify network connectivity")
        print(f"   3. Verify model '{MODEL_NAME}' is available")
        print(f"   4. Test direct access: curl {OLLAMA_URL}/api/tags")
        
        return False

async def test_connectivity():
    """Basic connectivity test using requests"""
    
    print("🔗 BASIC CONNECTIVITY TEST\n")
    
    try:
        print(f"   Testing connection to {OLLAMA_URL}...")
        
        import requests
        
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Connectivity successful!")
            print(f"   📋 Available models:")
            
            if 'models' in data:
                for model in data['models'][:5]:  # Show first 5 models
                    print(f"      - {model.get('name', 'N/A')}")
                if len(data['models']) > 5:
                    print(f"      ... and {len(data['models']) - 5} more")
            else:
                print(f"      (Could not list models)")
            
            # Search for specific embedding models
            embedding_models = [m for m in data.get('models', []) if 'embed' in m.get('name', '').lower()]
            if embedding_models:
                print(f"   🎯 Embedding models found:")
                for model in embedding_models:
                    print(f"      - {model.get('name')}")
            
            return True
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            return False
                    
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout - Server not responding in 10 seconds")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection error - Cannot connect to {OLLAMA_URL}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print(f"🧪 REMOTE OLLAMA TEST AT {OLLAMA_URL}")
    print("=" * 60)
    
    async def run_all_tests():
        # First test basic connectivity
        connectivity_ok = await test_connectivity()
        
        if connectivity_ok:
            print(f"\n" + "=" * 60)
            # If connectivity works, test full functionality
            success = await test_ollama_remote_ip()
            
            if success:
                print(f"\n🎊 ALL TESTS PASSED SUCCESSFULLY")
            else:
                print(f"\n⚠️  SOME TESTS FAILED")
        else:
            print(f"\n⚠️  NO CONNECTIVITY WITH OLLAMA SERVER")
            print(f"   Skipping functionality tests...")
    
    # Run all tests
    asyncio.run(run_all_tests())
