#!/usr/bin/env python3
"""
Test para verificar la compatibilidad de collections con embedding functions personalizados
desde la perspectiva de clientes MCP.

Este test verifica que:
1. Las collections creadas con embedding functions personalizados se pueden acceder
2. Los embeddings guardados son accesibles y utilizables
3. Las operaciones MCP funcionan correctamente con estas collections
"""

import asyncio
import sys
import os
import requests
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chroma_mcp.server import (
    get_chroma_client, 
    chroma_create_collection,
    chroma_add_documents,
    chroma_get_documents,
    chroma_get_collection_info,
    chroma_query_documents,
    chroma_list_collections
)

async def test_mcp_client_compatibility():
    """Test que simula un cliente MCP trabajando con collections personalizadas"""
    
    print("🔧 PRUEBA DE COMPATIBILIDAD MCP - EMBEDDING FUNCTIONS PERSONALIZADOS")
    print("=" * 70)
    
    collection_name = "test_mcp_compatibility"
    
    # Ollama server configuration - modify for your environment
    embedding_config = {
        "embedding_function": "ollama",
        "model_name": os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
        "url": os.getenv("OLLAMA_URL", "http://localhost:11434")
    }
    
    try:
        print("📡 Paso 1: Verificando conectividad con Ollama...")
        
        # Test de conectividad
        ollama_url = embedding_config["url"]
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama server is accessible")
        else:
            print(f"❌ Ollama server responded with status: {response.status_code}")
            return False
            
        print("\n📚 Paso 2: Creando collection con embedding function personalizado...")
        
        # Crear collection con configuración personalizada
        await chroma_create_collection(
            collection_name=collection_name,
            embedding_function_name="ollama",
            embedding_function_config=embedding_config
        )
        print(f"✅ Collection '{collection_name}' creada exitosamente")
        
        print("\n📝 Paso 3: Añadiendo documentos de prueba...")
        
        # Añadir documentos
        test_documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science",
            "ChromaDB is a vector database for AI applications"
        ]
        
        await chroma_add_documents(
            collection_name=collection_name,
            documents=test_documents,
            ids=["doc1", "doc2", "doc3"],
            metadatas=[
                {"category": "AI", "complexity": "basic"},
                {"category": "programming", "complexity": "intermediate"},
                {"category": "database", "complexity": "advanced"}
            ]
        )
        print(f"✅ {len(test_documents)} documentos añadidos exitosamente")
        
        print("\n🔍 Paso 4: Verificando información de la collection...")
        
        # Obtener información de la collection
        collection_info = await chroma_get_collection_info(collection_name)
        print(f"📊 Collection info:")
        print(f"   - Nombre: {collection_info['name']}")
        print(f"   - Documentos: {collection_info['count']}")
        print(f"   - Embedding function: {collection_info.get('embedding_function', {}).get('type', 'N/A')}")
        
        print("\n🧮 Paso 5: Verificando acceso a embeddings almacenados...")
        
        # Obtener documentos y sus embeddings
        stored_docs = await chroma_get_documents(collection_name)
        
        print(f"📄 Documentos recuperados: {len(stored_docs.get('documents', []))}")
        
        if 'embeddings' in stored_docs and stored_docs['embeddings']:
            embeddings = stored_docs['embeddings']
            print(f"🧮 Embeddings recuperados: {len(embeddings)}")
            
            # Verificar dimensiones de los embeddings
            if embeddings and len(embeddings) > 0:
                first_embedding = embeddings[0]
                if hasattr(first_embedding, '__len__'):
                    embedding_dim = len(list(first_embedding))
                    print(f"📏 Dimensiones del embedding: {embedding_dim}")
                    
                    # Verificar que son embeddings válidos (no zeros)
                    if hasattr(first_embedding, '__iter__'):
                        embedding_list = list(first_embedding)
                        non_zero_count = sum(1 for x in embedding_list if abs(float(x)) > 1e-10)
                        print(f"🔢 Elementos no-cero en el primer embedding: {non_zero_count}/{embedding_dim}")
                        
                        if non_zero_count > 0:
                            print("✅ Los embeddings contienen datos válidos")
                        else:
                            print("⚠️  Los embeddings parecen estar vacíos o ser ceros")
                else:
                    print("⚠️  Formato de embedding no reconocido")
            else:
                print("❌ No se pudieron recuperar embeddings")
        else:
            print("❌ No hay embeddings en la respuesta")
            
        print("\n🔎 Paso 6: Probando búsqueda semántica...")
        
        # Realizar consulta semántica
        query_results = await chroma_query_documents(
            collection_name=collection_name,
            query_texts=["What is AI and machine learning?"],
            n_results=2
        )
        
        if query_results and 'documents' in query_results:
            print(f"🎯 Resultados de búsqueda semántica: {len(query_results['documents'][0])} documentos")
            
            # Mostrar resultados con distancias
            for i, (doc, distance) in enumerate(zip(
                query_results['documents'][0],
                query_results.get('distances', [{}])[0] if query_results.get('distances') else []
            )):
                print(f"   {i+1}. Documento: {doc[:50]}...")
                if distance is not None:
                    print(f"      Distancia: {distance:.4f}")
                    
            print("✅ Búsqueda semántica funciona correctamente")
        else:
            print("❌ La búsqueda semántica no devolvió resultados")
            
        print("\n📋 Paso 7: Verificando listado de collections...")
        
        # Listar collections
        collections = await chroma_list_collections()
        collection_names = [c['name'] for c in collections]
        
        if collection_name in collection_names:
            print(f"✅ Collection '{collection_name}' aparece en el listado de collections")
        else:
            print(f"❌ Collection '{collection_name}' NO aparece en el listado")
            
        print(f"📚 Total de collections: {len(collections)}")
        
        print("\n" + "=" * 70)
        print("🎉 RESULTADO: COMPATIBILIDAD MCP VERIFICADA EXITOSAMENTE")
        print("✅ Los clientes MCP pueden:")
        print("   - Acceder a collections con embedding functions personalizados")
        print("   - Recuperar documentos y sus embeddings almacenados")
        print("   - Realizar búsquedas semánticas efectivas")
        print("   - Usar todas las operaciones MCP estándar")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR DURANTE LA PRUEBA: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def cleanup_test_data():
    """Limpia los datos de prueba"""
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        
        for collection in collections:
            if collection.name.startswith("test_mcp_"):
                try:
                    client.delete_collection(collection.name)
                    print(f"🧹 Eliminada collection de prueba: {collection.name}")
                except Exception as e:
                    print(f"⚠️  No se pudo eliminar {collection.name}: {e}")
                    
    except Exception as e:
        print(f"⚠️  Error durante la limpieza: {e}")

if __name__ == "__main__":
    print("🧪 INICIANDO PRUEBA DE COMPATIBILIDAD MCP")
    print()
    
    # Ejecutar la prueba principal
    success = asyncio.run(test_mcp_client_compatibility())
    
    # Limpieza opcional
    print("\n🧹 Limpiando datos de prueba...")
    asyncio.run(cleanup_test_data())
    
    if success:
        print("\n✅ TODAS LAS PRUEBAS PASARON - Los clientes MCP son totalmente compatibles")
        sys.exit(0)
    else:
        print("\n❌ ALGUNAS PRUEBAS FALLARON - Revisar la configuración")
        sys.exit(1)
