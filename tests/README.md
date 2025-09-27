# Tests

This folder contains tests to verify the functionality of Chroma MCP Server.

## Available Tests

### `test_remote_ollama.py`

Complete functionality test with Ollama on remote server. This test demonstrates that:

- ✅ Can connect to a remote Ollama server
- ✅ Can create collections with custom embedding configuration
- ✅ Documents are processed with the configured embedding function
- ✅ Queries return semantically correct results

**Configuration:**
Configure your Ollama server connection by setting environment variables or editing the test files:

```python
OLLAMA_URL = "http://localhost:11434"      # Change to your server
MODEL_NAME = "nomic-embed-text"            # Change to your model
```

**Environment Variables (recommended):**
```bash
export OLLAMA_URL="http://your-server:11434"
export OLLAMA_MODEL="nomic-embed-text"
```

**Run:**
```bash
python tests/test_remote_ollama.py
```

**Expected output:**
- Successful connectivity with Ollama server
- List of available models
- Creation of collection with custom configuration
- Addition of 5 test documents
- English queries with relevant results

### Upcoming tests

- `test_local_ollama.py` - Test with local Ollama
- `test_openai_embeddings.py` - Test with OpenAI embeddings  
- `test_multiple_collections.py` - Test with multiple collections
- `test_embedding_persistence.py` - Test embedding configuration persistence

## Requirements

To run the tests you need:

1. **Chroma MCP Server** running
2. **Ollama server** accessible (local or remote)
3. **Embedding models** installed in Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama pull mxbai-embed-large
   ```

## Verify connectivity

Before running the tests, verify connectivity:

```bash
# For local server
curl http://localhost:11434/api/tags

# For remote server  
3. **Test connectivity:**
   ```bash
   curl http://localhost:11434/api/tags
   # or for remote server:
   curl http://your-server:11434/api/tags
   ```
```
