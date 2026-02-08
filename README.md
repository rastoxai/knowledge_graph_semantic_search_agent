# knowledge_graph_semantic_search_agent

Proof-of-concept that combines a Neo4j knowledge graph with a ChromaDB vector store and an Ollama-powered agent for multi-step queries.

## Prereqs

- Neo4j running on `bolt://localhost:7687` with user `neo4j` and password `password`
- Ollama running with `llama3` pulled
- Python venv: `agent_poc_env`

## Quick Start

Seed the KG + vector store (and run built-in checks):

```
./agent_poc_env/bin/python setup_data.py
```

Run the agent:

```
./agent_poc_env/bin/python agent_orchestrator.py
```

## Quick Checks

Neo4j:

```
./agent_poc_env/bin/python test_neo4j.py
```

ChromaDB:

```
./agent_poc_env/bin/python test_chomadb1.py
```

## Notes

- `setup_data.py` uses a persistent Chroma client and writes to `./chroma_data`.
- The agent includes a deterministic semantic -> KG path for complex queries and avoids looping when the vector store is empty.
