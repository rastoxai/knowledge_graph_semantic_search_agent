# knowledge_graph_semantic_search_agent

Proof-of-concept that combines a Neo4j knowledge graph with a ChromaDB vector store and an Ollama-powered agent for multi-step queries. The goal is to uplift Uber's search toward an agentic, neuro-symbolic architecture where structured rules (KG) and semantic similarity (vector) are fused into one answer.

## Project Intent

- Domain: Uber search uplift using a hybrid KG + vector approach.
- Agent persona (system instruction): "Personalized Deal Finder."
- Hard context: User U1 is always a Gold Member.
- Tooling constraints: valid Cypher for KG queries, keyword-style inputs for vector search.

## Agent Logic

The agent uses a ReAct-style loop with a scratchpad to avoid repeated or circular tool calls:

1) Thought: the LLM decides the next move.
2) Action + Action Input: it selects a tool and provides parameters.
3) Observation: it receives results from Neo4j or ChromaDB.
4) Scratchpad: previous steps are retained via `{agent_scratchpad}` to keep state and prevent loops.

## Goal

Synthesize a single answer from two disconnected sources:
- Unstructured dish descriptions (vector search).
- Structured membership rules and promos (knowledge graph).

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
