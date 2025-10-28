import os
import json
from neo4j import GraphDatabase
from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
# FIX: Replacing the failing import with the most specific, current path for the ReAct agent creator.
from langchain_classic.agents.agent import AgentExecutor # CORRECTED IMPORT PATH to classic, https://python.langchain.com/api_reference/langchain/agents.html
from langchain_classic.agents.react.agent import create_react_agent # CORRECTED IMPORT PATH
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate # Using the core prompt class now
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any



# --- Configuration (Must match setup_data.py) ---
# Neo4j Config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")
NEO4J_DB = "neo4j"

# ChromaDB Config
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "eats_dishes"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Used by Chroma internally

# Ollama Config
OLLAMA_MODEL = "llama3" # Ensure you have run 'ollama pull llama3'

# --- 1. Custom Tool Definitions ---

# --- A. Knowledge Graph Tool (Structured Data) ---

@tool
def knowledge_graph_search(cypher_query: str) -> str:
    """
    Executes a Cypher query against the Neo4j Knowledge Graph. 
    Use this for questions about user membership, promotions, restaurants, 
    and explicit relationships between entities (e.g., 'who favors which restaurant?').
    Returns a JSON string of results.
    """
    # FIX: Strip markdown backticks and excessive whitespace added by the LLM
    cleaned_query = cypher_query.strip().strip('`')
    
    print(f"\n[KG TOOL EXECUTING]: {cleaned_query}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        with driver.session(database=NEO4J_DB) as session:
            # Use the cleaned query string for execution
            result = session.run(cleaned_query) 
            
            # Convert Neo4j Records to a list of dictionaries (JSON serializable)
            data = [record.data() for record in result]
            
            if not data:
                return "No results found for the Cypher query. Check your query syntax or node/relationship names."

            # Return the result as a JSON string for the LLM to process
            return json.dumps(data)
            
    except Exception as e:
        return f"Neo4j Query Error: {e}. Please try simplifying the query."
    finally:
        driver.close()

# --- B. Vector Search Tool (Semantic Data) ---

@tool
def semantic_dish_search(query: str) -> List[Dict[str, Any]]:
    """
    Performs a semantic similarity search on the ChromaDB Vector Store. 
    Use this for finding dishes based on their description, taste, texture, 
    or flavor profile (e.g., 'creamy', 'spicy', 'comfort food').
    Returns a list of matching dish metadata (dish_id, restaurant_id, name, rating, price).
    """
    print(f"\n[VECTOR TOOL EXECUTING]: Semantic search for '{query}'")
    
    # FIX: Using PersistentClient for robust disk loading
    try:
        chroma_client = PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        # Provide a specific error if loading fails, just in case
        return f"ChromaDB Load Error: Failed to initialize PersistentClient or find collection {COLLECTION_NAME}. Ensure 'chroma_data' exists and is populated. Error: {e}"


    # Perform the query
    # Chroma automatically uses the default embedding model for query embedding
    results = collection.query(
        query_texts=[query],
        n_results=3, # Retrieve top 3 results
        include=['metadatas'] # We only need the metadata, not the raw documents
    )
    
    # Process the results into a clean list
    if results and results['metadatas'] and results['metadatas'][0]:
        # Return the list of metadata dictionaries
        return results['metadatas'][0]
    else:
        return []

# --- 2. Agent Setup ---

def run_agent_poc(user_query: str):
    """Initializes and runs the LangChain Agent."""
    
    print("\n--- Initializing Agent ---")
    
    # 1. Initialize the local LLM via Ollama
    # NOTE: The LangChainDeprecationWarning about Ollama can be ignored for now, 
    # as the core functionality is still present.
    llm = Ollama(model=OLLAMA_MODEL)
    
    # 2. Define the Tools
    tools = [semantic_dish_search, knowledge_graph_search]
    
    # 3. Define the Agent's Prompt/Instruction using the ReAct Template
    # This standard template is required for create_react_agent
    system_instruction = (
        "You are the 'Personalized Deal Finder' Agent. Your goal is to combine information "
        "from the Knowledge Graph (KG) and the Vector Search Index (Semantic Search) to answer "
        "complex user queries about food dishes, deals, and user status. "
        "The user is a Gold Member (User ID: U1). "
        "For knowledge_graph_search, the Action Input MUST be a single, valid Cypher query."
        "For semantic_dish_search, the Action Input MUST clearly extract keywords from the query."
        "After gathering information, synthesize a concise final answer."
    )

    react_template = (
        "{system_instruction}\n\n"
        "You have access to the following tools:\n"
        "{tools}\n\n"
        "Use the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer to the original input question\n\n"
        "Begin!\n\n"
        "Question: {input}\n"
        "{agent_scratchpad}"
        "Thought:"
    )

    prompt = PromptTemplate.from_template(react_template).partial(
        system_instruction=system_instruction
    )
    
    # 4. Create the Agent (Now using the modern, stable create_react_agent)
    agent = create_react_agent(llm, tools, prompt)
    
    # 5. Create the Executor (Executor remains robust and stable)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        # Setting a higher max_iterations to allow the complex query to complete multiple tool steps
        max_iterations=8,
        handle_parsing_errors=True
    )
    
    # 6. Run the Agent
    print(f"\n--- Running Query: {user_query} ---\n")
    
    # The invoke method requires the input string
    result = agent_executor.invoke({"input": user_query})
    
    print("\n\n--- Final Result ---")
    print(result['output'])
    
    return result['output']


if __name__ == "__main__":
    # Ensure Ollama is running and Llama 3 is pulled before running this script!
    
    # This is the target complex query that requires two steps:
    # Step A (Semantic Search): Find a creamy Thai dish.
    # Step B (KG Search): Check if the associated restaurant has a promo for Gold members (U1).
    complex_query = "Find me a high-rated, creamy Thai dish that has a current promo for my Gold membership."

    # Run the orchestration
    run_agent_poc(complex_query)
    
    # Example simple query using only the KG
    # run_agent_poc("What is the promo code for Thai Basil House?")
    
    # Example simple query using only Vector Search
    # run_agent_poc("I am craving something that tastes like rich, sweet comfort food noodles.")
