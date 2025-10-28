import os
from neo4j import GraphDatabase
from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# --- Configuration ---
# Neo4j Config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")
NEO4J_DB = "neo4j"

# ChromaDB Config
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "eats_dishes"

# Embedding Model (must be local, small, and fast)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- 1. Data Model (Simulated E-commerce Data) ---

# Structured/Relational Data (for Neo4j KG)
restaurants_kg_data = [
    {"id": "R1", "name": "Thai Basil House", "cuisine": "Thai", "rating": 4.7, "address": "123 Market St"},
    {"id": "R2", "name": "Pizza Planet", "cuisine": "Italian", "rating": 4.2, "address": "456 Oak Ave"},
    {"id": "R3", "name": "Green Garden Grill", "cuisine": "Vegan", "rating": 4.9, "address": "789 Pine Ln"},
]

# Unstructured/Semantic Data (for ChromaDB Vector Store)
dishes_vector_data = [
    {"id": "D1", "restaurant_id": "R1", "name": "Red Curry Delight", "description": "A fragrant, creamy, and spicy coconut milk red curry with bamboo shoots and basil. Vegetarian option available.", "price": 15.00, "rating": 4.8},
    {"id": "D2", "restaurant_id": "R1", "name": "Pad See Ew Noodles", "description": "Wide rice noodles stir-fried with Chinese broccoli, egg, and a rich, sweet soy sauce. Classic comfort food.", "price": 14.50, "rating": 4.5},
    {"id": "D3", "restaurant_id": "R2", "name": "Pepperoni Classic", "description": "Traditional hand-tossed pepperoni pizza with slow-cooked tomato sauce and fresh mozzarella.", "price": 20.00, "rating": 4.1},
    {"id": "D4", "restaurant_id": "R3", "name": "Avocado Black Bean Burger", "description": "A dense, savory patty made from black beans and quinoa, topped with fresh avocado and chipotle mayo.", "price": 16.50, "rating": 4.9},
]

# Relationship/Transactional Data (for Neo4j KG)
relationships_data = [
    {"type": "MEMBERSHIP", "user_id": "U1", "level": "Gold"},
    {"type": "PROMO", "restaurant_id": "R1", "code": "GOLD20", "level_required": "Gold", "details": "20% off all Thai dishes."},
    {"type": "PROMO", "restaurant_id": "R3", "code": "FREEDEL", "level_required": "Silver", "details": "Free delivery on all orders over $15."},
    {"type": "FAVORITE", "user_id": "U1", "restaurant_id": "R1"},
]

# --- 2. Neo4j KG Setup ---

def setup_neo4j(uri, auth, db_name, restaurants, relationships):
    """Initializes and populates the Neo4j Knowledge Graph."""
    print("--- Setting up Neo4j KG ---")
    driver = GraphDatabase.driver(uri, auth=auth)

    def populate_data(tx):
        # 1. Clear existing data
        tx.run("MATCH (n) DETACH DELETE n")
        print("Existing graph cleared.")

        # 2. Create Restaurant Nodes
        for r in restaurants:
            tx.run(f"CREATE (r:Restaurant {{id: '{r['id']}', name: '{r['name']}', cuisine: '{r['cuisine']}', rating: {r['rating']}}})")
        print(f"Created {len(restaurants)} Restaurant nodes.")

        # 3. Create User and Membership Nodes/Relationships
        # We hardcode one user for simplicity
        tx.run("CREATE (u:User {id: 'U1', name: 'Agent User'})")
        tx.run("MATCH (u:User {id: 'U1'}) CREATE (m:Membership {level: 'Gold'}) CREATE (u)-[:HAS_MEMBERSHIP]->(m)")
        print("Created User and Membership nodes.")

        # 4. Create Promo and Favorite Relationships
        for rel in relationships:
            if rel['type'] == 'PROMO':
                # FIX: Split the complex query into two simple ones to avoid the Cypher Syntax Error.
                # Query 1: MATCH Restaurant and CREATE Promo/OFFERS relationship
                tx.run(f"""
                    MATCH (r:Restaurant {{id: '{rel['restaurant_id']}'}}) 
                    MERGE (p:Promo {{code: '{rel['code']}', details: '{rel['details']}'}}) 
                    MERGE (r)-[:OFFERS]->(p)
                """)
                # Query 2: MATCH Membership and link it to the created Promo
                tx.run(f"""
                    MATCH (p:Promo {{code: '{rel['code']}'}}) 
                    MATCH (m:Membership {{level: '{rel['level_required']}'}}) 
                    MERGE (p)-[:REQUIRES_LEVEL]->(m)
                """)
            elif rel['type'] == 'FAVORITE':
                tx.run(f"MATCH (u:User {{id: '{rel['user_id']}'}}), (r:Restaurant {{id: '{rel['restaurant_id']}'}}) "
                       f"MERGE (u)-[:FAVORS]->(r)")
        print(f"Created {len(relationships)} relationships.")

    with driver.session(database=db_name) as session:
        session.execute_write(populate_data)
        
    driver.close()
    print("Neo4j setup complete.")

# --- 3. ChromaDB Vector Store Setup ---

def setup_chromadb(data, path, collection_name, model_name):
    """Initializes and populates the ChromaDB Vector Store."""
    print("--- Setting up ChromaDB Vector Store ---")
    
    # Initialize Chroma Client (Persistent storage for local PoC)
    chroma_client = Client(Settings(persist_directory=path))
    
    # Delete collection if it exists to ensure a clean start
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass # Ignore if it doesn't exist
        
    # Create the collection
    # Note: We use a sentence-transformer model for local embedding generation.
    # In production, this would be a dedicated embeddings service.
    
    # NOTE: Chroma automatically uses the default "all-MiniLM-L6-v2" model
    # if you don't provide an embedding function, but we'll be explicit
    # and use the LangChain integration with a local model wrapper later.
    
    # For this setup script, we'll simply use the default embedding process.
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Convert data into Chroma-ready format
    documents = [d['description'] for d in data]
    metadatas = [
        {
            "dish_id": d['id'],
            "restaurant_id": d['restaurant_id'],
            "name": d['name'],
            "price": d['price'],
            "rating": d['rating'],
        } 
        for d in data
    ]
    ids = [d['id'] for d in data]
    
    # Add documents to the collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"ChromaDB setup complete. Indexed {len(documents)} dish descriptions.")

# --- 4. Main Execution ---

if __name__ == "__main__":
    # Ensure Chroma directory exists
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # 1. Setup the Neo4j Knowledge Graph
    setup_neo4j(NEO4J_URI, NEO4J_AUTH, NEO4J_DB, restaurants_kg_data, relationships_data)
    
    # 2. Setup the ChromaDB Vector Store
    setup_chromadb(dishes_vector_data, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME)
    
    print("\n\n*** PoC Data Setup Complete! ***")
    print("You can now connect LangChain to these two data sources in the next step.")
