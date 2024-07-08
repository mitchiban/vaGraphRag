#########################################################################################################
#                                               Cypher gen via python
#########################################################################################################

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


#########################################################################################################
#                                               Set up
#########################################################################################################
load_dotenv()
NEO4J_USER = os.getenv('NEO4J_USERNAME_F')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_F')
NEO4J_URI = os.getenv('NEO4J_URI_F')


#########################################################################################################
#                                               Create Query Contstruct Functions
#########################################################################################################
def fn1(src_node_type, src_node_name, tgt_node_type, tgt_node_name, rel_type):
    query = f"""
    MERGE (a:{src_node_type} {{name: $src_node_name}})
    MERGE (b:{tgt_node_type} {{name: $tgt_node_name}})
    MERGE (a)-[r:{rel_type}]->(b)
    RETURN a, b, r
    """
    return query, {'src_node_name': src_node_name, 'tgt_node_name': tgt_node_name}

# Function to execute the Cypher query
def execute_cypher_query(query, parameters):
    # Create a Neo4j driver instance
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        # Execute the query
        with driver.session() as session:
            result = session.run(query, parameters)
            
            # Print the results
            for record in result:
                print(record)


#########################################################################################################
#                                               Define Content Function
#########################################################################################################

def add_nodes():
    src_node_type = "Person"
    src_node_name = "Alice"
    tgt_node_type = "Food"
    tgt_node_name = "Peas"
    rel_type = "DISLIKES"

    query, parameters = fn1(src_node_type, src_node_name, tgt_node_type, tgt_node_name, rel_type)
    print("Cypher Query:\n", query)
    print("Parameters:\n", parameters)

    # Execute the Cypher query
    execute_cypher_query(query, parameters)

#########################################################################################################
#                                               Execute Functions
#########################################################################################################

add_nodes()
