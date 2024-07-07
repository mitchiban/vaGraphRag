
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph #02
from langchain_openai import AzureOpenAIEmbeddings # 03
from langchain_community.vectorstores import Neo4jVector # 03
from langchain.chains import GraphCypherQAChain #04


#########################################################################################################
#                                               Set up
#########################################################################################################

load_dotenv()
# api
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-02-15-preview"

# llm
llm = AzureChatOpenAI(
    azure_deployment="vaGraphRAGviaaos",
    api_version=api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# neo
neo4j_user = os.getenv('NEO4J_USERNAME_F')
neo4j_password = os.getenv('NEO4J_PASSWORD_F')
neo4j_uri = os.getenv('NEO4J_URI_F')

NEO4J_USER = os.getenv('NEO4J_USERNAME_F')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_F')
NEO4J_URI = os.getenv('NEO4J_URI_F')


URI = neo4j_uri
AUTH = (neo4j_user, neo4j_password)

graph = Neo4jGraph(neo4j_uri, neo4j_user, neo4j_password)
print(graph)



#########################################################################################################
#                                               Set up chain = GraphCypherQAChain

#########################################################################################################


chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
response1 = chain.invoke({"query": "What is Mr. Dursley's job?"})
response1

response2 = chain.invoke({"query": "How many pets does Dudley have?"})
response2

#########################################################################################################
#                                               create function
#########################################################################################################




#########################################################################################################
#                                               execute function
#########################################################################################################

#answer_question()