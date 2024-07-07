
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph #02
from langchain_openai import AzureOpenAIEmbeddings # 03
from langchain_community.vectorstores import Neo4jVector # 03

from langchain_openai import OpenAIEmbeddings # added for embeddings


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
#                                               Set up embeddings
#########################################################################################################

def set_up_embeddings():
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",  #do Ineed to  set up a model for embeddings?
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        openai_api_version=api_version,
    )
    
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    print(embeddings)

#set_up_embeddings()


def set_up_embeddingsv2():
    """
    UseOpenAI embeddings function because Azure seems to need a new model instantiated.
    Embeddings are created on document, but the function has several errors.
    Raised issue https://github.com/mitchiban/vaGraphRag/issues/3

    """
    embeddings = OpenAIEmbeddings()
    # query_result = embeddings.embed_query(question)
    # document_result = embeddings.embed_query(document)
    # len(query_result)
    

    vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
    )
    print(embeddings)
    return vector_index

#set_up_embeddingsv2()


#########################################################################################################
#                                               create function
#########################################################################################################

def answer_question(vector_index):
    embd = OpenAIEmbeddings()
    question = "Who is Dudley?"
    emb_question = embd.embed_query(question)
    results = vector_index.similarity_search(emb_question, k=1)
    print(results[0].page_content)


#########################################################################################################
#                                               execute function
#########################################################################################################

# Initialize embeddings and vector index
vector_index = set_up_embeddingsv2()

answer_question(vector_index)