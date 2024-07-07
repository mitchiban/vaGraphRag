
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

#02
from langchain_core.documents import Document
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from yfiles_jupyter_graphs import GraphWidget

from langchain.document_loaders import WikipediaLoader #05
from langchain_core.documents import Document #05
import pickle #05
from langchain.text_splitter import TokenTextSplitter #05
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
NEO4J_USER = os.getenv('NEO4J_USERNAME_F')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_F')
NEO4J_URI = os.getenv('NEO4J_URI_F')


graph = Neo4jGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
print(graph)



#########################################################################################################
#                                               Set up
#########################################################################################################

llm = AzureChatOpenAI(
    azure_deployment="vaGraphRAGviaaos",
    api_version=api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)


graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

#########################################################################################################
#                                               get docs
#########################################################################################################

raw_documents = WikipediaLoader(query="French Revolution", load_max_docs = 10).load()
len(raw_documents)

for i in range(len(raw_documents)):
    print(f'Document {i}')
    print(raw_documents[i].page_content)
     
print(raw_documents)

#attributing an empty dict to the metadata field of each document
raw_documents = [Document(page_content=doc.page_content, metadata={}) for doc in raw_documents]
raw_documents[0].metadata

# Define chunking strategy
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents)
     
# directly show the graph resulting from the given Cypher query
default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"

def showGraph(cypher: str = default_cypher):
    # create a neo4j session to run queries
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))
    session = driver.session()
    widget = GraphWidget(graph = session.run(cypher).graph())
    widget.node_label_mapping = 'id'
    #display(widget)
    return widget

#showGraph()


vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


from langchain.chains import RetrievalQA

qa_graph_chain = RetrievalQA.from_chain_type(
    llm, retriever=vector_index.as_retriever(), verbose = True
)

result = qa_graph_chain({"query": "Considering the financial crisis and resistance to reform by the Ancien Régime, how did the convocation of the Estates General in May 1789 specifically address these multifaceted issues?"})
result["result"]
     