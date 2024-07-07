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

#API
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-02-15-preview"

# import variables
load_dotenv()  # Load variables from .env file
neo4j_user = os.getenv('NEO4J_USERNAME_F')
neo4j_password = os.getenv('NEO4J_PASSWORD_F')
neo4j_uri = os.getenv('NEO4J_URI_F')

URI = neo4j_uri
AUTH = (neo4j_user, neo4j_password)

graph = Neo4jGraph(neo4j_uri, neo4j_user, neo4j_password)


#########################################################################################################
#                                     Set up LLM
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



########################################################################################################
#                                               Set up 

#########################################################################################################
raw_documents = WikipediaLoader(query="French Revolution", load_max_docs = 10).load()

"""
from langchain_community.document_loaders import HuggingFaceDatasetLoader
dataset_name = "evidence_infer_treatment"
page_content_column = "Text"

#loading only the first 100 rows of the dataset
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column, name = '2.0', )
"""

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
     
import pickle

# Save the documents variable
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)