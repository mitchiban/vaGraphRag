
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph #02
from langchain_openai import AzureOpenAIEmbeddings # 03
from langchain_community.vectorstores import Neo4jVector # 03
from langchain.chains import GraphCypherQAChain #05
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