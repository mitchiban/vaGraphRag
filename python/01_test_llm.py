

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI



#########################################################################################################
#                                               Set up
#########################################################################################################

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-02-15-preview"



llm = AzureChatOpenAI(
    azure_deployment="vaGraphRAGviaaos",
    api_version=api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    openai_api_version=api_version,
)



#retrieving environmental variables from the .env file
load_dotenv()

#alternative
os.environ["NEO4J_URI"] = "NEO4J_URI"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "NEO4J_PASSWORD"

#graph = Neo4jGraph()

#########################################################################################################
#                                               Test LLM?
#########################################################################################################

def test_llm():
    message = HumanMessage(
    content="Translate this sentence from English to French. I love programming."
    )
    response = llm.invoke([message])
    print(message)
    print(response.content)
    print(response)
    
test_llm()
