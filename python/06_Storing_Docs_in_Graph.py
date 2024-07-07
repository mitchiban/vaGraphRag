
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
