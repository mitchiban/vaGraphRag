"""
Description: RAG learning 
"""
#import 1

"""
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
"""

#import 2

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
#api_version = "2023-07-01-preview"
api_version = "2024-02-15-preview"

llm = AzureChatOpenAI(
    model="gpt-4-turbo",
    #azure_deployment="gpt-4-turbo",
    azure_deployment="vaGraphRAGviaaos",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    openai_api_version=api_version,
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

#########################################################################################################
#                                     Exploring AuraDB: Test Harry Potter
#########################################################################################################
"""
from langchain_core.documents import Document
"""
text = """
Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
that they were perfectly normal, thank you very much. They were the last
people you'd expect to be involved in anything strange or mysterious,
because they just didn't hold with such nonsense.
Mr. Dursley was the director of a firm called Grunnings, which made
drills. He was a big, beefy man with hardly any neck, although he did
have a very large mustache. Mrs. Dursley was thin and blonde and had
nearly twice the usual amount of neck, which came in very useful as she
spent so much of her time craning over garden fences, spying on the
neighbors. The Dursleys had a small son called Dudley and in their
opinion there was no finer boy anywhere.
The Dursleys had everything they wanted, but they also had a secret, and
their greatest fear was that somebody would discover it. They didn't
think they could bear it if anyone found out about the Potters. Mrs.
Potter was Mrs. Dursley's sister, but they hadn't met for several years;
in fact, Mrs. Dursley pretended she didn't have a sister, because her
sister and her good-for-nothing husband were as unDursleyish as it was
possible to be. The Dursleys shuddered to think what the neighbors would
say if the Potters arrived in the street. The Dursleys knew that the
Potters had a small son, too, but they had never even seen him. This boy
was another good reason for keeping the Potters away; they didn't want
Dudley mixing with a child like that.
"""

def convert_to_graph_docs():
    documents = [Document(page_content=text)]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")

    # What is going on here?
    graph.add_graph_documents(
    graph_documents, 
    baseEntityLabel=True, 
    include_source=True
    )

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

#########################################################################################################
#                                     Adding Embeddings
#########################################################################################################
"""
#note ms flavour!
from langchain_openai import AzureOpenAIEmbeddings

# set up params 
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    openai_api_version=api_version,
)

# execute call 
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# a different query? 
query = "Who is Dudley?"

results = vector_index.similarity_search(query, k=1)
print(results[0].page_content)

#########################################################################################################
#                                     Cypher Chain (GraphCypherQAChain)
#########################################################################################################
from langchain.chains import GraphCypherQAChain

chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
response = chain.invoke({"query": "What is Mr. Dursley's job?"})
response


#########################################################################################################
#                                     QA Chain (RetrievalQA)
#########################################################################################################

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vector_index.as_retriever(), verbose = True
)

result = qa_chain({"query": "What is Mr. Dursley's job?"})
result["result"]

#########################################################################################################
#                                     GraphRAG
#########################################################################################################

raw_documents = WikipediaLoader(query="French Revolution", load_max_docs = 10).load()


from langchain_community.document_loaders import HuggingFaceDatasetLoader
dataset_name = "evidence_infer_treatment"
page_content_column = "Text"

#loading only the first 100 rows of the dataset
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column, name = '2.0', )


len(raw_documents) #expect 10

#print raw_docs
raw_documents

# print raw docs with some formatting
for i in range(len(raw_documents)):
    print(f'Document {i}')
    print(raw_documents[i].page_content)


from langchain_core.documents import Document
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

#########################################################################################################
#                                     Storing documents in a GraphDB
#########################################################################################################

llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

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

showGraph()

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

#########################################################################################################
#                                     Graph Only
#########################################################################################################
# Defining each question as a variable
question1 = "How did the economic policies of the Ancien Régime contribute to the financial crisis that precipitated the French Revolution?"
question2 = "In what ways did the social and political structure of the Estates-General contribute to its transformation into the National Assembly?"
question3 = "What role did economic depression and military defeats play in the radicalization of the French Revolution in 1792?"
question4 = "How did the French Revolutionary Wars affect the internal political landscape of France from 1792 to 1799?"
question5 = "Examine the socio-economic reasons behind the calling of the Estates-General in 1789."
question6 = "How did Enlightenment ideas influence the legislative reforms of the National Assembly?"
question7 = "What event directly led to the transformation of the Estates-General into the National Assembly in June 1789?"
question8 = "Which radical measure taken by the National Assembly on July 14, 1789, symbolically marked the beginning of the French Revolution?"
question9 = "Which governing body replaced the National Convention after the fall of Robespierre in 1794?"
question10 = "What significant political change occurred in France on 18 Brumaire in 1799?"
question11 = "Considering the financial difficulties faced by the Ancien Régime, how did the complex and inconsistent tax system contribute to the financial instability and eventual calling of the Estates-General?"
question12 = "What role did the socio-economic pressures such as the increase in the population and the widening gap between the rich and the poor play in setting the stage for the French Revolution?"
question13 = "How did the financial crisis, exacerbated by poor harvests and high food prices, lead to the convening of the Estates-General in 1789?"
question14 = "Discuss the immediate political repercussions of the Storming of the Bastille on the French Revolution."

# Creating a list of all questions
questions = [question1, question2, question3, question4, question5, question6, question7, question8, question9, question10, question11, question12, question13, question14]

# set empty list and loop
graph_results = []
graph_source_documents = []
for q in questions:
    graph_results.append(graph_chain({"query": q})["result"])
    graph_source_documents.append(graph_chain({"query": q})["source_documents"])

#finally pandas :p
import pandas as pd

# Assuming questions, rag_results, graph_results are your lists
df = pd.DataFrame({
    'questions': questions,
    'graph_results': graph_results,
    'graph_source_documents': graph_source_documents
})

df.head(20)

# block
graphrag_text = []
for index, row in df.iterrows():
    graphrag_documents = row['graph_source_documents']
    combined_graphrag_documents = " ".join(doc.page_content for doc in graphrag_documents)
    graphrag_text.append(combined_graphrag_documents)


df.graph_source_documents = graphrag_text
df.head(20)


print('QUESTION')
print(df.iloc[0]['questions'])
#printing whitespace
print('')
print('ANSWER')
print(df.iloc[0]['graph_results'])
print('')
print('CONTEXT')
print(df.iloc[0]['graph_source_documents'])


#########################################################################################################
#                                     Evaluation Metrics
#########################################################################################################

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
"""
groundedness_critique_prompt = PromptTemplate.from_template("""
You will be given a context and answer about that context.
Your task is to provide a 'total rating' scoring how well the ANSWER is entailed by the CONTEXT. 
Give your answer on a scale of 1 to 5, where 1 means that the ANSWER is logically false from the information contained in the CONTEXT, and 5 means that the ANSWER follows logically from the information contained in the CONTEXT.

Provide your response in a list as follows:

Response:::
[Evaluation: (your rationale for the rating, as a text),
Total rating: (your rating, as a number between 1 and 5)]

You MUST provide values for 'Evaluation:' and 'Total rating:' in your response.

Now here are the context, question and answer.

Context: {context}\n
Answer: {answer}\n
Response::: """)


relevance_critique_prompt = PromptTemplate.from_template("""
You will be given a context, and question and answer about that context.
Your task is to provide a 'total rating' to measure how well the answer addresses the main aspects of the question, based on the context. 
Consider whether all and only the important aspects are contained in the answer when evaluating relevance. 
Given the context and question, score the relevance of the answer between one to five stars using the following rating scale: 

Give your response on a scale of 1 to 5, where 1 means that the answer doesn't address the question at all, and 5 means that the answer is perfectly matching the question.

Provide your response in a list as follows:

Response:::
[Evaluation: (your rationale for the rating, as a text),
Total rating: (your rating, as a number between 1 and 5)]

You MUST provide values for 'Evaluation:' and 'Total rating:' in your response.

Now here is the question.

Answer: {answer}\n
Question: {question}\n
Context: {context}\n
Response::: """)

coherence_critique_prompt = PromptTemplate.from_template("""
You will be given a question and answer.
Your task is to measure the coherence of the answer. Coherence is measured by how well all the sentences fit together and sound naturally as a whole. Consider the overall quality of the answer when evaluating coherence. 
Given the question and answer, score the coherence of answer on a scale of 1 to 5, where 1 means that the answer completely lacks coherence, 5 means that the answer has perfect coherency.

Provide your response in a list as follows:

Response:::
[Evaluation: (your rationale for the rating, as a text),
Total rating: (your rating, as a number between 1 and 5)]

You MUST provide values for 'Evaluation:' and 'Total rating:' in your response.

Now here is the question.

Question: {question}\n
Answer: {answer}\n
Response::: """)

#########################################################################################################
#                                     Graph evaluation
#########################################################################################################

"""
groundness = []
relevance = []
coherence = []
for i in range(len(df)):
    question = df.iloc[i]['questions']
    answer = df.iloc[i]['graph_results']
    context = df.iloc[i]['graph_source_documents']
    groundness_chain = LLMChain(llm=llm, prompt=groundedness_critique_prompt)
    groundness.append(groundness_chain.run(context=context, answer = answer))
    relevance_chain = LLMChain(llm=llm, prompt=relevance_critique_prompt)
    relevance.append(relevance_chain.run(question=question, answer = answer, context = context))
    coherence_chain = LLMChain(llm=llm, prompt=coherence_critique_prompt)
    coherence.append(coherence_chain.run(question=question, answer = answer))
"""