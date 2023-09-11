#Author: BinaryBills
#Creation Date: September 8, 2023
#Date Modified: September 11, 2023
#Purpose:  

import os
import pickle
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.environ.get('API_KEY')

#Captures
urls = ["https://en.wikipedia.org/wiki/Domenico_Grasso",
        "https://umdearborn.edu/about-um-dearborn", 
        "https://umdearborn.edu/cecs/departments/computer-and-information-science/our-faculty-research", 
        "http://catalog.umd.umich.edu/graduate/coursesaz/cis/",
        "http://catalog.umd.umich.edu/undergraduate/coursesaz/cis/",
        "https://umdearborn.edu/people-um-dearborn/zheng-song"]

#Tokenzing our data
loaders = UnstructuredURLLoader(urls=urls)

#Grabs all data from webpage
data = loaders.load()

#Splits into individual documents
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

#Generating EMbeddings for TOkens
#Embeddings are necessary to store the chunks in a vector database. We use OpenAI to generate embeddings and LangChain's OpenAIEmbedding to provide a wrapper
#around that OpenAI embedding model. 
embeddings = OpenAIEmbeddings()

#SToring embeddings in a vector database
#FAISS is an open source similarity search library developed by Facebook AI and written in C++ with bindings for Python. 
#FAISS is not a vector database that can permanently store embeddings, but rather an in-memory index of embeddings.
#LangChain agents use a llm as a reasoning engine and connect it to two components: tools and memory
#Tools connect the LLM to external data sources, allowing it to have new information and perform actions, like modifying files.
#Memory enables the agent to recall previous interactions with other entities or tools, which can be short-term or long-term.
#Faiss is in memory-index. We save it to the disk and reload it to make it persistent across sessons. 
# To use the index in LangChain, we expose it as a retriever, a generic LangChain interface that makes it easy to combine a vector store with language models
vectorSTore_openAI = FAISS.from_documents(docs, embeddings)
with open("faiss_store_openai.pk1", "wb") as f:
    pickle.dump(vectorSTore_openAI,f)

with open("faiss_store_openai.pk1", "rb") as f:
    VectorSTore = pickle.load(f)

#Information Retrival (RAG APPROACH)
#https://python.langchain.com/docs/integrations/chat/openai
llm = ChatOpenAI(temperature=0.9, model="ft:gpt-3.5-turbo-0613:personal::7xPsahmg")

qa_llm_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorSTore_openAI.as_retriever())

langchain_tool = Tool(
    name='UM-Dearborn Data',
    func=qa_llm_chain.run,
    description='Use tool for queries about CIS courses, CIS professors, and about the University of Michigan-Dearborn'
)

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='Use this tool for general purpose queries not about the UNiversity of Michigan-Dearborn'
)

#Provide context to the OpenAI Model by injecting the relevant chunks of our document into the prompt
#An agent operates through a cyclical process: user assigns a task, agent thinks what should do, decides on an action(tool and input), observes output of tool,
#and repeats these steps until the agent deems the task complete. It takes the problem and slices it into subproblems.

PREFIX = "You are a wise and polite AI assistant, speaking like Master Yoda, and knowledgeable about the University of Michigan-Dearborn ending every response with 'my young padawan."

FORMAT_INSTRUCTIONS = """

Answer the following questions as best you can. You have access to the following tools:

UM-Dearborn Data: Useful when you need to answer questions about CIS courses, CIS professors, and about the University of Michigan-Dearborn
Language Model: Useful for when you need to answer questions not discussed in UM-Dearborn Data.

Use the following format:

  Question: the input question you must answer
  Thought: you should always think about what to do
  Action: the action to take, should be one of [{tool_names}]
  Action Input: the input to the action
  Observation: the result of the action
  ... (this Thought/Action/Action Input/Observation can repeat N times)
  Thought: I now need to write the answer in the speaking pattern of Master Yoda, ending the sentence with my young padawan, and writing everything such that the object comes before the subject and the verb phrase in each sentence. 
  Final Answer: the final answer to the original input question
"""

SUFFIX = '''
Begin!
Instructions: {input}
{agent_scratchpad}
'''

agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=[langchain_tool, llm_tool],
    llm=llm, 
    verbose=True,
    agent_kwargs={
        'prefix': PREFIX, 
        'format_instructions': FORMAT_INSTRUCTIONS,
        'suffix': SUFFIX
    },
    handle_parsing_errors=True,
    )

while True:
    user_input = input("Enter your question (type 'exit' to end): ")
    
    if user_input.lower() == 'exit':
        break

    agent.run(user_input)





