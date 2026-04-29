
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool

from langchain.agents import create_agent

from langchain_ollama import ChatOllama, OllamaEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import os
from dotenv import load_dotenv


# make pinecone db
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("mcdb")


# embedding model (same as our ingestion)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")


# make our instance of DB
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    namespace="primary-corpus"
)


# allow the llm to call our vectorDB
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}
)


# me when claude fixes my errors
retriever_tool = create_retriever_tool(
    retriever,
    name="minecraft_search",
    description="Search Minecraft wiki and is the ground truth for Minecraft mechanics."
)


tools = [retriever_tool]


# ooga booga
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2
)


# system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Minecraft redstone expert. "
     "Use the retrieval tool when needed. "
     "Always base answers on retrieved context when available."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


# make agent
agent = create_agent(model=llm, tools=tools, system_prompt=prompt)

# wrapper to ask question
def ask(question: str):
    
    result = agent.invoke({
        "messages": [("human", question)]
    })

    return result["messages"][-1].content



# unga bunga
if __name__ == "__main__":
    response = ask("How does a redstone comparator work in subtraction mode?")
    print("\n\nOUTPUT:\n\n")
    print(response)