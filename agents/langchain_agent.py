
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool

from langchain.agents import create_agent

from langchain_ollama import ChatOllama, OllamaEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import os
from dotenv import load_dotenv

load_dotenv()
# make pinecone db
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("mcdb")


# embedding model (same as our ingestion)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")


# make our instance of DB
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    namespace="primary-corpus",
)







# ooga booga
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2
)

# system prompt
system_prompt = """
You are a Minecraft expert.
Use the retrieval tool when needed.
Always base answers on retrieved context when available.
List the context you are using to base your answer off of.
if asked on creating structures,redstone creations, or crafting, use ASCII to draw the query answer
anything in <> are instructions, do not show in output
Your answer should consist of the following:


<your answer to the query in plain text in conversational format>

<reasoning>

<drawing>

<context>
"""


# make agent
agent = create_agent(model=llm, system_prompt=system_prompt)

# wrapper to ask question
def ask(question: str):
    docs = vectorstore.similarity_search(question, k=7)
    documents = "\n".join(doc.page_content for doc in docs)
    print("found:",documents)
    query = f"""
    The following is the user's question, answer it using the retrieved documents as context:{question}


    =====================
    Context:
    {documents}



    """
    result = agent.invoke({
        "messages": [("human", query)]
    })

    return result["messages"][-1].content



# unga bunga
if __name__ == "__main__":
    response = ask("How do I beat Herdcraft 2026 April fools update?")
    #response = ask("how do I make a nether portal")
    print("\n\nOUTPUT:\n\n")
    print(response)