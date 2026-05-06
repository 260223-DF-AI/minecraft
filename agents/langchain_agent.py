
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool

from langchain.agents import create_agent

from langchain_ollama import ChatOllama, OllamaEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
#from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper

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




total_faith = 0
total_relevancy = 0
total_precision = 0


# ooga booga
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.1,
    format="json"
)
reranker = ChatOllama(
    model="llama3.2",
    temperature=0
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



def ollama_rerank(query: str, doc_text: str) -> float:
    """
    Uses LLM as a reranker.
    Returns score from 0–1.
    """

    prompt = f"""
You are a relevance scoring system.

Score how relevant the DOCUMENT is to the QUESTION.

Return ONLY a number between 0 and 1.

QUESTION:
{query}

DOCUMENT:
{doc_text}

SCORE:
"""

    try:
        response = reranker.invoke(prompt).content.strip()

        # extract float safely
        score = float(response)
        return max(0.0, min(1.0, score))

    except Exception:
        return 0.0

# ask but with pinecone's reranking
def ask2(question:str):
    docs = vectorstore.similarity_search_with_score(question, k=100)

    reranked_docs = sorted(
        docs,
        key=lambda d: d[1],
        reverse=True
    )[:20]

    for doc_tuple in reranked_docs:
        #print(doc)
        doc = doc_tuple[0]
        print(doc.metadata["source"])

    documents = "\n".join(doc[0].page_content for doc in reranked_docs)
    #print("found:",documents)
    query = f"""
    Instructions:
    - Use the retrieved context ONLY if it is relevant to the question.
    - If the context is not relevant, ignore it completely and answer using your own knowledge.
    - Do NOT force information from the context into your answer.
    - If you use the context, base your answer strictly on it and do not invent additional details.
    - If multiple documents are provided, use only the ones that are relevant.
    - If none of the documents are relevant, clearly ignore them and answer normally.

    Retrieved Context:
    {documents}

    Question:
    {question}

    """
    result = agent.invoke({
        "messages": [("human", query)]
    })

    return result["messages"][-1].content
import json
def load_golden_dataset(filepath: str) -> list[dict]:
    """
    Load the golden dataset from a JSON file.

    Expected format: see data/golden_dataset.json for the schema.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
ragas_llm = LangchainLLMWrapper(llm)
def run_ragas(question: str, answer: str, contexts: list[str], ground_truth: str):
    if not ground_truth:
        print("No ground truth provided, skipping evaluation.")
        return None
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    })
    result = evaluate(
        dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision
                ],
                llm = ragas_llm,
    )
    return result

# wrapper to ask question
def ask(question: str, gt:str):
    docs = vectorstore.similarity_search(question, k=100)
    # for doc in docs:
    #     #print(doc)
    #     print(doc.metadata["source"])

    # Sort by rerank score
    scored_docs = []
    for doc in docs:
        score = ollama_rerank(question, doc.page_content)
        doc.metadata["rerank_score"] = score
        scored_docs.append(doc)

    reranked_docs_temp = sorted(
        scored_docs,
        key=lambda d: d.metadata["rerank_score"],
        reverse=True
    )[:5]
    reranked_docs = []
    for doc in reranked_docs_temp:
        if doc.metadata["rerank_score"] > 0.65:
            reranked_docs.append(doc)
    #for doc in reranked_docs:
        #print(doc)
        #print(doc.metadata["source"])

    documents = "\n".join(doc.page_content for doc in reranked_docs)
    #print("found:",documents)
    query = f"""
    Instructions:
    - Use the retrieved context ONLY if it is relevant to the question.
    - If the context is not relevant, ignore it completely and answer using your own knowledge.
    - Do NOT force information from the context into your answer.
    - If you use the context, base your answer strictly on it and do not invent additional details.
    - If multiple documents are provided, use only the ones that are relevant.
    - If none of the documents are relevant, clearly ignore them and answer normally.

    Retrieved Context:
    {documents}

    Question:
    {question}

    Answer:
    """
    result = agent.invoke({
        "messages": [("human", query)]
    })

    contexts = [doc.page_content.strip() for doc in reranked_docs if doc.page_content and doc.page_content.strip()]
    ragas_result = run_ragas(
        question=question,
        answer=result["messages"][-1].content,
        contexts=contexts,
        ground_truth=gt
    )
    print("RAGAS:", ragas_result)
    print("\n\n")
    return result["messages"][-1].content



# unga bunga
if __name__ == "__main__":
    #response = ask("How do I beat 26w14a the april fools update in 2026?")
    #response = ask("What are the drop tables for chests in the ancient cities")
    #response = ask("I want to speedrun a bastion, it is the housing pattern")
    #response = ask("What's the drop percentage of an apple?")
    #response = ask("What's a moobloom in Minecraft?")
    #response = ask("How do I get the Xbox Cape?")
    #response = ask("What's eternal fire and how do we get it?")
    #response = ask2("How do I beat 26w14a the april fools update in 2026?")

    # truth = """
    # 10. IGNITION OPTIMIZATION

    # Fastest ignition priorities:

    # 1. Flint and Steel (best consistent)
    # 2. Fire Charge (inventory-based instant use)
    # 3. Lava spread (situational)
    # 4. Fire arrow (rare)
    # 5. Ghast fireball (Nether setup only)
    # """

    #response = ask("What's the best way to ignite a nether portal in a minecraft speedrun? Let me know the decision for early-game and mid-game.", gt=truth)
    
    golden = load_golden_dataset("./data/golden_dataset.json")
    for g in golden:
        print("question:", g["question"])
        response = ask(g["question"], gt=g["ground_truth_answer"])
        print()
        #print("answer:", g["ground_truth_answer"])
        #print(response)
    #print("\n\nOUTPUT:\n\n")
    #print(response)