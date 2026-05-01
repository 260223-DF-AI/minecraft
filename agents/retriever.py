"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

from agents.state import ResearchState

from langchain_ollama import ChatOllama, OllamaEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
from dotenv import load_dotenv


llm = ChatOllama(model="llama3.2")


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
        response = llm.invoke(prompt).content.strip()

        # extract float safely
        score = float(response)
        return max(0.0, min(1.0, score))

    except Exception:
        return 0.0


def compress_with_llm(llm, query: str, text: str) -> str:
    """
    Lightweight replacement for LLMChainExtractor.
    Forces model to extract only relevant info.
    """
    prompt = f"""
You are a context compressor.

Extract ONLY the parts of the text that help answer the question.
Remove all irrelevant information.

QUESTION:
{query}

TEXT:
{text}

RETURN ONLY THE COMPRESSED TEXT:
"""
    return llm.invoke(prompt).content.strip()


def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve relevant document chunks for the current sub-task.

    TODO:
    - Extract the current sub-task from state["plan"].
    - Query the Pinecone index with semantic search and metadata filters.
    - Apply context compression to reduce token noise.
    - Apply re-ranking to prioritize the most relevant results.
    - Return updated state with retrieved_chunks populated.
    - Log actions to the scratchpad.

    Returns:
        Dict with "retrieved_chunks" key containing a list of dicts,
        each with: content, relevance_score, source, page_number.
    """
    # make our instance of DB
    load_dotenv()
    # make pinecone db
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index = pc.Index("mcdb")

    # embedding model (same as our ingestion)
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        namespace="primary-corpus",
    )

    docs = vectorstore.similarity_search(state["question"], k=100)

    if not docs:
        state["retrieved_chunks"] = []
        state["scratchpad"].append("No documents retrieved.")
        return state

    compressed_docs = []
    for doc in docs:
        try:
            compressed_text = compress_with_llm(llm, state["question"], doc.page_content)
            doc.page_content = compressed_text
            compressed_docs.append(doc)
        except Exception:
            # fallback: keep original if compression fails
            compressed_docs.append(doc)

    # Sort by rerank score
    scored_docs = []
    for doc in compressed_docs:
        score = ollama_rerank(state["question"], doc.page_content)
        doc.metadata["rerank_score"] = score
        scored_docs.append(doc)

    reranked_docs = sorted(
        scored_docs,
        key=lambda d: d.metadata["rerank_score"],
        reverse=True
    )[:10]

    # --- Format output ---
    results = []
    for doc in reranked_docs:
        results.append({
            "content": doc.page_content,
            "relevance_score": doc.metadata.get("rerank_score"),
            "source": doc.metadata.get("source", "unknown"),
            "page_number": doc.metadata.get("page", "1"),
        })

    state["retrieved_chunks"] = results
    print(results)

    state["scratchpad"].append("Retrieved relevant documents from Pinecone.")

    return state

if __name__ == "__main__":
    state = ResearchState()
    state["retrieved_chunks"] = []
    state["question"] = "What's the drop rate of an apple in Minecraft?"
    state["scratchpad"] = []
    response = retriever_node(state)
