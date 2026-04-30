"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus/wiki_pages --namespace primary-corpus
"""

import argparse
import os

from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from datetime import datetime
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_community.embeddings import OllamaEmbeddings # deprecated swapping to langchain_ollama
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() # load environment variables from .env file
def parse_args() -> argparse.Namespace:
    """Parse ingestion CLI arguments."""
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing PDF/text documents.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="primary-corpus",
        help="Pinecone namespace to upsert into.",
    )
    return parser.parse_args()


def load_documents(input_dir: str) -> list:
    """
    Load and return raw documents from the input directory.

    TODO:
    - Support PDF files (e.g., using pypdf or LangChain's PyPDFLoader).
    - Support plain text files.
    - Return a list of Document objects with content and metadata
      (source filename, page number).
    """
    docs = [] # stores all our Document objects

    # loop over all files in the input directory and load them
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            # load pdf using pypdf or LangChain's PyPDFLoader
            filepath = os.path.join(input_dir, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            for page in pages:
                docs.append(
                    Document(
                        page_content=page.page_content,
                        metadata={# metadata for the document (source and page number)
                            "source": filename,
                            "page": page.metadata.get("page", "1")
                        }
                    )
                )
        elif filename.endswith(".txt"):
            # load .txt files
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read() # read the contents of the file
                if not text:
                    continue # empty document
            docs.append(
                Document(
                    page_content=text,
                    metadata={ # metadata for the document (source and page number)
                        "source": filename,
                        "page": "1"
                    }
                )
            )
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    return docs



def split_by_sections(text: str):
    pattern = r"(==+ .*? ==+)"
    parts = re.split(pattern, text)

    chunks = []
    current_section = "intro"

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("=="):
            current_section = part
        else:
            chunks.append((current_section, part))

    return chunks

def is_valid_text(text: str) -> bool:
    text = text.strip()

    if not text:
        return False
    if len(text) < 30:  # filter tiny junk chunks
        return False

    return True
def chunk_documents(documents: list) -> list:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    final_docs = []

    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    for doc in documents:
        sections = split_by_sections(doc.page_content)

        for i, (section, content) in enumerate(sections):

            if len(content) < 1200:
                chunks = [content]
            else:
                chunks = token_splitter.split_text(content)

            for j, chunk in enumerate(chunks):
                if not is_valid_text(chunk): # dont add junk chunks
                    continue
                chunk = chunk.strip()
                final_docs.append(
                    Document(
                        page_content=f"{section}\n{chunk}",
                        metadata={
                            "text": f"{section}\n{chunk}",  # need it to be called text
                            "source": str(doc.metadata.get("source", "unknown")),
                            "page_number": str(doc.metadata.get("page", "1")),
                            "section": section,
                            "chunk_id": f"{i}_{j}",
                            "timestamp": timestamp
                        }
                    )
                )

    return final_docs


def generate_embeddings(chunks: list) -> list:
    """
    Generate vector embeddings for document chunks in batches.

    TODO:
    - Use Sentence Transformers (e.g., all-MiniLM-L6-v2)
      or Bedrock Titan Embeddings.
    - Process in batches for efficiency (see W5 Monday — batch embedding).
    """
    # embeddings_model = BedrockEmbeddings(
    #     model_id="amazon.titan-embed-text-v2:0",
    #     region_name="us-east-1"
    # )
    # enrich the metadata with source and page number (for better context)

    # replace bedrock with ollama embedding for local testing

    # using nomic-embed-text because I saw that it has a large context window and still has good performance
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    texts = [
    f"[Source: {c.metadata.get('source', '')}, Page: {c.metadata.get('page_number', '')}]\n{c.page_content}"
    for c in chunks
    ]
    vectors = embeddings_model.embed_documents(texts)

    return vectors


def upsert_to_pinecone(embeddings: list, namespace: str, chunks: list) -> None:
    """
    Upsert embedding vectors and metadata into the Pinecone index.

    TODO:
    - Initialize the Pinecone client using env vars.
    - Upsert vectors with rich metadata into the specified namespace.
    """
    # 1. get pinecone key and call our index mcdb
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "mcdb"
    # code to create the index if not exists (from trainer code demo)
    if not pinecone.has_index(index_name):
        pinecone.create_index(
            name = index_name,
            dimension = 768, # nomic-embed-text has 768 dimensions
            metric = "cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pinecone.Index(name=index_name)


    # 2. Prepare metadata + IDs
    metadatas = [c.metadata for c in chunks]

    ids = [
        f"{m.get('source','unknown')}_p{m.get('page_number','na')}_c{m.get('chunk_id')}"
        for m in metadatas
    ]

    # 3. zip it up for upserting
    vectors = list(zip(ids, embeddings, metadatas))

    # 4. Batch upsert (recommended to me by chatGPT)
    batch_size = 100

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]

        index.upsert(
            vectors=batch,
            namespace=namespace
        )

def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()
    args = parse_args()

    documents = load_documents(args.input_dir)
    print("docs loaded...")
    chunks = chunk_documents(documents)
    print("done splitting chunks...")
    embeddings = generate_embeddings(chunks)
    print("finished embedding")
    upsert_to_pinecone(embeddings, args.namespace, chunks)

    print(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")



if __name__ == "__main__":
    main()