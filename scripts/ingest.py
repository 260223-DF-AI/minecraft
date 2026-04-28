"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
"""

import argparse
import os

from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
import datetime
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
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
            loader = PyPDFLoader(filename)
            pages = loader.load()
            for page in pages:
                docs.append(
                    Document(
                        page_content=page.page_content,
                        metadata={# metadata for the document (source and page number)
                            "source": filename,
                            "page": page.metadata.get("page", None)
                        }
                    )
                )
        elif filename.endswith(".txt"):
            # load .txt files
            with open(filename, "r") as f:
                text = f.read() # read the contents of the file
            docs.append(
                Document(
                    page_content=text,
                    metadata={ # metadata for the document (source and page number)
                        "source": filename,
                        "page": None
                    }
                )
            )
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    return docs


def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    TODO:
    - Use RecursiveCharacterTextSplitter or sentence-level splitting.
    - Attach chunk metadata (chunk_id, source, page_number, timestamp).
    """
    splitter = SentenceTransformersTokenTextSplitter(chunk_size=256) # make splitter 256 tokens per chunk

    chunks = splitter.split_documents(documents)


    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # time stamp for meta data

    return [
        Document(
            page_content=c.page_content,
            metadata={
                **c.metadata,  # keep any existing metadata from splitter
                "source": c.metadata.get("source",None),
                "page_number": c.metadata.get("page", c.metadata.get("page_number",None)),
                "chunk_id": i,
                "timestamp": timestamp
            }
        )
        for i, c in enumerate(chunks)
    ]


def generate_embeddings(chunks: list) -> list:
    """
    Generate vector embeddings for document chunks in batches.

    TODO:
    - Use Sentence Transformers (e.g., all-MiniLM-L6-v2)
      or Bedrock Titan Embeddings.
    - Process in batches for efficiency (see W5 Monday — batch embedding).
    """
    embeddings_model = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )

    texts = [c.page_content for c in chunks]
    vectors = embeddings_model.embed_documents(texts)

    return vectors


def upsert_to_pinecone(embeddings: list, namespace: str) -> None:
    """
    Upsert embedding vectors and metadata into the Pinecone index.

    TODO:
    - Initialize the Pinecone client using env vars.
    - Upsert vectors with rich metadata into the specified namespace.
    """
    vector_store = PineconeVectorStore(index = "mcdb", namespace = namespace, embeddings = embeddings)


def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()
    args = parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents)
    embeddings = generate_embeddings(chunks)
    upsert_to_pinecone(embeddings, args.namespace)

    print(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
