"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

from pydantic import BaseModel

from agents.state import ResearchState

from langchain_ollama import OllamaChat

import json
# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single supporting citation."""
    source: str
    page_number: int | None = None
    excerpt: str


class AnalysisResult(BaseModel):
    """Pydantic model enforcing structured analyst output."""
    answer: str
    citations: list[Citation]
    confidence: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------

# code to build a prompt given some stuff from the state
def build_prompt(question: str, sub_task: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']} | Page: {c.get('page_number')}]\n{c['text']}"
        for c in retrieved_chunks
    )

    return f"""
    You are a Minecraft expert.

    Answer the question using the provided context if it is relevant to the question.
    The context may not be relevant to the question. Only include information from the context that is necessary to answer the question.
    You may use your own knowledge to answer the question if the context is not relevant to the question.

    QUESTION:
    {question}

    SUB-TASK:
    {sub_task}

    CONTEXT:
    {context}

    INSTRUCTIONS:
    - Provide a clear, concise answer.
    - Extract atomic factual claims from your answer.
    - Each claim must be preferably verifiable from the context.
    - Include citations for every major statement if possible.

    Return JSON with:
    - answer (string)
    - claims (list of strings)
    - citations (list of objects with source, page_number, excerpt)
    - confidence (float between 0 and 1)

    Return ONLY valid JSON.

    SCHEMA:
    {{
    "answer": "string",
    "claims": ["string", "..."],
    "citations": [
        {{
        "source": "string",
        "page_number": int or null,
        "excerpt": "string"
        }}
    ],
    "confidence": float
    }}
    """

def stream_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    stream = OllamaChat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_response = ""

    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)  # real-time output
        full_response += token

    print()  # newline after stream ends
    return full_response
def analyst_node(state: ResearchState) -> dict:
    """
    Synthesize retrieved chunks into a structured research response.

    TODO:
    - Build a prompt from the question, sub-task, and retrieved_chunks.
    - Invoke AWS Bedrock (e.g., Claude) with structured output enforcement.
    - Parse the response into an AnalysisResult.
    - Support streaming for real-time feedback.
    - Log actions to the scratchpad.

    Returns:
        Dict with "analysis" key containing the AnalysisResult as a dict,
        and "confidence_score" updated from the model's self-assessment.
    """
    question = state["question"]
    sub_task = state.get("current_task", "No Subtask")
    retrieved_chunks = state.get("retrieved_chunks", [])


    # build prompt from question, subtask, and chunks from state
    prompt = build_prompt(question, sub_task, retrieved_chunks)

    # call our model here
    raw_output = stream_ollama(prompt)
    
    analysis_json = ""
    try:
        analysis_json = json.loads(raw_output) # attempt to parse json
        analysis = AnalysisResult(**analysis_json)
    except Exception as e:
        raise ValueError(f"Invalid model output: {e}\nOutput: {raw_output}")

    # Scratchpad logging
    scratchpad = state.get("scratchpad", [])
    scratchpad.append(f"Analysis performed got json:{analysis_json}")

    return {
        "analysis": analysis.model_dump(),
        "confidence_score": analysis.confidence
    }
