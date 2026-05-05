"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

from pydantic import BaseModel
import os
from agents.state import ResearchState

from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

import json
from dotenv import load_dotenv
load_dotenv()
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
def build_prompt(question: str, sub_task: str, retrieved_chunks: list[Document]) -> str:
    context = "\n\n".join(
        f"[Source: {doc.metadata['source']} | Page: {doc.metadata.get('page_number', 1)}]\n{doc.page_content}"
        for doc in retrieved_chunks
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
    - Only use Citations if they are from the context.
    - For confidence scores, a 10.0 would be directly verifiable from the context. Use lower score if this is not the case.
    - For confidence scores, dock points 3-5 points if you used information from your own training instead of the context (do not dock if context has the answer).
    Return JSON with:
    - answer (string)
    - claims (list of strings)
    - citations (list of objects with source, page_number, excerpt)
    - confidence (float between 0.0 and 10.0)

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

from langchain_ollama import ChatOllama

def stream_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    #print("Streaming response from Ollama... with prompt:", prompt)

    llm = ChatOllama(model=model, temperature=0)
    #llm = ChatBedrock(
    #    model_id=os.environ["BEDROCK_MODEL_ID"],
    #    region_name=os.environ["AWS_REGION"],
    #    model_kwargs={"max_tokens": 512, "temperature": 0.0},
    #)

    stream = llm.stream(prompt)

    full_response = ""

    for chunk in stream:
        token = chunk.content or ""
        print(token, end="", flush=True)
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
        #print("got raw output:", raw_output)
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


if __name__ == "__main__":
    print("start..")
    cave_spider_context = """SOURCE: Cave_spider.txt\n A cave spider is a smaller spider variant that behaves similarly, but has less health and inflicts Poison with its attacks. They are only spawned by spawners in various structures, or they can spawn naturally in sulfur caves.[upcoming: Chaos Cubed]


    == Spawning ==

    The cave spider is one of the only two mobs in the game to spawn exclusively from spawners,[until: Chaos Cubed] the other being the breeze.


    === Monster spawners ===
    Cave spiders spawn from monster spawners in mineshafts at a light level of 0. These monster spawners are surrounded by cobwebs in corridors.


    === Trial spawners ===
    Cave spiders have a 25% chance to be selected as the "small melee" mob for trial spawners in trial chambers.


    === Spider jockeys ===

    There is a 1% chance for a cave spider to spawn with a skeleton riding it, forming a cave spider jockey. The skeleton has an 80% chance to be replaced by a stray, bogged, parched, or wither skeleton in the biomes where they spawn. Similar to regular spiders, the skeleton controls how both mobs move. Cave spider jockeys can fit through smaller gaps than spider jockeys.


    == Drops ==


    === On death ===
    Java Edition:Bedrock Edition:
    5XP when killed by a player or tamed wolf.


    == Behavior ==
    Cave spiders inherit their behavior from spiders:

    They are neutral if under daylight or in light levels of 12 or above, otherwise hostile toward players and iron golems. Once they become hostile, light does not affect them.
    They attack by leaping and biting.
    They are unaffected by cobwebs and Poison.
    They are scared of armadillos.
    They have some distinctions from regular spiders:

    They can fit through a space that is one block wide and 12 block tall.
    They can go through the spaces between two different types of (unconnected) fences.
    They cannot spawn with status effects in Hard difficulty.[Java Edition  only]
    They flip 90 upon death.[Bedrock Edition  only]
    Being arthropods, they are weak against weapons with the Bane of Arthropods enchantment.
    Unlike other neutral mobs, cave spiders don't count towards the AngryAt tag.
    The red eyes of cave spiders are emissive with Vibrant Visuals, making them easily distinguishable in the dark caves where they spawn.


    === Poison ===
    On Normal or Hard difficulty, cave spiders inflict Poison upon attacking. On Normal difficulty, the Poison lasts for 7 seconds and causes 6HP damage. On Hard difficulty, it lasts for 15 seconds and causes 12HP  6. Poison damage is not taken when the player's health is at 1HP.
    """

    cave_spider_doc = Document(
    page_content=cave_spider_context,
    metadata={
        "source": "Cave_spider.txt",
        "page_number": "1",
    },
    )

    bad_context = """
    The sky is blue and the grass is green. I like Minecraft.
    """

    bad_doc = Document(
    page_content=bad_context,
    metadata={
        "source": "I_made_it_up.txt",
        "page_number": "1",
    },
    )
    print(bad_doc.metadata['source'])

    # test the agent with correct context
    state = {
        "question": "How much damage do cave spiders do? Can they spawn in the trial chambers?",
        "retrieved_chunks": [cave_spider_doc]
    }
    result = analyst_node(state)
    print("with good context:", result)

    # test the agent with irrelevant context
    state = {
        "question": "How much damage do cave spiders do? Can they spawn in the trial chambers?",
        "retrieved_chunks": [bad_doc]
    }
    result = analyst_node(state)
    print("with bad context:", result)