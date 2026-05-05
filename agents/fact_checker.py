"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
"""

from pydantic import BaseModel
from agents.state import ResearchState
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

import json
import os
import re

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

llm = ChatOllama(model="qwen2.5:3b", temperature=0,format="json")


# -----------------------------
# Models
# -----------------------------

class ClaimVerdict(BaseModel):
    claim: str
    verdict: str  # Supported | Unsupported | Inconclusive
    evidence: str | None = None


class FactCheckReport(BaseModel):
    verdicts: list[ClaimVerdict]
    overall_confidence: float

def extract_json(text: str):
    # remove markdown fences
    text = re.sub(r"```json|```", "", text).strip()

    # find first '{'
    start = text.find("{")
    if start == -1:
        return None

    # walk forward and balance braces
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1

        if brace_count == 0:
            return text[start:i+1]

    return None


def verify_claim(llm, claim: str, context: str) -> ClaimVerdict:
    prompt = f"""
You are a STRICT fact-checker. Given a claim and supporting evidence, decide one of: Supported, Unsupported, Inconclusive.
Supported = the evidence directly states or strongly implies the claim.
Unsupported = the evidence contradicts the claim.
Inconclusive = the evidence is silent on the claim.

Quote a short snippet from the evidence as your justification.

Output schema: return ONLY a JSON with 'verdict' (one of the three labels above, exactly as spelled) and 'evidence' (a short string snippet from the input):
{{
  "verdict": "Supported" | "Unsupported" | "Inconclusive",
  "evidence": string or null
}}

CLAIM:
{claim}

CONTEXT:
{context}
"""
    raw = llm.invoke(prompt)
    print("\n=== RAW LLM OUTPUT ===\n", repr(raw.content))

    try:
        data = json.loads(raw.content)
    except Exception:
        print("BAD RAW OUTPUT:", raw.content)
        return ClaimVerdict(
            claim=claim,
            verdict="Inconclusive",
            evidence="BAD_JSON"
        )
    #print(context)
    allowed_verdicts = {"Supported", "Unsupported", "Inconclusive"}

    if data.get("verdict") not in allowed_verdicts:
        return ClaimVerdict(
            claim=claim,
            verdict="Inconclusive",
            evidence="INVALID_VERDICT_SCHEMA"
        )

    return ClaimVerdict(
        claim=claim,
        verdict=data.get("verdict", "Inconclusive"),
        evidence=data.get("evidence")
    )




def compute_confidence(verdicts: list, weights: list[float] | None = None) -> float:
    score_map = {
        "Supported": 1.0,
        "Inconclusive": 0.5,
        "Unsupported": 0.0
    }

    if not verdicts:
        return 0.0

    if weights is None:
        weights = [1.0] * len(verdicts)

    total = 0.0
    weight_sum = 0.0

    for v, w in zip(verdicts, weights):

        if v is None:
            continue
        if not hasattr(v, "verdict"):
            continue
        if v.verdict not in score_map:
            continue

        total += score_map[v.verdict] * w
        weight_sum += w

    return total / weight_sum if weight_sum else 0.0


# -----------------------------
# Robust JSON extraction
# -----------------------------

def extract_json(text: str):
    """
    Extracts first valid JSON object from messy LLM output.
    """
    if not text:
        return None

    # remove markdown fences
    text = re.sub(r"```json|```", "", text).strip()

    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1

        if brace_count == 0:
            return text[start:i + 1]

    return None


# -----------------------------
# Fact checking prompt
# -----------------------------

def verify_claim(llm, claim: str, context: str) -> ClaimVerdict:
    prompt = f"""
    You are a strict fact-checking system.

    Your job is to evaluate whether a claim is supported by the context.

    Return ONLY valid JSON.
    Do not include markdown or explanations.

    ---

    OUTPUT FORMAT:

    {{
    "verdict": "Supported",
    "evidence": ""
    }}

    ---

    RULES:

    - verdict must be one of:
    "Supported", "Unsupported", "Inconclusive"

    - evidence must ALWAYS be a string
    - if no evidence exists, use ""

    - no extra keys allowed

    ---

    DEFINITIONS:

    - Supported = context directly supports claim
    - Unsupported = context contradicts claim
    - Inconclusive = not enough information

    ---

    CLAIM:
    {claim}

    CONTEXT:
    {context}
    """

    raw = llm.invoke(prompt)
    content = raw.content

    print("\n=== RAW LLM OUTPUT ===\n", repr(content))

    json_str = extract_json(content)

    if not json_str:
        print("BAD RAW OUTPUT:", content)
        return ClaimVerdict(
            claim=claim,
            verdict="Inconclusive",
            evidence="Empty JSON, Model can't find supporting evidence"
        )

    try:
        data = json.loads(json_str)
    except Exception as e:
        print("JSON PARSE ERROR:", e)
        return ClaimVerdict(
            claim=claim,
            verdict="Inconclusive",
            evidence="BAD_JSON"
        )

    verdict = data.get("verdict", "Inconclusive")
    evidence = data.get("evidence")
    #confidence = data.get("confidence", 0.01)

    allowed = {"Supported", "Unsupported", "Inconclusive"}
    if verdict not in allowed:
        verdict = "Inconclusive"

    return ClaimVerdict(
        claim=claim,
        verdict=verdict,
        evidence=evidence
    )


# -----------------------------
# Confidence scoring
# -----------------------------

def compute_confidence(verdicts: list[ClaimVerdict]) -> float:
    score_map = {
        "Supported": 1.0,
        "Inconclusive": 0.0,
        "Unsupported": .75
    }

    if not verdicts:
        return 0.0

    total = 0.0
    count = 0

    for v in verdicts:
        if not v or not hasattr(v, "verdict"):
            continue
        if v.verdict not in score_map:
            continue

        total += score_map[v.verdict]
        count += 1

    return total / count if count else 0.0



def confidence_calc2(verdicts: list[ClaimVerdict]) -> float:
    total = 0.0
    for v in verdicts:
        total += v.confidence
    return total/len(verdicts)

# -----------------------------
# Main node
# -----------------------------


def fact_checker_node(state: ResearchState) -> dict:
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("mcdb")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    namespace = "factchecker-corpus"

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        namespace=namespace,
    )

    analysis = state["analysis"]
    claims = analysis.get("claims", [])
    citations = analysis.get("citations", [])

    verdicts = []

    for claim, citation in zip(claims, citations):
        docs = vectorstore.similarity_search(
            query=claim,
            k=10,
            filter={"source": citation["source"]}
        )

        if not docs:
            docs = vectorstore.similarity_search(claim, k=10)

        context = "\n".join(
            f"[DOC {i}]\n{d.page_content}\n"
            for i, d in enumerate(docs)
        )

        verdict = verify_claim(llm, claim, context)
        verdicts.append(verdict)

    confidence = compute_confidence(verdicts)
    #confidence = confidence_calc2(verdicts)

    state["fact_check_report"] = {
        "verdicts": verdicts,
        "overall_confidence": confidence
    }

    if confidence < 0.7:
        state["needs_review"] = True

    return state


# -----------------------------
# Test harness
# -----------------------------

if __name__ == "__main__":
    sample_unsupported = {
        "answer": "Apples in Minecraft drop from oak and dark oak leaves.",
        "claims": [
            "Apples can be obtained by breaking any tree leaf."
        ],
        "citations": [
            {
                "source": "Minecraft Wiki",
                "page_number": 1,
                "excerpt": "Only oak and dark oak leaves can drop apples."
            }
        ],
        "confidence": 0.0
    }
    sample_analysis = {
        "answer": "Apples in Minecraft drop from oak leaves.",
        "claims": [
            "Apples drop from oak leaves in Minecraft."
        ],
        "citations": [
            {
                "source": "Minecraft Wiki",
                "page_number": 1,
                "excerpt": "Apples are dropped by oak leaves when decayed or broken."
            }
        ],
        "confidence": 0.0
    }
    sample_analysis1 = {
    "answer": "Apples in Minecraft drop from oak and dark oak leaves.",
    "claims": [
        "Apples drop from oak leaves in Minecraft.",
        "Apples drop from dark oak leaves in Minecraft.",
        "Apples can be obtained by breaking any tree leaf."
    ],
    "citations": [
        {
            "source": "Minecraft Wiki",
            "page_number": 1,
            "excerpt": "Apples are dropped by oak leaves when decayed or broken."
        },
        {
            "source": "Minecraft Wiki",
            "page_number": 1,
            "excerpt": "Dark oak leaves also have a chance to drop apples."
        },
        {
            "source": "Minecraft Wiki",
            "page_number": 1,
            "excerpt": "Only oak and dark oak leaves can drop apples."
        }
    ],
    "confidence": 0.0
}

    state = ResearchState()
    state["question"] = "How do you get apples in Minecraft?"
    state["analysis"] = sample_unsupported
    state["scratchpad"] = []

    result_state = fact_checker_node(state)

    print("\n=== FACT CHECK REPORT ===\n")

    report = result_state.get("fact_check_report", {})

    for v in report.get("verdicts", []):
        print(f"Claim: {v.claim}")
        print(f"Verdict: {v.verdict}")
        print(f"Evidence: {v.evidence}")
        print("-" * 40)

    print("\nOverall Confidence:", report.get("overall_confidence"))
    print("Needs Review:", result_state.get("needs_review", False))
