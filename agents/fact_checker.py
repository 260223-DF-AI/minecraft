"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

from pydantic import BaseModel
from agents.state import ResearchState
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone
import json
import os
import re
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
llm = ChatOllama(model="qwen3.5:9b", temperature=0, timeout=30, format="json")



class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""
    claim: str
    verdict: str  # "Supported" | "Unsupported" | "Inconclusive"
    evidence: str | None = None


class FactCheckReport(BaseModel):
    """Full verification report across all claims."""
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


def fact_checker_node(state: ResearchState) -> dict:
    """
    Verify the Analyst's response against trusted reference sources.

    TODO:
    - Extract claims from state["analysis"].
    - Query the 'fact-check-sources' Pinecone namespace for each claim.
    - Produce per-claim verdicts.
    - If confidence < threshold, trigger HITL interrupt.
    - Support Time Travel via state checkpointing.
    """

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

    analysis = state["analysis"]

    claims = analysis["claims"]
    citations = analysis["citations"]

    verdicts = []

    for claim, citation in zip(claims, citations):
        docs = vectorstore.similarity_search(
            query=claim,
            k=10,
            filter={
                "source": citation["source"],
        }
        )
        if not docs:
            docs = vectorstore.similarity_search(claim, k=10)

        context = "\n".join(f"[DOC {i}]\n{d.page_content}\n"for i, d in enumerate(docs))

        verdict = verify_claim(llm, claim, context)
        verdicts.append(verdict)

    confidence = compute_confidence(verdicts)

    state["fact_check_report"] = {
        "verdicts": verdicts,
        "overall_confidence": confidence
    }

    if confidence < 0.7:
        state["needs_review"] = True

    return state



if __name__ == "__main__":

    # -----------------------------
    # 1. Mock analyst output
    # -----------------------------
    sample_analysis = {
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

    sample_analysis1 = {
        "answer": "Apples in Minecraft drop from oak and dark oak leaves.",
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

    # -----------------------------
    # 2. Build initial state
    # -----------------------------
    state = ResearchState()
    state["question"] = "How do you get apples in Minecraft?"
    state["analysis"] = sample_analysis
    state["scratchpad"] = []

    # -----------------------------
    # 3. Run fact checker
    # -----------------------------
    result_state = fact_checker_node(state)

    # -----------------------------
    # 4. Print results
    # -----------------------------
    print("\n=== FACT CHECK REPORT ===\n")

    report = result_state.get("fact_check_report", {})

    for v in report.get("verdicts", []):
        print(f"Claim: {v.claim}")
        print(f"Verdict: {v.verdict}")
        print(f"Evidence: {v.evidence}")
        print("-" * 40)

    print("\nOverall Confidence:", report.get("overall_confidence"))
    print("\nNeeds Review:", result_state.get("needs_review", False))