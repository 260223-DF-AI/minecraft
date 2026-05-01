"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

import json
from agents.state import ResearchState

llm = ChatOllama(
    model="llama3.2",   # or mistral, phi3, etc.
    temperature=0
)


def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    question = state["question"]

    system_prompt = """You are a planning agent.

    Break the user's question into a concise sequence of actionable steps.

    Rules:
    - Return ONLY valid JSON
    - Format: {"steps": ["step 1", "step 2", ...]}
    - Steps should map to: search, analyze, fact-check, synthesize
    - Keep it short (3–6 steps max)
    """

    user_prompt = f"Question: {question}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    content = response.content.strip()

    # ---- Safe parsing ----
    try:
        parsed = json.loads(content)
        steps = parsed.get("steps", [])
    except Exception:
        # fallback if model fails JSON
        steps = [
            f"Search for information about: {question}",
            "Analyze the findings",
            "Fact-check key claims",
            "Generate final answer"
        ]

    return {
        "plan": steps,
        "current_step": 0,
        "scratchpad": state.get("scratchpad", []) + [
            f"Generated plan: {steps}"
        ]
    }


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if step >= len(plan):
        return "critique"

    task = plan[step].lower()

    if "search" in task:
        return "retriever"
    elif "analyze" in task:
        return "analyst"
    elif "fact" in task:
        return "fact_checker"
    else:
        return "analyst"


def critique_node(state: ResearchState) -> dict:
    """
    Evaluate the aggregated response and decide: accept, retry, or escalate.

    TODO:
    - Check confidence_score against the HITL threshold.
    - If below threshold and iterations < max, loop back for refinement.
    - If below threshold and iterations >= max, trigger HITL interrupt.
    - If above threshold, accept and route to END.
    - Increment iteration_count.
    """
    confidence = state.get("confidence_score", 0.5)
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    threshold = state.get("hitl_threshold", 0.8)

    iteration += 1

    # Accept
    if confidence >= threshold:
        return {
            "iteration_count": iteration,
            "scratchpad": state["scratchpad"] + ["Accepted response."]
        }

    # Retry loop
    if iteration < max_iter:
        return {
            "iteration_count": iteration,
            "current_step": 0,  # restart plan
            "scratchpad": state["scratchpad"] + ["Retrying with refinement."]
        }

    # Escalate (HITL)
    return {
        "iteration_count": iteration,
        "scratchpad": state["scratchpad"] + ["Escalating to human."]
    }


def build_supervisor_graph():
    """
    Construct and compile the Supervisor StateGraph.

    TODO:
    - Instantiate StateGraph with ResearchState.
    - Add nodes: planner, retriever, analyst, fact_checker, critique.
    - Add edges and conditional edges (router).
    - Set entry point to planner.
    - Compile and return the graph.

    Returns:
        A compiled LangGraph that can be invoked with an initial state.
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", lambda s: {"current_step": s["current_step"] + 1})
    graph.add_node("analyst", lambda s: {"current_step": s["current_step"] + 1})
    graph.add_node("fact_checker", lambda s: {"current_step": s["current_step"] + 1})
    graph.add_node("critique", critique_node)

    # Entry point
    graph.set_entry_point("planner")

    # Flow
    graph.add_conditional_edges("planner", router)

    graph.add_conditional_edges("retriever", router)
    graph.add_conditional_edges("analyst", router)
    graph.add_conditional_edges("fact_checker", router)

    # Critique branching
    def critique_router(state: ResearchState):
        confidence = state.get("confidence_score", 0.5)
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 3)
        threshold = state.get("hitl_threshold", 0.8)

        if confidence >= threshold:
            return END
        elif iteration < max_iter:
            return "planner"
        else:
            return END  # or "hitl_node" if you add one

    graph.add_conditional_edges("critique", critique_router)

    return graph.compile()
