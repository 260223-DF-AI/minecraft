from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from dotenv import load_dotenv

load_dotenv()

# import your graph builder
from agents.supervisor import build_supervisor_graph

app = FastAPI()

# build graph once at startup
graph = build_supervisor_graph()


# -------------------------
# Request schema
# -------------------------
class ResearchRequest(BaseModel):
    question: str


# -------------------------
# Endpoint
# -------------------------
@app.post("/research")
def research(req: ResearchRequest):

    # thread id is new for each request for now
    thread_id = str(uuid.uuid4())

    result = graph.invoke(
        {"question": req.question},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return result