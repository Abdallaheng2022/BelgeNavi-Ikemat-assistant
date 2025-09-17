from fastapi import FastAPI
from pydantic import BaseModel
from trial_matcher_langgraph import build_graph,run_chatbot
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="TrialMatcher")
graph = build_graph()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class Match(BaseModel):
    query: str
    lang: str = "auto"

@app.post("/match")
def match(body: Match):
    print(body)
    result=run_chatbot(app=graph,user_query=body.query)
    return result
