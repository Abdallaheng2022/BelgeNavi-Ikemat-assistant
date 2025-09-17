from fastapi import FastAPI
from pydantic import BaseModel
from BelgeNavi import build_app,run

app = FastAPI(title="BelgeNavi")
graph = build_app()

class Ask(BaseModel):
    query: str
    lang: str = "auto"

@app.post("/ask")
def ask(body: Ask):
    print(body)
    result=run(body.query,graph)
    return result