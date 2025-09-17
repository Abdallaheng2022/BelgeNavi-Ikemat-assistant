from fastapi import FastAPI
#To commununicate to fastapi through corsmiddleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#Allow react frontend to talk with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000"],
    allow_credentials= True,
    allow_methods=["*"],
    allow_headers = ["*"],
)

@app.get("/ask")
def read_root():
    return {"message":"Hello from fastAPI 🚀"}

@app.get("form/preview")
def form_preview():
    return {"message":"Hello from fastAPI 🚀"}
@app.get("/schedule")
def schedule():
    return {"message":"Hello from fastAPI 🚀"}