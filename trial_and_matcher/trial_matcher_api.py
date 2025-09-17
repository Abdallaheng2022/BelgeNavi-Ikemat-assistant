"""
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

@app.get("/match")
def read_root():
    return {"message":"Hello from fastAPI 🚀"}

"""