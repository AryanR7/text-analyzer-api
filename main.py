from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Allow frontend to connect (adjust later for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your Netlify domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load summarizer model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Request body model
class TextIn(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(data: TextIn):
    summary = summarizer(data.text, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}
