"""REST API for the support chatbot. Run with: uvicorn app:app --reload"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from chatbot import build_bot

app = FastAPI(title="NetWave Support Chatbot", version="1.0")
bot = build_bot()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Message] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    escalated: bool
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "mode": "llm" if bot.llm else "offline"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = [m.model_dump() for m in request.history]
    reply = bot.answer(request.message, history)
    return ChatResponse(**reply.__dict__)
