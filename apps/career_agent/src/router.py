from fastapi import APIRouter
from pydantic import BaseModel
from apps.career_agent.src.agent import CareerAgent

router = APIRouter()
agent = CareerAgent()

class ChatRequest(BaseModel):
    message: str
    history: list = []

@router.post("/chat")
async def chat_with_career_agent(request: ChatRequest):
    response = await agent.chat(request.message, history=request.history)
    return {"response": response}