import uvicorn
import os
from fastapi import FastAPI
# Import router
from apps.career_agent.src.router import router as career_router

app = FastAPI(
    title="Chien's AI Ecosystem",
    description="Huynh Trung Chien AI Ecosystem",
    version="1.0.0"
)

# Register the path
app.include_router(career_router, prefix="/career-agent", tags=["Career Agent"])

@app.get("/")
async def health_check():
    return {"status": "online", "owner": "Chien", "agents": ["/career-agent"]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)