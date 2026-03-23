import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import router
from apps.career_agent.src.router import router as career_router

app = FastAPI(
    title="Chien's AI Ecosystem",
    description="Huynh Trung Chien AI Ecosystem",
    version="1.0.0"
)

# add CORs middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://huynhtrungchien.dev",
        "https://www.huynhtrungchien.dev"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the path
app.include_router(career_router, prefix="/career-agent", tags=["Career Agent"])

@app.get("/")
async def health_check():
    return {"status": "online", "owner": "Chien", "agents": ["/career-agent"]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)