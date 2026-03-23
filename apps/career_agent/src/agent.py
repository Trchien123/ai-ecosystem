import json
import asyncio
from openai import AsyncOpenAI
from core.config import settings

# import from shared_utils
from packages.shared_utils.shared_utils.embeddings import get_embedding
from packages.shared_utils.shared_utils.pinecone_client import index 
from packages.shared_utils.shared_utils.pushover import notify_opportunity_lead, notify_unknown_questions

# initialize the client
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Schemas for tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Always use this tool FIRST to search information about education, skills, experience and projects of Huynh Trung Chien from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Core words to search. For example: 'Computer Science', 'IELTS', 'VNPT Hackathon'"
                    }
                },
                "required": ["search_query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "notify_unknown",
            "description": "use this tool to notify Chien when you CANNOT find the answer from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "questions that you cannot answer"
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "notify_opportunity",
            "description": "use when the employers or any person want to contact or leave their contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "name of that person"},
                    "email": {"type": "string", "description": "email of that person (if any)"},
                    "phone": {"type": "string", "description": "phone number of that person (if any)"},
                    "message": {"type": "string", "description": "the message that they want to say to Chien"}
                },
                "required": ["name", "message"],
                "additionalProperties": False
            }
        }
    }
]

# ---------------------------
# Functions for tools calling
# ---------------------------

# Function for searching in the knowledge base
async def execute_search_knowledge_base(search_query: str) -> str:
    """
    Tool 1 - Search in the Pinecone knowledge base.
    """

    vector = await get_embedding(search_query)
    if not vector:
        return "Error when generating search vector!"
    
    response = await asyncio.to_thread(
        index.query,
        vector=vector,
        top_k=3,
        include_metadata=True,
        namespace="career-portfolio"
    )

    contexts = [match.metadata["text"] for match in response.matches if "text" in match.metadata]
    if not contexts:
        return "Cannot find any information from the knowledge base"
    
    return "\n\n---\n\n".join(contexts)

# Function for notify the unknown question
async def execute_notify_unknown(question: str) -> str:
    """
    Tool 2 - Notify hard questions
    """
    await notify_unknown_questions(question=question)
    return("Successfull notified Chien. Tell the users that Chien is going to reply.")

# Function for notify the contact or opportunity
async def execute_notify_opportunity(name: str, message: str, email: str = "N/A", phone: str = "N/A") -> str:
    await notify_opportunity_lead(name=name, email=email, phone=phone, message=message)
    return("Sucessfully notified Chien. Thanks the users.")

# --------------
# The MAIN AGENT
# --------------
class CareerAgent:
    def __init__(self):
        self.system_prompt = self.system_prompt = """
You are the AI Career Assistant representing Huỳnh Trung Chiến (Chien), a final-year Computer Science student majoring in Artificial Intelligence at Swinburne University of Technology. 
Your mission is to engage with recruiters, potential employers, and collaborators on Chien's behalf with a professional, confident, yet humble tone.

CORE OPERATING RULES:
1. TRUTH ONLY: Never hallucinate or fabricate information about Chien's background, skills, or experiences.
2. DATA RETRIEVAL: Whenever a user asks for information about Chien, you MUST call the `search_knowledge_base` tool to retrieve factual data from the vector database first.
3. CONTEXT-BASED ANSWERS: Formulate your responses based strictly on the information returned by the search tool.
4. LEAD GENERATION: If a user expresses interest in hiring, collaborating on a project, or leaves contact details, proactively ask for their name/email and call the `notify_opportunity` tool.
5. FALLBACK: If the search tool returns "No information found," call the `notify_chien_unknown` tool to alert Chien, then politely inform the user that you have notified Chien and he will get back to them personally.

Always stay in character as Chien's digital representative.
"""
    async def chat(self, messages: str, history: list = None):
        if history is None:
            history = []
            
        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": messages}]
        
        done = False
        while not done:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS
            )
            
            ai_message = response.choices[0].message
            
            # If AI uses tools
            if ai_message.tool_calls:
                messages.append(ai_message) 
                
                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Choose the correct tool calls
                    if tool_name == "search_knowledge_base":
                        result = await execute_search_knowledge_base(**arguments)
                    elif tool_name == "notify_unknown":
                        result = await execute_notify_unknown(**arguments)
                    elif tool_name == "notify_opportunity":
                        result = await execute_notify_opportunity(**arguments)
                    else:
                        result = "Unknown tool"
                        
                    # return the message to AI
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
            else:
                # answer the users
                done = True
                return ai_message.content