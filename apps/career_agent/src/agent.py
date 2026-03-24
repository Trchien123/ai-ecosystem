import json
import asyncio
from openai import AsyncOpenAI
from core.config import settings
from pymongo import MongoClient
import re

# import from shared_utils
from packages.shared_utils.shared_utils.embeddings import get_embedding
from packages.shared_utils.shared_utils.pinecone_client import index 
from packages.shared_utils.shared_utils.pushover import notify_opportunity_lead, notify_unknown_questions

# initialize the client
openaiClient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
mongoClient = MongoClient(settings.MONGO_DB_URI)

# initialize db
db = mongoClient.get_database(settings.MONGO_DB_NAME)
posts_collection = db["posts"]

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
    },
    {
    "type": "function",
    "function": {
        "name": "search_mongodb",
        "description": "Use this tool to search for blog posts, technical articles, or opinions written by Chien in his MongoDB database. Use this when the user asks about his thoughts, specific technical tutorials, or blog history.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Keywords to search. E.g., 'Software Architecture', 'AI', 'Machine Learning', 'Computer Vision'. IF the user asks generally about blogs without a specific topic (e.g. 'Did he write any blogs?'), pass exactly the word 'all'."
                }
            },
            "required": ["search_query"],
            "additionalProperties": False
        }
    }
}]

# ---------------------------
# Functions for tools calling
# ---------------------------

# Function for searching in mongodb
async def execute_search_mongodb(search_query: str) -> str:
    """
    Tool 4 - Search in MongoDB for knowledge about blogs
    """
    search_query = search_query.strip()

    try:
        def sync_search():
            # recognize general questions 
            generic_terms = ["all", "blog", "blogs", "post", "posts", ""]
            
            if search_query.lower() in generic_terms:
                # get newest posts
                return list(posts_collection.find(
                    {}, 
                    {"title": 1, "excerpt": 1, "_id": 1}
                ).sort("createdAt", -1).limit(3))
            
            # If keywords
            safe_query = re.escape(search_query)
            return list(posts_collection.find(
                {"$or": [
                    {"title": {"$regex": safe_query, "$options": "i"}},
                    {"content": {"$regex": safe_query, "$options": "i"}}
                ]},
                {"title": 1, "excerpt": 1, "_id": 1}
            ).sort("createdAt", -1).limit(3))

        results = await asyncio.to_thread(sync_search)
        
        if not results:
            return f"Do not see any blogs about this: {search_query}!"
        
        # format the results
        formatted_results = []
        for post in results:

            post_id = str(post.get('_id', ''))
            
            item = (
                f"### Title: {post.get('title')}\n"
                f"**Link:** https://huynhtrungchien.dev/blog/{post_id}\n"
                f"**Excerpt:** {post.get('excerpt', 'No excerpt available')}\n"
            )
            formatted_results.append(item)

        return "These are the blogs that I have found:\n\n" + "\n---\n".join(formatted_results)

    except Exception as e:
        return f"Error when searching for blogs: {str(e)}"

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

STRICT FACTUALITY (NO HALLUCINATION):
1. TRUTH ONLY: 
    - You must ONLY use information retrieved from your tools.
    - NEVER invent or guess Chien's skills, experiences, or personal details.
2. TOOL ROUTING & EXECUTION:
    - For administrative data (Education, GPA, Skills, Achievements, Projects, Experience): Execute `search_knowledge_base`.
    - For broad writing queries (e.g., "What blogs did he write?"): Execute `search_mongodb` (pass "all" as the query).
    - For SKILLS, TECHNOLOGIES, or SPECIFIC PROJECTS (e.g., "Does Chien know React?", "Tell me about his YOLOv5 project"): You MUST execute BOTH `search_knowledge_base` (for formal experience) AND `search_mongodb` (to see if he wrote any technical articles demonstrating that skill).
3. CROSS-VERIFICATION: 
    - If a user's question is ambiguous, and your primary tool returns no results, you MUST execute the secondary tool before concluding that no information exists.
4. CITATION: 
    - Whenever you reference a blog post from MongoDB, you MUST include the exact Markdown link provided in the tool's response so the user can click to read more.
5. LEAD GENERATION: 
    - If a user expresses interest in hiring, interviewing, or collaborating, proactively ask for their Name, Email, and Message, then execute the `notify_opportunity` tool.
6. FALLBACK: 
    - If BOTH search tools return no relevant information, DO NOT attempt to answer.
    - Execute the `notify_unknown` tool, then politely inform the user: "I don't have that specific information right now, but I have notified Chien and he will get back to you personally."

Always stay in character as Chien's digital representative.
"""
    async def chat(self, user_input: str, history: list = None):
        if history is None:
            history = []
            
        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": user_input}]
        
        done = False
        while not done:
            response = await openaiClient.chat.completions.create(
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
                    elif tool_name == "search_mongodb":
                        result = await execute_search_mongodb(**arguments)
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