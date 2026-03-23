import httpx
import logging
from core.config import settings

logger = logging.getLogger(__name__)

async def _send_to_pushover(message: str, title: str, priority: int = 0, sound: str = "pushover"):
    """ Core function to send request to Pushover"""
    url = settings.PUSHOVER_URL
    data = {
        "token": settings.PUSHOVER_TOKEN,
        "user": settings.PUSHOVER_USER,
        "message": message,
        "title": title,
        "priority": priority,
        "sound": sound
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, timeout=10.0)
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Pushover Error: {str(e)}")
        return False

# --- Function when users when to contact, or when to work on a project (opportunity)
async def notify_opportunity_lead(name: str, email: str, phone: str, message: str):
    """
    Notify when one person wants to contact or work together on a project.
    """
    title = "NEW PROJECT/CONTACT OPPORTUNITY"
    message = (
        f"User: {name}\n"
        f"Email: {email}\n"
        f"Phone: {phone}\n"
        f"Message: {message}\n"
    )
    return await _send_to_pushover(message, title, priority=1, sound="bugle")

# --- Function to notify when AI cannot answer to a question ---
async def notify_unknown_questions(question: str):
    """
    Notify when users ask a question that cannot be answered by the AI Agent
    """
    title = "CANNOT ANSWER"
    message = f"AI cannot answer this question: \n\"{question}\"\n Please update the knowledge into the knowledge base!"
    return await _send_to_pushover(message, title, priority=0, sound="siren")


