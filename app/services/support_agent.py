import random
from typing import List
from app.models.support import UserQuery, AgentResponse, UserContext
from app.core.config import settings
from groq import Groq
import json
from pathlib import Path

FAQ_PATH= Path(__file__).parents[2] /"faq.json"
with open(FAQ_PATH, "r") as f:
    FAQ=json.load(f)

# Initialize LLM
llm_client = Groq(api_key=settings.GROQ_API_KEY)

async def query_llm(question: str, context: UserContext = None) -> str:
    """
    Interroger Groq/LLM avec la question et le contexte utilisateur.
    """
    context_str = ""
    if context:
        if context.profile:
            context_str += f"Profile info: {context.profile}\n"
        if context.history:
            context_str += f"Previous interactions: {context.history}\n"
        if context.environment:
            context_str += f"Environment: {context.environment}\n"

    prompt = f"{context_str}\nQuestion: {question}\nAnswer clearly and factually."

    # Use SDK Groq
    chat_completion = llm_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=4000
)
    llm_answer = chat_completion.choices[0].message.content
    return llm_answer


def calculate_similarity(user_question: str, faq_question: str) -> float:
    """Return a score [0,1] of similarity between two questions."""
    user_words = set(user_question.lower().split())
    faq_words = set(faq_question.lower().split())
    if not faq_words:
        return 0.0
    overlap = user_words & faq_words
    return len(overlap) / len(faq_words)

def evaluate_confidence(user_question: str) -> float:
    """Compute confidence based on FAQ similarity."""
    scores = [calculate_similarity(user_question, entry["question"]) for entry in FAQ]
    if scores:
        max_score = max(scores)
        # Map similarity to confidence between 0.6 and 0.95
        confidence = 0.6 + 0.35 * max_score
        return round(confidence, 2)
    return 0.6

def determine_routing(confidence: float) -> str:
    if confidence > 0.8:
        return "direct"
    elif 0.6 < confidence <= 0.8:
        return "validation"
    else:
        return "human"

async def process_query(payload: UserQuery) -> AgentResponse:
    # Query LLM
    llm_answer = await query_llm(payload.question, payload.context)
    sources = ["Groq LLM"]

    # Compute confidence using FAQ
    confidence = evaluate_confidence(payload.question)

    # Determine routing
    routing = determine_routing(confidence)

    return AgentResponse(
        answer=llm_answer,
        confidence=confidence,
        routing=routing,
        sources=sources
    )
 

