from pydantic import BaseModel

from typing import Dict, List, Optional

class UserContext(BaseModel):
    profile: Optional[Dict] = None
    history: Optional[List[str]] = None
    environment: Optional[Dict] = None

class UserQuery(BaseModel):
    user_id: str
    question: str
    context: Optional[UserContext] = None  # Contexte utilisateur

class AgentResponse(BaseModel):
    answer: str
    confidence: float  # real confidence score
    routing: str       # "direct", "validation", "human"
    sources: List[str] = []  # Sources used for response


