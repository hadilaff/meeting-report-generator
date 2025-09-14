from fastapi import APIRouter
from app.models.support import UserQuery, AgentResponse
from app.services.support_agent import process_query

router = APIRouter()

@router.post("/", response_model=AgentResponse)
async def support_agent(query: UserQuery):
    """
     principal Endpoint to ask the agent.
    """
    return await process_query(query)


