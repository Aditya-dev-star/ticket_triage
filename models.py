from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel
from typing import Optional, Literal, Dict, Any

class TicketTriageAction(Action):
    """Action for the Ticket Triage environment."""
    action_type: Literal['view_kb', 'query_db', 'reply', 'refund', 'escalate'] = Field(..., description="Action to perform")
    query: Optional[str] = Field(None, description="Query string for KB or DB")
    message: Optional[str] = Field(None, description="Message to reply to the user")
    tx_id: Optional[str] = Field(None, description="Transaction ID to refund")

class TicketTriageObservation(Observation):
    """Observation from the Ticket Triage environment."""
    current_ticket: str = Field(description="The customer's message.")
    system_feedback: str = Field(description="System response from the last action.")
    remaining_steps: int = Field(description="Steps left before failure.")
    task_difficulty: str = Field(description="Current task difficulty.")
    done: bool = Field(default=False, description="Whether the episode has ended.")
    reward: float = Field(default=0.0, description="Reward for the last action.")
