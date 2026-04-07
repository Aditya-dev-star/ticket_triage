"""
Ticket Triage Environment for handling customer support tasks.
"""
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TicketTriageAction, TicketTriageObservation
except (ModuleNotFoundError, ImportError):
    from models import TicketTriageAction, TicketTriageObservation

# Dictionary linking episode boundaries to active grader scores.
session_grader_scores = {}

class TicketTriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.current_task_idx = 0
        self.max_steps = 10
        self.progress = 0

    def reset(self, task_index: int = None, **kwargs) -> TicketTriageObservation:
        episode_id = kwargs.get("episode_id", str(uuid4()))
        self._state = State(episode_id=episode_id, step_count=0)
        # Allow callers to pin a specific task
        
        # Handle string inputs gracefully if passed by grader via POST /reset 
        if isinstance(task_index, str):
            if task_index.lower() == "easy":
                task_index = 0
            elif task_index.lower() == "medium":
                task_index = 1
            elif task_index.lower() == "hard":
                task_index = 2
            elif task_index.isdigit():
                task_index = int(task_index)
            else:
                task_index = None

        if task_index is not None and isinstance(task_index, int) and 0 <= task_index <= 2:
            self.current_task_idx = task_index
        else:
            self.current_task_idx = self._reset_count % 3
        self._reset_count += 1
        self.progress = 0

        # Initialize isolated session score 0.0 to grader dict
        session_grader_scores[self._state.episode_id] = 0.0

        tasks = [
            ("easy", "Hi, I need a refund for tx_123, it was a mistake."),
            ("medium", "Can you tell me the return policy for electronics?"),
            ("hard", "User reports unauthorized charge tx_999.")
        ]
        diff, ticket = tasks[self.current_task_idx]

        return TicketTriageObservation(
            current_ticket=ticket,
            system_feedback="New ticket assigned.",
            remaining_steps=self.max_steps,
            task_difficulty=diff,
            done=False,
            reward=0.0
        )

    def step(self, action: TicketTriageAction) -> TicketTriageObservation:
        self._state.step_count += 1
        steps_left = self.max_steps - self._state.step_count
        done = steps_left <= 0
        
        # RL Signal Requirements
        reward = 0.0
        feedback = "Action executed."
        
        diff = ["easy", "medium", "hard"][self.current_task_idx]
        ticket = [
            "Hi, I need a refund for tx_123, it was a mistake.",
            "Can you tell me the return policy for electronics?",
            "User reports unauthorized charge tx_999."
        ][self.current_task_idx]

        grader_score = session_grader_scores.setdefault(self._state.episode_id, 0.0)

        if self.current_task_idx == 0:
            if action.action_type == "refund" and action.tx_id == "tx_123" and self.progress == 0:
                self.progress = 1
                grader_score += 0.5
                reward = 0.15
                feedback = "Refund processed successfully for tx_123."
            elif action.action_type == "reply" and self.progress == 1:
                grader_score += 0.5
                reward = 0.15 + 0.3  # step + bonus
                feedback = "Reply sent to customer. Ticket resolved."
                done = True
            elif action.action_type == "reply":
                reward = -0.1
                feedback = "Cannot reply before completing the necessary actions."
            else:
                reward = 0.0
                feedback = "Irrelevant or neutral action."

        elif self.current_task_idx == 1:
            q = (action.query or "").lower()
            msg = (action.message or "").lower()
            
            if action.action_type == "view_kb" and "return" in q and self.progress == 0:
                self.progress = 1
                grader_score += 0.4
                reward = 0.15
                feedback = "KB Article: Electronics return policy is 30 days."
            elif action.action_type == "reply" and self.progress == 1:
                grader_score += 0.3
                reward = 0.15
                if "30" in msg:
                    grader_score += 0.3
                    reward += 0.3
                    feedback = "Customer updated with correct policy. Ticket resolved."
                else:
                    reward -= 0.1
                    feedback = "Reply sent but the resolution was incorrect. Ticket closed."
                done = True
            elif action.action_type == "reply":
                reward = -0.1
                feedback = "Cannot reply before querying the KB."
                done = True
            else:
                reward = 0.0
                feedback = "Irrelevant or neutral action."

        elif self.current_task_idx == 2:
            q = (action.query or "").lower()
            
            if action.action_type == "refund" and self.progress < 2:
                reward = -0.1
                feedback = "Action denied. You cannot refund without investigating the transaction first."
            elif action.action_type == "query_db":
                if "tx_999" in q and self.progress == 0:
                    self.progress = 1
                    grader_score += 0.25
                    reward = 0.15
                    feedback = "DB Record: Transaction tx_999 belongs to User u_456, amount: $50."
                elif "u_456" in q and self.progress == 1:
                    self.progress = 2
                    grader_score += 0.25
                    reward = 0.15
                    feedback = "DB Record: User u_456 account status is Hacked."
                else:
                    reward = 0.0
                    feedback = "DB query returned no relevant results or was executed out of order."
            elif action.action_type == "refund" and action.tx_id == "tx_999" and self.progress == 2:
                self.progress = 3
                grader_score += 0.25
                reward = 0.15
                feedback = "Refund processed for hacked account."
            elif action.action_type == "reply" and self.progress == 3:
                grader_score += 0.25
                reward = 0.15 + 0.3
                feedback = "Customer informed. Account locked and ticket closed."
                done = True
            elif action.action_type == "reply":
                reward = -0.1
                feedback = "Cannot reply yet. Investigate further."
                done = True
            else:
                reward = 0.0

        # Cap and save session score
        session_grader_scores[self._state.episode_id] = min(max(grader_score, 0.0), 1.0)

        return TicketTriageObservation(
            current_ticket=ticket,
            system_feedback=feedback,
            remaining_steps=steps_left,
            task_difficulty=diff,
            done=done,
            reward=round(reward, 2)
        )

    def state(self) -> State:
        # Return the State object directly — openenv's /state route_config expects a State instance.
        # State has additionalProperties=True so extra fields are safe to include.
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
        )
