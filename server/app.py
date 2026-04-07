import sys
import os

# Ensure the root directory is in sys.path so 'models' can be imported when running `python server/app.py`
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv is required. Install via `uv sync`.") from e

try:
    from ..models import TicketTriageAction, TicketTriageObservation
    from .ticket_triage_environment import TicketTriageEnvironment
except (ModuleNotFoundError, ImportError):
    from models import TicketTriageAction, TicketTriageObservation
    from server.ticket_triage_environment import TicketTriageEnvironment

import os
import json
import asyncio
from fastapi.responses import HTMLResponse, StreamingResponse

app = create_app(
    TicketTriageEnvironment,
    TicketTriageAction,
    TicketTriageObservation,
    env_name="ticket_triage",
    max_concurrent_envs=1000,
)

# OpenEnv's /state route generates a Pydantic ResponseValidationError in OpenEnv<0.3
# if the environment's state() returns the State object properly. We will let it be.

TASK_ACTIONS = [
    # Easy
    [{"action_type": "refund",   "tx_id": "tx_123"},
     {"action_type": "reply",    "message": "Your refund has been processed."}],
    # Medium
    [{"action_type": "view_kb",  "query": "return policy"},
     {"action_type": "reply",    "message": "Our electronics return policy is 30 days."}],
    # Hard
    [{"action_type": "query_db", "query": "tx_999"},
     {"action_type": "query_db", "query": "u_456"},
     {"action_type": "refund",   "tx_id": "tx_999"},
     {"action_type": "reply",    "message": "Refund issued and account secured."}],
]


@app.get("/simulate")
def simulate_task(task_index: int = 0):
    """Streams a full optimal-agent simulation step-by-step via Server-Sent Events."""
    task_index = max(0, min(2, task_index))
    
    async def sse_generator():
        env = TicketTriageEnvironment()
        env.current_task_idx = task_index
        env._reset_count = task_index + 1  # ensure task doesn't cycle
        obs = env.reset(task_index=task_index)

        init_data = {"type": "reset", "ticket": obs.current_ticket}
        yield f"data: {json.dumps(init_data)}\n\n"
        await asyncio.sleep(0.8)

        total_reward = 0.0
        for action_data in TASK_ACTIONS[task_index]:
            action = TicketTriageAction(**action_data)
            obs = env.step(action)
            step_data = {
                "type": "step",
                "action_type": action_data["action_type"],
                "feedback": obs.system_feedback,
                "reward": obs.reward,
                "done": obs.done,
            }
            total_reward += obs.reward
            yield f"data: {json.dumps(step_data)}\n\n"
            await asyncio.sleep(1.2)
            if obs.done:
                break
                
        yield f"data: {json.dumps({'type': 'done', 'total_reward': total_reward})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.get("/api/analytics")
def get_analytics():
    """Mock BI Analytics data for dashboard visualization."""
    return {
        "metrics": {
            "avg_resolution_time": "2m 14s",
            "agent_accuracy": "94.2%",
            "total_tickets": 1420
        },
        "trends": {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "agent_scores": [0.85, 0.88, 0.91, 0.89, 0.94, 0.96, 0.98],
            "human_scores": [0.92, 0.91, 0.93, 0.90, 0.92, 0.91, 0.93]
        },
        "categories": {
            "labels": ["Refunds", "Policy Questions", "Fraud", "Tech Support"],
            "data": [45, 25, 15, 15]
        }
    }

@app.get("/")
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        html_path = "server/index.html"
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found.</h1>")

@app.get("/baseline")
def run_baseline(naive: bool = False):
    try:
        try:
            from ..client import solve_task
        except (ModuleNotFoundError, ImportError):
            from client import solve_task
        
        api_key = os.environ.get("OPENAI_API_KEY", "")
        # If no API key and NOT explicitly testing the naive agent,
        # use the heuristic (optimal) path so the environment still demonstrates 1.0 scores.
        # Only use naive=True when the caller explicitly requests it (the "Test Naive" button).
        use_naive = naive  # respect the caller's explicit choice always
        scores = solve_task(api_key, naive=use_naive)
        return scores
    except Exception as e:
        return {"error": f"Failed to execute baseline: {e}"}

@app.get("/grader")
def get_grader_score(session_id: str = None):
    try:
        from .ticket_triage_environment import session_grader_scores
    except ImportError:
        from server.ticket_triage_environment import session_grader_scores
        
    if session_id and session_id in session_grader_scores:
        return {"grader_score": session_grader_scores[session_id]}
    elif session_grader_scores:
        # fallback to the latest if undefined
        latest = list(session_grader_scores.values())[-1]
        return {"grader_score": latest}
    return {"grader_score": 0.0}

@app.get("/tasks")
def get_tasks():
    schema = TicketTriageAction.model_json_schema()
    return {
        "tasks": [
            {"id": "easy", "description": "Refund an erroneous transaction."},
            {"id": "medium", "description": "Query the KB and articulate the return policy."},
            {"id": "hard", "description": "Investigate hacked account, secure, and refund."}
        ],
        "action_schema": schema
    }

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()
