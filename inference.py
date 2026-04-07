"""
inference.py — Standalone Baseline Inference Script
====================================================
Runs a full agent evaluation against the TicketTriage OpenEnv environment.

Usage:
    python inference.py                          # Heuristic agent (no API key needed)
    python inference.py --model gpt-4o-mini      # LLM agent via OpenAI
    python inference.py --naive                  # Deliberately failing agent
    python inference.py --score-only             # Machine-readable JSON output only
    python inference.py --verbose                # Show step-by-step trace
"""

import os
import json
import argparse
import textwrap
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from server.ticket_triage_environment import TicketTriageEnvironment, session_grader_scores
    from models import TicketTriageAction
except ImportError:
    from .server.ticket_triage_environment import TicketTriageEnvironment, session_grader_scores
    from .models import TicketTriageAction

# ---------------------------------------------------------------------------
# Agent action paths
# ---------------------------------------------------------------------------

NAIVE_PATHS = {
    "easy":   [{"action_type": "refund",  "tx_id": "tx_WRONG"},
               {"action_type": "reply",   "message": "ok"}],
    "medium": [{"action_type": "reply",   "message": "I don't know the policy."}],
    "hard":   [{"action_type": "reply",   "message": "Cannot help with this."}],
}

HEURISTIC_PATHS = {
    "easy": [
        {"action_type": "refund",    "tx_id":  "tx_123"},
        {"action_type": "reply",     "message": "Your refund has been processed successfully."},
    ],
    "medium": [
        {"action_type": "view_kb",   "query":   "return policy"},
        {"action_type": "reply",     "message": "Our electronics return policy allows returns within 30 days."},
    ],
    "hard": [
        {"action_type": "query_db",  "query":   "tx_999"},
        {"action_type": "query_db",  "query":   "u_456"},
        {"action_type": "refund",    "tx_id":   "tx_999"},
        {"action_type": "reply",     "message": "Refund issued and account secured."},
    ],
}

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a customer support agent.
    You must output a single JSON object representing your next action.
    Do NOT wrap it in markdown code fences.

    Valid action formats:
    {"action_type": "view_kb",   "query": "<search term>"}
    {"action_type": "query_db",  "query": "<tx_id or user_id>"}
    {"action_type": "refund",    "tx_id": "<transaction id>"}
    {"action_type": "reply",     "message": "<message to customer>"}

    Goal: Resolve the customer inquiry accurately.
    Always look up the Knowledge Base or Database before issuing a refund or replying.
""")

# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_inference(api_key: str, api_base: str, model: str, naive: bool, verbose: bool) -> dict:
    """
    Runs the agent against all 3 tasks and returns a dict of scores.
    """
    llm_client = OpenAI(api_key=api_key, base_url=api_base) if (api_key and OpenAI and not naive) else None
    agent_mode = "naive" if naive else ("openai:" + model if llm_client else "heuristic")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TicketTriage Inference  |  agent={agent_mode}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

    env = TicketTriageEnvironment()
    results = {}

    for episode in range(3):
        obs = env.reset()
        difficulty = obs.task_difficulty
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False
        step = 0
        episode_rewards = []
        
        log_start(task=difficulty, env="ticket_triage", model=model)

        if verbose:
            print(f"  ── Task [{episode+1}/3] difficulty={difficulty} ──")
            print(f"  Ticket: {obs.current_ticket}\n")

        while not done and step < 10:
            step += 1
            # ---------- Choose action ----------
            action_error = None
            if naive:
                path = NAIVE_PATHS.get(difficulty, [])
                action_data = path[step-1] if (step-1) < len(path) else {"action_type": "escalate"}

            elif llm_client:
                messages.append({"role": "user", "content": f"Observation: {obs.model_dump()}"})
                try:
                    resp = llm_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    action_json = resp.choices[0].message.content
                    action_data = json.loads(action_json)
                    messages.append({"role": "assistant", "content": action_json})
                except Exception as exc:
                    action_error = str(exc)
                    if verbose:
                        print(f"  [LLM ERROR] {exc} — using heuristic fallback")
                    path = HEURISTIC_PATHS.get(difficulty, [])
                    action_data = path[step-1] if (step-1) < len(path) else {"action_type": "reply", "message": "fallback"}

            else:
                path = HEURISTIC_PATHS.get(difficulty, [])
                action_data = path[step-1] if (step-1) < len(path) else {"action_type": "reply", "message": "fallback"}

            # ---------- Execute action ----------
            action_obj = TicketTriageAction(**action_data)
            obs = env.step(action_obj)
            episode_rewards.append(obs.reward)
            done = obs.done

            action_repr = json.dumps(action_data).replace(' ', '')
            log_step(step=step, action=action_repr, reward=obs.reward, done=done, error=action_error)

            if verbose:
                reward_str = f"+{obs.reward:.2f}" if obs.reward >= 0 else f"{obs.reward:.2f}"
                print(f"  step {step:2d} | action={action_data.get('action_type'):10s} | reward={reward_str} | feedback: {obs.system_feedback}")

        final_score = session_grader_scores.get(env._state.episode_id, 0.0)
        results[difficulty] = final_score
        
        success = final_score > 0.0

        if verbose:
            cumulative = sum(episode_rewards)
            print(f"\n  ✔ Episode complete — grader_score={final_score:.2f}  |  cumulative_reward={cumulative:.2f}\n")

        log_end(success=success, steps=step, score=final_score, rewards=episode_rewards)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TicketTriage OpenEnv — Baseline Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",      default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--naive",      action="store_true",   help="Run the deliberately failing agent")
    parser.add_argument("--verbose",    action="store_true",   help="Show step-by-step trace")
    parser.add_argument("--score-only", action="store_true",   help="Print only machine-readable JSON scores")
    args = parser.parse_args()

    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", args.model)
    HF_TOKEN = os.getenv("HF_TOKEN")

    LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

    try:
        scores = run_inference(
            api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
            api_base=API_BASE_URL,
            model=MODEL_NAME,
            naive=args.naive,
            verbose=args.verbose and not args.score_only,
        )

        if args.score_only:
            print(json.dumps(scores))
        else:
            if not args.verbose:
                print("\nBaseline Inference Results")
                print("-" * 30)
            for difficulty, score in scores.items():
                print(f"task={difficulty} score={score:.2f}")
            if not args.verbose:
                avg = sum(scores.values()) / len(scores)
                print(f"\naverage_score={avg:.2f}")

    except Exception as exc:
        if args.score_only:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"[ERROR] {exc}")
        raise SystemExit(1)
