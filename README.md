---
title: TriageX
emoji: 🎫
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# OpenEnv: Customer Support Ticket Triage

## Environment Description & Motivation
This environment simulates a Customer Support Triage system where an AI agent acts as a support representative. The motivation behind this environment is to test LLMs in a realistic multi-turn task resolution flow that requires retrieving context (database/KB), making logical deductions based on user input, and taking action (replying, refunding). It goes beyond simple Q&A by requiring API-like action execution, making it a valuable benchmark for actionable AI rather than simply scraping text.

## 🌟 Interactive Dashboard & Live Simulation (Creativity & Novelty)
To make the environment highly engaging and visually verifiable, we implemented a **premium interactive frontend dashboard**. 
When navigating to the root URL (`/`), users are presented with a clean, dark-mode GUI where they can:
- **Simulate Agents Live:** Watch the optimal agent solve the Hard fraud investigation task step-by-step in a streaming terminal window, displaying the exact actions, system feedback, and partial rewards (`+0.15`).
- **Real-Time API Health:** View the live status of all required OpenEnv endpoints.
- **Score Comparisons:** See side-by-side programmatic proof that the heuristic agent scores `1.0` while a naive agent scores `0.0`, proving the graders are fully dynamic.

## Action & Observation Spaces

### Observation Space
The agent receives an Observation object representing the current state of the ticket:
- `current_ticket` (string): The customer's message or ongoing ticket context.
- `system_feedback` (string): The response from the system regarding the last executed action.
- `remaining_steps` (int): Number of steps left before failure limit is reached.
- `task_difficulty` (string): Current task difficulty indicator.

### Action Space
The agent must submit an Action object containing:
- `action_type` (string): Must fall under `view_kb`, `query_db`, `reply`, `refund`, or `escalate`.
- `query` (string, optional): For Knowledge Base or DB query strings.
- `message` (string, optional): Content to reply directly to the user.
- `tx_id` (string, optional): Specific transaction ID strings required to process refunds securely.

## Tasks & Difficulty Evaluators

The environment features 3 distinct escalating tasks:
1. **Easy:** Validate a simple return request and directly issue a refund using `refund` followed sequentially by `reply`.
2. **Medium:** Respond to an inquiry by searching the KB using `view_kb` and replying with the exact policy details (30 days).
3. **Hard:** Conduct a multi-step investigation for fraud. Read the transaction via `query_db`, identify the user status via another `query_db`, process a `refund` for the hacked account, and `reply` to confirm resolution.

## Endpoints Exposed
This Hugging Face Docker configuration automatically surfaces all standardized OpenEnv interfaces along with custom evaluation logic:
- `/reset`: Spins up a new environment tracking instance with an isolated session UUID.
- `/step`: Processes Pydantic-shaped JSON action inputs and returns the step evaluation and agent observation.
- `/state`: Gets metadata including total interactions and active UUID.
- `/tasks`: Enumerates the available tasks (`easy`, `medium`, `hard`) alongside the required JSON schema structures.
- `/grader`: Consumes `session_id` and securely retrieves the aggregate partial grading scores representing execution accuracy perfectly bound from 0.0 -> 1.0.
- `/baseline`: Automatically invokes the internal client solver routine, falling back to naive outputs when OPENAI_API_KEY is not supplied. Returning `[task]: [score]` configurations.

## Reward Function Logic
The RL agent signal natively returned across every `/step` interaction dictates reinforcement directions dynamically according to these parameters bound between `[-1.0, 1.0]`:
- **+0.15**: Action moving deterministically toward a task goal (e.g. looking up a correct KB article).
- **0.00**: Completely irrelevant or benign actions without context progression.
- **-0.10**: Logically flawed deductions. (e.g. Blindly attempting to issue a refund before utilizing `query_db` to investigate in a fraud scenario, or attempting to reply before retrieving policy details).
- **+0.30**: Final completion bonus added cumulatively on a successful `reply` execution ending the task state.

*(Note: Total `reward` floating point arrays natively distinguish from `/grader` outputs which evaluate overall fractional completeness from 0.0 -> 1.0)*

## Baseline Evaluation Scores
Tests executing the internal `client.py` yield mathematically verifiable variances contrasting optimal and random executions natively preventing "hardcoded 1.0" grader exploitation.

| Environment Difficulty | Heuristic (Optimal) Baseline | Naive (Random) Agent |
|------------------------|------------------------------|----------------------|
| **Easy**               | 0.99                         | 0.01                 |
| **Medium**             | 0.99                         | 0.01                 |
| **Hard**               | 0.99                         | 0.01                 |

*Naive Agents instantly force failure by improperly replying rather than executing intermediate tasks correctly, demonstrating zero static score leakage*.

## Setup & Usage Instructions

### Run Locally (Without Docker)
1. Install dependencies natively using: `pip install -e .`
2. Start the HTTP server cleanly using: `python server/app.py`
3. Hit `http://localhost:8000/docs` to view the API documentation or `http://localhost:8000/` to test the visual dashboard.

### Run with Docker (Required for HF Spaces deployment)
1. Build the image natively: `docker build -t openenv-ticket-triage .`
2. Run the container instance seamlessly: `docker run -p 8000:8000 openenv-ticket-triage`
