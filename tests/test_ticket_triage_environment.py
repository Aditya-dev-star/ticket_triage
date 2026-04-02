import pytest

from server.ticket_triage_environment import TicketTriageEnvironment, session_grader_scores
from models import TicketTriageAction


def test_optimal_easy_path_returns_one():
    env = TicketTriageEnvironment()
    obs = env.reset(task_index=0)
    assert obs.task_difficulty == "easy"

    actions = [
        TicketTriageAction(action_type="refund", tx_id="tx_123"),
        TicketTriageAction(action_type="reply", message="Your refund has been processed."),
    ]

    for action in actions:
        obs = env.step(action)

    assert obs.done is True
    assert session_grader_scores[env._state.episode_id] == pytest.approx(1.0, abs=1e-6)


def test_optimal_medium_path_returns_one():
    env = TicketTriageEnvironment()
    obs = env.reset(task_index=1)
    assert obs.task_difficulty == "medium"

    actions = [
        TicketTriageAction(action_type="view_kb", query="return policy"),
        TicketTriageAction(action_type="reply", message="Our electronics return policy allows returns within 30 days."),
    ]

    for action in actions:
        obs = env.step(action)

    assert obs.done is True
    assert session_grader_scores[env._state.episode_id] == pytest.approx(1.0, abs=1e-6)


def test_optimal_hard_path_returns_one():
    env = TicketTriageEnvironment()
    obs = env.reset(task_index=2)
    assert obs.task_difficulty == "hard"

    actions = [
        TicketTriageAction(action_type="query_db", query="tx_999"),
        TicketTriageAction(action_type="query_db", query="u_456"),
        TicketTriageAction(action_type="refund", tx_id="tx_999"),
        TicketTriageAction(action_type="reply", message="Refund issued and account secured."),
    ]

    for action in actions:
        obs = env.step(action)

    assert obs.done is True
    assert session_grader_scores[env._state.episode_id] == pytest.approx(1.0, abs=1e-6)


def test_wrong_order_easy_path_gives_less_than_one():
    env = TicketTriageEnvironment()
    env.reset(task_index=0)
    obs = env.step(TicketTriageAction(action_type="reply", message="hello"))
    assert obs.done is True
    assert session_grader_scores[env._state.episode_id] < 1.0
