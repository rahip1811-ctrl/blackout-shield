"""Tests for BlackoutShield environment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BlackoutAction, BlackoutObservation
from server.blackout_environment import BlackoutEnvironment
from server.tasks import TASKS, grade_episode


def test_reset():
    env = BlackoutEnvironment(seed=42)
    obs = env.reset(task_id=0)
    assert isinstance(obs, BlackoutObservation)
    assert not obs.done
    assert obs.reward is not None


def test_step():
    env = BlackoutEnvironment(seed=42)
    obs = env.reset(task_id=0)
    obs = env.step(BlackoutAction(action_id=16))
    assert isinstance(obs, BlackoutObservation)
    assert obs.reward is not None


def test_three_tasks():
    for tid in [0, 1, 2]:
        env = BlackoutEnvironment(seed=42)
        obs = env.reset(task_id=tid)
        assert not obs.done
        assert obs.task_description != ""


def test_episode_boundaries():
    env = BlackoutEnvironment(seed=42)
    obs = env.reset(task_id=0)
    assert not obs.done
    obs = env.step(BlackoutAction(action_id=21))
    assert obs.done


def test_grader_scores():
    for tid in [0, 1, 2]:
        env = BlackoutEnvironment(seed=42)
        obs = env.reset(task_id=tid)
        for _ in range(15):
            obs = env.step(BlackoutAction(action_id=10))
        grade = grade_episode(tid, env.state_dict())
        assert 0.0 <= grade["score"] <= 1.0


def test_reward_varies():
    env = BlackoutEnvironment(seed=42)
    obs = env.reset(task_id=1)
    rewards = []
    for _ in range(10):
        obs = env.step(BlackoutAction(action_id=8))
        rewards.append(obs.reward)
    assert len(set(rewards)) > 1, "Rewards must vary"


def test_legal_actions():
    env = BlackoutEnvironment(seed=42)
    env.reset(task_id=0)
    legal = env.get_legal_actions()
    assert isinstance(legal, list)
    assert len(legal) > 0
    assert all(0 <= a <= 21 for a in legal)


def test_reasoning_fields():
    env = BlackoutEnvironment(seed=42)
    obs = env.reset(task_id=1)
    assert obs.recent_events != ""
    assert obs.action_guidance != ""
    assert obs.available_actions is not None
    assert len(obs.available_actions) > 0


def test_physics_context():
    env = BlackoutEnvironment(seed=42)
    obs = env.reset(task_id=0)
    assert obs.physics_context is not None
    assert "total_generation" in obs.physics_context
    assert "cascade_risk" in obs.physics_context


def test_baseline_passes():
    from baseline import run_episode
    for tid in [0, 1, 2]:
        result = run_episode(tid, seed=42)
        assert result["grade"]["passed"], f"Task {tid} should pass"


if __name__ == "__main__":
    tests = [f for f in dir() if f.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            eval(f"{t}()")
            print(f"  ✅ {t}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {t}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
