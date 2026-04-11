"""
BlackoutShield Server v3.4
FastAPI server matching OpenEnv reference project patterns.
Endpoints: /health, /reset, /step, /state, /tasks, /grader, /legal-actions
"""

import sys
import os
import logging

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from models import BlackoutAction, BlackoutObservation, BlackoutState, BlackoutStepResult, BlackoutResetResult
from server.blackout_environment import BlackoutEnvironment
from server.tasks import TASKS, grade_episode, apply_rubrics, detect_reward_hacking

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="BlackoutShield v3.4",
    description="AI-Powered Grid Cyber Defense — OpenEnv RL Environment",
    version="3.4.0",
    docs_url="/docs",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
env: Optional[BlackoutEnvironment] = None
active_task_id: int = 0


# ==================== CORE ENDPOINTS ====================

@app.get("/")
@app.get("/health")
def health():
    """Health check — required by OpenEnv."""
    return {
        "status": "healthy",
        "service": "blackout-shield",
        "version": "3.4.0",
        "tasks_available": len(TASKS),
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/grader", "/legal-actions"],
    }


@app.post("/reset")
def reset_env(task_id: int = 0, seed: int = 42):
    """
    Reset environment and start new episode.
    Accepts query params: POST /reset?task_id=0&seed=42
    Also accepts JSON body: {"task_id": 0, "seed": 42}
    Also works with empty POST body (defaults to task 0).
    """
    global env, active_task_id

    if task_id not in TASKS:
        raise HTTPException(400, f"task_id must be one of {list(TASKS.keys())}")

    active_task_id = task_id
    env = BlackoutEnvironment(seed=seed)
    obs = env.reset(task_id=active_task_id)

    logger.info(f"Reset: task={TASKS[task_id]['name']}, seed={seed}")

    return {
        "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs.__dict__,
        "reward": obs.reward,
        "done": obs.done,
        "info": {
            "task_id": task_id,
            "task_name": TASKS[task_id]["name"],
            "difficulty": TASKS[task_id]["difficulty"],
            "max_steps": TASKS[task_id]["max_steps"],
            "seed": seed,
        },
    }


@app.post("/step")
def step_env(action: BlackoutAction):
    """
    Execute action in environment.
    Body: {"action_id": 10}
    """
    global env, active_task_id

    if env is None:
        raise HTTPException(400, "Environment not initialized. Call POST /reset first.")
    if not (0 <= action.action_id <= 21):
        raise HTTPException(400, f"action_id must be 0-21, got {action.action_id}")

    # Validate against legal actions
    legal = env.get_legal_actions()
    if action.action_id not in legal:
        raise HTTPException(400, f"Action {action.action_id} is masked. Legal actions: {legal}")

    obs = env.step(BlackoutAction(action_id=action.action_id))

    resp = {
        "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs.__dict__,
        "reward": obs.reward,
        "done": obs.done,
        "info": {"step": len(env.hospital_history)},
    }

    if obs.done:
        resp["grade"] = grade_episode(active_task_id, env.state_dict())

    return resp


@app.get("/state")
def get_state():
    """Get full environment state for debugging."""
    if env is None:
        raise HTTPException(400, "Environment not initialized.")
    s = env.state
    return s.model_dump() if hasattr(s, 'model_dump') else s.__dict__


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        str(tid): {
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
            "pass_criteria": t.get("pass_criteria", ""),
            "valid_trajectories": t.get("valid_trajectories", []),
        }
        for tid, t in TASKS.items()
    }


@app.get("/actions")
def list_actions():
    """List all 22 actions with descriptions."""
    return {
        str(k): f"{v[0]}({v[1]})" if v[1] is not None else v[0]
        for k, v in BlackoutEnvironment.ACTION_MAP.items()
    }


@app.get("/legal-actions")
def legal_actions():
    """Get list of legal action IDs for current state."""
    if env is None:
        raise HTTPException(400, "Environment not initialized.")
    return {"legal_actions": env.get_legal_actions()}


@app.post("/grader")
def run_grader(task_id: Optional[int] = None):
    """
    Run grader on current episode. Returns rubric scores + hacking detection.
    """
    if env is None:
        raise HTTPException(400, "Environment not initialized.")

    tid = task_id if task_id is not None else active_task_id
    state = env.state_dict()
    grade = grade_episode(tid, state)
    rubrics = apply_rubrics(state)
    hacking = detect_reward_hacking(state)

    return {
        "grade": grade,
        "rubrics": rubrics,
        "reward_hacking": hacking,
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()