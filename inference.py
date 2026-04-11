"""
BlackoutShield Inference Script v3.4
====================================
MANDATORY: Root-level file judges run.

Uses EMBEDDED environment (no HTTP server needed).
Uses OpenAI Client for LLM calls.
Emits [START]/[STEP]/[END] to stdout.

Environment Variables:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       API key
"""

import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# Import environment directly (embedded, no HTTP needed)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import BlackoutAction, BlackoutObservation
from server.blackout_environment import BlackoutEnvironment
from server.tasks import TASKS, grade_episode

# --- Configuration ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_IDS = [0, 1, 2]
TASK_NAMES = {0: "easy_line_trip", 1: "medium_stealth_spoof", 2: "hard_texas_cascade"}
BENCHMARK = "blackout_shield"
MAX_STEPS = 50
TEMPERATURE = 0.3

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI grid defense operator protecting a hospital from cyber attacks on a 5-bus IEEE power grid.

You will receive:
- Grid status with voltages, thermal loads, and suspicion scores
- recent_events: what just happened on the grid
- action_guidance: suggested priorities based on current state
- available_actions: legal actions you can take right now

Actions (respond with ONLY the number):
0-4: isolate_bus_X (disconnect compromised bus)
5-9: scan_bus_X (detect SCADA spoofing)
10: reinforce_hospital (boost hospital power)
11: reinforce_bus1
12-13: shed_load (reduce demand)
14: island_mode (emergency hospital isolation)
15-16: restore_line (reconnect tripped line)
21: submit (end when stable)

Strategy:
1. Tripped lines → RESTORE (15/16)
2. Hospital < 0.80 → ISLAND MODE (14)
3. Suspicion > 0.3 → SCAN that bus (5-9)
4. Suspicion > 0.8 → ISOLATE (0-4, NOT bus 4!)
5. Thermal > 0.95 → SHED LOAD (12/13)
6. Hospital < 0.98 → REINFORCE (10)
7. Stable + time > 80% → SUBMIT (21)

IMPORTANT: Use the recent_events and action_guidance to reason about the best action.
Reply with ONLY one integer. Nothing else.
""").strip()


# --- Logging ---
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def format_obs(obs):
    """Format observation with reasoning fields for LLM."""
    hp = obs.hospital_power
    status = "CRITICAL!" if hp < 0.85 else ("LOW" if hp < 0.95 else "OK")

    text = (
        f"Hospital: {hp:.3f} {status} | Confidence: {obs.telemetry_confidence:.2f}\n"
        f"Voltages: {[round(v,3) for v in obs.scada_voltages]}\n"
        f"Thermal: {[round(t,3) for t in obs.line_thermal]}\n"
        f"Lines: {['ON' if s>0.5 else 'TRIPPED' for s in obs.line_status]}\n"
        f"Suspicion: {[round(s,2) for s in obs.suspicion_map]}\n"
        f"Load: {obs.load_served:.3f} | Time: {obs.time_step:.0%}\n"
    )

    # Add reasoning fields
    if obs.recent_events:
        text += f"\nRecent Events: {obs.recent_events}\n"
    if obs.action_guidance:
        text += f"Action Guidance: {obs.action_guidance}\n"
    if obs.available_actions:
        text += f"Legal Actions: {', '.join(obs.available_actions)}\n"

    return text


def get_llm_action(client, obs, step, history):
    """Ask LLM to choose action based on grid state + reasoning."""
    user_prompt = f"{format_obs(obs)}\n\nRecent:\n" + "\n".join(history[-5:]) + "\n\nAction (integer only):"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE, max_tokens=10, stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        for tok in text.split():
            try:
                a = int(tok)
                if 0 <= a <= 21: return a
            except ValueError:
                continue
        return 10  # Default: reinforce hospital
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return 10


def run_task(client, task_id):
    """Run one task using embedded environment (no HTTP)."""
    task_name = TASK_NAMES[task_id]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Create environment directly (embedded, no server needed)
    env = BlackoutEnvironment(seed=42)
    obs = env.reset(task_id=task_id)

    history, rewards, steps_taken = [], [], 0
    score, success = 0.0, False

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_id = get_llm_action(client, obs, step, history)
            action_name = f"action_{action_id}"

            obs = env.step(BlackoutAction(action_id=action_id))
            reward = obs.reward or 0.0

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_name, reward=reward, done=obs.done, error=None)
            history.append(f"Step {step}: {action_name} -> {reward:+.2f} hospital={obs.hospital_power:.3f}")

            if obs.done:
                break

        # Grade the episode
        grade = grade_episode(task_id, env.state_dict())
        score = grade.get("score", 0.0)
        success = grade.get("passed", False)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores = []
    for task_id in TASK_IDS:
        score = run_task(client, task_id)
        scores.append(score)

    avg = sum(scores) / len(scores)
    print(f"\n[SUMMARY] avg_score={avg:.3f} scores={','.join(f'{s:.3f}' for s in scores)}", flush=True)


if __name__ == "__main__":
    main()