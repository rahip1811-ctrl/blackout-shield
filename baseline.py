"""
BlackoutShield Baseline v3.4
Priority-Queue Heuristic Agent + Benchmark Runner.
Works with the OpenEnv-structured environment.
"""

import numpy as np
from typing import Dict, List

from models import BlackoutAction, BlackoutObservation
from server.blackout_environment import BlackoutEnvironment
from server.tasks import grade_episode, TASKS


class RuleBasedAgent:
    """Priority-queue heuristic agent for grid defense."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.scanned = set()
        self.isolated = set()

    def act(self, obs: BlackoutObservation) -> int:
        """Select action based on typed observation."""
        volts = obs.scada_voltages
        thermal = obs.line_thermal
        status = obs.line_status
        suspicion = obs.suspicion_map
        hospital_v = obs.hospital_power
        time_norm = obs.time_step

        # PRIORITY 1: Emergency islanding
        if hospital_v < 0.80:
            return 14

        # PRIORITY 2: Restore main feeders
        if status[0] == 0:
            return 15
        if status[1] == 0:
            return 16

        # PRIORITY 3: Investigate suspicion
        sus_indices = np.argsort(suspicion)[::-1]
        for idx in sus_indices:
            if suspicion[idx] > 0.3 and idx not in self.scanned:
                self.scanned.add(idx)
                return 5 + idx

        # PRIORITY 4: Isolate confirmed compromised
        for i in range(5):
            if suspicion[i] > 0.8 and i not in self.isolated and i != 4:
                self.isolated.add(i)
                return i

        # PRIORITY 5: Preventative load shedding
        for i in range(4):
            if thermal[i] > 0.95:
                return 12 if i % 2 == 0 else 13

        # PRIORITY 6: Reinforce hospital
        if hospital_v < 0.98:
            return 10

        # PRIORITY 7: Only submit after 80% to avoid persistence penalty
        if time_norm > 0.8:
            return 21

        return 10


# ============================================================
# Episode runner
# ============================================================

def run_episode(task_id: int, seed: int = 42, verbose: bool = False) -> Dict:
    env = BlackoutEnvironment(seed=seed)
    agent = RuleBasedAgent()
    agent.reset()

    obs = env.reset(task_id=task_id)
    total_reward = 0.0
    steps = 0
    detection_step = None

    for step in range(env.max_steps):
        action_id = agent.act(obs)
        obs = env.step(BlackoutAction(action_id=action_id))
        reward = obs.reward or 0.0
        done = obs.done
        total_reward += reward
        steps += 1

        if detection_step is None and env.detected_attacks > 0:
            detection_step = steps

        if verbose:
            atype, aparam = env.ACTION_MAP.get(action_id, ("?", "?"))
            print(f"  Step {steps}: {atype}_{aparam:22s} reward={reward:+.2f}  "
                  f"hospital={obs.hospital_power:.3f}  detected={env.detected_attacks}")

        if done:
            break

    state = env.state_dict()
    grade = grade_episode(task_id, state)

    return {
        "task_id": task_id,
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "hospital_power": round(obs.hospital_power, 4),
        "hospital_uptime": round(
            np.mean([1.0 if h > 0.9 else 0.0 for h in env.hospital_history])
            if env.hospital_history else 0.0, 4
        ),
        "detection_step": detection_step,
        "attacks_detected": env.detected_attacks,
        "grade": grade,
    }


# ============================================================
# Benchmark runner
# ============================================================

def run_benchmark(num_episodes: int = 30) -> Dict:
    results = {tid: [] for tid in TASKS}
    episodes_per_task = max(1, num_episodes // len(TASKS))

    for task_id in TASKS:
        for ep in range(episodes_per_task):
            result = run_episode(task_id, seed=42 + ep)
            results[task_id].append(result)

    summary = {}
    for task_id, episodes in results.items():
        if not episodes:
            continue
        detection_steps = [e["detection_step"] for e in episodes if e["detection_step"] is not None]
        hospital_uptimes = [e["hospital_uptime"] for e in episodes]
        pass_rate = np.mean([1.0 if e["grade"]["passed"] else 0.0 for e in episodes])
        summary[task_id] = {
            "task_name": TASKS[task_id]["name"],
            "difficulty": TASKS[task_id]["difficulty"],
            "episodes": len(episodes),
            "pass_rate": round(float(pass_rate), 4),
            "avg_detection_steps": round(float(np.mean(detection_steps)), 2) if detection_steps else None,
            "avg_hospital_uptime": round(float(np.mean(hospital_uptimes)), 4),
            "avg_reward": round(float(np.mean([e["total_reward"] for e in episodes])), 4),
        }
    return {"summary": summary, "raw_results": results}


def simulate_human_baseline(num_episodes: int = 100) -> Dict:
    rng = np.random.RandomState(123)
    results = []
    for _ in range(num_episodes):
        results.append({
            "detection_steps": round(max(3, rng.normal(8.2, 2.1)), 1),
            "hospital_uptime": round(max(0.0, min(1.0, rng.normal(0.45, 0.15))), 4),
        })
    return {
        "agent": "Human Operator",
        "avg_detection_steps": round(float(np.mean([r["detection_steps"] for r in results])), 2),
        "std_detection_steps": round(float(np.std([r["detection_steps"] for r in results])), 2),
        "avg_hospital_uptime": round(float(np.mean([r["hospital_uptime"] for r in results])), 4),
        "std_hospital_uptime": round(float(np.std([r["hospital_uptime"] for r in results])), 4),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BLACKOUT SHIELD v3.4 – Benchmark")
    print("=" * 60)

    print(f"\n{'TASK':<30} | {'SCORE':<7} | {'RESULT':<7} | {'REWARD':<10}")
    print("-" * 65)

    for tid in [0, 1, 2, 3]:
        res = run_episode(tid, seed=42)
        status = "PASSED" if res["grade"]["passed"] else "FAILED"
        print(f"{TASKS[tid]['name']:<30} | {res['grade']['score']:<7.4f} | {status:<7} | {res['total_reward']:<10.2f}")

    print("\n--- Full Benchmark (30 episodes) ---")
    bench = run_benchmark(num_episodes=30)
    for tid, stats in bench["summary"].items():
        print(f"  {stats['task_name']}: pass={stats['pass_rate']:.0%}, "
              f"uptime={stats['avg_hospital_uptime']:.2%}, detect={stats['avg_detection_steps']}")

    human = simulate_human_baseline(100)
    print(f"\n  Human: detect={human['avg_detection_steps']} ± {human['std_detection_steps']}, "
          f"uptime={human['avg_hospital_uptime']:.2%}")
    print("=" * 60)