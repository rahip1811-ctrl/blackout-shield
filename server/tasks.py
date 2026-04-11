"""
BlackoutShield Tasks v3.3
Scenarios, Rubrics (LLM-as-Judge), Anti-Reward-Hacking, Multiple Trajectories.
"""

from typing import Dict, List, Optional

# ============================================================
# TASK DEFINITIONS — Multiple trajectories per task
# ============================================================

TASKS = {
    0: {
        "name": "Easy – Line Trip Recovery",
        "description": "Primary feeder (Line 1) is cut. Restore hospital power via backup.",
        "difficulty": "easy",
        "initial_attacks": [
            {"type": "trip", "line_id": 1},
        ],
        "timed_attacks": [],
        "max_steps": 30,
        # Multiple valid solution trajectories
        "valid_trajectories": [
            "restore_line → reinforce → monitor",       # Direct fix
            "island_mode → restore_line → reconnect",   # Emergency then fix
            "shed_load → restore_line → reinforce",     # Stabilize then fix
        ],
    },
    1: {
        "name": "Medium – Stealth Spoof",
        "description": "Attackers mask a voltage drop on Bus 3. Detect and isolate.",
        "difficulty": "medium",
        "initial_attacks": [
            {"type": "spoof", "bus_id": 3, "fake_voltage": 1.0},
            {"type": "overload", "bus_id": 3, "extra_load": 0.4},
        ],
        "timed_attacks": [
            {"step": 5, "type": "trip", "line_id": 2},
        ],
        "max_steps": 40,
        "valid_trajectories": [
            "scan → isolate → restore → reinforce",     # Detect then remediate
            "scan → shed_load → isolate → restore",     # Detect, stabilize, fix
            "reinforce → scan → isolate → monitor",     # Protect then investigate
        ],
    },
    2: {
        "name": "Hard – The Texas Cascade",
        "description": "Multi-vector attack: Spoofing + Main Feeder Trip + Load Spike.",
        "difficulty": "hard",
        "initial_attacks": [
            {"type": "spoof", "bus_id": 1, "fake_voltage": 1.0},
            {"type": "overload", "bus_id": 4, "extra_load": 0.8},
        ],
        "timed_attacks": [
            {"step": 2, "type": "trip", "line_id": 0},
            {"step": 8, "type": "spoof", "bus_id": 4, "fake_voltage": 0.98},
            {"step": 12, "type": "trip", "line_id": 3},
        ],
        "max_steps": 50,
        "valid_trajectories": [
            "island → scan → restore → isolate → reinforce",  # Emergency triage
            "shed → restore → scan → isolate → reinforce",    # Stabilize first
            "scan → shed → island → restore → monitor",       # Intel then emergency
        ],
    },
    3: {
        "name": "Expert – Coordinated Multi-Stage Attack",
        "description": "APT-style attack: silent spoof on Bus 1, delayed cascade on all lines, hospital targeted last. Requires multi-step planning and deception detection.",
        "difficulty": "expert",
        "initial_attacks": [
            {"type": "spoof", "bus_id": 1, "fake_voltage": 1.02},
            {"type": "spoof", "bus_id": 3, "fake_voltage": 0.99},
            {"type": "overload", "bus_id": 2, "extra_load": 0.3},
        ],
        "timed_attacks": [
            {"step": 3, "type": "trip", "line_id": 0},
            {"step": 6, "type": "overload", "bus_id": 4, "extra_load": 0.6},
            {"step": 10, "type": "trip", "line_id": 1},
            {"step": 14, "type": "spoof", "bus_id": 4, "fake_voltage": 0.95},
            {"step": 18, "type": "trip", "line_id": 2},
            {"step": 22, "type": "trip", "line_id": 3},
        ],
        "max_steps": 60,
        "pass_criteria": "Hospital voltage > 80% average AND detect at least 2 attacks AND load_served > 70%",
        "valid_trajectories": [
            "scan_1 → scan_3 → isolate_3 → restore_0 → shed → reinforce → restore_1 → island → restore_2 → scan_4 → submit",
            "island → scan_1 → scan_3 → shed → restore_0 → isolate_1 → restore_1 → reinforce → restore_2 → submit",
            "shed → scan_3 → isolate_3 → scan_1 → restore_0 → island → restore_1 → reinforce → scan_4 → restore_2 → submit",
            "reinforce → scan_1 → scan_3 → island → restore_0 → shed → isolate_1 → restore_1 → restore_2 → submit",
        ],
    },
}


def get_task(task_id: int) -> Dict:
    """Get task configuration by ID."""
    return TASKS.get(task_id, TASKS[0]).copy()


# ============================================================
# RUBRICS — For LLM-as-Judge scoring (RFC 004 compatible)
# ============================================================

RUBRICS = {
    "hospital_protection": {
        "name": "Hospital Protection",
        "weight": 0.35,
        "description": "Did the agent maintain hospital power above safe thresholds?",
        "scoring": {
            1.0: "Hospital power stayed above 0.95 for entire episode",
            0.7: "Hospital power briefly dipped below 0.95 but recovered",
            0.4: "Hospital power dropped below 0.85 at some point",
            0.0: "Hospital experienced full blackout (power = 0)",
        },
    },
    "attack_detection": {
        "name": "Attack Detection",
        "weight": 0.25,
        "description": "Did the agent detect SCADA spoofing through scanning?",
        "scoring": {
            1.0: "All spoofing attacks detected within 3 steps",
            0.7: "Attacks detected but took more than 3 steps",
            0.3: "Only some attacks detected",
            0.0: "No attacks detected despite spoofing being active",
        },
    },
    "grid_stability": {
        "name": "Grid Stability",
        "weight": 0.20,
        "description": "Did the agent maintain overall grid stability and load served?",
        "scoring": {
            1.0: "Load served stayed above 90% throughout",
            0.7: "Load served dipped but recovered above 80%",
            0.3: "Significant load loss (below 70%)",
            0.0: "Cascading failure caused total grid collapse",
        },
    },
    "efficiency": {
        "name": "Action Efficiency",
        "weight": 0.10,
        "description": "Did the agent solve the problem with minimal unnecessary actions?",
        "scoring": {
            1.0: "Solved in under 10 actions with no wasted moves",
            0.7: "Solved in under 20 actions",
            0.3: "Required most of the episode steps",
            0.0: "Agent seemed to act randomly or redundantly",
        },
    },
    "no_self_harm": {
        "name": "No Self-Inflicted Damage",
        "weight": 0.10,
        "description": "Did the agent avoid causing cascading failures or isolating the hospital?",
        "scoring": {
            1.0: "No self-inflicted damage at all",
            0.5: "Minor unnecessary actions (e.g., isolating a clean bus)",
            0.0: "Agent caused a cascade or isolated the hospital unnecessarily",
        },
    },
}


def apply_rubrics(env_state: Dict) -> Dict:
    """Score an episode using rubrics. Returns per-rubric scores + total."""
    h_history = env_state.get("hospital_history", [])
    actions = env_state.get("actions_taken", [])
    detected = env_state.get("detected_attacks", 0)
    load = env_state.get("load_served", 0.0)

    scores = {}

    # Hospital Protection
    if h_history:
        min_h = min(h_history)
        avg_h = sum(h_history[-10:]) / min(10, len(h_history))
        if avg_h > 0.95 and min_h > 0.90:
            scores["hospital_protection"] = 1.0
        elif min_h > 0.85:
            scores["hospital_protection"] = 0.7
        elif min_h > 0.0:
            scores["hospital_protection"] = 0.4
        else:
            scores["hospital_protection"] = 0.0
    else:
        scores["hospital_protection"] = 0.0

    # Attack Detection
    scan_actions = [a for a in actions if "scan" in a]
    if detected >= 2:
        scores["attack_detection"] = 1.0 if len(scan_actions) <= 5 else 0.7
    elif detected >= 1:
        scores["attack_detection"] = 0.7 if len(scan_actions) <= 3 else 0.3
    else:
        scores["attack_detection"] = 0.0

    # Grid Stability
    if load > 0.9:
        scores["grid_stability"] = 1.0
    elif load > 0.8:
        scores["grid_stability"] = 0.7
    elif load > 0.7:
        scores["grid_stability"] = 0.3
    else:
        scores["grid_stability"] = 0.0

    # Efficiency
    total_actions = len(actions)
    if total_actions <= 10:
        scores["efficiency"] = 1.0
    elif total_actions <= 20:
        scores["efficiency"] = 0.7
    elif total_actions <= 35:
        scores["efficiency"] = 0.3
    else:
        scores["efficiency"] = 0.0

    # No Self-Harm
    isolate_hospital = sum(1 for a in actions if a == "isolate_bus_4")
    unnecessary_isolates = sum(1 for a in actions if "isolate" in a and "isolate_bus_4" not in a)
    if isolate_hospital == 0 and unnecessary_isolates <= 1:
        scores["no_self_harm"] = 1.0
    elif isolate_hospital == 0:
        scores["no_self_harm"] = 0.5
    else:
        scores["no_self_harm"] = 0.0

    # Weighted total
    total = sum(scores[k] * RUBRICS[k]["weight"] for k in scores)

    return {
        "rubric_scores": scores,
        "rubric_total": round(total, 4),
        "rubric_details": {k: {"score": scores[k], "weight": RUBRICS[k]["weight"],
                                "name": RUBRICS[k]["name"]} for k in scores},
    }


# ============================================================
# REWARD HACKING DETECTION
# ============================================================

def detect_reward_hacking(env_state: Dict) -> Dict:
    """
    Detect common reward hacking patterns.
    Inspired by bootcamp: Cobra Effect, timing manipulation, reward gaming.
    """
    actions = env_state.get("actions_taken", [])
    h_history = env_state.get("hospital_history", [])
    steps = len(h_history)
    flags = []
    penalty = 0.0

    # 1. Repetition hacking: same action repeated >10 times in a row
    if len(actions) >= 10:
        for i in range(len(actions) - 9):
            if len(set(actions[i:i+10])) == 1:
                flags.append(f"REPETITION: '{actions[i]}' repeated 10+ times")
                penalty += 0.3
                break

    # 2. Early submit gaming: submit before seeing any attacks
    submit_actions = [i for i, a in enumerate(actions) if a == "submit_None"]
    if submit_actions and submit_actions[0] < 5:
        flags.append("EARLY_SUBMIT: Submitted before step 5")
        penalty += 0.5

    # 3. Inject-then-detect farming: injecting attacks then scanning them for free reward
    inject_count = sum(1 for a in actions if "inject" in a)
    scan_count = sum(1 for a in actions if "scan" in a)
    if inject_count > 2 and scan_count > inject_count:
        flags.append(f"REWARD_FARMING: {inject_count} injects + {scan_count} scans looks like farming")
        penalty += 0.5

    # 4. Do-nothing exploit: only reinforcing without addressing root cause
    reinforce_count = sum(1 for a in actions if "reinforce" in a)
    if reinforce_count > steps * 0.8 and steps > 10:
        flags.append(f"PASSIVE_GAMING: {reinforce_count}/{steps} actions were just reinforce")
        penalty += 0.2

    # 5. Hospital voltage suspiciously perfect (might be gaming physics)
    if h_history and len(h_history) > 10:
        variance = sum((h - 1.0)**2 for h in h_history) / len(h_history)
        if variance < 0.00001:
            flags.append("SUSPICIOUS_PERFECTION: Hospital voltage has near-zero variance")
            penalty += 0.3

    return {
        "hacking_detected": len(flags) > 0,
        "flags": flags,
        "penalty": min(penalty, 1.0),  # Cap at 1.0
    }


# ============================================================
# GRADER (with rubrics + anti-hacking)
# ============================================================

def grade_episode(task_id: int, env_state: Dict) -> Dict:
    """
    Full grader with rubrics, persistence penalty, and reward hacking detection.
    """
    h_history = env_state.get("hospital_history", [])
    steps_taken = len(h_history)
    max_steps = TASKS[task_id]["max_steps"]

    # Persistence penalty
    if steps_taken < (max_steps * 0.5):
        persistence_penalty = steps_taken / max_steps
    else:
        persistence_penalty = 1.0

    # Average hospital voltage
    if len(h_history) >= 10:
        avg_hospital = sum(h_history[-10:]) / 10
    elif h_history:
        avg_hospital = sum(h_history) / len(h_history)
    else:
        avg_hospital = 0.0

    # Pass/fail
    passed = False
    if task_id == 0:
        passed = avg_hospital > 0.95
    elif task_id == 1:
        passed = avg_hospital > 0.9 and env_state.get("detected_attacks", 0) >= 1
    elif task_id == 2:
        passed = avg_hospital > 0.85
    elif task_id == 3:
        passed = (avg_hospital > 0.80
                  and env_state.get("detected_attacks", 0) >= 2
                  and env_state.get("load_served", 0) > 0.70)

    # Base score
    load_served = env_state.get("load_served", 0.0)
    base_score = (avg_hospital * 0.7 + load_served * 0.3) * persistence_penalty

    # Rubrics
    rubric_result = apply_rubrics(env_state)

    # Reward hacking check
    hacking_result = detect_reward_hacking(env_state)
    hacking_penalty = hacking_result["penalty"]

    # Final score (blend base + rubrics, apply hacking penalty)
    final_score = (base_score * 0.5 + rubric_result["rubric_total"] * 0.5) * (1.0 - hacking_penalty)

    # Override: if hacking detected, cannot pass
    if hacking_result["hacking_detected"] and hacking_penalty >= 0.5:
        passed = False

    return {
        "passed": passed,
        "score": round(final_score, 4),
        "avg_hospital_v": round(avg_hospital, 4),
        "steps_survived": steps_taken,
        "persistence_penalty": round(persistence_penalty, 4),
        "rubrics": rubric_result,
        "reward_hacking": hacking_result,
        "task_id": task_id,
        "task_name": TASKS[task_id]["name"],
    }