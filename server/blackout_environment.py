"""
BlackoutShield Environment (OpenEnv-compatible)
Extends OpenEnv Environment base class with grid cyber defense logic.
"""

import uuid
import numpy as np
from typing import Dict, Optional

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        """Fallback base class matching OpenEnv interface."""
        pass

try:
    from ..models import BlackoutAction, BlackoutObservation, BlackoutState
except ImportError:
    from models import BlackoutAction, BlackoutObservation, BlackoutState

try:
    from .grid_sim import GridSimulator
except ImportError:
    from grid_sim import GridSimulator

try:
    from .tasks import get_task, grade_episode, TASKS
except ImportError:
    from tasks import get_task, grade_episode, TASKS


class BlackoutEnvironment(Environment):
    """
    OpenEnv-compliant grid cyber defense environment.

    Observation: BlackoutObservation (typed fields instead of raw numpy)
    Action: BlackoutAction (action_id: int 0-21)
    State: BlackoutState (full internal state)
    """

    ACTION_MAP = {
        0: ("isolate_bus", 0), 1: ("isolate_bus", 1), 2: ("isolate_bus", 2),
        3: ("isolate_bus", 3), 4: ("isolate_bus", 4),
        5: ("scan_bus", 0), 6: ("scan_bus", 1), 7: ("scan_bus", 2),
        8: ("scan_bus", 3), 9: ("scan_bus", 4),
        10: ("reinforce", 4), 11: ("reinforce", 1),
        12: ("shed_load", 2), 13: ("shed_load", 3),
        14: ("island_mode", 4),
        15: ("restore_line", 0), 16: ("restore_line", 1),
        17: ("inject_attack", 1), 18: ("inject_attack", 2),
        19: ("inject_attack", 3), 20: ("inject_attack", 4),
        21: ("submit", None),
    }

    ACTION_DESCRIPTIONS = (
        "0-4: isolate_bus_X, 5-9: scan_bus_X, 10: reinforce_hospital, "
        "11: reinforce_bus1, 12-13: shed_load, 14: island_mode, "
        "15-16: restore_line, 17-20: inject_attack(demo), 21: submit"
    )

    def __init__(self, seed: int = 42, max_steps: int = 50):
        if hasattr(Environment, '__init__') and Environment.__init__ is not object.__init__:
            super().__init__()
        self.seed = seed
        self.max_steps = max_steps
        self.grid = GridSimulator(seed=seed)
        self.rng = np.random.RandomState(seed)
        self._reset_internal()

    def _reset_internal(self):
        self.suspicion_map = np.zeros(5)
        self.scanned_buses = set()
        self.detected_attacks = 0
        self.actions_taken = []
        self.episode_reward = 0.0
        self._done = False
        self.task_config = None
        self.hospital_history = []
        self._episode_id = str(uuid.uuid4())
        self._task_id = 0

    # ------------------------------------------------------------------
    # OpenEnv API: reset() → Observation
    # ------------------------------------------------------------------
    def reset(self, task_id: int = 1) -> BlackoutObservation:
        """Reset environment and return initial observation."""
        self.task_config = get_task(task_id)
        self.grid = GridSimulator(seed=self.seed)
        self.grid.reset()
        self.rng = np.random.RandomState(self.seed)
        self._reset_internal()
        self._task_id = task_id
        self.task_config = get_task(task_id)

        if self.task_config.get("initial_attacks"):
            for attack in self.task_config["initial_attacks"]:
                self._apply_attack(attack)

        return self._build_observation(reward=0.0)

    # ------------------------------------------------------------------
    # OpenEnv API: step(action) → Observation (with reward + done)
    # ------------------------------------------------------------------
    def step(self, action: BlackoutAction) -> BlackoutObservation:
        """Execute action and return observation with reward and done."""
        if self._done:
            return self._build_observation(reward=0.0)

        action_id = action.action_id
        # CRITICAL: numpy integers don't pass isinstance(x, int)
        if isinstance(action_id, (int, np.integer)):
            action_type, param = self.ACTION_MAP.get(int(action_id), ("noop", None))
        else:
            action_type, param = "noop", None

        # Execute action
        reward = self._execute_action(action_type, param)

        # Advance grid
        self.grid.step()

        # Timed attacks from task config
        if self.task_config and self.task_config.get("timed_attacks"):
            for timed in self.task_config["timed_attacks"]:
                if timed["step"] == self.grid.time_step:
                    self._apply_attack(timed)

        # Track hospital
        hospital_power = self.grid.get_hospital_power()
        self.hospital_history.append(hospital_power)

        # Step reward
        reward += self._compute_reward(hospital_power)
        self.episode_reward += reward

        # Termination
        if action_type == "submit" or self.grid.time_step >= self.max_steps:
            self._done = True

        # Log action
        action_str = f"{action_type}_{param}" if param is not None else action_type
        self.actions_taken.append(action_str)

        return self._build_observation(reward=reward)

    # ------------------------------------------------------------------
    # OpenEnv API: state property → State
    # ------------------------------------------------------------------
    @property
    def state(self) -> BlackoutState:
        """Full internal state for debugging/visualization."""
        grid_state = self.grid.get_state()
        return BlackoutState(
            episode_id=self._episode_id,
            step_count=self.grid.time_step,
            buses=grid_state["buses"],
            lines=grid_state["lines"],
            suspicion_map=self.suspicion_map.tolist(),
            hospital_power=round(self.grid.get_hospital_power(), 4),
            load_served=round(self.grid.get_load_served(), 4),
            detected_attacks=self.detected_attacks,
            cascade_count=grid_state["cascade_count"],
            hospital_history=self.hospital_history.copy(),
            fog_of_war=grid_state.get("fog_of_war", {}),
            actions_taken=self.actions_taken.copy(),
            episode_reward=self.episode_reward,
            task_id=self._task_id,
            done=self._done,
        )

    # ------------------------------------------------------------------
    # Action masking — legal actions for current state
    # ------------------------------------------------------------------
    def get_legal_actions(self):
        """Return list of legal action IDs based on current grid state."""
        legal = [10, 11, 12, 13]  # reinforce + shed always legal

        # Restore only if line is tripped
        for i in range(4):
            if not self.grid.lines[i].status:
                if i == 0: legal.append(15)
                elif i == 1: legal.append(16)

        # Scan — always legal (but penalized if clean)
        for i in range(5):
            legal.append(5 + i)

        # Isolate — only if suspicion > 0.3
        for i in range(5):
            if self.suspicion_map[i] > 0.3:
                legal.append(i)

        # Island mode — always legal in emergencies
        legal.append(14)

        # Submit — only after 20% of episode
        if self.grid.time_step >= self.max_steps * 0.2:
            legal.append(21)

        return sorted(list(set(legal)))

    def _action_id_to_name(self, action_id):
        """Convert action ID to human-readable name."""
        at, ap = self.ACTION_MAP.get(action_id, ("unknown", None))
        return f"{at}_{ap}" if ap is not None else at

    # ------------------------------------------------------------------
    # Reasoning log generation for LLM Chain-of-Thought
    # ------------------------------------------------------------------
    def _generate_reasoning(self):
        """Generate recent_events and action_guidance for LLM reasoning."""
        events = []
        guidance = []

        # Recent events
        tripped = [i for i in range(4) if not self.grid.lines[i].status]
        if tripped:
            events.append(f"Lines {tripped} are TRIPPED (disconnected)")

        hp = self.grid.get_hospital_power()
        if hp == 0.0:
            events.append("CRITICAL FAILURE: Hospital has ZERO power — complete blackout")
        elif hp < 0.50:
            events.append(f"CRITICAL: Hospital power at {hp:.1%} — emergency action needed")
        elif hp < 0.95:
            events.append(f"WARNING: Hospital power at {hp:.1%} — below safe threshold")

        for i in range(5):
            if self.suspicion_map[i] > 0.5:
                events.append(f"High suspicion on Bus {i} (score: {self.suspicion_map[i]:.0%}) — possible SCADA spoofing")

        thermal = self.grid.get_thermal_status()
        overloaded = [i for i in range(4) if thermal.get(i, 0) > 0.95]
        if overloaded:
            events.append(f"Thermal overload risk on lines {overloaded}")

        if self.detected_attacks > 0:
            events.append(f"{self.detected_attacks} attack(s) detected via scanning")

        # Action guidance (priority queue)
        if hp < 0.80:
            guidance.append("PRIORITY 1: Use island_mode (action 14) to save hospital")
        if tripped:
            guidance.append(f"PRIORITY 1: Restore tripped lines (actions {[15+i for i in range(min(2,len(tripped)))]})")
        if hp < 0.95 and hp >= 0.80:
            guidance.append("PRIORITY 2: Reinforce hospital (action 10)")
        sus_buses = [i for i in range(5) if self.suspicion_map[i] > 0.3]
        if sus_buses:
            guidance.append(f"PRIORITY 2: Scan suspicious buses {sus_buses} (actions {[5+i for i in sus_buses]})")
        if overloaded:
            guidance.append("PRIORITY 3: Shed load (actions 12-13) to prevent cascade")
        if not guidance:
            guidance.append("Grid stable. Monitor or submit when ready (action 21)")

        recent = " | ".join(events) if events else "Grid nominal. No immediate alerts."
        guide = " | ".join(guidance[:3])
        return recent, guide

    # ------------------------------------------------------------------
    # Observation builder (enhanced with reasoning)
    # ------------------------------------------------------------------
    def _build_observation(self, reward: float) -> BlackoutObservation:
        scada_v = self.grid.get_scada_voltages()
        thermal = self.grid.get_thermal_status()
        active_spoofs = len([s for s in self.grid.spoofs if s.active])
        confidence = 1.0 / (1.0 + active_spoofs)
        task_desc = self.task_config["description"] if self.task_config else ""

        recent_events, action_guidance = self._generate_reasoning()
        legal = self.get_legal_actions()

        hp = self.grid.get_hospital_power()
        ls = self.grid.get_load_served()

        physics_context = {
            "total_generation": float(sum(b.generation for b in self.grid.buses.values())),
            "total_load": float(sum(b.load for b in self.grid.buses.values())),
            "cascade_risk": min(1.0, float(max(thermal.get(i, 0) for i in range(4)) - 0.8) / 0.2) if max(thermal.get(i, 0) for i in range(4)) > 0.8 else 0.0,
            "voltage_margin": float(min(scada_v.get(i, 1.0) for i in range(5)) - 0.90),
            "lines_tripped": sum(1 for i in range(4) if not self.grid.lines[i].status),
        }

        return BlackoutObservation(
            done=self._done,
            reward=round(reward, 4),
            scada_voltages=[scada_v.get(i, 0.0) for i in range(5)],
            telemetry_confidence=round(confidence, 4),
            line_thermal=[round(min(thermal.get(i, 0.0), 2.0), 4) for i in range(4)],
            line_status=[1.0 if self.grid.lines[i].status else 0.0 for i in range(4)],
            bus_alerts=[1.0 if scada_v.get(i, 1.0) < 0.95 or scada_v.get(i, 1.0) > 1.05 else 0.0 for i in range(5)],
            line_alerts=[1.0 if min(thermal.get(i, 0.0), 2.0) > 0.9 else 0.0 for i in range(4)],
            suspicion_map=self.suspicion_map.tolist(),
            hospital_power=round(hp, 4),
            load_served=round(ls, 4),
            time_step=round(self.grid.time_step / self.max_steps, 4),
            recent_events=recent_events,
            action_guidance=action_guidance,
            task_description=task_desc,
            available_actions=[self._action_id_to_name(i) for i in legal],
            physics_context=physics_context,
        )

    # ------------------------------------------------------------------
    # Attack application (from task configs)
    # ------------------------------------------------------------------
    def _apply_attack(self, attack: Dict):
        atype = attack["type"]
        if atype == "spoof":
            self.grid.inject_spoof(attack["bus_id"], attack.get("fake_voltage", 1.0))
            self.suspicion_map[attack["bus_id"]] += 0.5  # IDS anomaly detected
        elif atype == "trip":
            self.grid.trip_line(attack["line_id"])
        elif atype == "overload":
            bus_id = attack["bus_id"]
            self.grid.buses[bus_id].load += attack.get("extra_load", 0.5)
            self.grid.solve_power_flow()

    # ------------------------------------------------------------------
    # Action execution with reward shaping (EXACT copy from environment.py)
    # ------------------------------------------------------------------
    def _execute_action(self, action_type: str, param) -> float:
        reward = -0.01  # Existence penalty

        if action_type == "scan_bus" and param is not None:
            result = self.grid.scan_bus(param)
            self.scanned_buses.add(param)
            reward -= 0.05
            if result["is_spoofed"]:
                self.suspicion_map[param] = 1.0
                self.detected_attacks += 1
                self.grid.remove_spoof(param)
                reward += 1.0
            else:
                self.suspicion_map[param] = max(0, self.suspicion_map[param] - 0.2)

        elif action_type == "isolate_bus" and param is not None:
            self.grid.isolate_bus(param)
            if param == 4:
                reward -= 2.0
            elif self.suspicion_map[param] > 0.7:
                reward += 0.5
            else:
                reward -= 0.5

        elif action_type == "reinforce" and param is not None:
            self.grid.reinforce_bus(param, amount=0.2)
            reward -= 0.1

        elif action_type == "shed_load" and param is not None:
            self.grid.shed_load(param, amount=0.2)
            reward -= 0.2

        elif action_type == "island_mode" and param is not None:
            self.grid.trip_line(1)  # Trip BUS1↔BUS4
            self.grid.reinforce_bus(4, amount=0.8)  # Heavy local gen
            reward -= 0.5

        elif action_type == "restore_line" and param is not None:
            if self.grid.restore_line(param):
                reward += 0.3
            else:
                reward -= 0.1

        elif action_type == "inject_attack" and param is not None:
            fake_v = 1.0 + self.rng.uniform(-0.05, 0.05)
            self.grid.inject_spoof(param, fake_v)
            self.suspicion_map[param] += 0.4

        return reward

    # ------------------------------------------------------------------
    # Per-step reward computation
    # ------------------------------------------------------------------
    def _compute_reward(self, hospital_power: float) -> float:
        reward = 0.0

        if hospital_power > 0.95:
            reward += 0.5
        elif hospital_power < 0.5:
            reward -= 1.0

        thermal = self.grid.get_thermal_status()
        for ratio in thermal.values():
            if ratio > 0.8:
                reward -= 0.1

        if self.grid.cascade_count > 0:
            reward -= 2.0 * self.grid.cascade_count
            self.grid.cascade_count = 0

        return reward

    # ------------------------------------------------------------------
    # Helper: convert state to dict (for grading compatibility)
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        """Return state as a plain dict for grading functions."""
        s = self.state
        return {
            "hospital_power": s.hospital_power,
            "load_served": s.load_served,
            "detected_attacks": s.detected_attacks,
            "hospital_history": s.hospital_history,
            "actions_taken": s.actions_taken,
            "episode_reward": s.episode_reward,
            "done": s.done,
        }