"""
BlackoutShield Models v3.4
Pydantic models with reasoning fields for LLM Chain-of-Thought.
Works with or without openenv-core.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ============ ACTIONS ============

class BlackoutAction(BaseModel):
    """Agent action: integer 0-21 mapping to grid defense commands."""
    action_id: int = Field(..., description="Action ID (0-21)")

    class Config:
        json_schema_extra = {"example": {"action_id": 10}}


# ============ OBSERVATIONS ============

class BlackoutObservation(BaseModel):
    """Enhanced observation with reasoning guidance for LLM agents."""

    # Core grid state
    scada_voltages: List[float] = Field(default_factory=lambda: [1.0]*5,
        description="Bus voltages (may be SPOOFED) [5 dims]")
    telemetry_confidence: float = Field(default=1.0,
        description="SCADA trust score (0-1, low=spoofing active)")
    line_thermal: List[float] = Field(default_factory=lambda: [0.0]*4,
        description="Thermal load per line (>1.0=overloaded) [4 dims]")
    line_status: List[float] = Field(default_factory=lambda: [1.0]*4,
        description="Line status (1=active, 0=tripped) [4 dims]")
    bus_alerts: List[float] = Field(default_factory=lambda: [0.0]*5,
        description="IDS voltage alerts per bus [5 dims]")
    line_alerts: List[float] = Field(default_factory=lambda: [0.0]*4,
        description="Thermal overload alerts per line [4 dims]")
    suspicion_map: List[float] = Field(default_factory=lambda: [0.0]*5,
        description="Attack probability per bus (0-1) [5 dims]")
    hospital_power: float = Field(default=1.0,
        description="Hospital voltage (must stay > 0.95)")
    load_served: float = Field(default=1.0,
        description="Fraction of load being served (0-1)")
    time_step: float = Field(default=0.0,
        description="Episode progress (0-1)")

    # Reasoning fields for LLM Chain-of-Thought
    recent_events: str = Field(default="",
        description="Text description of recent grid state changes and anomalies")
    action_guidance: str = Field(default="",
        description="Natural language priority suggestions for LLM reasoning")

    # Task context
    task_description: str = Field(default="",
        description="Human-readable scenario description")
    available_actions: List[str] = Field(default_factory=list,
        description="List of legal action names available this step")

    # Physics context for advanced reasoning
    physics_context: Optional[Dict[str, Any]] = Field(default=None,
        description="Grid physics: total_generation, total_load, cascade_risk, etc.")

    # OpenEnv compatibility
    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============ STEP / RESET RESULTS ============

class BlackoutStepResult(BaseModel):
    observation: BlackoutObservation
    reward: float = Field(..., description="Reward for this step")
    done: bool = Field(..., description="Episode finished")
    info: Dict[str, Any] = Field(default_factory=dict)


class BlackoutResetResult(BaseModel):
    observation: BlackoutObservation
    info: Dict[str, Any] = Field(default_factory=dict)


# ============ STATE ============

class BlackoutState(BaseModel):
    """Full internal state for debugging and visualization."""
    episode_id: str = ""
    step_count: int = 0
    buses: Dict = Field(default_factory=dict)
    lines: Dict = Field(default_factory=dict)
    suspicion_map: List[float] = Field(default_factory=lambda: [0.0]*5)
    hospital_power: float = 1.0
    load_served: float = 1.0
    detected_attacks: int = 0
    cascade_count: int = 0
    hospital_history: List[float] = Field(default_factory=list)
    fog_of_war: Dict = Field(default_factory=dict)
    actions_taken: List[str] = Field(default_factory=list)
    episode_reward: float = 0.0
    task_id: int = 0
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)