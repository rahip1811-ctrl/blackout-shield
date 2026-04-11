"""
BlackoutShield Client v3.4
Async + sync HTTP client for interacting with the environment server.
Also supports embedded (in-process) mode.
"""

import os
import asyncio
from typing import Optional, Dict, Any

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    try:
        import requests
        HAS_HTTPX = False
    except ImportError:
        HAS_HTTPX = False

from models import BlackoutAction, BlackoutObservation


class BlackoutShieldEnv:
    """
    Client for BlackoutShield environment.
    
    Usage (HTTP):
        env = BlackoutShieldEnv(base_url="http://localhost:8000")
        obs = env.reset(task_id=0)
        obs = env.step(BlackoutAction(action_id=10))
    
    Usage (Embedded):
        env = BlackoutShieldEnv.embedded()
        obs = env.reset(task_id=0)
        obs = env.step(BlackoutAction(action_id=10))
    """

    def __init__(self, base_url: str = "https://rahictrl-blackout-shield.hf.space"):
        self.base_url = base_url.rstrip("/")
        self._embedded_env = None
        self._session = None

    @classmethod
    def embedded(cls):
        """Create an embedded (in-process) environment — no HTTP needed."""
        from server.blackout_environment import BlackoutEnvironment
        instance = cls.__new__(cls)
        instance.base_url = None
        instance._embedded_env = BlackoutEnvironment(seed=42)
        instance._session = None
        return instance

    def reset(self, task_id: int = 0, seed: int = 42) -> BlackoutObservation:
        """Reset environment and start new episode."""
        if self._embedded_env:
            return self._embedded_env.reset(task_id=task_id)
        
        import requests
        resp = requests.post(f"{self.base_url}/reset", params={"task_id": task_id, "seed": seed})
        resp.raise_for_status()
        data = resp.json()
        return BlackoutObservation(**data.get("observation", data))

    def step(self, action: BlackoutAction) -> BlackoutObservation:
        """Execute action in environment."""
        if self._embedded_env:
            return self._embedded_env.step(action)
        
        import requests
        resp = requests.post(f"{self.base_url}/step", json={"action_id": action.action_id})
        resp.raise_for_status()
        data = resp.json()
        return BlackoutObservation(**data.get("observation", data))

    def get_state(self) -> Dict[str, Any]:
        """Get full environment state."""
        if self._embedded_env:
            s = self._embedded_env.state
            return s.model_dump() if hasattr(s, 'model_dump') else s.__dict__
        
        import requests
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def get_legal_actions(self):
        """Get list of legal action IDs."""
        if self._embedded_env:
            return self._embedded_env.get_legal_actions()
        
        import requests
        resp = requests.get(f"{self.base_url}/legal-actions")
        resp.raise_for_status()
        return resp.json().get("legal_actions", [])

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        if self._embedded_env:
            return {"status": "healthy", "mode": "embedded"}
        
        import requests
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Clean up resources."""
        self._embedded_env = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()