"""Closed-world simulator of open AI-agent networks under Sybil and eclipse attacks."""

from sybagent.config import SimConfig, load_config
from sybagent.simulation import Simulation

__all__ = ["SimConfig", "load_config", "Simulation"]
__version__ = "0.1.0"
