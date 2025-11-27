"""
Benchmark scenarios.
"""

from .base import Scenario
from .durability.crash_recovery import CrashRecoveryScenario
from .consistency.fact_correction import FactCorrectionScenario
from .consistency.temporal_versioning import TemporalVersioningScenario
from .auditability.provenance_chain import ProvenanceChainScenario
from .replay.point_in_time import PointInTimeScenario
from .replay.determinism import DeterminismScenario
from .baseline.niah import NIAHScenario

# Organized by guarantee
SCENARIOS = {
    "durability": [
        CrashRecoveryScenario,
    ],
    "consistency": [
        FactCorrectionScenario,
        TemporalVersioningScenario,
    ],
    "auditability": [
        ProvenanceChainScenario,
    ],
    "replay": [
        PointInTimeScenario,
        DeterminismScenario,
    ],
    "baseline": [
        NIAHScenario,
    ],
}

# Flat list of all scenarios
ALL_SCENARIOS = [
    CrashRecoveryScenario,
    FactCorrectionScenario,
    TemporalVersioningScenario,
    ProvenanceChainScenario,
    PointInTimeScenario,
    DeterminismScenario,
    NIAHScenario,
]

# Name -> class mapping
SCENARIO_MAP = {s.name: s for s in ALL_SCENARIOS}


def get_scenario(name: str) -> type:
    """Get a scenario class by name"""
    if name not in SCENARIO_MAP:
        raise KeyError(f"Unknown scenario: {name}. Available: {list(SCENARIO_MAP.keys())}")
    return SCENARIO_MAP[name]


def list_scenarios() -> list[str]:
    """List all available scenario names"""
    return list(SCENARIO_MAP.keys())


__all__ = [
    "Scenario",
    "SCENARIOS",
    "ALL_SCENARIOS",
    "get_scenario",
    "list_scenarios",
    "CrashRecoveryScenario",
    "FactCorrectionScenario",
    "TemporalVersioningScenario",
    "ProvenanceChainScenario",
    "PointInTimeScenario",
    "DeterminismScenario",
    "NIAHScenario",
]
