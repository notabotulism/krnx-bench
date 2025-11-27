"""
Shared data models for the benchmark harness.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
import json
import uuid


@dataclass
class Event:
    """An event to be stored in the memory system"""
    
    content: str
    event_type: str = "generic"
    workspace_id: str = "default"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    parent_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()
        if "event_id" not in self.metadata:
            self.metadata["event_id"] = str(uuid.uuid4())
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "event_type": self.event_type,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "parent_hash": self.parent_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(**data)


@dataclass
class QueryResult:
    """Result of a memory query"""
    
    response: str
    context_events: list[dict]
    context_tokens: int
    query_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    total_time_ms: float = 0.0


@dataclass
class State:
    """Reconstructed state at a point in time"""
    
    timestamp: float
    events: list[Event]
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> "State":
        events = [Event.from_dict(e) for e in data.get("events", [])]
        return cls(
            timestamp=data["timestamp"],
            events=events,
            metadata=data.get("metadata", {})
        )


@dataclass
class ProvenanceChain:
    """Hash-chain provenance for an event"""
    
    target_hash: str
    chain: list[dict]
    verified: bool
    gaps: list[str] = field(default_factory=list)


@dataclass
class TrialResult:
    """Result of a single trial within a scenario"""
    
    trial_id: int
    success: bool
    metrics: dict
    raw_output: Optional[str] = None
    error: Optional[str] = None
    timing_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "success": self.success,
            "metrics": self.metrics,
            "raw_output": self.raw_output,
            "error": self.error,
            "timing_ms": self.timing_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrialResult":
        return cls(**data)


@dataclass
class ScenarioResult:
    """Aggregated result of running a scenario"""
    
    scenario_name: str
    adapter_name: str
    trials: list[TrialResult]
    aggregate: dict
    config: dict = field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "adapter_name": self.adapter_name,
            "trials": [t.to_dict() for t in self.trials],
            "aggregate": self.aggregate,
            "config": self.config,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioResult":
        trials = [TrialResult.from_dict(t) for t in data.get("trials", [])]
        return cls(
            scenario_name=data["scenario_name"],
            adapter_name=data["adapter_name"],
            trials=trials,
            aggregate=data.get("aggregate", {}),
            config=data.get("config", {}),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )
    
    def to_jsonl(self) -> str:
        """Serialize to JSONL format (one line per trial)"""
        lines = []
        for trial in self.trials:
            line = {
                "scenario": self.scenario_name,
                "adapter": self.adapter_name,
                **trial.to_dict()
            }
            lines.append(json.dumps(line))
        return "\n".join(lines)


@dataclass
class SuiteResult:
    """Result of running the complete benchmark suite"""
    
    results: list[ScenarioResult]
    started_at: str
    completed_at: str
    config: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SuiteResult":
        results = [ScenarioResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            results=results,
            started_at=data["started_at"],
            completed_at=data["completed_at"],
            config=data.get("config", {}),
        )


@dataclass
class LLMResponse:
    """Response from an LLM call"""
    
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_ms: float = 0.0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
