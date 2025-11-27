"""
Fact Correction Scenario

Tests: Consistency - Always get the current truth

This scenario tests whether the memory system returns the most recent
version of a fact after multiple updates, with distractors in between.
"""

import time
import random
import logging
from typing import Optional, Callable

from ..base import BaseScenario
from ...models import Event, TrialResult
from ...adapters.base import MemoryAdapter
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)


# Distractor topics for generating filler conversation
DISTRACTOR_TOPICS = [
    "weather", "sports", "cooking", "travel", "movies", "music",
    "technology", "science", "history", "art", "books", "games",
    "fitness", "gardening", "photography", "pets", "fashion", "food",
]

DISTRACTOR_TEMPLATES = [
    "I really enjoy {topic}. It's one of my favorite things.",
    "Have you heard the latest news about {topic}?",
    "I was thinking about {topic} the other day.",
    "My friend told me something interesting about {topic}.",
    "I read an article about {topic} recently.",
    "What do you think about {topic}?",
    "Let me tell you about my experience with {topic}.",
    "I'm planning to learn more about {topic}.",
]


class FactCorrectionScenario(BaseScenario):
    """
    Fact Correction Scenario
    
    Tests consistency by:
    1. Planting a fact (e.g., "My email is v1@test.com")
    2. Adding distractor conversation turns
    3. Updating the fact (e.g., "My email is v2@test.com")
    4. Repeating until we have N versions
    5. Querying for the current fact
    
    Success: The system returns the most recent version (vN)
    Failure: The system returns an older version or hallucinates
    
    Metrics:
    - Correct rate (returns latest version)
    - Stale rate (returns older version)
    - Hallucination rate (returns something else)
    """
    
    name = "fact_correction"
    description = "Test retrieval of most recent fact version"
    guarantee = "consistency"
    supported_adapters = ["krnx", "naive_rag", "baseline"]
    
    def __init__(self):
        super().__init__()
        self.versions = 5  # Number of fact versions
        self.distractors_per_version = 100  # Filler turns between versions
        self.fact_type = "email"  # Type of fact to test
    
    def configure(self, config: dict) -> None:
        super().configure(config)
        self.versions = config.get("versions", 5)
        self.distractors_per_version = config.get("distractors_per_version", 100)
        self.fact_type = config.get("fact_type", "email")
    
    def _run_trial(
        self,
        adapter: MemoryAdapter,
        llm: LLMClient,
        trial_id: int,
    ) -> TrialResult:
        """Run a single fact correction trial"""
        
        start = time.time()
        
        # Generate fact versions
        fact_values = self._generate_fact_values()
        
        # Plant facts with distractors
        events_written = 0
        version_timestamps = []
        
        for i, value in enumerate(fact_values):
            # Record timestamp of this version
            version_timestamps.append(time.time())
            
            # Write the fact
            fact_event = Event(
                content=self._format_fact(value),
                event_type="fact_update",
                metadata={
                    "fact_type": self.fact_type,
                    "version": i + 1,
                    "value": value,
                },
            )
            adapter.write_event(fact_event)
            events_written += 1
            
            # Add distractors (except after last version)
            if i < len(fact_values) - 1:
                for _ in range(self.distractors_per_version):
                    distractor = Event(
                        content=self._generate_distractor(),
                        event_type="conversation",
                    )
                    adapter.write_event(distractor)
                    events_written += 1
        
        write_time = (time.time() - start) * 1000
        
        # Query for the current fact
        query = self._format_query()
        
        query_start = time.time()
        result = adapter.query(query, llm)
        query_time = (time.time() - query_start) * 1000
        
        # Grade the response
        grade, matched_version = self._grade_response(result.response, fact_values)
        
        total_time = (time.time() - start) * 1000
        
        return TrialResult(
            trial_id=trial_id,
            success=(grade == "correct"),
            metrics={
                "grade": grade,
                "expected_value": fact_values[-1],
                "matched_version": matched_version,
                "versions": self.versions,
                "distractors_per_version": self.distractors_per_version,
                "events_written": events_written,
                "context_events": len(result.context_events),
                "context_tokens": result.context_tokens,
                "write_time_ms": write_time,
                "query_time_ms": query_time,
            },
            raw_output=result.response,
            timing_ms=total_time,
        )
    
    def _generate_fact_values(self) -> list[str]:
        """Generate fact values for each version"""
        
        if self.fact_type == "email":
            return [f"user_v{i}@example.com" for i in range(1, self.versions + 1)]
        elif self.fact_type == "phone":
            return [f"555-000-{i:04d}" for i in range(1, self.versions + 1)]
        elif self.fact_type == "address":
            return [f"{i * 100} Main Street, City {i}" for i in range(1, self.versions + 1)]
        elif self.fact_type == "budget":
            return [f"${i * 10000}" for i in range(1, self.versions + 1)]
        else:
            return [f"value_v{i}" for i in range(1, self.versions + 1)]
    
    def _format_fact(self, value: str) -> str:
        """Format a fact as a natural statement"""
        
        templates = {
            "email": f"My email address is {value}",
            "phone": f"My phone number is {value}",
            "address": f"My address is {value}",
            "budget": f"My budget is {value}",
        }
        return templates.get(self.fact_type, f"The value is {value}")
    
    def _format_query(self) -> str:
        """Format the query for the fact"""
        
        queries = {
            "email": "What is my email address?",
            "phone": "What is my phone number?",
            "address": "What is my address?",
            "budget": "What is my budget?",
        }
        return queries.get(self.fact_type, "What is the value?")
    
    def _generate_distractor(self) -> str:
        """Generate a random distractor message"""
        
        topic = random.choice(DISTRACTOR_TOPICS)
        template = random.choice(DISTRACTOR_TEMPLATES)
        return template.format(topic=topic)
    
    def _grade_response(
        self, 
        response: str, 
        fact_values: list[str]
    ) -> tuple[str, Optional[int]]:
        """
        Grade the response.
        
        Returns:
            Tuple of (grade, matched_version)
            grade: "correct" | "stale" | "hallucinated"
            matched_version: Which version was found (None if hallucinated)
        """
        
        response_lower = response.lower()
        
        # Check for latest version (correct)
        latest = fact_values[-1].lower()
        if latest in response_lower:
            return "correct", len(fact_values)
        
        # Check for older versions (stale)
        for i, value in enumerate(fact_values[:-1]):
            if value.lower() in response_lower:
                return "stale", i + 1
        
        # No match (hallucinated)
        return "hallucinated", None
    
    def _compute_aggregate(self, results: list[TrialResult]) -> dict:
        """Compute aggregate statistics"""
        
        if not results:
            return {}
        
        grades = [r.metrics.get("grade", "error") for r in results]
        
        correct = grades.count("correct")
        stale = grades.count("stale")
        hallucinated = grades.count("hallucinated")
        errors = len(grades) - correct - stale - hallucinated
        
        # Timing stats
        timings = [r.timing_ms for r in results if r.timing_ms > 0]
        query_times = [r.metrics.get("query_time_ms", 0) for r in results]
        
        # Version distribution for stale results
        stale_versions = [
            r.metrics.get("matched_version") 
            for r in results 
            if r.metrics.get("grade") == "stale"
        ]
        
        return {
            "correct_rate": correct / len(results),
            "stale_rate": stale / len(results),
            "hallucination_rate": hallucinated / len(results),
            "error_rate": errors / len(results),
            "total_trials": len(results),
            "correct_count": correct,
            "stale_count": stale,
            "hallucinated_count": hallucinated,
            "mean_timing_ms": sum(timings) / len(timings) if timings else 0,
            "mean_query_time_ms": sum(query_times) / len(query_times) if query_times else 0,
            "stale_version_distribution": self._count_versions(stale_versions),
        }
    
    def _count_versions(self, versions: list[Optional[int]]) -> dict[int, int]:
        """Count occurrences of each stale version"""
        counts: dict[int, int] = {}
        for v in versions:
            if v is not None:
                counts[v] = counts.get(v, 0) + 1
        return counts
