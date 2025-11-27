"""
Base adapter protocol and exceptions.
"""

from typing import Protocol, runtime_checkable, Optional, Callable, TYPE_CHECKING
from abc import abstractmethod

if TYPE_CHECKING:
    from ..models import Event, QueryResult, State, ProvenanceChain
    from ..llm.client import LLMClient


class NotSupported(Exception):
    """Raised when an adapter doesn't support an operation"""
    pass


class AdapterError(Exception):
    """Raised when an adapter encounters an error"""
    pass


@runtime_checkable
class MemoryAdapter(Protocol):
    """
    Common interface for memory systems under test.
    
    All adapters must implement the core operations.
    Extended operations may raise NotSupported for adapters that don't support them.
    """
    
    name: str
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    @abstractmethod
    def setup(self) -> None:
        """
        Initialize the adapter and any required infrastructure.
        For Docker adapters, this starts containers and waits for health.
        """
        ...
    
    @abstractmethod
    def teardown(self) -> None:
        """
        Clean up the adapter and release resources.
        For Docker adapters, this stops and removes containers.
        """
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all data (between trials).
        Should be fast - used to reset state between test runs.
        """
        ...
    
    # =========================================================================
    # Core Operations (all adapters must implement)
    # =========================================================================
    
    @abstractmethod
    def write_event(self, event: "Event") -> str:
        """
        Write an event to the memory system.
        
        Args:
            event: The event to store
            
        Returns:
            A unique identifier (hash) for the stored event
        """
        ...
    
    @abstractmethod
    def query(self, query: str, llm: "LLMClient") -> "QueryResult":
        """
        Query the memory system and generate a response.
        
        This is the main retrieval + generation operation:
        1. Retrieve relevant context from the memory system
        2. Build a prompt with the context
        3. Call the LLM to generate a response
        
        Args:
            query: The user's question
            llm: LLM client to use for generation
            
        Returns:
            QueryResult with response and metadata
        """
        ...
    
    # =========================================================================
    # Extended Operations (may raise NotSupported)
    # =========================================================================
    
    def get_event(self, event_hash: str) -> "Event":
        """
        Retrieve an event by its hash.
        
        Args:
            event_hash: The hash returned from write_event
            
        Returns:
            The stored event
            
        Raises:
            NotSupported: If adapter doesn't support direct retrieval
            KeyError: If event not found
        """
        raise NotSupported(f"{self.name} does not support get_event")
    
    def replay_to(self, timestamp: float) -> "State":
        """
        Reconstruct state at a specific point in time.
        
        Args:
            timestamp: Unix timestamp to replay to
            
        Returns:
            State object representing the reconstructed state
            
        Raises:
            NotSupported: If adapter doesn't support temporal replay
        """
        raise NotSupported(f"{self.name} does not support temporal replay")
    
    def get_provenance(self, event_hash: str) -> "ProvenanceChain":
        """
        Get the causal chain (provenance) for an event.
        
        Args:
            event_hash: Hash of the event to trace
            
        Returns:
            ProvenanceChain with the full hash chain
            
        Raises:
            NotSupported: If adapter doesn't support provenance
        """
        raise NotSupported(f"{self.name} does not support provenance tracking")
    
    # =========================================================================
    # Fault Injection (for durability tests)
    # =========================================================================
    
    def kill(self) -> None:
        """
        Forcefully terminate the memory system (SIGKILL).
        Used for crash recovery testing.
        
        Raises:
            NotSupported: If adapter doesn't support fault injection
        """
        raise NotSupported(f"{self.name} does not support fault injection")
    
    def restart(self) -> None:
        """
        Restart the memory system after a kill.
        
        Raises:
            NotSupported: If adapter doesn't support fault injection
        """
        raise NotSupported(f"{self.name} does not support restart")
    
    def is_alive(self) -> bool:
        """
        Check if the memory system is responding.
        
        Returns:
            True if healthy, False otherwise
        """
        return True
    
    # =========================================================================
    # Capabilities
    # =========================================================================
    
    def supports(self, capability: str) -> bool:
        """
        Check if this adapter supports a capability.
        
        Capabilities:
        - "replay": Temporal replay to any timestamp
        - "provenance": Hash-chain provenance tracking
        - "fault_injection": Kill/restart for crash testing
        - "versioning": Multiple versions of same fact
        
        Returns:
            True if supported, False otherwise
        """
        return False


class BaseAdapter:
    """
    Base class with common functionality for adapters.
    
    Subclasses should override the abstract methods and optionally
    override extended methods if they support those capabilities.
    """
    
    name: str = "base"
    
    def __init__(self, config: dict):
        self.config = config
        self._setup_complete = False
    
    def __enter__(self):
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
        return False
    
    def _ensure_setup(self):
        """Raise error if setup() wasn't called"""
        if not self._setup_complete:
            raise AdapterError(f"{self.name} adapter not set up. Call setup() first.")
    
    def _build_prompt(self, query: str, context_events: list[dict]) -> str:
        """Build a standard prompt with retrieved context"""
        
        if not context_events:
            return f"Question: {query}\n\nAnswer:"
        
        context_lines = []
        for event in context_events:
            timestamp = event.get("timestamp", "")
            content = event.get("content", "")
            if timestamp:
                context_lines.append(f"[{timestamp}] {content}")
            else:
                context_lines.append(content)
        
        context_str = "\n".join(context_lines)
        
        return f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {query}

Answer:"""
