"""
KRNX Docker adapter for benchmarking.
"""

import time
import logging
from typing import Optional

import httpx

from .base import BaseAdapter, AdapterError, NotSupported
from .docker_utils import DockerManager, ServiceConfig
from ..models import Event, QueryResult, State, ProvenanceChain

logger = logging.getLogger(__name__)


class KRNXDockerAdapter(BaseAdapter):
    """
    KRNX running in Docker for realistic benchmark testing.
    
    This adapter starts the full KRNX stack (Redis + KRNX server) in Docker
    and communicates via HTTP API.
    
    Supports all KRNX capabilities:
    - Temporal replay
    - Hash-chain provenance
    - Fault injection (crash/restart)
    """
    
    name = "krnx"
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.docker = DockerManager()
        self.base_url = f"http://localhost:{config.get('port', 8100)}"
        self.client: Optional[httpx.Client] = None
        
        # Docker config
        self.krnx_image = config.get("image", "krnx:latest")
        self.krnx_port = config.get("port", 8100)
        self.redis_port = config.get("redis_port", 6379)
        self.timeout = config.get("timeout", 60)
    
    def setup(self) -> None:
        """Start KRNX + Redis containers"""
        
        logger.info("Setting up KRNX Docker adapter...")
        
        # Start Redis first
        redis_config = ServiceConfig(
            name="redis",
            image="redis/redis-stack:latest",
            ports={"6379": self.redis_port},
            healthcheck_cmd=["redis-cli", "ping"],
        )
        self.docker.start_service(redis_config, timeout=30)
        
        # Start KRNX
        krnx_config = ServiceConfig(
            name="krnx",
            image=self.krnx_image,
            ports={"8000": self.krnx_port},
            environment={
                "KRNX_REDIS_URL": "redis://redis:6379",
                "KRNX_SQLITE_PATH": "/data/krnx.db",
            },
            healthcheck_url=f"{self.base_url}/health",
            depends_on=["redis"],
        )
        self.docker.start_service(krnx_config, timeout=self.timeout)
        
        # Create HTTP client
        self.client = httpx.Client(base_url=self.base_url, timeout=30)
        
        self._setup_complete = True
        logger.info("KRNX Docker adapter ready")
    
    def teardown(self) -> None:
        """Stop all containers"""
        
        logger.info("Tearing down KRNX Docker adapter...")
        
        if self.client:
            self.client.close()
            self.client = None
        
        self.docker.cleanup_all()
        self._setup_complete = False
    
    def clear(self) -> None:
        """Clear all data between trials"""
        
        self._ensure_setup()
        
        try:
            resp = self.client.post("/admin/clear")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            # If endpoint doesn't exist, try alternative methods
            logger.warning(f"Clear endpoint failed: {e}. Attempting restart...")
            self.kill()
            self.restart()
    
    def write_event(self, event: Event) -> str:
        """Write event to KRNX, return hash"""
        
        self._ensure_setup()
        
        try:
            resp = self.client.post("/events", json=event.to_dict())
            resp.raise_for_status()
            data = resp.json()
            return data.get("hash") or data.get("event_id")
        
        except httpx.HTTPError as e:
            raise AdapterError(f"Failed to write event: {e}")
    
    def get_event(self, event_hash: str) -> Event:
        """Retrieve event by hash"""
        
        self._ensure_setup()
        
        try:
            resp = self.client.get(f"/events/{event_hash}")
            resp.raise_for_status()
            return Event.from_dict(resp.json())
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise KeyError(f"Event not found: {event_hash}")
            raise AdapterError(f"Failed to get event: {e}")
    
    def query(self, query: str, llm: "LLMClient") -> QueryResult:
        """Query KRNX for context, then call LLM"""
        
        self._ensure_setup()
        
        start = time.time()
        
        # Get context from KRNX
        try:
            resp = self.client.post("/recall", json={
                "query": query,
                "top_k": self.config.get("top_k", 10),
            })
            resp.raise_for_status()
            context = resp.json()
        except httpx.HTTPError as e:
            raise AdapterError(f"Failed to recall context: {e}")
        
        query_time = (time.time() - start) * 1000
        
        # Build prompt with context
        events = context.get("events", [])
        prompt = self._build_prompt(query, events)
        
        # Call LLM
        llm_start = time.time()
        response = llm.complete(prompt)
        llm_time = (time.time() - llm_start) * 1000
        
        return QueryResult(
            response=response.text,
            context_events=events,
            context_tokens=response.prompt_tokens,
            query_time_ms=query_time,
            llm_time_ms=llm_time,
            total_time_ms=(time.time() - start) * 1000,
        )
    
    def replay_to(self, timestamp: float) -> State:
        """Reconstruct state at timestamp"""
        
        self._ensure_setup()
        
        try:
            resp = self.client.post("/replay", json={"timestamp": timestamp})
            resp.raise_for_status()
            return State.from_dict(resp.json())
        
        except httpx.HTTPError as e:
            raise AdapterError(f"Failed to replay: {e}")
    
    def get_provenance(self, event_hash: str) -> ProvenanceChain:
        """Get causal chain for event"""
        
        self._ensure_setup()
        
        try:
            resp = self.client.get(f"/provenance/{event_hash}")
            resp.raise_for_status()
            data = resp.json()
            
            return ProvenanceChain(
                target_hash=event_hash,
                chain=data.get("chain", []),
                verified=data.get("verified", False),
                gaps=data.get("gaps", []),
            )
        
        except httpx.HTTPError as e:
            raise AdapterError(f"Failed to get provenance: {e}")
    
    def kill(self) -> None:
        """SIGKILL the KRNX container"""
        
        logger.info("Killing KRNX container...")
        self.docker.kill_service("krnx")
    
    def restart(self) -> None:
        """Restart KRNX after kill"""
        
        logger.info("Restarting KRNX container...")
        self.docker.restart_service("krnx", timeout=self.timeout)
        
        # Recreate HTTP client
        if self.client:
            self.client.close()
        self.client = httpx.Client(base_url=self.base_url, timeout=30)
    
    def is_alive(self) -> bool:
        """Check if KRNX is responding"""
        
        if not self.docker.is_service_alive("krnx"):
            return False
        
        try:
            resp = self.client.get("/health", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def supports(self, capability: str) -> bool:
        """KRNX supports all capabilities"""
        
        return capability in {
            "replay",
            "provenance", 
            "fault_injection",
            "versioning",
        }
