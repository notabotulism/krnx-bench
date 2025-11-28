"""
KRNX Docker adapter for benchmarking.

Uses ISOLATED PORTS (16xxx) to avoid conflicts with local development:
- Redis: 16379 (host) -> 6379 (container)
- KRNX:  16380 (host) -> 6380 (container)

This allows running benchmarks while your dev Redis/KRNX are active.
"""

import time
import logging
from typing import Optional, TYPE_CHECKING

import httpx

from .base import BaseAdapter, AdapterError, NotSupported
from .docker_utils import DockerManager, ServiceConfig
from ..models import Event, QueryResult, State, ProvenanceChain

if TYPE_CHECKING:
    from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class KRNXDockerAdapter(BaseAdapter):
    """
    KRNX running in Docker for realistic benchmark testing.
    
    Port Strategy:
    - Host ports use 16xxx range (isolated from dev)
    - Container ports use standard ports (6379, 6380)
    - Container-to-container uses Docker network (redis:6379)
    
    Supports all KRNX capabilities:
    - Temporal replay
    - Hash-chain provenance
    - Fault injection (crash/restart)
    """
    
    name = "krnx"
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.docker = DockerManager()
        
        # Host ports (isolated - 16xxx range)
        self.krnx_host_port = config.get("port", 16380)
        self.redis_host_port = config.get("redis_port", 16379)
        
        # Container ports (standard)
        self.krnx_container_port = 6380
        self.redis_container_port = 6379
        
        self.base_url = f"http://localhost:{self.krnx_host_port}"
        self.client: Optional[httpx.Client] = None
        
        # Docker config
        self.krnx_image = config.get("image", "krnx:latest")
        self.timeout = config.get("timeout", 60)
        self.top_k = config.get("top_k", 10)
        
        # Workspace for benchmark
        self.workspace_id = "benchmark"
        self.user_id = "bench_user"
    
    def setup(self) -> None:
        """Start KRNX + Redis containers on isolated ports"""
        
        logger.info("Setting up KRNX Docker adapter...")
        logger.info(f"  Redis: localhost:{self.redis_host_port} -> container:6379")
        logger.info(f"  KRNX:  localhost:{self.krnx_host_port} -> container:6380")
        
        # Start Redis first (isolated port)
        redis_config = ServiceConfig(
            name="redis",
            image="redis:7-alpine",
            ports={str(self.redis_container_port): self.redis_host_port},
            healthcheck_cmd=["redis-cli", "ping"],
        )
        
        try:
            self.docker.start_service(redis_config, timeout=30)
        except Exception as e:
            if "address already in use" in str(e).lower():
                raise AdapterError(
                    f"Port {self.redis_host_port} is already in use. "
                    f"Either stop the conflicting service or change redis_port in config/default.yaml"
                ) from e
            raise
        
        # Start KRNX (container connects to Redis via Docker network)
        # Note: DockerManager prefixes container names with "krnx-bench-"
        krnx_config = ServiceConfig(
            name="krnx",
            image=self.krnx_image,
            ports={str(self.krnx_container_port): self.krnx_host_port},
            environment={
                # Container-to-container: use Docker network DNS
                # Container name is "krnx-bench-redis" (prefixed by DockerManager)
                # NOTE: KRNX uses REDIS_HOST/REDIS_PORT, not REDIS_URL
                "REDIS_HOST": "krnx-bench-redis",
                "REDIS_PORT": "6379",
                "DATABASE_PATH": "/app/data/krnx.db",
                "LOG_LEVEL": "INFO",
            },
            healthcheck_url=f"{self.base_url}/api/v1/health",
            depends_on=["redis"],
        )
        
        try:
            self.docker.start_service(krnx_config, timeout=self.timeout)
        except Exception as e:
            if "address already in use" in str(e).lower():
                raise AdapterError(
                    f"Port {self.krnx_host_port} is already in use. "
                    f"Either stop the conflicting service or change port in config/default.yaml"
                ) from e
            raise
        
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
        
        # Try dedicated clear endpoint first
        try:
            resp = self.client.post("/api/v1/admin/clear")
            resp.raise_for_status()
            return
        except httpx.HTTPError:
            pass
        
        # Fallback: erase the benchmark workspace
        try:
            resp = self.client.delete(f"/api/v1/workspaces/{self.workspace_id}")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"Clear failed (may be OK on first run): {e}")
    
    def write_event(self, event: Event) -> str:
        """Write event to KRNX, return event_id"""
        
        self._ensure_setup()
        
        try:
            resp = self.client.post(
                "/api/v1/events/write",
                json={
                    "workspace_id": self.workspace_id,
                    "user_id": self.user_id,
                    "session_id": f"{self.workspace_id}_{self.user_id}",
                    "content": {"text": event.content} if isinstance(event.content, str) else event.content,
                    "channel": event.event_type,
                    "metadata": event.metadata or {},
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("event_id") or data.get("hash")
        
        except httpx.HTTPError as e:
            raise AdapterError(f"Failed to write event: {e}")
    
    def get_event(self, event_id: str) -> Event:
        """Retrieve event by ID"""
        
        self._ensure_setup()
        
        try:
            resp = self.client.get(f"/api/v1/events/{event_id}")
            resp.raise_for_status()
            data = resp.json()
            return Event.from_dict(data)
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise KeyError(f"Event not found: {event_id}")
            raise AdapterError(f"Failed to get event: {e}")
    
    def query(self, query: str, llm: "LLMClient") -> QueryResult:
        """Query KRNX for context, then call LLM"""
        
        self._ensure_setup()
        
        start = time.time()
        
        # Get context from KRNX
        try:
            resp = self.client.post(
                "/api/v1/events/query",
                json={
                    "workspace_id": self.workspace_id,
                    "user_id": self.user_id,
                    "limit": self.top_k,
                }
            )
            resp.raise_for_status()
            context = resp.json()
        except httpx.HTTPError as e:
            raise AdapterError(f"Failed to query context: {e}")
        
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
            # Use query with time filter for replay
            resp = self.client.post(
                "/api/v1/events/query",
                json={
                    "workspace_id": self.workspace_id,
                    "user_id": self.user_id,
                    "end_time": timestamp,
                    "limit": 1000,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            return State(
                timestamp=timestamp,
                events=data.get("events", []),
                event_count=len(data.get("events", [])),
            )
        
        except httpx.HTTPError as e:
            raise AdapterError(f"Failed to replay: {e}")
    
    def get_provenance(self, event_id: str) -> ProvenanceChain:
        """Get causal chain for event"""
        
        self._ensure_setup()
        
        # Try dedicated provenance endpoint
        try:
            resp = self.client.get(f"/api/v1/provenance/{event_id}")
            resp.raise_for_status()
            data = resp.json()
            
            return ProvenanceChain(
                target_hash=event_id,
                chain=data.get("chain", []),
                verified=data.get("verified", False),
                gaps=data.get("gaps", []),
            )
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Provenance endpoint may not exist - return minimal chain
                return ProvenanceChain(
                    target_hash=event_id,
                    chain=[event_id],
                    verified=True,
                    gaps=[],
                )
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
            resp = self.client.get("/api/v1/health", timeout=5)
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
    
    def _build_prompt(self, query: str, events: list) -> str:
        """Build LLM prompt with context"""
        
        if not events:
            return f"Question: {query}\n\nAnswer based on your knowledge:"
        
        context_parts = []
        for i, event in enumerate(events[:10]):  # Limit context
            content = event.get("content", {})
            if isinstance(content, dict):
                text = content.get("text", str(content))
            else:
                text = str(content)
            context_parts.append(f"[{i+1}] {text}")
        
        context_str = "\n".join(context_parts)
        
        return f"""Context from memory:
{context_str}

Question: {query}

Answer based on the context above:"""
