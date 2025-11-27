"""
Naive RAG adapter using Qdrant for benchmarking.

This represents the "industry standard" approach:
- Simple vector store
- Top-k retrieval by embedding similarity
- No temporal awareness, no provenance
"""

import time
import uuid
import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct,
    Filter,
)

from .base import BaseAdapter, AdapterError, NotSupported
from .docker_utils import DockerManager, ServiceConfig
from ..models import Event, QueryResult

logger = logging.getLogger(__name__)


class NaiveRAGAdapter(BaseAdapter):
    """
    Simple vector store RAG baseline using Qdrant.
    
    This adapter represents what most production systems use today:
    - Embed content with OpenAI
    - Store in Qdrant
    - Retrieve top-k by cosine similarity
    
    Limitations (by design):
    - No temporal awareness
    - No provenance tracking
    - No versioning (overwrites based on embedding similarity)
    - Loses everything on crash (no persistence configured)
    """
    
    name = "naive_rag"
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.docker = DockerManager()
        self.qdrant: Optional[QdrantClient] = None
        
        # Config
        self.port = config.get("port", 6333)
        self.collection_name = "benchmark"
        self.embedding_dim = 1536  # OpenAI ada-002
        self.top_k = config.get("top_k", 10)
        self.timeout = config.get("timeout", 30)
        
        # Embedding client (lazy init)
        self._embedding_client = None
    
    @property
    def embedding_client(self):
        """Lazy initialization of OpenAI client for embeddings"""
        if self._embedding_client is None:
            from openai import OpenAI
            self._embedding_client = OpenAI()
        return self._embedding_client
    
    def setup(self) -> None:
        """Start Qdrant container"""
        
        logger.info("Setting up Naive RAG adapter (Qdrant)...")
        
        # Start Qdrant
        qdrant_config = ServiceConfig(
            name="qdrant",
            image="qdrant/qdrant:latest",
            ports={"6333": self.port},
            healthcheck_url=f"http://localhost:{self.port}/health",
        )
        self.docker.start_service(qdrant_config, timeout=self.timeout)
        
        # Connect client
        self.qdrant = QdrantClient(host="localhost", port=self.port)
        
        # Create collection
        self._create_collection()
        
        self._setup_complete = True
        logger.info("Naive RAG adapter ready")
    
    def teardown(self) -> None:
        """Stop Qdrant container"""
        
        logger.info("Tearing down Naive RAG adapter...")
        
        self.qdrant = None
        self.docker.cleanup_all()
        self._setup_complete = False
    
    def clear(self) -> None:
        """Clear collection between trials"""
        
        self._ensure_setup()
        
        try:
            self.qdrant.delete_collection(self.collection_name)
        except:
            pass
        
        self._create_collection()
    
    def _create_collection(self) -> None:
        """Create the benchmark collection"""
        
        try:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        except Exception as e:
            # Collection might already exist
            logger.debug(f"Collection creation: {e}")
    
    def write_event(self, event: Event) -> str:
        """Embed and store in Qdrant"""
        
        self._ensure_setup()
        
        # Get embedding
        embedding = self._embed(event.content)
        
        # Generate ID
        event_id = str(uuid.uuid4())
        
        # Store
        try:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=event_id,
                    vector=embedding,
                    payload={
                        "content": event.content,
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "workspace_id": event.workspace_id,
                        "user_id": event.user_id,
                        "metadata": event.metadata,
                    }
                )]
            )
        except Exception as e:
            raise AdapterError(f"Failed to write to Qdrant: {e}")
        
        return event_id
    
    def query(self, query: str, llm: "LLMClient") -> QueryResult:
        """Standard RAG: embed query, find similar, call LLM"""
        
        self._ensure_setup()
        
        start = time.time()
        
        # Embed query
        query_embedding = self._embed(query)
        
        # Search
        try:
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.top_k,
            )
        except Exception as e:
            raise AdapterError(f"Failed to search Qdrant: {e}")
        
        query_time = (time.time() - start) * 1000
        
        # Build context
        events = [r.payload for r in results]
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
    
    def _embed(self, text: str) -> list[float]:
        """Get embedding from OpenAI"""
        
        response = self.embedding_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return response.data[0].embedding
    
    # =========================================================================
    # Unsupported operations
    # =========================================================================
    
    def get_event(self, event_hash: str) -> Event:
        """Retrieve by ID (limited support)"""
        
        self._ensure_setup()
        
        try:
            results = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=[event_hash],
            )
            
            if not results:
                raise KeyError(f"Event not found: {event_hash}")
            
            payload = results[0].payload
            return Event(
                content=payload["content"],
                event_type=payload.get("event_type", "generic"),
                timestamp=payload.get("timestamp"),
                workspace_id=payload.get("workspace_id", "default"),
                user_id=payload.get("user_id"),
                metadata=payload.get("metadata", {}),
            )
        
        except Exception as e:
            if "not found" in str(e).lower():
                raise KeyError(f"Event not found: {event_hash}")
            raise AdapterError(f"Failed to retrieve event: {e}")
    
    def replay_to(self, timestamp: float):
        raise NotSupported("Naive RAG does not support temporal replay")
    
    def get_provenance(self, event_hash: str):
        raise NotSupported("Naive RAG does not support provenance tracking")
    
    # Fault injection - limited support
    def kill(self) -> None:
        """Kill Qdrant container"""
        logger.info("Killing Qdrant container...")
        self.docker.kill_service("qdrant")
    
    def restart(self) -> None:
        """Restart Qdrant (loses all data!)"""
        logger.info("Restarting Qdrant container (data will be lost)...")
        self.docker.restart_service("qdrant", timeout=self.timeout)
        
        # Reconnect
        self.qdrant = QdrantClient(host="localhost", port=self.port)
        self._create_collection()
    
    def is_alive(self) -> bool:
        """Check if Qdrant is responding"""
        
        if not self.docker.is_service_alive("qdrant"):
            return False
        
        try:
            self.qdrant.get_collections()
            return True
        except:
            return False
    
    def supports(self, capability: str) -> bool:
        """Naive RAG has limited capabilities"""
        
        # Only supports basic fault injection (though it loses data)
        return capability == "fault_injection"
