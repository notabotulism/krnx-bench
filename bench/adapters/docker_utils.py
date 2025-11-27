"""
Docker container management utilities for benchmark adapters.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

import docker
from docker.models.containers import Container
from docker.models.networks import Network
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a Docker service"""
    
    name: str
    image: str
    ports: dict[str, int]  # container_port -> host_port
    environment: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, dict] = field(default_factory=dict)  # host_path -> {"bind": path, "mode": "rw"}
    healthcheck_url: Optional[str] = None
    healthcheck_cmd: Optional[list[str]] = None
    depends_on: list[str] = field(default_factory=list)
    network: Optional[str] = None


class DockerManager:
    """
    Manages Docker containers for benchmark adapters.
    
    Handles container lifecycle, health checking, and fault injection.
    """
    
    def __init__(self, network_name: str = "krnx-bench-network"):
        self.client = docker.from_env()
        self.network_name = network_name
        self.containers: dict[str, Container] = {}
        self.service_configs: dict[str, ServiceConfig] = {}
        self._network: Optional[Network] = None
    
    def ensure_network(self) -> Network:
        """Create or get the benchmark network"""
        
        if self._network is not None:
            return self._network
        
        try:
            self._network = self.client.networks.get(self.network_name)
            logger.debug(f"Using existing network: {self.network_name}")
        except docker.errors.NotFound:
            self._network = self.client.networks.create(
                self.network_name,
                driver="bridge"
            )
            logger.info(f"Created network: {self.network_name}")
        
        return self._network
    
    def start_service(
        self,
        config: ServiceConfig,
        timeout: int = 60
    ) -> Container:
        """
        Start a container and wait for it to be healthy.
        
        Args:
            config: Service configuration
            timeout: Seconds to wait for health
            
        Returns:
            Running container
            
        Raises:
            TimeoutError: If health check fails
            docker.errors.APIError: If container fails to start
        """
        
        container_name = f"krnx-bench-{config.name}"
        
        # Remove existing container if present
        try:
            existing = self.client.containers.get(container_name)
            logger.debug(f"Removing existing container: {container_name}")
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass
        
        # Ensure network exists
        network = self.ensure_network()
        
        # Build port bindings
        port_bindings = {}
        for container_port, host_port in config.ports.items():
            # Ensure port format is correct
            if "/" not in container_port:
                container_port = f"{container_port}/tcp"
            port_bindings[container_port] = host_port
        
        # Start container
        logger.info(f"Starting container: {container_name} (image: {config.image})")
        
        container = self.client.containers.run(
            image=config.image,
            name=container_name,
            ports=port_bindings,
            environment=config.environment,
            volumes=config.volumes or None,
            network=self.network_name,
            detach=True,
            remove=False,  # We'll remove manually for inspection
        )
        
        self.containers[config.name] = container
        self.service_configs[config.name] = config
        
        # Wait for health
        if config.healthcheck_url:
            self._wait_for_http_health(config.healthcheck_url, timeout)
        elif config.healthcheck_cmd:
            self._wait_for_cmd_health(container, config.healthcheck_cmd, timeout)
        else:
            # Basic wait for container to be running
            self._wait_for_running(container, timeout)
        
        logger.info(f"Container healthy: {container_name}")
        return container
    
    def stop_service(self, name: str, timeout: int = 10) -> None:
        """Gracefully stop a container"""
        
        if name not in self.containers:
            return
        
        container = self.containers[name]
        logger.info(f"Stopping container: {name}")
        
        try:
            container.stop(timeout=timeout)
        except Exception as e:
            logger.warning(f"Error stopping container {name}: {e}")
            container.kill()
        
        del self.containers[name]
    
    def kill_service(self, name: str) -> None:
        """
        SIGKILL a container (for crash testing).
        
        This simulates an unexpected crash - no graceful shutdown.
        """
        
        if name not in self.containers:
            raise ValueError(f"No container with name: {name}")
        
        container = self.containers[name]
        logger.info(f"Killing container (SIGKILL): {name}")
        
        container.kill(signal="SIGKILL")
        
        # Don't remove from tracking - we'll restart it
    
    def restart_service(self, name: str, timeout: int = 60) -> Container:
        """
        Restart a killed container.
        
        Uses the original configuration to start a fresh container.
        """
        
        if name not in self.service_configs:
            raise ValueError(f"No config for service: {name}")
        
        config = self.service_configs[name]
        
        # Remove the dead container
        if name in self.containers:
            try:
                self.containers[name].remove(force=True)
            except:
                pass
            del self.containers[name]
        
        # Start fresh
        return self.start_service(config, timeout)
    
    def is_service_alive(self, name: str) -> bool:
        """Check if a service is running and healthy"""
        
        if name not in self.containers:
            return False
        
        container = self.containers[name]
        
        try:
            container.reload()
            return container.status == "running"
        except:
            return False
    
    def get_service_logs(self, name: str, tail: int = 100) -> str:
        """Get recent logs from a container"""
        
        if name not in self.containers:
            return ""
        
        container = self.containers[name]
        return container.logs(tail=tail).decode("utf-8")
    
    def cleanup_all(self) -> None:
        """Stop and remove all benchmark containers"""
        
        logger.info("Cleaning up all containers...")
        
        for name in list(self.containers.keys()):
            try:
                container = self.containers[name]
                container.stop(timeout=5)
                container.remove()
                logger.debug(f"Removed container: {name}")
            except Exception as e:
                logger.warning(f"Error cleaning up {name}: {e}")
                try:
                    self.containers[name].remove(force=True)
                except:
                    pass
        
        self.containers.clear()
        
        # Optionally remove network
        if self._network:
            try:
                self._network.remove()
                logger.debug(f"Removed network: {self.network_name}")
            except:
                pass
            self._network = None
    
    def _wait_for_http_health(self, url: str, timeout: int) -> None:
        """Poll HTTP endpoint until healthy"""
        
        start = time.time()
        last_error = None
        
        while time.time() - start < timeout:
            try:
                resp = httpx.get(url, timeout=5)
                if resp.status_code == 200:
                    return
                last_error = f"HTTP {resp.status_code}"
            except httpx.ConnectError:
                last_error = "Connection refused"
            except httpx.TimeoutException:
                last_error = "Timeout"
            except Exception as e:
                last_error = str(e)
            
            time.sleep(1)
        
        raise TimeoutError(
            f"Service not healthy after {timeout}s. Last error: {last_error}"
        )
    
    def _wait_for_cmd_health(
        self, 
        container: Container, 
        cmd: list[str], 
        timeout: int
    ) -> None:
        """Run health check command in container"""
        
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                result = container.exec_run(cmd)
                if result.exit_code == 0:
                    return
            except:
                pass
            
            time.sleep(1)
        
        raise TimeoutError(f"Health check command failed after {timeout}s")
    
    def _wait_for_running(self, container: Container, timeout: int) -> None:
        """Wait for container to be in running state"""
        
        start = time.time()
        
        while time.time() - start < timeout:
            container.reload()
            if container.status == "running":
                # Give it a moment to initialize
                time.sleep(2)
                return
            elif container.status in ("exited", "dead"):
                logs = container.logs(tail=50).decode("utf-8")
                raise RuntimeError(f"Container exited unexpectedly. Logs:\n{logs}")
            
            time.sleep(0.5)
        
        raise TimeoutError(f"Container not running after {timeout}s")
