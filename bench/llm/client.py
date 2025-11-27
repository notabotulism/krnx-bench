"""
Unified LLM client for benchmarks.

Supports OpenAI and Anthropic models.
"""

import time
import logging
from typing import Optional

from ..models import LLMResponse

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client for benchmark runs.
    
    Supports:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    
    Configuration:
        provider: "openai" or "anthropic"
        model: Model name (e.g., "gpt-4-turbo-preview")
        temperature: Sampling temperature (default: 0.0 for determinism)
        max_tokens: Maximum response tokens
    """
    
    def __init__(self, config: dict):
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4-turbo-preview")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        
        # Initialize provider client
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
        
        logger.info(f"LLM client initialized: {self.provider}/{self.model}")
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            prompt: The prompt text
            **kwargs: Override default parameters
            
        Returns:
            LLMResponse with text and token counts
        """
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        start = time.time()
        
        if self.provider == "openai":
            response = self._complete_openai(prompt, temperature, max_tokens)
        elif self.provider == "anthropic":
            response = self._complete_anthropic(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        response.latency_ms = (time.time() - start) * 1000
        
        logger.debug(
            f"LLM completion: {response.prompt_tokens} prompt + "
            f"{response.completion_tokens} completion tokens in {response.latency_ms:.0f}ms"
        )
        
        return response
    
    def _complete_openai(
        self, 
        prompt: str, 
        temperature: float, 
        max_tokens: int
    ) -> LLMResponse:
        """Generate completion via OpenAI"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return LLMResponse(
            text=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=self.model,
        )
    
    def _complete_anthropic(
        self, 
        prompt: str, 
        temperature: float, 
        max_tokens: int
    ) -> LLMResponse:
        """Generate completion via Anthropic"""
        
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return LLMResponse(
            text=response.content[0].text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            model=self.model,
        )
    
    def embed(self, text: str) -> list[float]:
        """
        Get embedding for text.
        
        Currently only supports OpenAI embeddings.
        """
        
        if self.provider != "openai":
            # Use OpenAI for embeddings even with Anthropic LLM
            from openai import OpenAI
            client = OpenAI()
        else:
            client = self.client
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        
        return response.data[0].embedding
