"""
LLM Provider Integration Module

Supports multiple LLM providers with unified interface.
Allows users to choose and configure their preferred LLM.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import aiohttp
import json
import os
from abc import ABC, abstractmethod
import asyncio

# OpenAI compatible
import openai
from openai import AsyncOpenAI

# Anthropic
try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Google
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Cohere
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    TOGETHER = "together"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class LLMProviderInfo:
    """Information about LLM providers"""
    
    PROVIDERS = {
        LLMProvider.OPENAI: {
            "name": "OpenAI",
            "api_base": "https://api.openai.com/v1",
            "models": [
                "gpt-4-turbo-preview",
                "gpt-4-1106-preview", 
                "gpt-4",
                "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo",
            ],
            "requires_api_key": True,
            "supports_streaming": True,
            "supports_functions": True
        },
        LLMProvider.ANTHROPIC: {
            "name": "Anthropic Claude",
            "api_base": "https://api.anthropic.com",
            "models": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-2.1",
                "claude-2.0",
                "claude-instant-1.2"
            ],
            "requires_api_key": True,
            "supports_streaming": True,
            "supports_functions": False
        },
        LLMProvider.GOOGLE: {
            "name": "Google Gemini",
            "api_base": "https://generativelanguage.googleapis.com",
            "models": [
                "gemini-pro",
                "gemini-pro-vision",
                "gemini-ultra"
            ],
            "requires_api_key": True,
            "supports_streaming": True,
            "supports_functions": True
        },
        LLMProvider.COHERE: {
            "name": "Cohere",
            "api_base": "https://api.cohere.ai",
            "models": [
                "command-r-plus",
                "command-r",
                "command",
                "command-light"
            ],
            "requires_api_key": True,
            "supports_streaming": True,
            "supports_functions": False
        },
        LLMProvider.TOGETHER: {
            "name": "Together AI",
            "api_base": "https://api.together.xyz/v1",
            "models": [
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "meta-llama/Llama-2-70b-chat-hf",
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                "zero-one-ai/Yi-34B-Chat"
            ],
            "requires_api_key": True,
            "supports_streaming": True,
            "supports_functions": False
        },
        LLMProvider.REPLICATE: {
            "name": "Replicate",
            "api_base": "https://api.replicate.com/v1",
            "models": [
                "meta/llama-2-70b-chat",
                "mistralai/mixtral-8x7b-instruct-v0.1",
                "01-ai/yi-34b-chat"
            ],
            "requires_api_key": True,
            "supports_streaming": True,
            "supports_functions": False
        },
        LLMProvider.HUGGINGFACE: {
            "name": "Hugging Face",
            "api_base": "https://api-inference.huggingface.co/models",
            "models": [
                "meta-llama/Llama-2-70b-chat-hf",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "tiiuae/falcon-180B-chat"
            ],
            "requires_api_key": True,
            "supports_streaming": False,
            "supports_functions": False
        },
        LLMProvider.GROQ: {
            "name": "Groq",
            "api_base": "https://api.groq.com/openai/v1",
            "models": [
                "llama2-70b-4096",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ],
            "requires_api_key": True,
            "supports_streaming": True,
            "supports_functions": False
        },
        LLMProvider.MISTRAL: {
            "name": "Mistral AI",
            "api_base": "https://api.mistral.ai/v1",
            "models": [
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest",
                "mistral-embed"
            ],
            "requires_api_key": True,
            "supports_streaming": True,
            "supports_functions": True
        },
        LLMProvider.OLLAMA: {
            "name": "Ollama (Local)",
            "api_base": "http://localhost:11434",
            "models": [
                "llama2",
                "mistral",
                "codellama",
                "neural-chat",
                "starling-lm",
                "orca-mini"
            ],
            "requires_api_key": False,
            "supports_streaming": True,
            "supports_functions": False
        },
        LLMProvider.CUSTOM: {
            "name": "Custom Provider",
            "api_base": "",
            "models": [],
            "requires_api_key": True,
            "supports_streaming": False,
            "supports_functions": False
        }
    }
    
    @classmethod
    def get_info(cls, provider: LLMProvider) -> Dict[str, Any]:
        """Get information about a provider"""
        return cls.PROVIDERS.get(provider, cls.PROVIDERS[LLMProvider.CUSTOM])


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt"""
        pass
    
    @abstractmethod
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion for prompt"""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the LLM provider"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (also works with OpenAI-compatible APIs)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout
        )
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion using OpenAI API"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **kwargs
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"OpenAI API streaming error: {str(e)}")
    
    async def validate_connection(self) -> bool:
        """Validate OpenAI API connection"""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            timeout=config.timeout
        )
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using Anthropic API"""
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion using Anthropic API"""
        try:
            stream = await self.client.messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **kwargs
            )
            async for chunk in stream:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    yield chunk.delta.text
        except Exception as e:
            raise Exception(f"Anthropic API streaming error: {str(e)}")
    
    async def validate_connection(self) -> bool:
        """Validate Anthropic API connection"""
        try:
            # Try a minimal API call
            await self.complete("Hi", max_tokens=5)
            return True
        except Exception:
            return False


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "http://localhost:11434"
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using Ollama"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    result = await response.json()
                    return result.get("response", "")
            except Exception as e:
                raise Exception(f"Ollama API error: {str(e)}")
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion using Ollama"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens
                        }
                    }
                ) as response:
                    async for line in response.content:
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
            except Exception as e:
                raise Exception(f"Ollama streaming error: {str(e)}")
    
    async def validate_connection(self) -> bool:
        """Validate Ollama connection"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.api_base}/api/tags") as response:
                    return response.status == 200
            except Exception:
                return False


class GenericOpenAICompatibleClient(BaseLLMClient):
    """Generic client for OpenAI-compatible APIs (Together, Groq, etc.)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout
        )
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI-compatible API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"{self.config.provider.value} API error: {str(e)}")
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion using OpenAI-compatible API"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **kwargs
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"{self.config.provider.value} streaming error: {str(e)}")
    
    async def validate_connection(self) -> bool:
        """Validate API connection"""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False


class LLMManager:
    """Manager for LLM providers"""
    
    def __init__(self):
        self.configs: Dict[str, LLMConfig] = {}
        self.clients: Dict[str, BaseLLMClient] = {}
        self.active_provider: Optional[str] = None
    
    def add_provider(self, name: str, config: LLMConfig) -> None:
        """Add a new LLM provider configuration"""
        self.configs[name] = config
        
        # Create appropriate client
        if config.provider == LLMProvider.OPENAI:
            self.clients[name] = OpenAIClient(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            self.clients[name] = AnthropicClient(config)
        elif config.provider == LLMProvider.OLLAMA:
            self.clients[name] = OllamaClient(config)
        elif config.provider in [LLMProvider.TOGETHER, LLMProvider.GROQ, 
                               LLMProvider.MISTRAL, LLMProvider.CUSTOM]:
            self.clients[name] = GenericOpenAICompatibleClient(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    def set_active_provider(self, name: str) -> None:
        """Set the active LLM provider"""
        if name not in self.configs:
            raise ValueError(f"Provider {name} not found")
        self.active_provider = name
    
    def get_active_client(self) -> BaseLLMClient:
        """Get the active LLM client"""
        if not self.active_provider:
            raise ValueError("No active provider set")
        return self.clients[self.active_provider]
    
    async def complete(self, prompt: str, provider: Optional[str] = None, **kwargs) -> str:
        """Generate completion using specified or active provider"""
        provider_name = provider or self.active_provider
        if not provider_name:
            raise ValueError("No provider specified")
        
        client = self.clients.get(provider_name)
        if not client:
            raise ValueError(f"Provider {provider_name} not found")
        
        return await client.complete(prompt, **kwargs)
    
    async def stream_complete(self, prompt: str, provider: Optional[str] = None, **kwargs):
        """Stream completion using specified or active provider"""
        provider_name = provider or self.active_provider
        if not provider_name:
            raise ValueError("No provider specified")
        
        client = self.clients.get(provider_name)
        if not client:
            raise ValueError(f"Provider {provider_name} not found")
        
        async for chunk in client.stream_complete(prompt, **kwargs):
            yield chunk
    
    async def validate_provider(self, name: str) -> bool:
        """Validate a provider's connection"""
        client = self.clients.get(name)
        if not client:
            return False
        return await client.validate_connection()
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers"""
        providers = []
        for name, config in self.configs.items():
            info = LLMProviderInfo.get_info(config.provider)
            providers.append({
                "name": name,
                "provider": config.provider.value,
                "provider_name": info["name"],
                "model": config.model,
                "active": name == self.active_provider
            })
        return providers
    
    def get_provider_models(self, provider: LLMProvider) -> List[str]:
        """Get available models for a provider"""
        info = LLMProviderInfo.get_info(provider)
        return info.get("models", [])
    
    def save_config(self, path: str) -> None:
        """Save configuration to file"""
        config_data = {
            "providers": {
                name: {
                    "provider": config.provider.value,
                    "api_key": config.api_key,
                    "api_base": config.api_base,
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "timeout": config.timeout,
                    "extra_params": config.extra_params
                }
                for name, config in self.configs.items()
            },
            "active_provider": self.active_provider
        }
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, path: str) -> None:
        """Load configuration from file"""
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        for name, provider_config in config_data.get("providers", {}).items():
            config = LLMConfig(
                provider=LLMProvider(provider_config["provider"]),
                api_key=provider_config.get("api_key"),
                api_base=provider_config.get("api_base"),
                model=provider_config.get("model", ""),
                temperature=provider_config.get("temperature", 0.7),
                max_tokens=provider_config.get("max_tokens", 2000),
                timeout=provider_config.get("timeout", 60),
                extra_params=provider_config.get("extra_params", {})
            )
            self.add_provider(name, config)
        
        if config_data.get("active_provider"):
            self.set_active_provider(config_data["active_provider"])