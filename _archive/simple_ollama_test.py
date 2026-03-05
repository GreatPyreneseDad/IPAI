#!/usr/bin/env python3
"""
Simple Ollama Integration Test

Basic test to verify Ollama + Llama 3.2 integration works.
"""

import asyncio
import json
import aiohttp
from typing import AsyncGenerator

class OllamaClient:
    """Simple Ollama API client for testing"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2:latest"):
        self.host = host.rstrip('/')
        self.model = model
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=30.0)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session
    
    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate response from Ollama"""
        session = await self._get_session()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            async with session.post(f"{self.host}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Ollama connection error: {e}")
    
    async def stream_generate(self, prompt: str, max_tokens: int = 50) -> AsyncGenerator[str, None]:
        """Stream response from Ollama"""
        session = await self._get_session()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            async with session.post(f"{self.host}/api/generate", json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    token = data['response']
                                    if token:
                                        yield token
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Ollama connection error: {e}")
    
    async def check_model(self) -> bool:
        """Check if model is available"""
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    return self.model in models
                return False
        except aiohttp.ClientError:
            return False
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()

async def main():
    """Run basic Ollama tests"""
    print("🚀 Starting Simple Ollama + Llama 3.2 Test\\n")
    
    client = OllamaClient()
    
    try:
        # Test 1: Check if model is available
        print("🔍 Checking if llama3.2:latest is available...")
        model_available = await client.check_model()
        print(f"📋 Model available: {model_available}")
        
        if not model_available:
            print("❌ Model not available. Please run: ollama pull llama3.2")
            return False
        
        # Test 2: Basic generation
        print("\\n🤖 Testing basic generation...")
        response = await client.generate("Hello, how are you?", max_tokens=50)
        print(f"📝 Response: {response}")
        
        # Test 3: Streaming
        print("\\n📡 Testing streaming generation...")
        print("📤 Streamed response: ", end="", flush=True)
        
        async for token in client.stream_generate("Tell me a short joke", max_tokens=50):
            print(token, end="", flush=True)
        print()
        
        # Test 4: GCT-style prompt
        print("\\n🧠 Testing GCT-style interaction...")
        gct_prompt = """You are a coherence-focused assistant. Please provide a brief, supportive response to help with life direction.

User: I'm feeling confused about my career path. Can you help?"""
        
        gct_response = await client.generate(gct_prompt, max_tokens=100)
        print(f"📤 GCT Response: {gct_response}")
        
        await client.close()
        
        print("\\n🎉 All basic tests passed! Ollama + Llama 3.2 integration is working.")
        print("\\n📋 Test Summary:")
        print("  ✅ Ollama connection established")
        print("  ✅ Llama 3.2 model available")
        print("  ✅ Basic text generation working")
        print("  ✅ Streaming generation working")
        print("  ✅ GCT-style prompting working")
        
        print("\\n🚀 Ready for IPAI integration!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        await client.close()
        return False

if __name__ == "__main__":
    import sys
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Unexpected error: {e}")
        sys.exit(1)