"""Provider boundary; route code never imports a vendor SDK."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt:str, *, model:str|None=None) -> str: ...
    async def stream(self,prompt:str,*,model:str|None=None)->AsyncIterator[str]:
        yield await self.complete(prompt,model=model)
class LocalModelProvider(LLMProvider):
    async def complete(self,prompt:str,*,model:str|None=None)->str:
        # A fabricated acknowledgement could be mistaken for grounded output.
        # Production requests execute through SovereignAgent/RAGPipeline.
        raise RuntimeError("LocalModelProvider is not wired for direct completion; use SovereignAgent.")
