from abc import ABC,abstractmethod
class ModelProvider(ABC):
 @abstractmethod
 async def generate(self,messages:list[dict],context:dict|None=None)->str: ...
class EchoModelProvider(ModelProvider):
 async def generate(self,messages:list[dict],context:dict|None=None)->str: return f"Received: {messages[-1]['content']}"
