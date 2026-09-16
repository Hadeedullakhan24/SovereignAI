from abc import ABC,abstractmethod
class ToolProvider(ABC):
 @abstractmethod
 async def invoke(self,name:str,arguments:dict)->dict: ...
