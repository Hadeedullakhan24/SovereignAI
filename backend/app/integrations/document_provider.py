from abc import ABC,abstractmethod
from pathlib import Path
class DocumentProvider(ABC):
 @abstractmethod
 async def retrieve(self,file_id:str,path:Path)->dict: ...
class LocalDocumentProvider(DocumentProvider):
 async def retrieve(self,file_id:str,path:Path)->dict:
  if path.suffix.lower() in {".txt",".md"}: return {"file_id":file_id,"content":path.read_text(encoding="utf-8",errors="replace"),"provider":"local"}
  return {"file_id":file_id,"content":None,"provider":"local","message":"No extractor configured for this type."}
