from backend.app.integrations.model_provider import EchoModelProvider,ModelProvider
class AgentService:
 def __init__(self,model:ModelProvider|None=None): self.model=model or EchoModelProvider()
 async def execute(self,task_type:str,input_data:dict)->dict:
  response=await self.model.generate([{"role":"user","content":input_data.get("instruction",input_data.get("message",task_type))}],input_data)
  return {"result":response,"model_used":type(self.model).__name__,"agent_action":task_type}
