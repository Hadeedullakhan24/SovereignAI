from sqlalchemy.orm import Session
from backend.app.database.models import AuditLog
def audit(db:Session,*,user_id:str|None,action:str,status:str,request_id:str|None,**kwargs)->None: db.add(AuditLog(user_id=user_id,action=action,status=status,request_id=request_id,**kwargs))
