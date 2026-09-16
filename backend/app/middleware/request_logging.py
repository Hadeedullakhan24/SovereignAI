import logging,time,uuid
from starlette.middleware.base import BaseHTTPMiddleware
logger=logging.getLogger("app")
class RequestLoggingMiddleware(BaseHTTPMiddleware):
 async def dispatch(self,request,call_next):
  request.state.request_id=request.headers.get("X-Request-ID",str(uuid.uuid4())); start=time.perf_counter(); response=await call_next(request); response.headers["X-Request-ID"]=request.state.request_id; logger.info("request_complete",extra={"request_id":request.state.request_id,"method":request.method,"path":request.url.path,"status":response.status_code,"duration_ms":int((time.perf_counter()-start)*1000)}); return response
