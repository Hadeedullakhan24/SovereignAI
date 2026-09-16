import logging,uuid
from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.app.api.router import router
from backend.app.configuration.settings import settings
from backend.app.database.session import init_db
from backend.app.middleware.request_logging import RequestLoggingMiddleware
from backend.app.services.lifecycle_service import cleanup_temporary_files
logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s")
app=FastAPI(title=settings.app_name,version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=settings.origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware)
@app.on_event("startup")
def startup():
 if len(settings.jwt_secret_key)<32 and settings.app_env.lower() not in {"development","test"}: raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters")
 for directory in (settings.upload_dir,settings.output_dir,settings.temp_dir): directory.mkdir(parents=True,exist_ok=True)
 init_db()
 cleanup_temporary_files()
def error(request:Request,status:int,code:str,message:str): return JSONResponse(status_code=status,content={"success":False,"error":{"code":code,"message":message,"request_id":getattr(request.state,"request_id",None)}})
@app.exception_handler(StarletteHTTPException)
async def http_error(request:Request,exc:StarletteHTTPException):
 detail=exc.detail if isinstance(exc.detail,dict) else {}
 return error(request,exc.status_code,detail.get("code",{401:"AUTHENTICATION_ERROR",403:"AUTHORIZATION_ERROR",404:"RESOURCE_NOT_FOUND",409:"CONFLICT",413:"FILE_TOO_LARGE",415:"UNSUPPORTED_MEDIA_TYPE"}.get(exc.status_code,"REQUEST_ERROR")),detail.get("message",str(exc.detail)))
@app.exception_handler(RequestValidationError)
async def validation_error(request:Request,exc:RequestValidationError): return error(request,422,"VALIDATION_ERROR","Request validation failed")
@app.exception_handler(Exception)
async def unexpected(request:Request,exc:Exception): logging.getLogger("app").exception("unhandled request_id=%s",getattr(request.state,"request_id",None)); return error(request,500,"INTERNAL_ERROR","An unexpected error occurred")
app.include_router(router)
