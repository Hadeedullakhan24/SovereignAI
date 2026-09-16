from __future__ import annotations
import csv, io, json, secrets, zipfile
from pathlib import Path
from fastapi import HTTPException, UploadFile
from backend.app.core.config import get_settings
EXTENSIONS={".pdf":"application/pdf",".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".gif":"image/gif",".webp":"image/webp",".docx":"application/vnd.openxmlformats-officedocument.wordprocessingml.document",".xlsx":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",".csv":"text/csv",".txt":"text/plain",".json":"application/json",".md":"text/markdown"}
def detect_content(data:bytes)->str:
    if data.startswith(b"%PDF-"): return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"): return "image/png"
    if data.startswith(b"\xff\xd8\xff"): return "image/jpeg"
    if data.startswith((b"GIF87a",b"GIF89a")): return "image/gif"
    if data.startswith(b"RIFF") and data[8:12]==b"WEBP": return "image/webp"
    if data.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names=set(z.namelist())
            if "[Content_Types].xml" not in names: raise ValueError("Unrecognised ZIP")
            if "word/document.xml" in names: return EXTENSIONS[".docx"]
            if "xl/workbook.xml" in names: return EXTENSIONS[".xlsx"]
        raise ValueError("Unsupported Office container")
    try: text=data.decode("utf-8")
    except UnicodeDecodeError: raise ValueError("Non-text content")
    if text.lstrip().startswith(("{","[")): json.loads(text); return "application/json"
    if "\x00" in text: raise ValueError("Binary content")
    lines=[x for x in text.splitlines() if x.strip()]
    if len(lines)>1:
        try:
            dialect=csv.Sniffer().sniff("\n".join(lines[:20]),delimiters=",;\t")
            if all(len(row)>1 for row in csv.reader(lines[:20],dialect)): return "text/csv"
        except csv.Error: pass
    return "text/markdown" if any(x in text for x in ("# ","## ","```")) else "text/plain"
class LocalStorage:
    def __init__(self): self.root=get_settings().upload_dir.resolve(); self.root.mkdir(parents=True,exist_ok=True)
    async def save(self,upload:UploadFile)->tuple[str,str,int,str]:
        name=Path(upload.filename or "").name
        if not name or name in {".",".."} or name!=upload.filename: raise HTTPException(400,"Unsafe filename")
        suffix=Path(name).suffix.lower()
        if suffix not in EXTENSIONS: raise HTTPException(415,"Unsupported file extension")
        parts=[]; size=0
        while chunk:=await upload.read(1024*1024):
            size+=len(chunk)
            if size>get_settings().max_upload_bytes: raise HTTPException(413,"File exceeds size limit")
            parts.append(chunk)
        if not size: raise HTTPException(400,"Upload is empty")
        try: detected=detect_content(b"".join(parts))
        except (ValueError,zipfile.BadZipFile,json.JSONDecodeError) as exc: raise HTTPException(415,"Invalid or unsupported file content") from exc
        if detected!=EXTENSIONS[suffix]: raise HTTPException(415,"Filename extension does not match file content")
        key=f"{secrets.token_hex(16)}{suffix}"; target=(self.root/key).resolve()
        if target.parent!=self.root: raise HTTPException(400,"Unsafe storage path")
        try: target.write_bytes(b"".join(parts))
        except Exception: target.unlink(missing_ok=True); raise
        return key,name,size,detected
    def path(self,key:str)->Path:
        path=(self.root/Path(key).name).resolve()
        if path.parent!=self.root or not path.is_file(): raise HTTPException(404,"File not found")
        return path
