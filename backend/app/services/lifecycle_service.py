"""Safe startup cleanup for unreferenced temporary worker files."""
from datetime import datetime,timezone
from backend.app.configuration.settings import settings
def cleanup_temporary_files() -> int:
 settings.temp_dir.mkdir(parents=True,exist_ok=True); cutoff=datetime.now(timezone.utc).timestamp()-settings.cleanup_age_hours*3600; removed=0
 for path in settings.temp_dir.iterdir():
  if path.is_file() and path.stat().st_mtime<cutoff:
   path.unlink(); removed+=1
 return removed
