"""Celery application; imported by API only to enqueue and by worker to execute."""
from celery import Celery
from backend.app.core.config import get_settings
settings=get_settings()
broker=settings.celery_broker_url or settings.redis_url or "redis://localhost:6379/0"
celery_app=Celery("sovereignai",broker=broker,backend=settings.celery_result_backend or broker,include=["backend.app.workers.tasks"])
celery_app.conf.update(task_track_started=True,task_acks_late=True,worker_prefetch_multiplier=1,task_serializer="json",result_serializer="json",accept_content=["json"],timezone="UTC")
