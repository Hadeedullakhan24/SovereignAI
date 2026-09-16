Configure Celery with SOVEREIGNAI_REDIS_URL in production. Workers must invoke
the LLMProvider/tool adapter and write Task state transitions transactionally;
the HTTP process only queues work.
