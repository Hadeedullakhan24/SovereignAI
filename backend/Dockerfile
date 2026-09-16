FROM python:3.12-slim
WORKDIR /srv/app
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . .
ENV SOVEREIGNAI_ENVIRONMENT=production
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
