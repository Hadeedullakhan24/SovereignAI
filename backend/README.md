# SovereignAI Integration Backend

Production-oriented FastAPI integration layer for authenticated chat, uploads, document access, task/agent orchestration, result downloads, and audit records. Its API layer is provider-neutral: connect the existing `rag_engine` and `agent` packages by replacing `ModelProvider`, `DocumentProvider`, or `ToolProvider` implementations rather than changing routes.

## Run locally

From the repository root, use Python 3.11+:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r backend/requirements.txt
copy backend/.env.example .env
python -m alembic -c backend/alembic.ini upgrade head
uvicorn backend.app.main:app --reload
```

Use `DATABASE_URL=postgresql+psycopg://user:password@host/db` for PostgreSQL. Set a unique `JWT_SECRET_KEY` before deployment. The SQLite default is appropriate only for local development. Docker uses `docker compose up --build`.

## API contract

All endpoints return `{ "success": true, "data": ... }`; errors return `{ "success": false, "error": { "code", "message", "request_id" } }`. Interactive OpenAPI documentation is available at `/docs`, `/redoc`, and `/openapi.json`.

| Method | Endpoint | Auth | Purpose |
| --- | --- | --- | --- |
| POST | `/api/v1/auth/register` | No | Register `{email,password,name}` |
| POST | `/api/v1/auth/login` | No | Obtain JWT from `{email,password}` |
| GET | `/api/v1/auth/me` | Bearer | Current user |
| POST/GET | `/api/v1/chats` | Bearer | Create/list chats |
| GET | `/api/v1/chats/{id}` | Bearer | Chat with messages |
| POST | `/api/v1/chats/{id}/messages` | Bearer | Store message and invoke agent |
| POST | `/api/v1/files/upload` | Bearer | Multipart `file` upload |
| GET | `/api/v1/files/{file_id}/download` | Bearer | Download original upload |
| GET | `/api/v1/documents/{file_id}` | Bearer | File metadata |
| GET | `/api/v1/documents/{file_id}/content` | Bearer | Provider-based retrieval |
| POST | `/api/v1/tasks` | Bearer | Execute `{task_type,input}` |
| GET | `/api/v1/results/{task_id}` | Bearer | Task result metadata |
| GET | `/api/v1/results/{task_id}/download` | Bearer | Generated output, when one exists |
| GET | `/api/v1/health` | No | Health check |

Ownership is enforced for every chat, file, document, task, and download. Uploads use generated filenames and resolved paths, reject disallowed MIME types and configured oversize files, and are stored beneath `storage/uploads`. Generated files belong beneath `storage/outputs`; temporary workers should use `storage/temporary` and honor `CLEANUP_AGE_HOURS`.

## Architecture

`route → service → provider → database/audit`. `AgentService` uses a safe local echo provider by default, so tests and local startup do not need an external AI service. Adapter implementations can invoke the repository's air-gapped RAG and agent modules. Audit entries track action, request ID, actor, file/task, model/agent usage, state, and timing; passwords and JWTs are never logged.

Run tests with `pytest backend/tests -q`.
