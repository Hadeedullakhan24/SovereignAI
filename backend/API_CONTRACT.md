# Member 5 API contract

Base URL: `/api/v1`. Authenticate protected calls with `Authorization: Bearer <access_token>`.

| Feature | Endpoint | Contract |
|---|---|---|
| Auth | `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout` | Login/refresh return access and rotating refresh JWTs. Logout body is `{ "refresh_token": "..." }`. |
| Chat | `POST /chat` | Request: `{message, conversation_id?, agent?, model?, file_ids?, task_id?, stream?}`. Response (and SSE `message` data): `{conversation_id, message_id, content, status, is_verified, insufficient_evidence, citations: [], artifact: null, tool_results: [], error: null}`. `citations`, `artifact`, and `tool_results` are optional outputs and never determine whether `content` is valid. With `stream=true`, response is SSE `status`, `message`, then `done`. |
| Conversations | `GET /chat/history`, `GET/DELETE /chat/{conversation_id}` | Caller ownership is enforced. |
| Files | `POST /files/upload`, `GET /files`, `GET /files/{id}`, `/preview`, `/download`, `DELETE /files/{id}` | Multipart field name is `file`; server reports verified MIME type. |
| Tasks | `POST /tasks`, `GET /tasks`, `GET /tasks/{id}`, `POST /tasks/{id}/cancel`, `/retry` | Creation returns `202` and `QUEUED`; clients poll until a terminal state. |
| Approvals | `POST /approvals/{task_id}?approved=true` | Only task owner may decide an awaiting approval. |
| Results | `GET /results/{task_id}` | Returns terminal result/error, ownership enforced. |
| Events | Chat SSE; task polling currently. | Task SSE/WebSocket is not yet available. |

Canonical machine-readable contract: `GET /api/v1/openapi.json`.
