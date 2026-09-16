# API integration

The browser base URL is `VITE_API_BASE_URL` (default `http://localhost:8000/api/v1`). `VITE_WS_BASE_URL` is reserved but unused: the backend contract explicitly states that task SSE/WebSocket is not available.

Authentication uses `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, and authenticated `POST /auth/logout`. Access and rotating refresh tokens are stored as one browser session record; a 401 triggers one refresh and repeats the request. A failed refresh clears the session.

Implemented mappings: chat history/message/delete and `POST /chat`; files upload/list/detail/preview/download/delete; tasks create/list/detail/cancel/retry and `GET /results/{task_id}`; agents list; approval decision; audit list; health. Protected calls carry `Authorization: Bearer <access_token>`.

Chat supports the documented SSE mode at the transport level in the API contract, but the current UI sends non-streaming chat because the backend's SSE response contains one final `message` event then `done` rather than incremental content. Task updates use safe 4–10 second polling until terminal state.

Backend discrepancy / capability gaps: task list responses intentionally contain only `id`, `type`, `state`, `result`, `error`, and `attempts`; no timestamps, steps, agent/model execution detail, generated-result download endpoint, or task event stream is exposed. Approval has a decision endpoint but no list endpoint, so pending approvals are derived only from the caller's tasks in `WAITING_FOR_APPROVAL`. File preview/download endpoints require authorization; the frontend fetches them with the bearer token and opens a short-lived Blob URL.

The frontend never invents or displays hidden reasoning. Server errors are reduced to their public `detail` and request ID where available.
