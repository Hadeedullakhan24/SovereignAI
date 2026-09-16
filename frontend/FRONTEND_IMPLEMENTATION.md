# SovereignAI frontend implementation

## Completed

A Vite React/TypeScript workspace provides protected routes for dashboard, chat, files, tasks, task detail, agents, approvals, audit, and settings, along with registration/login/logout, centralized API handling, responsive shell/navigation, query caching/polling, loading/empty/error states, keyboard-native controls, status badges, task execution timeline, and safe Markdown rendering.

## Backend integration

See [API_INTEGRATION.md](API_INTEGRATION.md) for the exact endpoint mapping and documented API limitations. No production mocks or invented response fields are used.

## Realtime

Chat SSE is documented by the backend. The backend does not provide a task SSE/WebSocket channel, so task status is controlled polling. No private reasoning is requested or rendered.

## Files, tasks, approvals, audit

Files are uploaded under multipart `file`, listed, previewed/downloaded through server URLs, and deleted. Task operations use their actual state-machine endpoints. Approval UI derives pending actions from `WAITING_FOR_APPROVAL` task state because no approval-list API exists. Audit displays the user-scoped audit endpoint.

## Verification

`npm.cmd run build` passed (TypeScript check plus Vite production build). `npm.cmd run lint` passed (TypeScript check). A Vitest API-client bearer-token test was added and invoked; no backend services, Docker stack, browser authentication flow, or live API integration were verified in this environment. The backend CORS configuration already permits Vite's default origin; Docker Compose was not changed.
