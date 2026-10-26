/**
 * Server-side forwarding of requests to the Python backend.
 *
 * The browser calls only the routes of this Next.js app (/api/...). These
 * routes run on the Next.js server and forward the request to FastAPI, so
 * the browser never talks to the backend directly: no CORS configuration is
 * needed, and the backend address is set in one place (BACKEND_URL).
 */

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const BACKEND_DOWN_MESSAGE =
  "The Python backend is not reachable at " +
  BACKEND_URL +
  ". Start it from the repository root with: python -m chap1.api";

/**
 * Send a request to the backend and return its response to the browser.
 *
 * @param path - The backend path, for example "/tokenize".
 * @param init - The fetch options (method, body, ...).
 * @returns The backend response, or a 503 error if the backend is down.
 */
export async function forwardToBackend(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${BACKEND_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });
  } catch {
    return Response.json({ detail: BACKEND_DOWN_MESSAGE }, { status: 503 });
  }

  // FastAPI answers with JSON; an unexpected server error may not.
  try {
    const body = await response.json();
    return Response.json(body, { status: response.status });
  } catch {
    const detail = `The backend answered with HTTP ${response.status} and no JSON body.`;
    return Response.json({ detail }, { status: 502 });
  }
}
