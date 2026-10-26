/**
 * Browser-side calls to the routes of this app (see app/api/).
 */

import type {
  EncodingsResponse,
  ErrorResponse,
  FetchUrlResponse,
  TokenizeResponse,
} from "@/lib/types";

/** Turn the "detail" of an error response into a readable message. */
function errorMessage(body: ErrorResponse, status: number): string {
  if (typeof body.detail === "string") {
    return body.detail;
  }
  // FastAPI validation errors are a list of objects with a "msg" field.
  if (Array.isArray(body.detail) && body.detail.length > 0) {
    const first = body.detail[0] as { msg?: string };
    if (first.msg) {
      return first.msg;
    }
  }
  return `Request failed with HTTP status ${status}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(body as ErrorResponse, response.status));
  }
  return body as T;
}

export function getEncodings(): Promise<EncodingsResponse> {
  return requestJson<EncodingsResponse>("/api/encodings");
}

export function tokenize(
  text: string,
  encoding: string,
): Promise<TokenizeResponse> {
  return requestJson<TokenizeResponse>("/api/tokenize", {
    method: "POST",
    body: JSON.stringify({ text, encoding }),
  });
}

export function fetchUrl(url: string): Promise<FetchUrlResponse> {
  return requestJson<FetchUrlResponse>("/api/fetch-url", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
}
