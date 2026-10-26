import { connection } from "next/server";

import { forwardToBackend } from "@/lib/backend";

/** GET /api/encodings: the encodings available in the backend. */
export async function GET() {
  // Ask for the list at request time, not once at build time: the backend
  // is not running when the app is built.
  await connection();
  return forwardToBackend("/encodings");
}
