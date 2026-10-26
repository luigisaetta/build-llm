import { forwardToBackend } from "@/lib/backend";

/** POST /api/tokenize: tokenize a text. Body: { text, encoding }. */
export async function POST(request: Request) {
  return forwardToBackend("/tokenize", {
    method: "POST",
    body: await request.text(),
  });
}
