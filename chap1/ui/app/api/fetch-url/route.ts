import { forwardToBackend } from "@/lib/backend";

/** POST /api/fetch-url: read the text of a web page. Body: { url }. */
export async function POST(request: Request) {
  return forwardToBackend("/fetch-url", {
    method: "POST",
    body: await request.text(),
  });
}
