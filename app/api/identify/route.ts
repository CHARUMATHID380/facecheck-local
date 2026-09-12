/**
 * POST /api/identify
 *
 * Proxies an identification request to the Python backend.
 * Accepts multipart/form-data with:
 *   - image     : File   (JPEG / PNG / WEBP)
 *   - top_k     : string (optional, default "3")
 *   - threshold : string (optional, default "0.4")
 *
 * Returns JSON:
 * {
 *   identified : boolean
 *   name       : string   ("unknown" if no match passes threshold)
 *   face_id    : string | null
 *   similarity : number   (0–100)
 *   distance   : number   (0–1 cosine distance)
 *   candidates : Array<{ face_id, name, distance, similarity }>
 *   message    : string
 * }
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const imageFile = formData.get("image") as File | null;
    const topK = formData.get("top_k") as string | null;
    const threshold = formData.get("threshold") as string | null;

    if (!imageFile) {
      return NextResponse.json(
        { error: "'image' (file) is required." },
        { status: 400 }
      );
    }

    const arrayBuffer = await imageFile.arrayBuffer();
    const base64 = Buffer.from(arrayBuffer).toString("base64");
    const mimeType = imageFile.type || "image/jpeg";
    const dataUri = `data:${mimeType};base64,${base64}`;

    const payload: Record<string, unknown> = { image: dataUri };
    if (topK) payload.top_k = parseInt(topK, 10);
    if (threshold) payload.threshold = parseFloat(threshold);

    const backendRes = await fetch(`${BACKEND_URL}/identify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: data.detail ?? "Identification failed." },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
      return NextResponse.json(
        { error: "Face recognition backend is not running. Start it with: cd backend && python server.py" },
        { status: 503 }
      );
    }
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
