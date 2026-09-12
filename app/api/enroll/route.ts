/**
 * POST /api/enroll
 *
 * Proxies an enrollment request to the Python backend (http://localhost:8000/enroll).
 * Accepts a multipart/form-data body with:
 *   - image : File  (JPEG / PNG / WEBP)
 *   - name  : string
 *   - replace : "true" | "false"  (optional, default "false")
 *
 * Returns JSON from the Python backend.
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const imageFile = formData.get("image") as File | null;
    const name = formData.get("name") as string | null;
    const replace = formData.get("replace") as string | null;

    if (!imageFile || !name) {
      return NextResponse.json(
        { error: "Both 'image' (file) and 'name' (string) are required." },
        { status: 400 }
      );
    }

    // Convert the uploaded file to a base64 data-URI so the Python backend
    // can accept it in a JSON body (avoids multipart forwarding complexity).
    const arrayBuffer = await imageFile.arrayBuffer();
    const base64 = Buffer.from(arrayBuffer).toString("base64");
    const mimeType = imageFile.type || "image/jpeg";
    const dataUri = `data:${mimeType};base64,${base64}`;

    const payload = {
      image: dataUri,
      name: name.trim(),
      replace: replace === "true",
    };

    const backendRes = await fetch(`${BACKEND_URL}/enroll`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: data.detail ?? "Enrollment failed." },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 201 });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    // Surface backend-unavailable errors clearly
    if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
      return NextResponse.json(
        { error: "Face recognition backend is not running. Start it with: cd backend && python server.py" },
        { status: 503 }
      );
    }
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
