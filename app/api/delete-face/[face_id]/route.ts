/**
 * DELETE /api/delete-face/[face_id]
 *
 * Deletes a single enrolled face by its UUID.
 *
 * Path param: face_id — the UUID returned during enrollment
 *
 * Response: { message: string }
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ face_id: string }> }
) {
  const { face_id } = await params;

  if (!face_id) {
    return NextResponse.json(
      { error: "face_id path parameter is required." },
      { status: 400 }
    );
  }

  try {
    const backendRes = await fetch(`${BACKEND_URL}/faces/${encodeURIComponent(face_id)}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: data.detail ?? "Deletion failed." },
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
