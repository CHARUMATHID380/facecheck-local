/**
 * GET  /api/enrolled-faces
 * Returns the list of all enrolled faces from the Python backend.
 *
 * Response: Array<{ face_id: string; name: string; enrolled_at: string }>
 *
 * DELETE /api/enrolled-faces
 * Wipes the entire enrolled face database.
 *
 * Response: { message: string }
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

// ── GET ─────────────────────────────────────────────────────────────────────

export async function GET(_req: NextRequest) {
  try {
    const backendRes = await fetch(`${BACKEND_URL}/faces`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      // Opt out of Next.js fetch cache so we always get live data
      cache: "no-store",
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: data.detail ?? "Failed to fetch enrolled faces." },
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

// ── DELETE (clear all) ───────────────────────────────────────────────────────

export async function DELETE(_req: NextRequest) {
  try {
    const backendRes = await fetch(`${BACKEND_URL}/faces`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: data.detail ?? "Failed to clear the database." },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
      return NextResponse.json(
        { error: "Face recognition backend is not running." },
        { status: 503 }
      );
    }
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
