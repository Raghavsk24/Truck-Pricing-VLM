import { NextResponse } from "next/server";
import { loadAnalysis, persistAndRespond } from "@/lib/pipeline";
import type { AnalyzeResponse } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 30;

function fail(message: string, status = 400) {
  const body: AnalyzeResponse = { status: "error", user_message: message };
  return NextResponse.json(body, { status });
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { analysisId?: string; brand?: string };
    const analysisId = body.analysisId?.trim();
    const brand = body.brand?.trim();
    if (!analysisId) return fail("Missing analysis. Please upload the photo again.");
    if (!brand) return fail("Please enter the truck brand.");

    const row = await loadAnalysis(analysisId);
    if (!row.vlm_output?.output) {
      return fail("That analysis is incomplete. Please upload the photo again.");
    }

    const result = await persistAndRespond(
      row.image_path || "",
      row.vlm_output,
      analysisId,
      brand,
    );
    return NextResponse.json(result);
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Could not price this truck with that brand.";
    console.error(err);
    return fail(message, 500);
  }
}
