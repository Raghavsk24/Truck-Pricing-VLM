import "server-only";
import { randomUUID } from "crypto";
import { ANALYSES_TABLE, getSupabase, UPLOAD_BUCKET } from "./supabase";
import type { AnalyzeResponse, VlmResult } from "./types";
import { priceFromVlm } from "./pricing";
import { isPriceable, needsBrand, rejectionMessage } from "./vlm";

type AnalysisRow = {
  id: string;
  status: "rejected" | "needs_brand" | "ok";
  image_path: string | null;
  vlm_output: VlmResult;
  truck_type: string | null;
  brand: string | null;
  era: string | null;
  overall_score: number | null;
  center: number | null;
  low: number | null;
  high: number | null;
  contributions: unknown;
  user_message: string | null;
};

export async function uploadJpeg(jpeg: Buffer): Promise<{ id: string; path: string }> {
  const id = randomUUID();
  const path = `${id}.jpg`;
  const { error } = await getSupabase()
    .storage.from(UPLOAD_BUCKET)
    .upload(path, jpeg, {
      contentType: "image/jpeg",
      upsert: false,
    });
  if (error) {
    throw new Error(`Failed to store image: ${error.message}`);
  }
  return { id, path };
}

export async function insertAnalysis(row: AnalysisRow): Promise<void> {
  const { error } = await getSupabase().from(ANALYSES_TABLE).insert(row);
  if (error) {
    throw new Error(`Failed to save analysis: ${error.message}`);
  }
}

export async function loadAnalysis(id: string): Promise<AnalysisRow> {
  const { data, error } = await getSupabase()
    .from(ANALYSES_TABLE)
    .select("*")
    .eq("id", id)
    .single();
  if (error || !data) {
    throw new Error("Analysis not found. Please upload the photo again.");
  }
  return data as AnalysisRow;
}

export async function updateAnalysis(
  id: string,
  patch: Partial<AnalysisRow>,
): Promise<void> {
  const { error } = await getSupabase().from(ANALYSES_TABLE).update(patch).eq("id", id);
  if (error) {
    throw new Error(`Failed to update analysis: ${error.message}`);
  }
}

export async function persistAndRespond(
  imagePath: string,
  vlm: VlmResult,
  existingId?: string,
  brandOverride?: string,
): Promise<AnalyzeResponse> {
  const output = vlm.output;
  const id = existingId ?? randomUUID();

  if (!isPriceable(output)) {
    const user_message = rejectionMessage(output);
    if (!existingId) {
      await insertAnalysis({
        id,
        status: "rejected",
        image_path: imagePath,
        vlm_output: vlm,
        truck_type: String(output.truck_type),
        brand: null,
        era: output.age?.era ?? null,
        overall_score: null,
        center: null,
        low: null,
        high: null,
        contributions: null,
        user_message,
      });
    }
    return { status: "rejected", user_message };
  }

  if (!brandOverride && needsBrand(output)) {
    const user_prompt =
      output.brand.user_prompt ||
      "I couldn't find a visible logo, badge, or nameplate on this truck — do you know the brand/manufacturer?";
    if (!existingId) {
      await insertAnalysis({
        id,
        status: "needs_brand",
        image_path: imagePath,
        vlm_output: vlm,
        truck_type: String(output.truck_type),
        brand: null,
        era: output.age?.era ?? null,
        overall_score: output.condition.overall_score,
        center: null,
        low: null,
        high: null,
        contributions: null,
        user_message: user_prompt,
      });
    }
    return {
      status: "needs_brand",
      analysisId: id,
      user_prompt,
      truck_type: String(output.truck_type),
      era: output.age?.era ?? "unknown",
      model_series: output.age?.model_series ?? null,
      condition: output.condition,
    };
  }

  const priced = priceFromVlm(output, brandOverride);
  const row = {
    id,
    status: "ok" as const,
    image_path: imagePath,
    vlm_output: vlm,
    truck_type: priced.truck_type,
    brand: priced.brand,
    era: priced.era,
    overall_score: priced.overall_score,
    center: priced.center,
    low: priced.low,
    high: priced.high,
    contributions: priced.contributions,
    user_message: null,
  };

  if (existingId) {
    await updateAnalysis(id, row);
  } else {
    await insertAnalysis(row);
  }

  return {
    status: "ok",
    analysisId: id,
    ...priced,
  };
}
