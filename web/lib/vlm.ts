import "server-only";
import Anthropic from "@anthropic-ai/sdk";
import masterSchema from "@/data/truck_feature_extraction_master_instructions.json";
import type { VlmOutput, VlmResult } from "./types";
import { CAT_KEYS, PRICEABLE_SUBJECTS } from "./types";

const DEFAULT_MODEL = "claude-sonnet-4-5-20250929";

function client(): Anthropic {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY is not set");
  }
  return new Anthropic({ apiKey });
}

function extractJsonObject(text: string): unknown {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start < 0 || end <= start) {
    throw new Error("VLM did not return JSON");
  }
  return JSON.parse(candidate.slice(start, end + 1));
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("VLM output was not an object");
  }
  return value as Record<string, unknown>;
}

function emptyCondition(): VlmOutput["condition"] {
  const zero = { score: null, weight: 0, penalty_percent: 0 };
  return {
    categories: {
      chassis_and_frame: { ...zero, weight: 0.35 },
      front_end_engine_compartment_hood: { ...zero, weight: 0.25 },
      tires_wheels_suspension: { ...zero, weight: 0.2 },
      cab_sleeper_aero_fairings: { ...zero, weight: 0.2 },
    },
    overall_score: null,
    overall_condition_label: null,
    total_penalty_percent: 0,
    explanation: "",
  };
}

function parseCategory(raw: unknown, weight: number) {
  const rec = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const scoreRaw = rec.score;
  const score =
    scoreRaw === null || scoreRaw === undefined ? null : Number(scoreRaw);
  return {
    score: score != null && Number.isFinite(score) ? score : null,
    weight: Number(rec.weight) || weight,
    penalty_percent: Number(rec.penalty_percent) || 0,
  };
}

export function parseVlmResult(value: unknown): VlmResult {
  const root = asRecord(value);
  const output = asRecord(root.output ?? root);
  const brandRaw = asRecord(output.brand ?? {});
  const ageRaw = asRecord(output.age ?? {});
  const conditionRaw = asRecord(output.condition ?? {});
  const catsRaw = asRecord(conditionRaw.categories ?? {});

  const truckType = String(output.truck_type ?? "none");
  if (!truckType) {
    throw new Error("VLM output missing truck_type");
  }

  const categories = {
    chassis_and_frame: parseCategory(catsRaw.chassis_and_frame, 0.35),
    front_end_engine_compartment_hood: parseCategory(
      catsRaw.front_end_engine_compartment_hood,
      0.25,
    ),
    tires_wheels_suspension: parseCategory(catsRaw.tires_wheels_suspension, 0.2),
    cab_sleeper_aero_fairings: parseCategory(catsRaw.cab_sleeper_aero_fairings, 0.2),
  };

  const overall =
    conditionRaw.overall_score === null || conditionRaw.overall_score === undefined
      ? null
      : Number(conditionRaw.overall_score);

  const parsed: VlmResult = {
    reasoning: String(root.reasoning ?? ""),
    output: {
      truck_type: truckType,
      truck_type_user_message: String(output.truck_type_user_message ?? ""),
      primary_subject: String(output.primary_subject ?? "unusable"),
      primary_subject_user_message: String(
        output.primary_subject_user_message ?? "",
      ),
      brand: {
        has_brand: Boolean(brandRaw.has_brand),
        brand_name:
          brandRaw.brand_name == null ? null : String(brandRaw.brand_name),
        visible_identifier_type: String(brandRaw.visible_identifier_type ?? "none"),
        identifier_location:
          brandRaw.identifier_location == null
            ? null
            : String(brandRaw.identifier_location),
        brand_list_verified: Boolean(brandRaw.brand_list_verified),
        confidence: String(brandRaw.confidence ?? "low"),
        reasoning: String(brandRaw.reasoning ?? ""),
        needs_user_input: Boolean(brandRaw.needs_user_input),
        user_prompt:
          brandRaw.user_prompt == null ? null : String(brandRaw.user_prompt),
      },
      age: {
        model_series:
          ageRaw.model_series == null ? null : String(ageRaw.model_series),
        era: String(ageRaw.era ?? "unknown"),
        era_confidence: String(ageRaw.era_confidence ?? "low"),
        era_evidence: String(ageRaw.era_evidence ?? ""),
      },
      condition: {
        categories,
        overall_score: overall != null && Number.isFinite(overall) ? overall : null,
        overall_condition_label:
          conditionRaw.overall_condition_label == null
            ? null
            : String(conditionRaw.overall_condition_label),
        total_penalty_percent: Number(conditionRaw.total_penalty_percent) || 0,
        explanation: String(conditionRaw.explanation ?? ""),
      },
    },
  };

  for (const key of CAT_KEYS) {
    if (!parsed.output.condition.categories[key]) {
      parsed.output.condition = emptyCondition();
      break;
    }
  }

  return parsed;
}

export function isPriceable(output: VlmOutput): boolean {
  const type = String(output.truck_type).toLowerCase();
  const subject = String(output.primary_subject).toLowerCase();
  return type !== "none" && PRICEABLE_SUBJECTS.includes(subject as "front" | "side");
}

export function rejectionMessage(output: VlmOutput): string {
  const type = String(output.truck_type).toLowerCase();
  if (type === "none") {
    return (
      output.truck_type_user_message ||
      "This photo is not a Class 7/8 day-cab, dump, or sleeper truck. Please upload a Class 7/8 truck photo."
    );
  }
  if (String(output.primary_subject).toLowerCase() === "unusable") {
    return (
      output.primary_subject_user_message ||
      "Please upload a clearer photo of the truck that shows the front or side."
    );
  }
  return (
    output.primary_subject_user_message ||
    "Please upload a clearer Class 7/8 truck photo showing the front or side."
  );
}

export function needsBrand(output: VlmOutput): boolean {
  if (!isPriceable(output)) return false;
  return (
    Boolean(output.brand?.needs_user_input) ||
    !output.brand?.has_brand ||
    !output.brand?.brand_name
  );
}

export async function extractFeaturesFromJpeg(jpeg: Buffer): Promise<VlmResult> {
  const anthropic = client();
  const schemaText = JSON.stringify(masterSchema, null, 2);
  const response = await anthropic.messages.create({
    model: process.env.ANTHROPIC_MODEL || DEFAULT_MODEL,
    max_tokens: 8192,
    system: [
      {
        type: "text",
        text:
          "You are a truck-photo feature extractor. Follow this JSON schema and rubric EXACTLY. " +
          'On every call write reasoning first, then output. Return a single JSON object matching TruckFeatureExtractionMaster. ' +
          "Do not wrap it in markdown. Do not guess brand from shape or paint. " +
          "Do not infer condition for parts that are not visible.\n\n" +
          schemaText,
        cache_control: { type: "ephemeral" },
      },
    ],
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: jpeg.toString("base64"),
            },
          },
          {
            type: "text",
            text:
              "Inspect this photo. Return ONLY valid JSON matching TruckFeatureExtractionMaster " +
              '(keys: reasoning, output). Stage order: truck type, primary subject, brand, age, condition.',
          },
        ],
      },
    ],
  });

  const text = response.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();

  if (!text) {
    throw new Error("VLM returned no text");
  }

  try {
    return parseVlmResult(extractJsonObject(text));
  } catch (err) {
    const message = err instanceof Error ? err.message : "Could not parse VLM JSON";
    throw new Error(message);
  }
}
