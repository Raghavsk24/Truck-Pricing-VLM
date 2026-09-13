import {
  CAT_KEYS,
  CAT_LABELS,
  PRICEABLE_SUBJECTS,
  type AnalyzeOk,
  type AnalyzeResponse,
  type CatKey,
  type Contribution,
} from "./types";
import { eraLabel, formatUsd, formatUsdSigned, typeLabel } from "./format";

export type PhotoShot = {
  url: string;
  name: string;
  subject?: string;
  used?: boolean;
  reasoning?: string;
};

export type ChecklistItem = {
  key: string;
  label: string;
  done: boolean;
  status: string;
};

export type ConditionCategoryRow = {
  key: CatKey;
  score: number | null;
  weight: number;
  penalty_percent: number;
  reasoning: string;
};

export type ReportPhoto = {
  url: string | null;
  name: string;
  subject: string;
  used: boolean;
  reasoning: string;
};

export type AppraisalView = {
  reportId: string;
  title: string;
  subtitle: string;
  low: number;
  high: number;
  center: number;
  asking: number;
  brandName: string;
  brandPricingReasoning: string;
  conditionLabel: string | null;
  overallScore: number | null;
  totalPenaltyPercent: number;
  categories: ConditionCategoryRow[];
  photos: ReportPhoto[];
  contributions: Contribution[];
  vehicleLine: string;
  specLine: string;
  isSample: boolean;
};

const CHECKLIST: { key: string; label: string; optional?: boolean }[] = [
  { key: "front", label: "Front three-quarter" },
  { key: "side", label: "Driver side" },
  { key: "container", label: "Bed / body" },
  { key: "back", label: "Rear three-quarter" },
  { key: "interior", label: "Interior" },
  { key: "dashboard", label: "Odometer" },
  { key: "engine", label: "Engine bay", optional: true },
];

export const OEM_CHIPS = [
  "Freightliner",
  "Kenworth",
  "Peterbilt",
  "Mack",
  "International",
  "Volvo",
];

const CATEGORY_HINTS: Record<CatKey, string[]> = {
  chassis_and_frame: ["chassis", "frame", "leaf spring", "crossmember", "suspension hardware"],
  front_end_engine_compartment_hood: ["front end", "front-end", "grille", "hood", "bumper", "engine"],
  tires_wheels_suspension: ["tire", "tread", "rim", "wheel"],
  cab_sleeper_aero_fairings: ["cab", "sleeper", "aero", "fairing", "interior", "skirt", "extender"],
};

function round100(n: number): number {
  return Math.round(n / 100) * 100;
}

function snippetForCategory(explanation: string, key: CatKey): string {
  if (!explanation) return "No visual notes provided.";
  const needles = [
    CAT_LABELS[key].toLowerCase(),
    key.replace(/_/g, " "),
    ...CATEGORY_HINTS[key],
  ];
  const parts = explanation.split(/(?<=\.)\s+/);
  const hit = parts.find((p) => {
    const lower = p.toLowerCase();
    return needles.some((n) => lower.includes(n));
  });
  return hit || explanation;
}

export function buildChecklist(photos: PhotoShot[]): ChecklistItem[] {
  const subjects = new Set(photos.map((p) => (p.subject || "").toLowerCase()).filter(Boolean));
  const doneCount = CHECKLIST.filter((c) => subjects.has(c.key)).length;
  return CHECKLIST.map((c) => {
    const done = subjects.has(c.key);
    return {
      key: c.key,
      label: c.label,
      done,
      status: done ? "Read" : c.optional ? "Optional" : doneCount > 0 ? "Suggested" : "Needed",
    };
  });
}

export function vehicleTitle(result: Pick<AnalyzeOk, "brand" | "era" | "model_series" | "truck_type">): string {
  const series = result.model_series ? ` ${result.model_series}` : "";
  return `${result.brand}${series}`.trim() || typeLabel(String(result.truck_type));
}

export function isUsedSubject(subject: string | undefined): boolean {
  return PRICEABLE_SUBJECTS.includes((subject || "").toLowerCase() as "front" | "side");
}

export function photoMetaFromResponse(data: AnalyzeResponse): Pick<PhotoShot, "subject" | "used" | "reasoning"> {
  if (data.status === "ok" || data.status === "needs_brand") {
    const subject = String(data.primary_subject || "unusable");
    const used = isUsedSubject(subject);
    return {
      subject,
      used,
      reasoning: used
        ? `Primary subject is ${subject}. Front and side photographs are used for the appraisal.`
        : data.primary_subject_user_message ||
          `Primary subject is ${subject}. Only front and side photographs are used for pricing.`,
    };
  }
  if (data.status === "rejected") {
    const subject = String(data.primary_subject || "unusable");
    return {
      subject,
      used: false,
      reasoning:
        data.truck_type_user_message ||
        data.primary_subject_user_message ||
        `Primary subject is ${subject}. ${data.user_message}`,
    };
  }
  return {
    subject: "unusable",
    used: false,
    reasoning: data.user_message,
  };
}

function brandPricingReasoning(result: AnalyzeOk): string {
  const brandUsd = result.contributions.find((c) => c.key === "brand")?.usd ?? 0;
  const name = result.brand || "Unknown";
  const type = typeLabel(String(result.truck_type));
  const seen = result.brand_reasoning?.trim();
  const effect =
    Math.abs(brandUsd) < 50
      ? `${name} listings for this ${type} sit near the type average, so brand did not move the midpoint.`
      : brandUsd > 0
        ? `${name} listings for this ${type} typically ask more than the type average, which raised the midpoint by ${formatUsd(brandUsd)}.`
        : `${name} listings for this ${type} typically ask less than the type average, which lowered the midpoint by ${formatUsd(Math.abs(brandUsd))}.`;
  return seen ? `${effect} ${seen}` : effect;
}

function categoryRows(result: AnalyzeOk): ConditionCategoryRow[] {
  return CAT_KEYS.map((key) => {
    const cat = result.condition.categories[key];
    return {
      key,
      score: cat?.score ?? null,
      weight: cat?.weight ?? 0,
      penalty_percent: cat?.penalty_percent ?? 0,
      reasoning: snippetForCategory(result.condition.explanation, key),
    };
  });
}

function reportPhotos(photos: PhotoShot[]): ReportPhoto[] {
  return photos.map((p) => ({
    url: p.url,
    name: p.name,
    subject: p.subject || "unusable",
    used: Boolean(p.used),
    reasoning:
      p.reasoning ||
      (p.used
        ? `Primary subject is ${p.subject}. Front and side photographs are used for the appraisal.`
        : `Primary subject is ${p.subject || "unusable"}. This photograph was filtered out.`),
  }));
}

export function liveAppraisal(result: AnalyzeOk, photos: PhotoShot[]): AppraisalView {
  const asking = round100(result.center * 1.05);

  return {
    reportId: `RGL-${result.analysisId.slice(0, 4).toUpperCase()}`,
    title: vehicleTitle(result),
    subtitle: [
      typeLabel(String(result.truck_type)),
      eraLabel(result.era),
      result.model_series,
      new Date().toLocaleDateString("en-US", {
        day: "numeric",
        month: "short",
        year: "numeric",
      }),
    ]
      .filter(Boolean)
      .join(" · "),
    low: result.low,
    high: result.high,
    center: result.center,
    asking,
    brandName: result.brand,
    brandPricingReasoning: brandPricingReasoning(result),
    conditionLabel: result.overall_condition_label,
    overallScore: result.overall_score,
    totalPenaltyPercent: result.condition.total_penalty_percent,
    categories: categoryRows(result),
    photos: reportPhotos(photos),
    contributions: result.contributions,
    vehicleLine: vehicleTitle(result),
    specLine: `${typeLabel(String(result.truck_type))} · ${eraLabel(result.era)}`,
    isSample: false,
  };
}

export const SAMPLE_APPRAISAL: AppraisalView = {
  reportId: "RGL-4471",
  title: "2019 Ram 2500 Tradesman",
  subtitle: "Crew cab · 6.4L V8 · 4x4 · 118,400 mi · Boise, ID · 12 Sep 2026",
  low: 32400,
  high: 35600,
  center: 34100,
  asking: 35900,
  brandName: "RAM",
  brandPricingReasoning:
    "RAM listings for this class typically ask less than the type average, which lowered the midpoint by $1,400. RAM wordmark on the grille was readable and used as the brand cell in the price model.",
  conditionLabel: "Good",
  overallScore: 4,
  totalPenaltyPercent: 0,
  categories: [
    {
      key: "chassis_and_frame",
      score: 4,
      weight: 0.35,
      penalty_percent: 0,
      reasoning: "Chassis visible portion looks clean with only light suspension discoloration (4).",
    },
    {
      key: "front_end_engine_compartment_hood",
      score: 4,
      weight: 0.25,
      penalty_percent: 0,
      reasoning: "Front end intact with minor scuffs (4).",
    },
    {
      key: "tires_wheels_suspension",
      score: 4,
      weight: 0.2,
      penalty_percent: 0,
      reasoning: "Tires show deep tread (4).",
    },
    {
      key: "cab_sleeper_aero_fairings",
      score: 4,
      weight: 0.2,
      penalty_percent: 0,
      reasoning: "Exterior cab/aero intact; interior not visible so scored on exterior only (4).",
    },
  ],
  photos: [
    {
      url: null,
      name: "front.jpg",
      subject: "front",
      used: true,
      reasoning: "Primary subject is front. Front and side photographs are used for the appraisal.",
    },
    {
      url: null,
      name: "side.jpg",
      subject: "side",
      used: true,
      reasoning: "Primary subject is side. Front and side photographs are used for the appraisal.",
    },
    {
      url: null,
      name: "interior.jpg",
      subject: "interior",
      used: false,
      reasoning: "Primary subject is interior. Only front and side photographs are used for pricing.",
    },
    {
      url: null,
      name: "odometer.jpg",
      subject: "dashboard",
      used: false,
      reasoning: "Primary subject is dashboard. Only front and side photographs are used for pricing.",
    },
  ],
  contributions: [],
  vehicleLine: "2019 Ram 2500 Tradesman",
  specLine: "Crew cab · 6.4L V8 · 4x4",
  isSample: true,
};

export function axisBounds(low: number, high: number): { min: number; max: number } {
  const pad = Math.max(4000, (high - low) * 0.8);
  return {
    min: Math.max(0, round100(low - pad)),
    max: round100(high + pad),
  };
}

export { formatUsd, formatUsdSigned };
