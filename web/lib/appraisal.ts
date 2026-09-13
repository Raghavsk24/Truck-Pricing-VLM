import { CAT_KEYS, CAT_LABELS, type AnalyzeOk, type CatKey, type Contribution } from "./types";
import { eraLabel, formatUsd, formatUsdSigned, typeLabel } from "./format";

export type PhotoShot = {
  url: string;
  name: string;
  subject?: string;
};

export type ChecklistItem = {
  key: string;
  label: string;
  done: boolean;
  status: string;
};

export type AreaRow = {
  area: string;
  grade: string;
  note: string;
  effect: string;
};

export type AppraisalView = {
  reportId: string;
  title: string;
  subtitle: string;
  low: number;
  high: number;
  center: number;
  asking: number;
  tradeIn: number;
  quickSale: number;
  confidencePct: number;
  conditionLabel: string;
  overallScore: number | null;
  photosRead: number;
  photosTotal: number;
  areas: AreaRow[];
  photos: { url: string | null; caption: string }[];
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

function round100(n: number): number {
  return Math.round(n / 100) * 100;
}

function scoreGrade(score: number | null, fallback: string | null): string {
  if (fallback) return fallback;
  if (score == null) return "—";
  const rounded = Math.round(score);
  if (rounded >= 5) return "Excellent";
  if (rounded === 4) return "Good";
  if (rounded === 3) return "Fair";
  if (rounded === 2) return "Poor";
  return "Very Poor";
}

function snippetForCategory(explanation: string, key: CatKey): string {
  if (!explanation) return "No visual notes provided.";
  const needles = [CAT_LABELS[key].toLowerCase(), key.replace(/_/g, " ")];
  const parts = explanation.split(/(?<=\.)\s+/);
  const hit = parts.find((p) => {
    const lower = p.toLowerCase();
    return needles.some((n) => lower.includes(n));
  });
  return hit || explanation.slice(0, 160) + (explanation.length > 160 ? "…" : "");
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

export function liveAppraisal(result: AnalyzeOk, photos: PhotoShot[]): AppraisalView {
  const asking = round100(result.center * 1.05);
  const tradeIn = round100(result.low * 0.91);
  const quickSale = round100((result.low + result.center) / 2);
  const effectByKey = Object.fromEntries(result.contributions.map((c) => [c.key, c.usd]));

  const areas: AreaRow[] = CAT_KEYS.map((key) => {
    const cat = result.condition.categories[key];
    const unscored = cat?.score == null;
    const usd = effectByKey[key];
    return {
      area: CAT_LABELS[key],
      grade: unscored ? "Not visible" : scoreGrade(cat.score, null),
      note: unscored
        ? "This part of the truck was not in frame and was excluded from the overall score."
        : snippetForCategory(result.condition.explanation, key),
      effect:
        usd == null || Math.abs(usd) < 50 ? "—" : formatUsdSigned(usd),
    };
  });

  const captions: Record<string, string> = {
    front: "Front three-quarter",
    side: "Driver side",
    back: "Rear three-quarter",
    engine: "Engine bay",
    container: "Body / bed",
    interior: "Interior",
    dashboard: "Odometer / gauges",
    tires: "Tires & wheels",
  };

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
    tradeIn,
    quickSale,
    confidencePct: Math.max(0, Math.min(99, Math.round(100 - result.error_bound_pct))),
    conditionLabel: scoreGrade(result.overall_score, result.overall_condition_label),
    overallScore: result.overall_score,
    photosRead: photos.length,
    photosTotal: 7,
    areas,
    photos: photos.slice(0, 4).map((p) => ({
      url: p.url,
      caption: captions[p.subject || ""] || p.name,
    })),
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
  tradeIn: 29300,
  quickSale: 31500,
  confidencePct: 86,
  conditionLabel: "Good",
  overallScore: 3.8,
  photosRead: 7,
  photosTotal: 7,
  areas: [
    {
      area: "Exterior & paint",
      grade: "Good",
      note: "Consistent gloss across panels, no repaint detected",
      effect: "—",
    },
    {
      area: "Rear bumper",
      grade: "Fair",
      note: "Palm-sized dent, passenger corner, unrepaired",
      effect: "−$600",
    },
    {
      area: "Bed & liner",
      grade: "Good",
      note: "Spray-in liner, light scuffing, no rust",
      effect: "+$300",
    },
    {
      area: "Tires & wheels",
      grade: "Good",
      note: "Even wear, roughly 60% tread remaining",
      effect: "—",
    },
    {
      area: "Interior",
      grade: "Good",
      note: "Driver bolster wear normal for mileage",
      effect: "−$250",
    },
    {
      area: "Glass & lights",
      grade: "Excellent",
      note: "No chips or clouding visible",
      effect: "+$150",
    },
  ],
  photos: [
    { url: null, caption: "Front three-quarter · paint consistent" },
    { url: null, caption: "Rear three-quarter · bumper dent" },
    { url: null, caption: "Interior · seat wear normal" },
    { url: null, caption: "Odometer · 118,400 mi verified" },
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
