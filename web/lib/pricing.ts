import "server-only";
import type {
  CatKey,
  ConditionOutput,
  Contribution,
  PriceRangeModel,
  PriceResult,
  VlmOutput,
} from "./types";
import { CAT_KEYS, CAT_LABELS, CAT_WEIGHTS } from "./types";
import priceModelJson from "@/data/price_range_model.json";
import { eraLabel, formatUsd, typeLabel } from "./format";

export { eraLabel, formatUsd, typeLabel };

const BRAND_ALIASES: Record<string, string> = {
  FREIGHTLINER: "FREIGHTLINER",
  INTERNATIONAL: "INTERNATIONAL",
  NAVISTAR: "INTERNATIONAL",
  "INTERNATIONAL/NAVISTAR": "INTERNATIONAL",
  KENWORTH: "KENWORTH",
  PETERBILT: "PETERBILT",
  MACK: "MACK",
  VOLVO: "VOLVO",
  "WESTERN STAR": "WESTERN STAR",
  WESTERNSTAR: "WESTERN STAR",
  STERLING: "STERLING",
  FORD: "FORD",
  CHEVROLET: "CHEVROLET",
  CHEVY: "CHEVROLET",
  GMC: "GMC",
  HINO: "HINO",
  ISUZU: "ISUZU",
  UD: "UD",
  "NISSAN DIESEL": "UD",
  AUTOCAR: "AUTOCAR",
  OSHKOSH: "OSHKOSH",
};

const TYPE_MAP: Record<string, string> = {
  day_cab: "day_cab",
  "day-cab": "day_cab",
  day_cab_tractor: "day_cab",
  "day-cab-truck": "day_cab",
  day_cab_truck: "day_cab",
  sleeper: "sleeper",
  sleeper_tractor: "sleeper",
  "sleeper-truck": "sleeper",
  sleeper_truck: "sleeper",
  dump: "dump",
  dumper: "dump",
  "dump-truck": "dump",
  dump_truck: "dump",
  heavy_dump_truck: "dump",
  "heavy-dump-truck": "dump",
};

export const model = priceModelJson as PriceRangeModel;

export function canonicalBrand(brand: string | null | undefined): string {
  if (!brand) return model.unknown_brand || "UNKNOWN";
  const b = brand.trim().toUpperCase().replace(/\s+/g, " ");
  return BRAND_ALIASES[b] ?? b;
}

export function normalizeTruckType(value: string | null | undefined): string {
  if (!value) return "other";
  const v = value.trim().toLowerCase().replace(/ /g, "_");
  const hyphen = v.replace(/_/g, "-");
  if (TYPE_MAP[v]) return TYPE_MAP[v];
  if (TYPE_MAP[hyphen]) return TYPE_MAP[hyphen];
  if (v.includes("dump")) return "dump";
  if (v.includes("sleeper")) return "sleeper";
  if (v.includes("day") && v.includes("cab")) return "day_cab";
  return "other";
}

export function normalizeEra(value: string | null | undefined): string {
  if (!value) return model.unknown_era || "unknown";
  const v = value
    .trim()
    .toLowerCase()
    .replace(/-/g, "_")
    .replace(/ /g, "_")
    .replace(/\+/g, "_plus");
  const aliases: Record<string, string> = {
    pre_2010: "pre_2010",
    pre2010: "pre_2010",
    before_2010: "pre_2010",
    "2010_2015": "2010_2015",
    "2016_2020": "2016_2020",
    "2021_plus": "2021_plus",
    "2021plus": "2021_plus",
    "2021_": "2021_plus",
    unknown: "unknown",
  };
  return aliases[v] ?? "unknown";
}

function roundMoney(n: number): number {
  return Math.round(n * 100) / 100;
}

function lookupMus(
  truckType: string,
  brand: string,
  era: string,
): { globalMu: number; muType: number; muBrand: number; muEra: number } {
  const { cells } = model;
  const globalMu = cells.global_mu;
  const muType = cells.types[truckType]?.mu ?? globalMu;
  const muBrand = cells.type_brand[`${truckType}|${brand}`]?.mu ?? muType;
  if (!era || era === (model.unknown_era || "unknown")) {
    return { globalMu, muType, muBrand, muEra: muBrand };
  }
  const muEra = cells.type_brand_era[`${truckType}|${brand}|${era}`]?.mu ?? muBrand;
  return { globalMu, muType, muBrand, muEra };
}

const MIN_ERROR_BOUND = 0.15;
const MAX_ERROR_BOUND = 0.4;

function clampErrorBound(bound: number): number {
  return Math.max(MIN_ERROR_BOUND, Math.min(MAX_ERROR_BOUND, bound));
}

export function lookupErrorBound(truckType: string, era: string): number {
  const table = model.error_table;
  const hit = table.type_era?.[`${truckType}|${era}`];
  if (hit) return clampErrorBound(hit.bound);
  const byType = table.types?.[truckType];
  if (byType) return clampErrorBound(byType.bound);
  return clampErrorBound(table.global.bound);
}

function splitConditionUsd(
  condTotal: number,
  condition: ConditionOutput,
  scoreBar: number,
): Contribution[] {
  const scored = CAT_KEYS.filter((k) => condition.categories?.[k]?.score != null);
  if (scored.length === 0) {
    return [{ key: "condition", label: "Condition", usd: condTotal }];
  }

  let wsum = 0;
  const weights: Partial<Record<CatKey, number>> = {};
  for (const k of scored) {
    const w = Number(condition.categories[k].weight) || CAT_WEIGHTS[k];
    weights[k] = w;
    wsum += w;
  }
  if (wsum <= 0) wsum = 1;

  const pulls: Partial<Record<CatKey, number>> = {};
  let pullSum = 0;
  for (const k of scored) {
    const score = Number(condition.categories[k].score);
    const pull = ((weights[k] ?? 0) / wsum) * (score - scoreBar);
    pulls[k] = pull;
    pullSum += pull;
  }

  if (Math.abs(pullSum) < 1e-9) {
    return scored.map((k) => ({
      key: k,
      label: CAT_LABELS[k],
      usd: condTotal * ((weights[k] ?? 0) / wsum),
    }));
  }

  return scored.map((k) => ({
    key: k,
    label: CAT_LABELS[k],
    usd: condTotal * ((pulls[k] ?? 0) / pullSum),
  }));
}

function reconcile(parts: Contribution[], target: number): Contribution[] {
  const rounded = parts.map((p) => ({ ...p, usd: roundMoney(p.usd) }));
  const drift = roundMoney(target - rounded.reduce((s, p) => s + p.usd, 0));
  if (rounded.length && drift !== 0) {
    rounded[rounded.length - 1] = {
      ...rounded[rounded.length - 1],
      usd: roundMoney(rounded[rounded.length - 1].usd + drift),
    };
  }
  return rounded;
}

export function predictFromFeatures(
  truckTypeRaw: string,
  brandRaw: string | null,
  eraRaw: string | null,
  condition: ConditionOutput,
  modelSeries?: string | null,
): PriceResult {
  const truck_type = normalizeTruckType(truckTypeRaw);
  const brand = canonicalBrand(brandRaw);
  const era = normalizeEra(eraRaw);
  const { globalMu, muType, muBrand, muEra } = lookupMus(truck_type, brand, era);

  const score = condition.overall_score;
  const { beta, score_bar, condition_scale } = model.condition;
  let effect = 0;
  if (score != null && Number.isFinite(score)) {
    effect = condition_scale * beta * (Number(score) - score_bar);
  }

  const center = Math.exp(muEra + effect);
  const error_bound = lookupErrorBound(truck_type, era);
  const low = Math.max(0, center * (1 - error_bound));
  const high = center * (1 + error_bound);

  const baseline = Math.exp(globalMu);
  const typeUsd = Math.exp(muType) - baseline;
  const brandUsd = Math.exp(muBrand) - Math.exp(muType);
  const eraUsd = Math.exp(muEra) - Math.exp(muBrand);
  const condTotal = center - Math.exp(muEra);

  const contributions = reconcile(
    [
      { key: "type", label: "Truck type", usd: typeUsd },
      { key: "brand", label: "Brand", usd: brandUsd },
      { key: "era", label: "Era / age", usd: eraUsd },
      ...splitConditionUsd(condTotal, condition, score_bar),
    ],
    center - baseline,
  );

  return {
    truck_type,
    brand,
    era,
    model_series: modelSeries ?? null,
    overall_score: score,
    overall_condition_label: condition.overall_condition_label,
    center: roundMoney(center),
    low: roundMoney(low),
    high: roundMoney(high),
    error_bound,
    error_bound_pct: roundMoney(100 * error_bound),
    contributions,
    condition,
  };
}

export function priceFromVlm(output: VlmOutput, brandOverride?: string): PriceResult {
  const brand = brandOverride || output.brand?.brand_name || null;
  return predictFromFeatures(
    String(output.truck_type),
    brand,
    output.age?.era ?? null,
    output.condition,
    output.age?.model_series ?? null,
  );
}
