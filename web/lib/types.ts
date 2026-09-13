export const PRICEABLE_SUBJECTS = ["front", "side"] as const;

export const CAT_KEYS = [
  "chassis_and_frame",
  "front_end_engine_compartment_hood",
  "tires_wheels_suspension",
  "cab_sleeper_aero_fairings",
] as const;

export type CatKey = (typeof CAT_KEYS)[number];

export const CAT_WEIGHTS: Record<CatKey, number> = {
  chassis_and_frame: 0.35,
  front_end_engine_compartment_hood: 0.25,
  tires_wheels_suspension: 0.2,
  cab_sleeper_aero_fairings: 0.2,
};

export const CAT_LABELS: Record<CatKey, string> = {
  chassis_and_frame: "Chassis & frame",
  front_end_engine_compartment_hood: "Front end / hood",
  tires_wheels_suspension: "Tires / wheels",
  cab_sleeper_aero_fairings: "Cab / sleeper / aero",
};

export type TruckTypeVlm =
  | "day-cab-truck"
  | "dump-truck"
  | "sleeper-truck"
  | "none";

export type TruckType = "day_cab" | "sleeper" | "dump";

export type PrimarySubject =
  | "front"
  | "side"
  | "back"
  | "engine"
  | "container"
  | "interior"
  | "dashboard"
  | "tires"
  | "unusable";

export type CategoryScore = {
  score: number | null;
  weight: number;
  penalty_percent: number;
};

export type ConditionOutput = {
  categories: Record<CatKey, CategoryScore>;
  overall_score: number | null;
  overall_condition_label: string | null;
  total_penalty_percent: number;
  explanation: string;
};

export type BrandOutput = {
  has_brand: boolean;
  brand_name: string | null;
  visible_identifier_type: string;
  identifier_location: string | null;
  brand_list_verified: boolean;
  confidence: string;
  reasoning: string;
  needs_user_input: boolean;
  user_prompt: string | null;
};

export type AgeOutput = {
  model_series: string | null;
  era: string;
  era_confidence: string;
  era_evidence: string;
};

export type VlmOutput = {
  truck_type: TruckTypeVlm | string;
  truck_type_user_message: string;
  primary_subject: PrimarySubject | string;
  primary_subject_user_message: string;
  brand: BrandOutput;
  age: AgeOutput;
  condition: ConditionOutput;
};

export type VlmResult = {
  reasoning: string;
  output: VlmOutput;
};

export type Contribution = {
  key: string;
  label: string;
  usd: number;
};

export type PriceResult = {
  truck_type: TruckType | string;
  brand: string;
  era: string;
  model_series: string | null;
  overall_score: number | null;
  overall_condition_label: string | null;
  center: number;
  low: number;
  high: number;
  error_bound: number;
  error_bound_pct: number;
  contributions: Contribution[];
  condition: ConditionOutput;
};

export type AnalyzeOk = {
  status: "ok";
  analysisId: string;
  previewHint?: string;
} & PriceResult;

export type AnalyzeNeedsBrand = {
  status: "needs_brand";
  analysisId: string;
  user_prompt: string;
  truck_type: string;
  era: string;
  model_series: string | null;
  condition: ConditionOutput;
};

export type AnalyzeRejected = {
  status: "rejected";
  user_message: string;
};

export type AnalyzeError = {
  status: "error";
  user_message: string;
};

export type AnalyzeResponse =
  | AnalyzeOk
  | AnalyzeNeedsBrand
  | AnalyzeRejected
  | AnalyzeError;

export type CellMu = { n: number; mu: number };

export type PriceRangeModel = {
  model_family: string;
  condition: {
    beta: number;
    score_bar: number;
    condition_scale: number;
    n: number;
  };
  error_table: {
    type_era: Record<string, { n: number; bound: number }>;
    types: Record<string, { n: number; bound: number }>;
    global: { n: number; bound: number };
  };
  cells: {
    global_mu: number;
    types: Record<string, CellMu>;
    type_brand: Record<string, CellMu>;
    type_brand_era: Record<string, CellMu>;
  };
  unknown_brand: string;
  unknown_era: string;
};
