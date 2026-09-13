"use client";

import dynamic from "next/dynamic";
import type { AnalyzeOk } from "@/lib/types";

const Inner = dynamic(() => import("./ConditionPdfInner").then((m) => m.ConditionPdfInner), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-sm text-slate-400">
      Preparing condition report…
    </div>
  ),
});

type Props = Pick<
  AnalyzeOk,
  | "truck_type"
  | "brand"
  | "era"
  | "model_series"
  | "overall_score"
  | "overall_condition_label"
  | "condition"
>;

export function ConditionPdf(props: Props) {
  return (
    <div className="h-[900px] w-full overflow-hidden rounded-xl border border-slate-700 bg-slate-900">
      <Inner {...props} />
    </div>
  );
}
