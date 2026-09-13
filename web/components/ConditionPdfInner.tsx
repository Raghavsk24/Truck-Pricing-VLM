"use client";

import { PDFViewer } from "@react-pdf/renderer";
import { ConditionDocument } from "./ConditionDocument";
import type { AnalyzeOk } from "@/lib/types";

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

export function ConditionPdfInner(props: Props) {
  return (
    <PDFViewer width="100%" height="100%" showToolbar style={{ border: 0 }}>
      <ConditionDocument {...props} />
    </PDFViewer>
  );
}
