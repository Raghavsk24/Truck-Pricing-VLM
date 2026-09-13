import { Document, Page, Text, View, StyleSheet } from "@react-pdf/renderer";
import type { AnalyzeOk } from "@/lib/types";
import { CAT_KEYS, CAT_LABELS } from "@/lib/types";
import { eraLabel, typeLabel } from "@/lib/format";

const styles = StyleSheet.create({
  page: {
    paddingTop: 28,
    paddingBottom: 24,
    paddingHorizontal: 32,
    fontFamily: "Helvetica",
    fontSize: 9.5,
    color: "#0f172a",
    backgroundColor: "#ffffff",
  },
  headerBar: {
    backgroundColor: "#0f2744",
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginBottom: 14,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  brand: {
    color: "#f8fafc",
    fontSize: 16,
    fontFamily: "Helvetica-Bold",
    letterSpacing: 1.2,
  },
  headerSub: {
    color: "#cbd5e1",
    fontSize: 8,
    marginTop: 2,
  },
  meta: {
    color: "#e2e8f0",
    fontSize: 8,
    textAlign: "right",
  },
  identity: {
    flexDirection: "row",
    marginBottom: 12,
    gap: 8,
  },
  chip: {
    borderWidth: 1,
    borderColor: "#cbd5e1",
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 3,
  },
  chipLabel: {
    fontSize: 7,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 0.6,
  },
  chipValue: {
    fontSize: 10,
    fontFamily: "Helvetica-Bold",
    marginTop: 1,
  },
  scoreRow: {
    flexDirection: "row",
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  scoreBox: {
    width: 110,
    backgroundColor: "#0f2744",
    padding: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  scoreNum: {
    color: "#fbbf24",
    fontSize: 28,
    fontFamily: "Helvetica-Bold",
  },
  scoreDenom: {
    color: "#94a3b8",
    fontSize: 9,
  },
  scoreLabel: {
    color: "#f8fafc",
    fontSize: 11,
    fontFamily: "Helvetica-Bold",
    marginTop: 2,
  },
  scoreCopy: {
    flex: 1,
    padding: 10,
    justifyContent: "center",
  },
  scoreCopyTitle: {
    fontFamily: "Helvetica-Bold",
    fontSize: 11,
    marginBottom: 4,
  },
  scoreCopyBody: {
    color: "#334155",
    lineHeight: 1.35,
  },
  sectionTitle: {
    fontFamily: "Helvetica-Bold",
    fontSize: 10,
    marginBottom: 6,
    color: "#0f2744",
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  catRow: {
    flexDirection: "row",
    borderBottomWidth: 1,
    borderBottomColor: "#e2e8f0",
    paddingVertical: 6,
    alignItems: "flex-start",
  },
  catName: {
    width: 130,
    fontFamily: "Helvetica-Bold",
    fontSize: 9,
  },
  catMeta: {
    width: 150,
    fontSize: 8,
    color: "#475569",
  },
  catNotes: {
    flex: 1,
    fontSize: 8,
    color: "#334155",
    lineHeight: 1.3,
  },
  explanation: {
    marginTop: 10,
    padding: 8,
    backgroundColor: "#f8fafc",
    borderWidth: 1,
    borderColor: "#e2e8f0",
    lineHeight: 1.35,
    fontSize: 8.5,
    color: "#1e293b",
  },
  footer: {
    position: "absolute",
    bottom: 16,
    left: 32,
    right: 32,
    fontSize: 7.5,
    color: "#64748b",
    borderTopWidth: 1,
    borderTopColor: "#e2e8f0",
    paddingTop: 6,
  },
});

function snippetForCategory(explanation: string, key: string, label: string): string {
  if (!explanation) return "No visual notes provided.";
  const needles = [label.toLowerCase(), key.replace(/_/g, " ")];
  const parts = explanation.split(/(?<=\.)\s+/);
  const hit = parts.find((p) => {
    const lower = p.toLowerCase();
    return needles.some((n) => lower.includes(n));
  });
  return hit || explanation.slice(0, 180) + (explanation.length > 180 ? "…" : "");
}

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

export function ConditionDocument({
  truck_type,
  brand,
  era,
  model_series,
  overall_score,
  overall_condition_label,
  condition,
}: Props) {
  const date = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  return (
    <Document title="Kamion condition assessment">
      <Page size="LETTER" style={styles.page} wrap={false}>
        <View style={styles.headerBar}>
          <View>
            <Text style={styles.brand}>KAMION</Text>
            <Text style={styles.headerSub}>Vehicle condition assessment</Text>
          </View>
          <View>
            <Text style={styles.meta}>{date}</Text>
            <Text style={styles.meta}>Rubric: TruckConditionAssessment</Text>
          </View>
        </View>

        <View style={styles.identity}>
          <View style={styles.chip}>
            <Text style={styles.chipLabel}>Type</Text>
            <Text style={styles.chipValue}>{typeLabel(String(truck_type))}</Text>
          </View>
          <View style={styles.chip}>
            <Text style={styles.chipLabel}>Brand</Text>
            <Text style={styles.chipValue}>{brand}</Text>
          </View>
          <View style={styles.chip}>
            <Text style={styles.chipLabel}>Era</Text>
            <Text style={styles.chipValue}>{eraLabel(era)}</Text>
          </View>
          <View style={styles.chip}>
            <Text style={styles.chipLabel}>Series</Text>
            <Text style={styles.chipValue}>{model_series || "Not read"}</Text>
          </View>
        </View>

        <View style={styles.scoreRow}>
          <View style={styles.scoreBox}>
            <Text style={styles.scoreNum}>{overall_score ?? "—"}</Text>
            <Text style={styles.scoreDenom}>out of 5</Text>
            <Text style={styles.scoreLabel}>{overall_condition_label || "Unscored"}</Text>
          </View>
          <View style={styles.scoreCopy}>
            <Text style={styles.scoreCopyTitle}>Overall condition</Text>
            <Text style={styles.scoreCopyBody}>
              Weighted average of scored categories (chassis 35%, front end 25%, tires 20%,
              cab/aero 20%). Categories that were not visible are excluded and do not drag
              the score down. Total estimated value deduction:{" "}
              {Number(condition.total_penalty_percent || 0).toFixed(1)}%.
            </Text>
          </View>
        </View>

        <Text style={styles.sectionTitle}>Category scores</Text>
        {CAT_KEYS.map((key) => {
          const cat = condition.categories[key];
          const unscored = cat?.score == null;
          return (
            <View key={key} style={styles.catRow} wrap={false}>
              <Text style={styles.catName}>{CAT_LABELS[key]}</Text>
              <Text style={styles.catMeta}>
                {unscored
                  ? "Not visible — excluded"
                  : `Score ${cat.score}/5   weight ${Math.round(cat.weight * 100)}%   penalty ${cat.penalty_percent}%`}
              </Text>
              <Text style={styles.catNotes}>
                {unscored
                  ? "This part of the truck was not in frame, so it was disregarded from the overall score."
                  : snippetForCategory(condition.explanation, key, CAT_LABELS[key])}
              </Text>
            </View>
          );
        })}

        <Text style={[styles.sectionTitle, { marginTop: 10 }]}>Inspector notes</Text>
        <Text style={styles.explanation}>
          {condition.explanation || "No explanation was returned for this photo."}
        </Text>

        <Text style={styles.footer}>
          Scores follow the TruckConditionAssessment rubric (1 = worst, 5 = best). Null
          categories were excluded, not guessed. This one-page report is generated from a
          single vision pass on the listing photo.
        </Text>
      </Page>
    </Document>
  );
}
