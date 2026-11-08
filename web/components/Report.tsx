"use client";

import { IconPrinter } from "./ui";
import { type AppraisalView, axisBounds, formatUsd } from "@/lib/appraisal";
import { CAT_LABELS } from "@/lib/types";

const pct = (n: number) => `${Math.round(n * 100)}%`;

export function Report({
  view,
  onBack,
}: {
  view: AppraisalView;
  onBack: () => void;
}) {
  const { min, max } = axisBounds(view.low, view.high);
  const span = max - min || 1;
  const leftPct = ((view.low - min) / span) * 100;
  const widthPct = ((view.high - view.low) / span) * 100;
  const midPct = ((view.center - min) / span) * 100;
  const usedPhotos = view.photos.filter((p) => p.used);
  const filteredPhotos = view.photos.filter((p) => !p.used);

  return (
    <main className="page scrollthin">
      <div className="container report">
        <div className="report-head">
          <div>
            <div className="eyebrow">
              {view.isSample ? "Sample report" : `Appraisal report · ${view.reportId}`}
            </div>
            <h1>{view.title}</h1>
            {!view.isSample && <p className="muted" style={{ marginTop: 6 }}>{view.subtitle}</p>}
          </div>
          <div style={{ display: "flex", gap: 8 }} className="no-print">
            <button type="button" className="btn btn-secondary" onClick={onBack}>
              {view.isSample ? "Back" : "Back to conversation"}
            </button>
            <button type="button" className="btn btn-primary" onClick={() => window.print()}>
              Print report <IconPrinter />
            </button>
          </div>
        </div>

        <div className="report-grid">
          <div className="card">
            <div className="label">Estimated private-party value</div>
            <div className="big num">
              {formatUsd(view.low)} – {formatUsd(view.high)}
            </div>
            <div style={{ position: "relative", height: 44 }}>
              <div style={{ position: "absolute", top: 8, left: 0, right: 0, height: 6, borderRadius: 3, background: "#eceded" }} />
              <div style={{ position: "absolute", top: 8, left: `${leftPct}%`, width: `${widthPct}%`, height: 6, borderRadius: 3, background: "var(--accent)" }} />
              <div style={{ position: "absolute", top: 2, left: `${midPct}%`, width: 2, height: 18, background: "var(--text)" }} />
              <div className="small num" style={{ position: "absolute", top: 26, left: `${midPct}%`, transform: "translateX(-50%)", whiteSpace: "nowrap" }}>
                {formatUsd(view.center)} midpoint
              </div>
            </div>
            <div className="small muted num" style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
              <span>{formatUsd(min)}</span>
              <span>{formatUsd(max)}</span>
            </div>
          </div>

          <div className="card card-accent">
            <div className="label">Suggested asking price</div>
            <div className="big num" style={{ marginBottom: 12 }}>{formatUsd(view.asking)}</div>
            <p className="muted" style={{ fontSize: 14 }}>
              Listed about 5% above the midpoint, which leaves room to negotiate down to the range
              without going under it.
            </p>
          </div>
        </div>

        <section className="report-section">
          <h2>Brand</h2>
          <div className="card">
            <div className="stat-value" style={{ fontSize: 24 }}>{view.brandName}</div>
            <p className="muted" style={{ marginTop: 8 }}>{view.brandPricingReasoning}</p>
          </div>
        </section>

        <section className="report-section">
          <h2>Condition assessment</h2>
          <div className="stats">
            <Stat label="Overall condition" value={view.conditionLabel ?? "N/A"} />
            <Stat label="Overall score" value={view.overallScore == null ? "N/A" : String(view.overallScore)} />
            <Stat label="Total penalty" value={`${view.totalPenaltyPercent}%`} />
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Area</th>
                  <th>Score</th>
                  <th>Weight</th>
                  <th>Penalty</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {view.categories.map((a) => (
                  <tr key={a.key}>
                    <td style={{ fontWeight: 600 }}>{CAT_LABELS[a.key]}</td>
                    <td className="num">{a.score == null ? "N/A" : a.score}</td>
                    <td className="num">{pct(a.weight)}</td>
                    <td className="num">{a.penalty_percent}%</td>
                    <td className="muted">{a.reasoning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="report-section">
          <h2>Photos</h2>
          <PhotoGroup title="Used" photos={usedPhotos} empty="No photographs were used." />
          <PhotoGroup title="Filtered out" photos={filteredPhotos} empty="No photographs were filtered out." />
        </section>

        <p className="disclaimer">
          This appraisal is generated by a model from the photos and answers you provided. It is an
          estimate of market value, not an offer to buy, and does not account for mechanical faults
          that are not visible in the images.
        </p>
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="stat-value num">{value}</div>
    </div>
  );
}

function PhotoGroup({
  title,
  photos,
  empty,
}: {
  title: string;
  photos: AppraisalView["photos"];
  empty: string;
}) {
  return (
    <div style={{ marginBottom: 28 }}>
      <h3 style={{ marginBottom: 12 }}>{title}</h3>
      {photos.length === 0 ? (
        <p className="muted small">{empty}</p>
      ) : (
        <div className="photo-grid">
          {photos.map((p, i) => (
            <div key={`${p.url ?? p.name}-${p.subject}-${i}`}>
              {p.url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img className="photo" src={p.url} alt={p.subject} />
              ) : (
                <div className="photo-empty">{p.subject}</div>
              )}
              <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 8 }}>
                <span className="tag" style={{ textTransform: "capitalize" }}>{p.subject}</span>
                <span className={`tag ${p.used ? "tag-accent" : ""}`}>{p.used ? "Used" : "Filtered"}</span>
              </div>
              <p className="muted small" style={{ marginTop: 6 }}>{p.reasoning}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
