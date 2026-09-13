"use client";

import { Blueprint, IconPrinter } from "./ui";
import { type AppraisalView, axisBounds, formatUsd } from "@/lib/appraisal";

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
    <main className="scrollthin" style={{ flex: 1, overflowY: "auto", padding: "40.8px 27.2px 68px" }}>
      <div style={{ maxWidth: 1120, margin: "0 auto" }}>
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            justifyContent: "space-between",
            gap: 20.4,
            flexWrap: "wrap",
            paddingBottom: 20.4,
            borderBottom: "1px solid var(--color-divider)",
          }}
        >
          <div>
            {!view.isSample && (
              <div
                style={{
                  fontSize: 11,
                  letterSpacing: "0.18em",
                  textTransform: "uppercase",
                  color: "var(--color-accent-700)",
                  marginBottom: 6.8,
                }}
              >
                Appraisal report · {view.reportId}
              </div>
            )}
            <h1 style={{ fontSize: 42, margin: 0, lineHeight: 1.05 }}>{view.title}</h1>
            {!view.isSample && (
              <div
                style={{
                  fontSize: 14,
                  color: "color-mix(in srgb, var(--color-text) 60%, transparent)",
                  marginTop: 3.4,
                }}
              >
                {view.subtitle}
              </div>
            )}
          </div>
          <div style={{ display: "flex", gap: 6.8 }}>
            <button type="button" className="btn btn-secondary no-print" onClick={onBack}>
              {view.isSample ? "Back" : "Back to conversation"}
            </button>
            <button type="button" className="btn btn-primary no-print" onClick={() => window.print()}>
              Print report <IconPrinter />
            </button>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1.6fr) minmax(0, 1fr)",
            gap: 27.2,
            marginTop: 34,
            alignItems: "start",
          }}
        >
          <Blueprint className="p-report-value">
            <div style={{ padding: 27.2 }}>
              <div
                style={{
                  fontSize: 11,
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  color: "color-mix(in srgb, var(--color-text) 55%, transparent)",
                }}
              >
                Estimated private-party value
              </div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: 58, lineHeight: 1, margin: "10.2px 0 17px" }}>
                {formatUsd(view.low)} – {formatUsd(view.high)}
              </div>
              <div style={{ position: "relative", height: 42, marginBottom: 10.2 }}>
                <div style={{ position: "absolute", top: 19, left: 0, right: 0, height: 1, background: "var(--color-divider)" }} />
                <div
                  style={{
                    position: "absolute",
                    top: 14,
                    left: `${leftPct}%`,
                    width: `${widthPct}%`,
                    height: 11,
                    background: "var(--color-accent)",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    top: 6,
                    left: `${midPct}%`,
                    width: 1,
                    height: 27,
                    background: "var(--color-accent-900)",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    top: 34,
                    left: `${midPct}%`,
                    transform: "translateX(-50%)",
                    fontSize: 11,
                    whiteSpace: "nowrap",
                    color: "var(--color-accent-900)",
                  }}
                >
                  {formatUsd(view.center)} midpoint
                </div>
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 11.5,
                  color: "color-mix(in srgb, var(--color-text) 50%, transparent)",
                }}
              >
                <span>{formatUsd(min)}</span>
                <span>{formatUsd(max)}</span>
              </div>
            </div>
          </Blueprint>

          <Blueprint>
            <div style={{ padding: 27.2, background: "var(--color-accent-900)", color: "var(--color-bg)", borderRadius: 14 }}>
              <div style={{ fontSize: 11, letterSpacing: "0.16em", textTransform: "uppercase", opacity: 0.75 }}>
                Suggested asking price
              </div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: 52, lineHeight: 1, margin: "10.2px 0" }}>
                {formatUsd(view.asking)}
              </div>
              <p style={{ fontSize: 14, lineHeight: 1.6, opacity: 0.85, margin: 0 }}>
                Listed about 5% above the midpoint, which leaves room to negotiate down to the range
                without going under it.
              </p>
            </div>
          </Blueprint>
        </div>

        <h2 style={{ fontSize: 24, margin: "40.8px 0 13.6px" }}>Brand</h2>
        <Blueprint>
          <div style={{ padding: 24 }}>
            <div
              style={{
                fontSize: 11,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                color: "color-mix(in srgb, var(--color-text) 55%, transparent)",
                marginBottom: 6,
              }}
            >
              Brand name
            </div>
            <div style={{ fontFamily: "var(--font-heading)", fontSize: 32, lineHeight: 1.1, marginBottom: 12 }}>
              {view.brandName}
            </div>
            <p style={{ fontSize: 15, lineHeight: 1.6, margin: 0, color: "color-mix(in srgb, var(--color-text) 72%, transparent)" }}>
              {view.brandPricingReasoning}
            </p>
          </div>
        </Blueprint>

        <h2 style={{ fontSize: 24, margin: "40.8px 0 13.6px" }}>Condition Assessment</h2>
        <div
          style={{
            display: "flex",
            gap: 28,
            flexWrap: "wrap",
            marginBottom: 16,
            fontSize: 13.5,
            color: "color-mix(in srgb, var(--color-text) 68%, transparent)",
          }}
        >
          <span>
            <strong>overall_score</strong>: {view.overallScore ?? "null"}
          </span>
          <span>
            <strong>overall_condition_label</strong>: {view.conditionLabel ?? "null"}
          </span>
          <span>
            <strong>total_penalty_percent</strong>: {view.totalPenaltyPercent}
          </span>
        </div>
        <table className="table" style={{ width: "100%" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>categories</th>
              <th style={{ textAlign: "left" }}>score</th>
              <th style={{ textAlign: "left" }}>weight</th>
              <th style={{ textAlign: "left" }}>penalty_percent</th>
              <th style={{ textAlign: "left" }}>reasoning</th>
            </tr>
          </thead>
          <tbody>
            {view.categories.map((a) => (
              <tr key={a.key}>
                <td style={{ fontFamily: "var(--font-heading)", fontSize: 16 }}>{a.key}</td>
                <td>
                  <span className="tag tag-outline">{a.score == null ? "null" : a.score}</span>
                </td>
                <td>{a.weight}</td>
                <td>{a.penalty_percent}</td>
                <td style={{ fontSize: 13.5, color: "color-mix(in srgb, var(--color-text) 72%, transparent)" }}>
                  {a.reasoning}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <h2 style={{ fontSize: 24, margin: "40.8px 0 13.6px" }}>Photos</h2>
        <PhotoGroup title="Used" photos={usedPhotos} empty="No photographs were used." />
        <PhotoGroup title="Filtered out" photos={filteredPhotos} empty="No photographs were filtered out." />

        <div
          style={{
            marginTop: 40.8,
            paddingTop: 20.4,
            borderTop: "1px solid var(--color-divider)",
            fontSize: 12.5,
            color: "color-mix(in srgb, var(--color-text) 55%, transparent)",
            maxWidth: "70ch",
          }}
        >
          This appraisal is generated by a model from the photos and answers you provided. It is an
          estimate of market value, not an offer to buy, and does not account for mechanical faults
          that are not visible in the images.
        </div>
      </div>
    </main>
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
    <div style={{ marginBottom: 24 }}>
      <div
        style={{
          fontSize: 11,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "color-mix(in srgb, var(--color-text) 55%, transparent)",
          marginBottom: 12,
        }}
      >
        {title}
      </div>
      {photos.length === 0 ? (
        <div style={{ fontSize: 14, color: "color-mix(in srgb, var(--color-text) 55%, transparent)" }}>{empty}</div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 20.4 }}>
          {photos.map((p, i) => (
            <div key={`${p.url ?? p.name}-${p.subject}-${i}`}>
              <Blueprint duotone className="photo-frame">
                <div style={{ position: "relative", aspectRatio: "4 / 3", overflow: "hidden", borderRadius: 14 }}>
                  {p.url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={p.url} alt={p.subject} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                  ) : (
                    <div
                      style={{
                        width: "100%",
                        height: "100%",
                        display: "grid",
                        placeItems: "center",
                        fontSize: 12,
                        color: "color-mix(in srgb, var(--color-text) 45%, transparent)",
                        background: "var(--color-accent-100)",
                      }}
                    >
                      {p.subject}
                    </div>
                  )}
                </div>
              </Blueprint>
              <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <span className="tag tag-outline">{p.subject}</span>
                <span
                  style={{
                    fontSize: 11,
                    letterSpacing: "0.1em",
                    textTransform: "uppercase",
                    color: p.used ? "var(--color-accent-700)" : "color-mix(in srgb, var(--color-text) 50%, transparent)",
                  }}
                >
                  {p.used ? "Used" : "Filtered"}
                </span>
              </div>
              <div style={{ fontSize: 12.5, marginTop: 6, lineHeight: 1.5, color: "color-mix(in srgb, var(--color-text) 65%, transparent)" }}>
                {p.reasoning}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
