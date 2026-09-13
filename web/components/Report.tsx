"use client";

import { Blueprint, IconPrinter } from "./ui";
import {
  type AppraisalView,
  axisBounds,
  formatUsd,
} from "@/lib/appraisal";

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
          <Blueprint className="p-report-value" >
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
                  marginBottom: 20.4,
                }}
              >
                <span>{formatUsd(min)}</span>
                <span>{formatUsd(max)}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: `repeat(${view.isSample ? 2 : 3}, minmax(0, 1fr))`, gap: 10 }}>
                {!view.isSample && (
                  <MiniStat label="Confidence" value={`${view.confidencePct >= 70 ? "High" : view.confidencePct >= 50 ? "Medium" : "Wide"} · ${view.confidencePct}%`} />
                )}
                <MiniStat
                  label="Condition grade"
                  value={
                    view.overallScore != null
                      ? `${view.conditionLabel} (${Number(view.overallScore).toFixed(1)}/5)`
                      : view.conditionLabel
                  }
                />
                <MiniStat label="Photos read" value={`${view.photosRead} of ${view.photosTotal}`} />
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
              <p style={{ fontSize: 14, lineHeight: 1.6, opacity: 0.85, margin: "0 0 13.6px" }}>
                Listed about 5% above the midpoint, which leaves room to negotiate down to the range
                without going under it.
              </p>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                  gap: 16,
                  borderTop: "1px solid color-mix(in srgb, var(--color-bg) 30%, transparent)",
                  paddingTop: 13.6,
                }}
              >
                <div>
                  <div style={{ fontSize: 10.5, letterSpacing: "0.08em", textTransform: "uppercase", opacity: 0.7, whiteSpace: "nowrap" }}>
                    Trade-in
                  </div>
                  <div style={{ fontFamily: "var(--font-heading)", fontSize: 20 }}>{formatUsd(view.tradeIn)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 10.5, letterSpacing: "0.08em", textTransform: "uppercase", opacity: 0.7, whiteSpace: "nowrap" }}>
                    Quick sale
                  </div>
                  <div style={{ fontFamily: "var(--font-heading)", fontSize: 20 }}>{formatUsd(view.quickSale)}</div>
                </div>
              </div>
            </div>
          </Blueprint>
        </div>

        <h2 style={{ fontSize: 24, margin: "40.8px 0 13.6px" }}>Condition by area</h2>
        <table className="table" style={{ width: "100%" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Area</th>
              <th style={{ textAlign: "left" }}>Grade</th>
              <th style={{ textAlign: "left" }}>What the model saw</th>
              <th style={{ textAlign: "right" }}>Value effect</th>
            </tr>
          </thead>
          <tbody>
            {view.areas.map((a) => (
              <tr key={a.area}>
                <td style={{ fontFamily: "var(--font-heading)", fontSize: 16 }}>{a.area}</td>
                <td>
                  <span className="tag tag-outline">{a.grade}</span>
                </td>
                <td style={{ fontSize: 13.5, color: "color-mix(in srgb, var(--color-text) 72%, transparent)" }}>
                  {a.note}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--font-heading)", fontSize: 16 }}>{a.effect}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h2 style={{ fontSize: 24, margin: "40.8px 0 13.6px" }}>Photo evidence</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 20.4 }}>
          {(view.photos.length ? view.photos : [{ url: null, caption: "No photos attached" }]).map((p, i) => (
            <div key={`${p.url ?? "placeholder"}-${p.caption}-${i}`}>
              <Blueprint duotone className="photo-frame">
                <div style={{ position: "relative", aspectRatio: "4 / 3", overflow: "hidden", borderRadius: 14 }}>
                  {p.url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={p.url} alt={p.caption} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
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
                      {p.caption.split("·")[0]}
                    </div>
                  )}
                </div>
              </Blueprint>
              <div style={{ fontSize: 12, marginTop: 6.8, color: "color-mix(in srgb, var(--color-text) 60%, transparent)" }}>
                {p.caption}
              </div>
            </div>
          ))}
        </div>

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

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ background: "#fff", border: "1px solid var(--color-divider)", borderRadius: 14, padding: 13.6 }}>
      <div
        style={{
          fontSize: 10.5,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "color-mix(in srgb, var(--color-text) 55%, transparent)",
        }}
      >
        {label}
      </div>
      <div style={{ fontFamily: "var(--font-heading)", fontSize: 22 }}>{value}</div>
    </div>
  );
}
