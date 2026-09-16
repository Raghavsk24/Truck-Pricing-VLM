"use client";

import { BrandMark } from "./ui";

type Screen = "landing" | "chat" | "report";

export function AppHeader({
  screen,
  onLogo,
  onNew,
}: {
  screen: Screen;
  onLogo: () => void;
  onNew: () => void;
}) {
  return (
    <header
      className="no-print"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 20.4,
        padding: "13.6px 27.2px",
        borderBottom: "1px solid var(--color-divider)",
        flexWrap: "wrap",
      }}
    >
      <button
        type="button"
        onClick={onLogo}
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 10.2,
          cursor: "pointer",
          background: "none",
          border: 0,
          padding: 0,
          color: "inherit",
        }}
      >
        <BrandMark />
        <span
          style={{
            fontFamily: "var(--font-heading)",
            fontWeight: 600,
            fontSize: 20,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
          }}
        >
          Truck Pricing VLM
        </span>
        <span
          style={{
            fontSize: 11,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "color-mix(in srgb, var(--color-text) 55%, transparent)",
          }}
        >
          Truck appraisal
        </span>
      </button>
      <div style={{ display: "flex", alignItems: "center", gap: 6.8 }}>
        <span
          style={{
            fontSize: 13,
            color: "color-mix(in srgb, var(--color-text) 55%, transparent)",
          }}
        >
          Appraisals are free · no account needed to start
        </span>
        {screen !== "landing" && (
          <button type="button" className="btn btn-secondary" onClick={onNew} style={{ fontSize: 13 }}>
            New appraisal
          </button>
        )}
      </div>
    </header>
  );
}
