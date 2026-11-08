"use client";

import type { ReactNode } from "react";
import { SAMPLE_APPRAISAL, formatUsd } from "@/lib/appraisal";
import { ArrowRight, IconCamera, IconChart, IconChat } from "./ui";

export function Landing({
  onStart,
  onSample,
}: {
  onStart: () => void;
  onSample: () => void;
}) {
  const s = SAMPLE_APPRAISAL;
  return (
    <main className="page scrollthin">
      <div className="container">
        <section className="hero">
          <div>
            <div className="eyebrow">Class 7 &amp; 8 truck appraisal</div>
            <h1>Find out what your truck is worth.</h1>
            <p className="hero-lede">
              Upload photos of your truck. Get a value range, a condition breakdown, and an asking
              price you can defend.
            </p>
            <div className="cta-row">
              <button type="button" className="btn btn-primary btn-lg" onClick={onStart}>
                Start an appraisal <ArrowRight />
              </button>
              <button type="button" className="btn btn-secondary btn-lg" onClick={onSample}>
                See a sample report
              </button>
            </div>
          </div>

          <div className="card" aria-label="Sample appraisal preview">
            <div className="label">Sample appraisal</div>
            <div className="big num" style={{ marginBottom: 16, fontSize: 30, whiteSpace: "nowrap" }}>
              {formatUsd(s.low)} – {formatUsd(s.high)}
            </div>
            <PreviewRow label="Suggested asking price" value={formatUsd(s.asking)} />
            <PreviewRow label="Overall condition" value={s.conditionLabel ?? "N/A"} />
            <PreviewRow label="Areas graded" value={String(s.categories.length)} />
          </div>
        </section>

        <section className="section">
          <div className="eyebrow">How it works</div>
          <h2 style={{ marginTop: 10, fontSize: 30 }}>From photos to a number, with the reasoning attached.</h2>
          <div className="features">
            <Feature icon={<IconCamera />} title="Reads your photos" body="Panel damage, rust, tire tread, cab and aero, graded area by area." />
            <Feature icon={<IconChat />} title="Asks what it cannot see" body="If a badge is unreadable, it asks for the brand. A short conversation, not a form." />
            <Feature icon={<IconChart />} title="Prices against the market" body="Comparable asking prices, weighted by type, brand, era and condition." />
          </div>
        </section>

        <section className="section">
          <div className="cta-band">
            <div>
              <h2 style={{ fontSize: 30 }}>A report you can hand to a buyer.</h2>
              <p className="muted" style={{ marginTop: 10, maxWidth: "52ch" }}>
                Every number carries its reasoning: which photo it came from, what it added or took
                off, and how wide the range is.
              </p>
            </div>
            <button type="button" className="btn btn-primary btn-lg" onClick={onStart}>
              Start an appraisal <ArrowRight />
            </button>
          </div>
        </section>
      </div>
    </main>
  );
}

function PreviewRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="preview-row">
      <span className="muted">{label}</span>
      <strong className="num">{value}</strong>
    </div>
  );
}

function Feature({ icon, title, body }: { icon: ReactNode; title: string; body: string }) {
  return (
    <div className="card">
      <div className="feature-icon">{icon}</div>
      <h3>{title}</h3>
      <p className="muted" style={{ marginTop: 6, fontSize: 14 }}>{body}</p>
    </div>
  );
}
