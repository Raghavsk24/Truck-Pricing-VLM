"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { ArrowDown, ArrowRight, IconCamera, IconChart, IconChat } from "./ui";

export function Landing({
  onStart,
  onSample,
}: {
  onStart: () => void;
  onSample: () => void;
}) {
  const loopRef = useRef<SVGPathElement>(null);
  const landingRef = useRef<HTMLElement>(null);
  const lenRef = useRef(0);

  const drawLoop = () => {
    const path = loopRef.current;
    const box = landingRef.current;
    if (!path || !box) return;
    if (!lenRef.current) {
      lenRef.current = path.getTotalLength();
      path.style.strokeDasharray = String(lenRef.current);
    }
    const max = box.scrollHeight - box.clientHeight;
    const p = max > 0 ? Math.min(1, box.scrollTop / max) : 0;
    path.style.strokeDashoffset = String(lenRef.current * (1 - (0.12 + 0.88 * p)));
  };

  useEffect(() => {
    requestAnimationFrame(drawLoop);
  }, []);

  return (
    <main
      ref={landingRef}
      className="scrollthin"
      onScroll={drawLoop}
      style={{ flex: 1, minHeight: 0, overflowY: "auto" }}
    >
      <div style={{ position: "relative" }}>
        <svg
          viewBox="0 0 1000 3000"
          preserveAspectRatio="none"
          aria-hidden="true"
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            overflow: "visible",
          }}
        >
          <path
            ref={loopRef}
            d="M500 40 C500 240 200 300 200 460 S500 600 820 680 C980 730 980 940 800 1000 S300 1120 250 1340 C210 1520 700 1600 760 1800 S500 2140 420 2340 C380 2500 620 2600 620 2840 L620 2990"
            fill="none"
            stroke="var(--color-accent)"
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.38"
            vectorEffect="non-scaling-stroke"
          />
          <circle cx="200" cy="460" r="9" fill="none" stroke="var(--color-accent)" strokeWidth="2" opacity="0.3" vectorEffect="non-scaling-stroke" />
          <circle cx="800" cy="1000" r="9" fill="none" stroke="var(--color-accent)" strokeWidth="2" opacity="0.3" vectorEffect="non-scaling-stroke" />
          <circle cx="620" cy="2840" r="9" fill="none" stroke="var(--color-accent)" strokeWidth="2" opacity="0.3" vectorEffect="non-scaling-stroke" />
        </svg>

        <section
          style={{
            position: "relative",
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            textAlign: "center",
            padding: "72px 48px",
            boxSizing: "border-box",
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 0,
              pointerEvents: "none",
              background:
                "radial-gradient(120% 80% at 50% 100%, color-mix(in srgb, var(--color-accent) 22%, transparent) 0%, transparent 62%)",
            }}
          />
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 0,
              pointerEvents: "none",
              background:
                "linear-gradient(to bottom, var(--color-bg) 0%, color-mix(in srgb, var(--color-bg) 72%, transparent) 45%, color-mix(in srgb, var(--color-bg) 92%, transparent) 100%)",
            }}
          />
          <div style={{ position: "relative", zIndex: 1, maxWidth: 720, width: "100%" }}>
            <div
              style={{
                fontSize: 11,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: "var(--color-accent-900)",
                marginBottom: 20.4,
                fontFamily: "var(--font-heading)",
                fontWeight: 600,
                fontSize: 22,
                letterSpacing: "0.2em",
              }}
            >
              BLUEOOP
            </div>
            <h1
              style={{
                fontSize: "clamp(40px, 8vw, 66px)",
                lineHeight: 1.04,
                margin: "0 0 36px",
                letterSpacing: "-0.02em",
                textWrap: "pretty",
              }}
            >
              Find out what your truck is worth.
            </h1>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
              <button type="button" className="btn btn-primary" onClick={onStart} style={{ padding: "12px 24px", fontSize: 15 }}>
                Start an appraisal <ArrowRight />
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={onSample}
                style={{
                  padding: "12px 24px",
                  fontSize: 15,
                  background: "color-mix(in srgb, var(--color-bg) 85%, transparent)",
                }}
              >
                See a sample report <ArrowRight />
              </button>
            </div>
            <div
              style={{
                position: "relative",
                zIndex: 1,
                marginTop: 56,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 8,
                fontSize: 12,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: "color-mix(in srgb, var(--color-text) 45%, transparent)",
              }}
            >
              How it works
              <ArrowDown />
            </div>
          </div>
        </section>

        <section style={{ position: "relative", padding: "88px 48px", maxWidth: 1120, margin: "0 auto", boxSizing: "border-box" }}>
          <div style={{ fontSize: 11, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--color-accent-700)", marginBottom: 14 }}>
            What Blueoop does
          </div>
          <h2 style={{ fontSize: 38, lineHeight: 1.1, margin: "0 0 16px", maxWidth: "20ch" }}>
            Appraisals for your class 7 and 8 trucks based on your photographs
          </h2>
          <p
            style={{
              fontSize: 16,
              lineHeight: 1.65,
              maxWidth: "60ch",
              margin: "0 0 44px",
              color: "color-mix(in srgb, var(--color-text) 70%, transparent)",
            }}
          >
            Our VLM is trained to filter through your photographs and assess your trucks, condition brand model to generate you an appraisal report.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16 }}>
            <Feature icon={<IconCamera />} title="Reads your photos" body="Panel damage, rust, tire tread, cab and aero — graded area by area." />
            <Feature icon={<IconChat />} title="Asks what it cannot see" body="If a badge is unreadable, it asks for the brand. A short conversation, not a form." />
            <Feature icon={<IconChart />} title="Prices against the market" body="Comparable asking prices, weighted by type, brand, era and condition." />
          </div>
        </section>

        <section style={{ position: "relative", padding: "0 48px 88px", maxWidth: 1120, margin: "0 auto", boxSizing: "border-box" }}>
          <div
            style={{
              background: "var(--color-accent-900)",
              color: "var(--color-bg)",
              borderRadius: 20,
              padding: "56px 48px",
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: 48,
              alignItems: "center",
            }}
          >
            <div>
              <div style={{ fontSize: 11, letterSpacing: "0.18em", textTransform: "uppercase", opacity: 0.7, marginBottom: 14 }}>
                What you get
              </div>
              <h2 style={{ fontSize: 34, lineHeight: 1.12, margin: "0 0 16px" }}>A report you can hand to a buyer.</h2>
              <p style={{ fontSize: 15.5, lineHeight: 1.65, opacity: 0.85, margin: "0 0 28px" }}>
                Every number carries its reasoning — which photo it came from, what it added or took
                off, and how wide the range is for that type and era.
              </p>
              <button
                type="button"
                className="btn"
                onClick={onSample}
                style={{ background: "var(--color-bg)", color: "var(--color-accent-900)", padding: "12px 24px", fontSize: 15 }}
              >
                See a sample report <ArrowRight />
              </button>
            </div>
            <div style={{ display: "grid", gap: 12 }}>
              <DarkStat kicker="Value range & confidence" value="$32,400 – $35,600 · 86%" />
              <DarkStat kicker="Condition, by area" value="Four areas graded" />
              <DarkStat kicker="Suggested asking price" value="$35,900" />
            </div>
          </div>
        </section>

        <section style={{ position: "relative", padding: "0 48px 96px", maxWidth: 1120, margin: "0 auto", boxSizing: "border-box", textAlign: "center" }}>
          <h2 style={{ fontSize: 34, margin: "0 0 28px" }}>It takes about four minutes.</h2>
          <button type="button" className="btn btn-primary" onClick={onStart} style={{ padding: "12px 24px", fontSize: 15 }}>
            Start an appraisal <ArrowRight />
          </button>
        </section>
      </div>
    </main>
  );
}

function Feature({ icon, title, body }: { icon: ReactNode; title: string; body: string }) {
  return (
    <div style={{ background: "#fff", border: "1px solid var(--color-divider)", borderRadius: 14, padding: "26px 24px" }}>
      <div style={{ color: "var(--color-accent)", marginBottom: 14 }}>{icon}</div>
      <div style={{ fontFamily: "var(--font-heading)", fontWeight: 600, fontSize: 20, marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 14, lineHeight: 1.6, color: "color-mix(in srgb, var(--color-text) 65%, transparent)" }}>{body}</div>
    </div>
  );
}

function DarkStat({ kicker, value }: { kicker: string; value: string }) {
  return (
    <div style={{ border: "1px solid color-mix(in srgb, var(--color-bg) 26%, transparent)", borderRadius: 12, padding: "16px 18px" }}>
      <div style={{ fontSize: 10.5, letterSpacing: "0.12em", textTransform: "uppercase", opacity: 0.7 }}>{kicker}</div>
      <div style={{ fontFamily: "var(--font-heading)", fontSize: 24 }}>{value}</div>
    </div>
  );
}
