"use client";

import { useRef } from "react";
import type { AnalyzeOk } from "@/lib/types";
import { type PhotoShot } from "@/lib/appraisal";
import { AiAvatar, Blueprint, IconUpload } from "./ui";

export type ChatMessage = {
  id: string;
  role: "ai" | "user";
  text?: string;
  photos?: PhotoShot[];
  questions?: { tag: string; text: string }[];
  chips?: string[];
};

export function Conversation({
  messages,
  result,
  loading,
  draft,
  inputError,
  onDraft,
  onSend,
  onChip,
  onFiles,
  onOpenReport,
  onBack,
}: {
  messages: ChatMessage[];
  result: AnalyzeOk | null;
  loading: boolean;
  draft: string;
  inputError: string | null;
  onDraft: (v: string) => void;
  onSend: () => void;
  onChip: (label: string) => void;
  onFiles: (files: FileList | null) => void;
  onOpenReport: () => void;
  onBack: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <main className="chat-grid">
      <section className="chat-thread">
        <div
          className="scrollthin"
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "24px 40px 24px",
            display: "flex",
            flexDirection: "column",
            gap: 30,
            maxWidth: 780,
            width: "100%",
            margin: "0 auto",
            boxSizing: "border-box",
          }}
        >
          <div>
            <button type="button" className="btn btn-secondary" onClick={onBack} style={{ fontSize: 13 }}>
              Back
            </button>
          </div>
          {messages.map((m) =>
            m.role === "user" ? (
              <UserBubble key={m.id} message={m} />
            ) : (
              <AiBubble key={m.id} message={m} onChip={onChip} />
            ),
          )}
          {loading && (
            <div style={{ display: "flex", gap: 13.6 }}>
              <AiAvatar />
              <div style={{ fontSize: 15, lineHeight: 1.6, color: "color-mix(in srgb, var(--color-text) 65%, transparent)" }}>
                Reading those now…
              </div>
            </div>
          )}
        </div>

        <div style={{ borderTop: "1px solid var(--color-divider)", padding: "17px 34px 20.4px" }}>
          <div style={{ maxWidth: 780, margin: "0 auto" }}>
            <div
              style={{
                minHeight: inputError ? undefined : 0,
                marginBottom: inputError ? 10.2 : 0,
                fontSize: 14,
                lineHeight: 1.55,
                color: "var(--color-accent-800)",
              }}
            >
              {inputError}
            </div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 10.2 }}>
              <input
                className="input"
                type="text"
                placeholder="Upload your photographs to get your appraisal"
                value={draft}
                onChange={(e) => onDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") onSend();
                }}
                style={{ flex: 1, minHeight: 40 }}
              />
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => fileRef.current?.click()}
                style={{ flex: "none", height: 40, paddingInline: 17 }}
              >
                Upload
                <IconUpload />
              </button>
              {result && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={onOpenReport}
                  style={{ flex: "none", height: 40, paddingInline: 17 }}
                >
                  Open report
                </button>
              )}
            </div>
          </div>
        </div>
      </section>

      <input
        ref={fileRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        multiple
        className="hidden"
        onChange={(e) => {
          onFiles(e.target.files);
          e.target.value = "";
        }}
      />
    </main>
  );
}

function UserBubble({ message }: { message: ChatMessage }) {
  if (message.photos?.length) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6.8 }}>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(3, message.photos.length)}, 116px)`, gap: 10.2 }}>
          {message.photos.map((p) => (
            <Blueprint key={p.url} duotone>
              <div style={{ position: "relative", height: 86, overflow: "hidden", borderRadius: 14 }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={p.url} alt={p.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
              </div>
            </Blueprint>
          ))}
        </div>
        <div style={{ fontSize: 12, color: "color-mix(in srgb, var(--color-text) 55%, transparent)" }}>
          {message.photos.length} photo{message.photos.length === 1 ? "" : "s"} · uploaded
        </div>
      </div>
    );
  }
  return (
    <div style={{ display: "flex", justifyContent: "flex-end" }}>
      <div
        style={{
          maxWidth: "72%",
          background: "var(--color-accent-100)",
          border: "1px solid color-mix(in srgb, var(--color-accent) 22%, transparent)",
          borderRadius: "18px 18px 4px 18px",
          padding: "12px 16px",
          fontSize: 15,
          lineHeight: 1.65,
        }}
      >
        {message.text}
      </div>
    </div>
  );
}

function AiBubble({
  message,
  onChip,
}: {
  message: ChatMessage;
  onChip: (label: string) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 13.6 }}>
      <AiAvatar />
      <div style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 13.6 }}>
        {message.text && (
          <div style={{ fontSize: 15, lineHeight: 1.6, textWrap: "pretty" }}>{message.text}</div>
        )}
        {message.questions && (
          <div style={{ display: "grid", gap: 10 }}>
            {message.questions.map((q) => (
              <div
                key={q.tag}
                style={{
                  background: "#fff",
                  border: "1px solid var(--color-divider)",
                  borderRadius: 14,
                  padding: "11px 13.6px",
                  display: "flex",
                  gap: 10.2,
                  alignItems: "flex-start",
                }}
              >
                <span className="tag tag-outline" style={{ flex: "none" }}>
                  {q.tag}
                </span>
                <span style={{ fontSize: 14, lineHeight: 1.5 }}>{q.text}</span>
              </div>
            ))}
          </div>
        )}
        {message.chips && (
          <div style={{ display: "flex", gap: 6.8, flexWrap: "wrap" }}>
            {message.chips.map((c) => (
              <button key={c} type="button" className="btn btn-secondary" style={{ fontSize: 13 }} onClick={() => onChip(c)}>
                {c}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
