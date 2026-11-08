"use client";

import { useRef } from "react";
import type { AnalyzeOk } from "@/lib/types";
import { type PhotoShot } from "@/lib/appraisal";
import { IconUpload } from "./ui";

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
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const pick = () => fileRef.current?.click();

  return (
    <main className="chat">
      <div className="thread scrollthin">
        <div className="thread-inner">
          {messages.length === 0 && !loading && (
            <div className="empty">
              <h2>Upload photos of your truck</h2>
              <p className="muted">
                Front and side shots work best. Rear, interior and odometer shots help too.
              </p>
              <button type="button" className="btn btn-primary btn-lg" onClick={pick}>
                Upload photos <IconUpload />
              </button>
              <p className="muted small">JPEG, PNG or WebP · up to 8 MB each</p>
            </div>
          )}
          {messages.map((m) =>
            m.role === "user" ? (
              <UserMessage key={m.id} message={m} />
            ) : (
              <AiMessage key={m.id} message={m} onChip={onChip} />
            ),
          )}
          {loading && <p className="muted">Reading those now…</p>}
        </div>
      </div>

      <div className="composer">
        <div className="composer-inner">
          {inputError && <div className="composer-error" role="alert">{inputError}</div>}
          <div className="composer-row">
            <input
              className="input"
              type="text"
              placeholder="Type a message…"
              aria-label="Message"
              value={draft}
              onChange={(e) => onDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") onSend();
              }}
            />
            <button type="button" className="btn btn-primary" onClick={pick}>
              Upload <IconUpload />
            </button>
            {result && (
              <button type="button" className="btn btn-secondary" onClick={onOpenReport}>
                Open report
              </button>
            )}
          </div>
        </div>
      </div>

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

function UserMessage({ message }: { message: ChatMessage }) {
  if (message.photos?.length) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
        <div className="thumbs" style={{ gridTemplateColumns: `repeat(${Math.min(3, message.photos.length)}, auto)` }}>
          {message.photos.map((p) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img key={p.url} className="thumb" src={p.url} alt={p.name} />
          ))}
        </div>
        <div className="muted small">
          {message.photos.length} photo{message.photos.length === 1 ? "" : "s"} uploaded
        </div>
      </div>
    );
  }
  return (
    <div style={{ display: "flex", justifyContent: "flex-end" }}>
      <div className="bubble-user">{message.text}</div>
    </div>
  );
}

function AiMessage({
  message,
  onChip,
}: {
  message: ChatMessage;
  onChip: (label: string) => void;
}) {
  return (
    <div className="msg-ai">
      {message.text && <p style={{ textWrap: "pretty" }}>{message.text}</p>}
      {message.questions?.map((q) => (
        <div key={q.tag} className="note">
          <span className="tag tag-accent">{q.tag}</span>
          <span>{q.text}</span>
        </div>
      ))}
      {message.chips && (
        <div className="chips">
          {message.chips.map((c) => (
            <button key={c} type="button" className="btn btn-secondary btn-sm" onClick={() => onChip(c)}>
              {c}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
