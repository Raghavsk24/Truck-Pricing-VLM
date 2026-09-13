"use client";

import { useCallback, useState } from "react";
import { Landing } from "./Landing";
import { Conversation, type ChatMessage } from "./Conversation";
import { Report } from "./Report";
import type { AnalyzeNeedsBrand, AnalyzeOk, AnalyzeResponse } from "@/lib/types";
import { SAMPLE_APPRAISAL, liveAppraisal, type PhotoShot } from "@/lib/appraisal";
import { isAllowedImage, MAX_UPLOAD_BYTES, resizeImageFile } from "@/lib/resize-client";

type Screen = "landing" | "chat" | "report";

function uid() {
  return crypto.randomUUID();
}

function withRetry(message: string) {
  const text = message.trim();
  if (/try again/i.test(text)) return text;
  return `${text} Please try again.`;
}

export function PricingApp() {
  const [screen, setScreen] = useState<Screen>("landing");
  const [sample, setSample] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [photos, setPhotos] = useState<PhotoShot[]>([]);
  const [result, setResult] = useState<AnalyzeOk | null>(null);
  const [pending, setPending] = useState<AnalyzeNeedsBrand | null>(null);
  const [loading, setLoading] = useState(false);
  const [draft, setDraft] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);

  const reset = () => {
    photos.forEach((p) => URL.revokeObjectURL(p.url));
    setMessages([]);
    setPhotos([]);
    setResult(null);
    setPending(null);
    setLoading(false);
    setDraft("");
    setInputError(null);
    setSample(false);
    setScreen("landing");
  };

  const goChat = () => {
    setSample(false);
    setInputError(null);
    setScreen("chat");
  };

  const goSample = () => {
    setSample(true);
    setScreen("report");
  };

  const applyResponse = useCallback((data: AnalyzeResponse, uploaded: PhotoShot[]) => {
    const subject =
      data.status === "ok" || data.status === "needs_brand" ? data.primary_subject : undefined;
    if (subject) {
      setPhotos((prev) => {
        const copy = [...prev];
        let left = uploaded.length;
        for (let i = copy.length - 1; i >= 0 && left > 0; i -= 1) {
          copy[i] = { ...copy[i], subject };
          left -= 1;
        }
        return copy;
      });
    }

    if (data.status === "rejected" || data.status === "error") {
      setInputError(withRetry(data.user_message));
      setPending(null);
      return;
    }
    setInputError(null);
    if (data.status === "needs_brand") {
      setPending(data);
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "ai",
          text: data.user_prompt,
          questions: [
            {
              tag: "Brand",
              text: "No readable badge, hood ornament, or wordmark in this shot.",
            },
          ],
          chips: ["Freightliner", "Kenworth", "Peterbilt", "Mack", "International", "Volvo"],
        },
      ]);
      return;
    }
    setPending(null);
    setResult(data);
    setMessages((prev) => [
      ...prev,
      {
        id: uid(),
        role: "ai",
        text: `I read a ${String(data.truck_type).replace(/_/g, " ")}${data.brand ? ` · ${data.brand}` : ""}${data.model_series ? ` ${data.model_series}` : ""}. Condition looks ${data.overall_condition_label ?? "unscored"}. Open the full report when you're ready.`,
      },
    ]);
  }, []);

  const sendFiles = useCallback(
    async (list: FileList | null) => {
      if (!list?.length) return;
      const files = [...list];
      const valid: File[] = [];
      const reasons: string[] = [];
      for (const file of files) {
        if (!isAllowedImage(file)) {
          reasons.push("That file is not a JPEG, PNG, or WebP image of your truck.");
          continue;
        }
        if (file.size > MAX_UPLOAD_BYTES) {
          reasons.push("That file is larger than 8 MB.");
          continue;
        }
        valid.push(file);
      }
      if (!valid.length) {
        setInputError(withRetry(reasons[0] ?? "That upload is not a valid truck photo."));
        return;
      }
      if (reasons.length) {
        setInputError(withRetry(reasons[0]));
      } else {
        setInputError(null);
      }

      const shots: PhotoShot[] = valid.map((f) => ({
        url: URL.createObjectURL(f),
        name: f.name,
      }));
      setMessages((prev) => [...prev, { id: uid(), role: "user", photos: shots }]);
      setPhotos((prev) => [...prev, ...shots]);
      setLoading(true);
      try {
        const file = valid[0];
        const blob = await resizeImageFile(file);
        const body = new FormData();
        body.append("image", blob, "truck.jpg");
        const res = await fetch("/api/analyze", { method: "POST", body });
        const data = (await res.json()) as AnalyzeResponse;
        applyResponse(data, shots);
      } catch (err) {
        console.error(err);
        setInputError("Could not inspect this photo. Please upload another image and try again.");
      } finally {
        setLoading(false);
      }
    },
    [applyResponse],
  );

  const submitBrand = async (brand: string) => {
    const trimmed = brand.trim();
    if (!trimmed) {
      setInputError("Please enter the truck brand, then try again.");
      return;
    }
    setInputError(null);
    setMessages((prev) => [...prev, { id: uid(), role: "user", text: trimmed }]);
    setDraft("");
    if (!pending) {
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "ai",
          text: "Thanks — I'll hold that. Upload a front or side photo if you haven't already, and I'll fold it into the range.",
        },
      ]);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/price", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysisId: pending.analysisId, brand: trimmed }),
      });
      const data = (await res.json()) as AnalyzeResponse;
      if (data.status === "ok") {
        setPending(null);
        setResult(data);
        setInputError(null);
        setMessages((prev) => [
          ...prev,
          {
            id: uid(),
            role: "ai",
            text: `Noted — ${data.brand}. Open the full report when you're ready.`,
          },
        ]);
      } else if (data.status === "rejected" || data.status === "error") {
        setInputError(withRetry(data.user_message));
      }
    } catch {
      setInputError("Could not price this truck with that brand. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const onSend = () => {
    const text = draft.trim();
    if (!text) {
      setInputError(
        pending
          ? "Please enter the truck brand, then try again."
          : "Please upload your photographs to get your appraisal, then try again.",
      );
      return;
    }
    if (pending) {
      void submitBrand(text);
      return;
    }
    setDraft("");
    setInputError(null);
    setMessages((prev) => [
      ...prev,
      { id: uid(), role: "user", text },
      {
        id: uid(),
        role: "ai",
        text: result
          ? "Thanks — factored in. I'll hold the range until more photos land."
          : "Got it. Upload a front or side photo and I'll put a number on it.",
      },
    ]);
  };

  const view = sample
    ? SAMPLE_APPRAISAL
    : result
      ? liveAppraisal(result, photos)
      : SAMPLE_APPRAISAL;

  return (
    <div className="app-shell">
      {screen === "landing" && <Landing onStart={goChat} onSample={goSample} />}
      {screen === "chat" && (
        <Conversation
          messages={messages}
          result={result}
          loading={loading}
          draft={draft}
          inputError={inputError}
          onDraft={(value) => {
            setDraft(value);
            if (inputError) setInputError(null);
          }}
          onSend={onSend}
          onChip={(label) => void submitBrand(label)}
          onFiles={(files) => void sendFiles(files)}
          onOpenReport={() => {
            if (!result) return;
            setSample(false);
            setScreen("report");
          }}
          onBack={() => setScreen("landing")}
        />
      )}
      {screen === "report" && (
        <Report
          view={view}
          onBack={() => {
            if (sample) {
              setScreen("landing");
              setSample(false);
            } else {
              setScreen("chat");
            }
          }}
        />
      )}
    </div>
  );
}
