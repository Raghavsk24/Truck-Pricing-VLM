"use client";

import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { ContributionChart } from "./ContributionChart";
import { ConditionPdf } from "./ConditionPdf";
import type { AnalyzeNeedsBrand, AnalyzeOk, AnalyzeResponse } from "@/lib/types";
import { eraLabel, formatUsd, typeLabel } from "@/lib/format";
import { isAllowedImage, MAX_UPLOAD_BYTES, resizeImageFile } from "@/lib/resize-client";

type Phase = "idle" | "loading" | "needs_brand" | "ok";

export function PricingApp() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeOk | null>(null);
  const [brandPrompt, setBrandPrompt] = useState<AnalyzeNeedsBrand | null>(null);
  const [brandInput, setBrandInput] = useState("");
  const [dragOver, setDragOver] = useState(false);

  const resetFileInput = () => {
    if (inputRef.current) inputRef.current.value = "";
  };

  const applyResponse = useCallback((data: AnalyzeResponse) => {
    if (data.status === "rejected" || data.status === "error") {
      toast.error(data.user_message);
      setPhase("idle");
      setResult(null);
      setBrandPrompt(null);
      return;
    }
    if (data.status === "needs_brand") {
      setBrandPrompt(data);
      setBrandInput("");
      setPhase("needs_brand");
      return;
    }
    setBrandPrompt(null);
    setResult(data);
    setPhase("ok");
  }, []);

  const sendImage = useCallback(
    async (file: File) => {
      if (!isAllowedImage(file)) {
        toast.error("Please upload a JPEG, PNG, or WebP image of your truck.");
        resetFileInput();
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        toast.error("That file is larger than 8 MB. Please upload a smaller photo.");
        resetFileInput();
        return;
      }

      setPhase("loading");
      setResult(null);
      setBrandPrompt(null);
      const objectUrl = URL.createObjectURL(file);
      setPreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return objectUrl;
      });

      try {
        const blob = await resizeImageFile(file);
        const body = new FormData();
        body.append("image", blob, "truck.jpg");
        const res = await fetch("/api/analyze", { method: "POST", body });
        const data = (await res.json()) as AnalyzeResponse;
        applyResponse(data);
      } catch (err) {
        console.error(err);
        toast.error("Could not inspect this photo. Close this notice and try another image.");
        setPhase("idle");
      } finally {
        resetFileInput();
      }
    },
    [applyResponse],
  );

  const submitBrand = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!brandPrompt) return;
    const brand = brandInput.trim();
    if (!brand) {
      toast.error("Please enter the truck brand.");
      return;
    }
    setPhase("loading");
    try {
      const res = await fetch("/api/price", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysisId: brandPrompt.analysisId, brand }),
      });
      const data = (await res.json()) as AnalyzeResponse;
      applyResponse(data);
    } catch {
      toast.error("Could not price this truck with that brand. Try again.");
      setPhase("needs_brand");
    }
  };

  const onPick = (files: FileList | null) => {
    const file = files?.[0];
    if (file) void sendImage(file);
  };

  const startOver = () => {
    setPhase("idle");
    setResult(null);
    setBrandPrompt(null);
    setBrandInput("");
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    resetFileInput();
  };

  return (
    <div className="min-h-full bg-[#0b1016] text-slate-100">
      <header className="border-b border-slate-800 bg-[#0d1520]">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <p className="text-xs font-semibold tracking-[0.28em] text-amber-500">KAMION</p>
            <h1 className="text-lg font-semibold text-white">Truck pricing from a photo</h1>
          </div>
          <p className="hidden max-w-sm text-right text-xs text-slate-400 sm:block">
            Class 7/8 day cab, dump, and sleeper. Front or side shot. Asking range grounded
            in TruckPaper comps.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <section
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            onPick(e.dataTransfer.files);
          }}
          className={`rounded-2xl border-2 border-dashed px-6 py-10 text-center transition ${
            dragOver
              ? "border-amber-400 bg-amber-400/10"
              : "border-slate-700 bg-slate-900/60"
          } ${phase === "loading" ? "pointer-events-none opacity-70" : ""}`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => onPick(e.target.files)}
          />
          <p className="text-sm font-medium text-slate-200">
            {phase === "loading" ? "Inspecting photo…" : "Upload a front or side photo of the truck"}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            JPEG, PNG, or WebP · max 8 MB · long edge resized to 1568 px
          </p>
          <button
            type="button"
            disabled={phase === "loading"}
            onClick={() => inputRef.current?.click()}
            className="mt-5 rounded-full bg-amber-500 px-6 py-2.5 text-sm font-semibold text-slate-950 hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {phase === "loading" ? "Working…" : "Upload image"}
          </button>
        </section>

        {preview && (
          <div className="mt-8 grid gap-6 lg:grid-cols-[280px_1fr]">
            <figure className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={preview} alt="Uploaded truck" className="h-64 w-full object-cover" />
              <figcaption className="px-3 py-2 text-xs text-slate-400">Uploaded photo</figcaption>
            </figure>

            {phase === "loading" && (
              <div className="flex items-center rounded-xl border border-slate-800 bg-slate-900 px-6 text-sm text-slate-300">
                Reading type, brand, era, and condition from the photo. This can take a few
                seconds.
              </div>
            )}

            {phase === "ok" && result && (
              <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
                <div className="flex flex-wrap gap-2 text-xs">
                  <Chip>{typeLabel(String(result.truck_type))}</Chip>
                  <Chip>{result.brand}</Chip>
                  <Chip>{eraLabel(result.era)}</Chip>
                  {result.model_series && <Chip>{result.model_series}</Chip>}
                  {result.overall_condition_label && (
                    <Chip>{result.overall_condition_label}</Chip>
                  )}
                </div>
                <p className="mt-5 text-xs uppercase tracking-wider text-slate-400">
                  Asking range
                </p>
                <p className="text-3xl font-semibold text-white">
                  {formatUsd(result.low)} – {formatUsd(result.high)}
                </p>
                <p className="mt-1 text-sm text-slate-400">
                  Center {formatUsd(result.center)} · range ±{result.error_bound_pct}% for this
                  type and era
                </p>
                <button
                  type="button"
                  onClick={startOver}
                  className="mt-4 text-xs text-amber-400 hover:text-amber-300"
                >
                  Price another truck
                </button>
              </div>
            )}
          </div>
        )}

        {phase === "ok" && result && (
          <section className="mt-8 space-y-8">
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                What each variable contributes
              </h2>
              <p className="mt-1 text-xs text-slate-400">
                Dollar move versus the market baseline, split by type, brand, era, and the
                four condition areas.
              </p>
              <ContributionChart contributions={result.contributions} />
            </div>

            <div>
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">
                Condition assessment
              </h2>
              <ConditionPdf
                truck_type={result.truck_type}
                brand={result.brand}
                era={result.era}
                model_series={result.model_series}
                overall_score={result.overall_score}
                overall_condition_label={result.overall_condition_label}
                condition={result.condition}
              />
            </div>
          </section>
        )}
      </main>

      {phase === "needs_brand" && brandPrompt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <form
            onSubmit={submitBrand}
            className="w-full max-w-md rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-2xl"
          >
            <p className="text-xs font-semibold tracking-[0.2em] text-amber-500">BRAND NEEDED</p>
            <h2 className="mt-2 text-lg font-semibold text-white">
              We can price this truck once we know the OEM
            </h2>
            <p className="mt-2 text-sm text-slate-300">{brandPrompt.user_prompt}</p>
            <input
              autoFocus
              value={brandInput}
              onChange={(e) => setBrandInput(e.target.value)}
              placeholder="e.g. Freightliner"
              className="mt-4 w-full rounded-lg border border-slate-600 bg-slate-950 px-3 py-2 text-sm text-white outline-none ring-amber-400 focus:ring-2"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={startOver}
                className="rounded-lg px-3 py-2 text-sm text-slate-300 hover:text-white"
              >
                Re-upload
              </button>
              <button
                type="submit"
                className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-amber-400"
              >
                Price truck
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full border border-slate-600 bg-slate-800 px-3 py-1 text-slate-200">
      {children}
    </span>
  );
}
