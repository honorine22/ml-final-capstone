"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Camera,
  CheckCircle2,
  ClipboardCheck,
  Database,
  ExternalLink,
  FileImage,
  FlaskConical,
  Layers3,
  Leaf,
  LineChart,
  Loader2,
  MapPin,
  PackageCheck,
  PlayCircle,
  ShieldCheck,
  Upload,
  WifiOff,
} from "lucide-react";

type QualityKey = "good" | "broken" | "impurity" | "discolored" | "mold";
type Risk = "Low" | "Medium" | "High";

type AnalyzeResponse = {
  key?: QualityKey;
  fallback?: {
    key?: QualityKey;
  };
};

type Scenario = {
  label: string;
  shortLabel: string;
  confidence: number;
  risk: Risk;
  action: string;
  detail: string;
  priority: string;
  tone: "success" | "warning" | "danger";
};

const scenarios: Record<QualityKey, Scenario> = {
  good: {
    label: "Good maize grain",
    shortLabel: "Good",
    confidence: 94,
    risk: "Low",
    action: "Store safely or prepare for sale",
    detail:
      "The batch appears clean and suitable for normal storage with routine monitoring.",
    priority:
      "Classified as good only when most checks agree and no strong risk evidence appears.",
    tone: "success",
  },
  broken: {
    label: "Broken or damaged grain",
    shortLabel: "Broken",
    confidence: 87,
    risk: "Medium",
    action: "Sort before storage",
    detail:
      "Remove visibly damaged kernels before storage or sale to reduce rejection risk.",
    priority:
      "Broken kernels receive priority over good maize when enough patch evidence is detected.",
    tone: "warning",
  },
  impurity: {
    label: "Impurity-contaminated grain",
    shortLabel: "Impurity",
    confidence: 89,
    risk: "Medium",
    action: "Clean and re-screen",
    detail:
      "Separate stones, husks, dust and foreign matter before aggregation or sale.",
    priority:
      "Impurity evidence is prioritized because foreign matter reduces batch value.",
    tone: "warning",
  },
  discolored: {
    label: "Discolored grain",
    shortLabel: "Discolored",
    confidence: 82,
    risk: "Medium",
    action: "Sell quickly or refer for review",
    detail:
      "Discoloration may lower the quality grade; avoid mixing with clean grain.",
    priority:
      "Discolored patches are a warning sign before a batch is called good.",
    tone: "warning",
  },
  mold: {
    label: "Visible mold-risk grain",
    shortLabel: "Mold risk",
    confidence: 91,
    risk: "High",
    action: "Do not store — refer for checking",
    detail:
      "Visible mold risk requires careful handling and further quality assessment.",
    priority:
      "Mold-risk evidence has the highest priority, even if some patches look good.",
    tone: "danger",
  },
};

const history = [
  {
    site: "Kayonza cooperative",
    result: "Good maize grain",
    risk: "Low",
    time: "09:12",
  },
  {
    site: "Nyagatare market",
    result: "Impurity-contaminated",
    risk: "Medium",
    time: "10:25",
  },
  {
    site: "Local sample test",
    result: "Visible mold-risk grain",
    risk: "High",
    time: "11:04",
  },
];

const metrics = [
  { label: "Public sources", value: "3" },
  { label: "Classes mapped", value: "5" },
  { label: "Training stack", value: "PyTorch" },
  { label: "API route", value: "Ready" },
];

const datasetSources = [
  {
    name: "CK-CNN",
    purpose: "Good / defective / impurity",
    detail:
      "Main public dataset for kernel-level quality categories used in training.",
    href: "https://github.com/vision-cidis/CK-CNNLW",
  },
  {
    name: "GrainSet maize",
    purpose: "Visual grain quality",
    detail:
      "Additional maize grain data used to support public-only training.",
    href: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10632488/",
  },
  {
    name: "EfficientMaize",
    purpose: "Good / bad support",
    detail:
      "Used as extra good and broad bad maize support, without forcing unclear labels.",
    href: "https://data.mendeley.com/datasets/r6vvm5jkh6/1",
  },
];

const navItems = [
  { label: "Project", href: "#project" },
  { label: "How it works", href: "#scan" },
  { label: "Workflow", href: "#results" },
  { label: "Demo", href: "#data" },
];

const priorityRules: { label: string; tone: string }[] = [
  { label: "Mold", tone: "bg-danger text-white" },
  { label: "Impurity", tone: "bg-primary text-primary-foreground" },
  { label: "Broken", tone: "bg-warning text-white" },
  { label: "Discolored", tone: "bg-clay text-white" },
  { label: "Good", tone: "bg-success text-white" },
];

function toneClasses(tone: Scenario["tone"]) {
  if (tone === "success") {
    return {
      chip: "bg-success/10 text-success border-success/25",
      bar: "bg-success",
      icon: "bg-success/10 text-success",
      panel: "border-success/20 bg-success/5",
    };
  }

  if (tone === "danger") {
    return {
      chip: "bg-danger/10 text-danger border-danger/25",
      bar: "bg-danger",
      icon: "bg-danger/10 text-danger",
      panel: "border-danger/20 bg-danger/5",
    };
  }

  return {
    chip: "bg-warning/10 text-warning border-warning/25",
    bar: "bg-warning",
    icon: "bg-warning/10 text-warning",
    panel: "border-warning/20 bg-warning/5",
  };
}

export default function Home() {
  const [selected, setSelected] = useState<QualityKey>("good");
  const [fileName, setFileName] = useState("sample-maize-batch.jpg");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("Ready to assess");
  const inputRef = useRef<HTMLInputElement>(null);

  const result = scenarios[selected];
  const tones = toneClasses(result.tone);
  const confidenceWidth = useMemo(
    () => `${result.confidence}%`,
    [result.confidence]
  );

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function analyzeFile(file: File) {
    setFileName(file.name);

    const nextPreview = URL.createObjectURL(file);

    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return nextPreview;
    });

    setIsAnalyzing(true);
    setLastUpdated("Assessing image…");

    const localVisual = await classifyImageAppearance(nextPreview);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      let apiResult: AnalyzeResponse = {};

      try {
        apiResult = (await response.json()) as AnalyzeResponse;
      } catch {
        apiResult = {};
      }

      const predicted =
        apiResult.key ?? apiResult.fallback?.key ?? localVisual ?? "good";

      setSelected(predicted);
      setLastUpdated(
        response.ok ? "Assessment completed" : "Preview assessment completed"
      );
    } catch {
      setSelected(localVisual ?? "good");
      setLastUpdated("Preview assessment completed");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f6f8f4] text-ink">
      {/* HERO */}
      <section
        id="project"
        className="relative min-h-screen overflow-hidden bg-black text-white"
      >
        <img
          src="/farmers-maize-harvest-background.jpg"
          alt="Farmers harvesting maize"
          className="absolute inset-0 h-full w-full object-cover object-center"
        />

        <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(4,9,7,0.88)_0%,rgba(5,10,8,0.78)_38%,rgba(7,10,8,0.48)_68%,rgba(7,10,8,0.3)_100%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(0,0,0,0.32)_0%,rgba(0,0,0,0.08)_40%,rgba(0,0,0,0.66)_100%)]" />

        <header className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-5 py-7 md:px-8">
          <a href="#" className="flex items-center gap-2 text-lg font-semibold">
            <span className="text-[#f3c84f]">MaizeGuard</span>
            <span className="text-white">Rwanda</span>
          </a>

          <nav className="hidden items-center gap-10 text-sm font-medium text-white/82 md:flex">
            {navItems.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="transition hover:text-white"
              >
                {item.label}
              </a>
            ))}
          </nav>

          <a
            href="#scan"
            className="hidden items-center gap-2 rounded-full border border-white/15 bg-primary/90 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-black/20 transition hover:bg-primary md:inline-flex"
          >
            View demo
            <ArrowRight className="h-4 w-4" />
          </a>
        </header>

        <div className="relative z-10 mx-auto grid min-h-[calc(100vh-100px)] max-w-7xl items-center gap-10 px-5 pb-16 md:px-8 lg:grid-cols-[0.72fr_0.28fr]">
          <div className="max-w-4xl">
            <p className="mb-8 text-sm font-bold uppercase tracking-[0.22em] text-[#f3c84f]">
              Maize quality assessment
            </p>

            <h1 className="max-w-4xl font-display text-[clamp(4rem,7vw,7.35rem)] font-normal leading-[0.98] tracking-[-0.055em] text-white">
              Know your maize before storage or sale.
            </h1>

            <p className="mt-8 max-w-2xl text-lg leading-8 text-white/82 md:text-xl">
              Upload a maize image to get instant quality insights, confidence
              score, and practical post-harvest guidance.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <a
                href="#scan"
                className="inline-flex items-center gap-3 rounded-full bg-primary px-7 py-4 text-sm font-bold text-white shadow-xl shadow-black/25 transition hover:-translate-y-0.5 hover:bg-[#2d7447]"
              >
                <Upload className="h-4 w-4" />
                Assess a sample
                <ArrowRight className="h-4 w-4" />
              </a>

              <a
                href="#scan"
                className="inline-flex items-center gap-3 rounded-full border border-white/38 bg-black/20 px-7 py-4 text-sm font-bold text-white backdrop-blur transition hover:bg-white hover:text-ink"
              >
                <PlayCircle className="h-4 w-4" />
                See how it works
              </a>
            </div>
          </div>

          <div className="hidden lg:block">
            <div className="overflow-hidden rounded-[2rem] border border-white/24 bg-black/30 shadow-2xl shadow-black/40 backdrop-blur-md">
              <div className="relative aspect-[4/5]">
                <img
                  src="/farmers-maize-harvest-background.jpg"
                  alt="Maize quality check"
                  className="h-full w-full object-cover object-center"
                />

                <div className="absolute left-5 top-5 inline-flex items-center gap-2 rounded-full bg-black/62 px-4 py-2 text-xs font-bold uppercase tracking-[0.12em] text-white backdrop-blur">
                  <span className="h-2 w-2 rounded-full bg-success" />
                  Live demo
                </div>

                <div className="absolute inset-x-0 bottom-0 bg-[linear-gradient(180deg,transparent,rgba(0,0,0,0.84))] p-6">
                  <div className="flex items-center gap-4">
                    <div className="grid h-14 w-14 place-items-center rounded-2xl bg-primary text-white">
                      <Leaf className="h-6 w-6" />
                    </div>
                    <div>
                      <p className="text-xl font-bold text-white">
                        Quality check
                      </p>
                      <p className="mt-1 text-sm text-white/70">
                        Post-harvest workflow
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      {/* END HERO */}

      <section id="scan" className="px-5 py-20 md:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="mb-10 flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-clay">
                Assessment workspace
              </p>
              <h2 className="mt-3 max-w-3xl font-display text-4xl font-semibold leading-tight text-ink md:text-5xl">
                Upload a batch photo and review the result.
              </h2>
            </div>

            <span className="inline-flex items-center gap-2 rounded-full border border-border bg-white px-4 py-2 text-sm font-semibold text-ink shadow-sm">
              {isAnalyzing ? (
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
              ) : (
                <CheckCircle2 className="h-4 w-4 text-primary" />
              )}
              {lastUpdated}
            </span>
          </div>

          <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
            <div className="rounded-3xl bg-white p-6 shadow-xl shadow-black/[0.04] md:p-8">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="font-display text-2xl font-semibold text-ink">
                    Batch image
                  </h3>
                  <p className="mt-2 text-sm leading-6 text-ink-soft">
                    Use a clear photo of shelled maize on a plain surface.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  className="grid h-11 w-11 place-items-center rounded-xl border border-border bg-white text-primary shadow-sm transition hover:-translate-y-0.5 hover:border-primary/40"
                  title="Open camera"
                >
                  <Camera className="h-5 w-5" />
                </button>
              </div>

              <label className="group mt-6 flex min-h-[22rem] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed border-border bg-surface-2 px-5 py-6 text-center transition hover:border-primary/50 hover:bg-surface-3/70">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Selected maize batch preview"
                    className="h-72 w-full rounded-xl object-cover shadow-sm"
                  />
                ) : (
                  <span className="grid h-20 w-20 place-items-center rounded-2xl bg-primary text-white shadow-sm transition group-hover:-translate-y-1">
                    <Upload className="h-8 w-8" />
                  </span>
                )}

                <span className="mt-5 inline-flex max-w-full items-center gap-2 truncate text-sm font-semibold text-ink">
                  <FileImage className="h-4 w-4 shrink-0 text-primary" />
                  <span className="truncate">{fileName}</span>
                </span>

                <span className="mt-2 max-w-sm text-sm leading-6 text-ink-soft">
                  {isAnalyzing
                    ? "Checking the uploaded image…"
                    : "Upload a close image to classify visible quality signs."}
                </span>

                <input
                  ref={inputRef}
                  className="sr-only"
                  type="file"
                  accept="image/*"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void analyzeFile(file);
                  }}
                />
              </label>

              <div className="mt-6">
                <p className="mb-3 text-xs font-bold uppercase tracking-[0.16em] text-ink-soft">
                  Test with a sample condition
                </p>

                <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                  {(Object.entries(scenarios) as [QualityKey, Scenario][]).map(
                    ([key, item]) => {
                      const isSelected = selected === key;

                      return (
                        <button
                          key={key}
                          type="button"
                          onClick={() => {
                            setSelected(key);
                            setLastUpdated("Sample condition selected");
                          }}
                          className={`rounded-2xl border px-3 py-3 text-left text-sm font-semibold transition hover:-translate-y-0.5 ${
                            isSelected
                              ? "border-primary bg-primary text-primary-foreground shadow-sm"
                              : "border-border bg-white text-ink hover:border-clay/40"
                          }`}
                        >
                          <span className="block">{item.shortLabel}</span>
                          <span
                            className={`mt-1 block text-[11px] font-medium ${
                              isSelected
                                ? "text-primary-foreground/75"
                                : "text-ink-soft"
                            }`}
                          >
                            {item.risk} risk
                          </span>
                        </button>
                      );
                    }
                  )}
                </div>
              </div>
            </div>

            <div
              id="results"
              className="flex flex-col rounded-3xl bg-white p-6 shadow-xl shadow-black/[0.04] md:p-8"
            >
              <div className="flex items-start justify-between gap-5">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.2em] text-clay">
                    Assessment result
                  </p>
                  <h3 className="mt-3 font-display text-4xl font-semibold leading-[1.05] text-ink md:text-5xl">
                    {result.label}
                  </h3>
                </div>

                <div
                  className={`grid h-14 w-14 shrink-0 place-items-center rounded-2xl ${tones.icon}`}
                >
                  {isAnalyzing ? (
                    <Loader2 className="h-6 w-6 animate-spin" />
                  ) : result.risk === "High" ? (
                    <AlertTriangle className="h-6 w-6" />
                  ) : (
                    <CheckCircle2 className="h-6 w-6" />
                  )}
                </div>
              </div>

              <div className="mt-6 flex flex-wrap items-center gap-2">
                <span
                  className={`inline-flex items-center rounded-full border px-3.5 py-1.5 text-xs font-semibold ${tones.chip}`}
                >
                  {result.risk} risk
                </span>

                <span className="inline-flex items-center rounded-full border border-border bg-white px-3.5 py-1.5 text-xs font-semibold text-ink">
                  {result.confidence}% confidence
                </span>
              </div>

              <div className="mt-6">
                <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                  <div
                    className={`${tones.bar} h-full rounded-full transition-all duration-700`}
                    style={{ width: confidenceWidth }}
                  />
                </div>

                <div className="mt-2 flex items-center justify-between text-xs">
                  <span className="font-medium text-ink-soft">
                    Model confidence
                  </span>
                  <span className="font-semibold text-ink">
                    {result.confidence}%
                  </span>
                </div>
              </div>

              <div className={`mt-6 rounded-2xl border p-5 ${tones.panel}`}>
                <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-primary">
                  Recommended action
                </p>

                <h4 className="mt-2 font-display text-2xl font-semibold leading-snug text-ink">
                  {result.action}
                </h4>

                <p className="mt-2 text-sm leading-6 text-ink-soft">
                  {result.detail}
                </p>
              </div>

              <div className="mt-4 rounded-2xl border border-border bg-surface-2/70 p-5">
                <div className="flex items-start gap-3">
                  <Layers3 className="mt-0.5 h-5 w-5 shrink-0 text-clay" />

                  <div className="min-w-0">
                    <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-ink">
                      Mixed-risk priority
                    </p>

                    <p className="mt-1.5 text-sm leading-6 text-ink-soft">
                      {result.priority}
                    </p>

                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {priorityRules.map((rule, index) => (
                        <span
                          key={rule.label}
                          className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${rule.tone}`}
                        >
                          {index + 1}. {rule.label}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <Step
                  icon={<ClipboardCheck className="h-4 w-4" />}
                  title="Assess"
                  text="Image is uploaded and prepared."
                />
                <Step
                  icon={<LineChart className="h-4 w-4" />}
                  title="Classify"
                  text="The model checks visible quality."
                />
                <Step
                  icon={<Database className="h-4 w-4" />}
                  title="Recommend"
                  text="The result becomes an action."
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="relative overflow-hidden px-5 pb-20 md:px-8">
        <div className="relative z-10 mx-auto grid max-w-7xl gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-3xl bg-white p-6 shadow-sm md:p-7">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              <h2 className="font-display text-xl font-semibold text-ink">
                Training readiness
              </h2>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-3">
              {metrics.map((metric) => (
                <div
                  key={metric.label}
                  className="rounded-2xl border border-border bg-surface-2 p-4"
                >
                  <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-ink-soft">
                    {metric.label}
                  </p>
                  <p className="mt-1 font-display text-2xl font-semibold tracking-tight text-ink">
                    {metric.value}
                  </p>
                </div>
              ))}
            </div>

            <p className="mt-5 flex items-center gap-2 text-sm text-ink-soft">
              <WifiOff className="h-4 w-4 text-primary" />
              Prepared for a local demonstration and later API deployment.
            </p>
          </div>

          <div className="rounded-3xl bg-white p-6 shadow-sm md:p-7">
            <div className="flex items-center gap-2">
              <PackageCheck className="h-5 w-5 text-primary" />
              <h2 className="font-display text-xl font-semibold text-ink">
                Recent assessments
              </h2>
            </div>

            <div className="mt-5 overflow-hidden rounded-2xl border border-border">
              {history.map((item, index) => (
                <div
                  key={`${item.site}-${item.time}`}
                  className={`flex items-center justify-between gap-4 bg-white p-4 ${
                    index !== history.length - 1 ? "border-b border-border" : ""
                  }`}
                >
                  <div className="min-w-0">
                    <p className="truncate font-semibold text-ink">
                      {item.site}
                    </p>
                    <p className="mt-0.5 truncate text-sm text-ink-soft">
                      {item.result}
                    </p>
                  </div>

                  <div className="flex items-center gap-3 text-right">
                    <span
                      className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                        item.risk === "High"
                          ? "bg-danger/10 text-danger"
                          : item.risk === "Medium"
                            ? "bg-warning/10 text-warning"
                            : "bg-success/10 text-success"
                      }`}
                    >
                      {item.risk}
                    </span>

                    <span className="text-xs font-medium text-ink-soft">
                      {item.time}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section id="data" className="relative overflow-hidden px-5 pb-24 pt-4 md:px-8">
        <div className="relative z-10 mx-auto max-w-7xl overflow-hidden rounded-[1.75rem] bg-white shadow-sm">
          <div className="grid gap-10 p-6 md:grid-cols-[0.85fr_1.15fr] md:items-end md:p-10">
            <div>
              <p className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3.5 py-1.5 text-xs font-bold uppercase tracking-[0.16em] text-primary">
                <FlaskConical className="h-3.5 w-3.5" />
                Public training data
              </p>

              <h2 className="mt-5 font-display text-3xl font-semibold leading-tight text-ink md:text-4xl">
                Public datasets prepared for the maize quality model.
              </h2>

              <p className="mt-4 max-w-md text-sm leading-7 text-ink-soft">
                Local farmer images are kept for manual testing only. Training
                relies on public datasets to keep the model workflow consistent
                and reproducible.
              </p>
            </div>

            <div className="grid gap-3 md:grid-cols-3">
              {datasetSources.map((source) => (
                <a
                  key={source.name}
                  href={source.href}
                  target="_blank"
                  rel="noreferrer"
                  className="group relative flex flex-col rounded-2xl border border-border bg-surface-2 p-5 transition hover:-translate-y-1 hover:border-clay/40 hover:bg-white hover:shadow-md"
                >
                  <div className="flex items-start justify-between">
                    <div className="grid h-9 w-9 place-items-center rounded-xl bg-clay/10 text-clay">
                      <FlaskConical className="h-4 w-4" />
                    </div>

                    <ExternalLink className="h-4 w-4 text-ink-soft transition group-hover:text-clay" />
                  </div>

                  <p className="mt-5 font-display text-lg font-semibold text-ink">
                    {source.name}
                  </p>

                  <p className="mt-1 text-xs font-bold uppercase tracking-[0.12em] text-primary">
                    {source.purpose}
                  </p>

                  <p className="mt-3 text-sm leading-6 text-ink-soft">
                    {source.detail}
                  </p>
                </a>
              ))}
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-border bg-white px-5 py-10 md:px-8">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 text-sm text-ink-soft">
          <div className="flex items-center gap-2">
            <Leaf className="h-4 w-4 text-primary" />
            <span className="font-medium text-ink">MaizeGuard Rwanda</span>
            <span>· Field assessment prototype</span>
          </div>

          <p>Post-harvest decision support · {new Date().getFullYear()}</p>
        </div>
      </footer>
    </main>
  );
}

function MiniStat({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-border bg-white p-3 shadow-sm">
      <div className="mb-2 grid h-7 w-7 place-items-center rounded-lg bg-primary/10 text-primary">
        {icon}
      </div>

      <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-ink-soft">
        {label}
      </p>

      <p className="mt-0.5 text-sm font-semibold text-ink">{value}</p>
    </div>
  );
}

function Step({
  icon,
  title,
  text,
}: {
  icon: ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-surface-2 p-4">
      <div className="mb-3 grid h-9 w-9 place-items-center rounded-xl bg-primary/10 text-primary">
        {icon}
      </div>

      <p className="font-semibold text-ink">{title}</p>
      <p className="mt-1 text-xs leading-5 text-ink-soft">{text}</p>
    </div>
  );
}

function classifyImageAppearance(src: string): Promise<QualityKey | null> {
  return new Promise((resolve) => {
    const image = new Image();

    image.onload = () => {
      const canvas = document.createElement("canvas");
      const size = 72;
      canvas.width = size;
      canvas.height = size;

      const context = canvas.getContext("2d", {
        willReadFrequently: true,
      });

      if (!context) {
        resolve(null);
        return;
      }

      context.drawImage(image, 0, 0, size, size);

      const pixels = context.getImageData(0, 0, size, size).data;

      let dark = 0;
      let yellow = 0;
      let redBrown = 0;
      let blueBlack = 0;
      let veryLight = 0;

      const total = pixels.length / 4;

      for (let index = 0; index < pixels.length; index += 4) {
        const red = pixels[index];
        const green = pixels[index + 1];
        const blue = pixels[index + 2];
        const lightness = (red + green + blue) / 3;

        if (lightness < 64) dark += 1;
        if (lightness > 218) veryLight += 1;
        if (red > 115 && green > 85 && blue < 120 && red >= green * 0.82) {
          yellow += 1;
        }
        if (red > 75 && green < 115 && blue < 95 && red > blue * 1.25) {
          redBrown += 1;
        }
        if (blue > red * 1.1 && blue > green * 1.05 && lightness < 130) {
          blueBlack += 1;
        }
      }

      const darkRatio = dark / total;
      const yellowRatio = yellow / total;
      const redBrownRatio = redBrown / total;
      const blueBlackRatio = blueBlack / total;
      const veryLightRatio = veryLight / total;

      if (darkRatio > 0.28 || blueBlackRatio > 0.08) {
        resolve("mold");
        return;
      }

      if (redBrownRatio > 0.12 || yellowRatio < 0.32 || veryLightRatio > 0.22) {
        resolve("discolored");
        return;
      }

      if (yellowRatio > 0.55) {
        resolve("good");
        return;
      }

      resolve("broken");
    };

    image.onerror = () => resolve(null);
    image.src = src;
  });
}