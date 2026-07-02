import { NextResponse } from "next/server";

export const runtime = "nodejs";

type QualityKey = "good" | "broken" | "impurity" | "mold";

type ModelApiResponse = {
  label: string;
  raw_label?: string;
  confidence: number;
  confidence_percent?: number;
  confidencePercent?: number;
  needs_review?: boolean;
  needsReview?: boolean;
  review_reason?: string | null;
  probabilities?: Record<string, number>;
  risk: string;
  action: string;
  recommendation?: string;
  detail?: string;
};

function normalizeLabel(label: string): QualityKey | null {
  const value = label.toLowerCase().trim();

  if (value.includes("good") || value.includes("healthy") || value.includes("normal")) return "good";
  if (value.includes("broken") || value.includes("damage") || value.includes("defect")) return "broken";
  if (value.includes("impurity") || value.includes("dirty") || value.includes("foreign")) return "impurity";
  if (value.includes("mold") || value.includes("rotten") || value.includes("fung")) return "mold";

  return null;
}

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("image");

  if (!(file instanceof File)) {
    return NextResponse.json(
      { error: "No image file was uploaded." },
      { status: 400 }
    );
  }

  const modelApiUrl = process.env.MODEL_API_URL;

  if (!modelApiUrl) {
    return NextResponse.json(
      {
        label: "Needs review",
        confidence: 0,
        needsReview: true,
        risk: "Needs review",
        action: "Model unavailable",
        detail: "MODEL_API_URL is not configured.",
        source: "model-api-error"
      },
      { status: 503 }
    );
  }

  try {
    const forwardFormData = new FormData();
    forwardFormData.append("image", file, file.name);

    const response = await fetch(`${modelApiUrl.replace(/\/$/, "")}/predict`, {
      method: "POST",
      body: forwardFormData
    });

    if (!response.ok) {
      const details = await response.text();

      return NextResponse.json(
      {
        error: "Model API failed.",
        details,
        label: "Needs review",
        confidence: 0,
        confidenceRaw: 0,
        confidencePercent: 0,
        needsReview: true,
        risk: "Needs review",
        action: "Needs review",
        detail: "The model API returned an error. Check the backend server logs.",
        source: "model-api-error"
      },
      { status: response.status }
      );
    }

    const result = (await response.json()) as ModelApiResponse;
    const key = normalizeLabel(result.label);

    if (!key) {
      return NextResponse.json(
        {
          label: "Needs review",
          confidence: 0,
          needsReview: true,
          risk: "Needs review",
          action: "Unsupported model label",
          detail: `The model returned an unsupported label: ${result.label}`,
          source: "model-api-error"
        },
        { status: 502 }
      );
    }
    const confidencePercent = result.confidence_percent ?? result.confidencePercent ?? result.confidence * 100;
    const needsReview = result.needs_review ?? result.needsReview ?? false;

    return NextResponse.json({
      key,
      label: result.label,
      rawLabel: result.raw_label ?? result.label,
      confidence: Math.round(confidencePercent),
      confidenceRaw: result.confidence,
      confidencePercent,
      needsReview,
      probabilities: result.probabilities ?? {},
      risk: needsReview ? "Needs review" : result.risk,
      action: needsReview ? "Needs review" : result.action,
      detail: result.review_reason ?? result.recommendation ?? result.detail ?? "",
      source: "model-api"
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to call model API.",
        details: error instanceof Error ? error.message : "Unknown error",
        label: "Needs review",
        confidence: 0,
        confidenceRaw: 0,
        confidencePercent: 0,
        needsReview: true,
        risk: "Needs review",
        action: "Needs review",
        detail: "The model API is not reachable. Start the backend server and try again.",
        source: "model-api-error"
      },
      { status: 502 }
    );
  }
}
