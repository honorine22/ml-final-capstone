import { NextResponse } from "next/server";

export const runtime = "nodejs";

type QualityKey = "good" | "broken" | "impurity" | "discolored" | "mold";

type ModelApiResponse = {
  label: string;
  confidence: number;
  confidence_percent?: number;
  confidencePercent?: number;
  probabilities?: Record<string, number>;
  risk: string;
  action: string;
  recommendation?: string;
  detail?: string;
};

const rules = [
  {
    match: ["mold", "fungus", "aflatoxin"],
    key: "mold",
    label: "Visible mold-risk grain",
    confidence: 91,
    risk: "High",
    action: "Do not store; refer to cooperative facility",
    detail: "Visible mold risk requires careful handling and professional quality checks."
  },
  {
    match: ["broken", "damage", "crack", "defective"],
    key: "broken",
    label: "Broken or damaged grain",
    confidence: 87,
    risk: "Medium",
    action: "Sort before storage",
    detail: "Remove visibly damaged kernels to reduce buyer rejection and storage risk."
  },
  {
    match: ["impurity", "dirty", "dust", "stone", "husk"],
    key: "impurity",
    label: "Impurity-contaminated grain",
    confidence: 89,
    risk: "Medium",
    action: "Clean and re-screen",
    detail: "Separate stones, husks, dust, and foreign matter before aggregation."
  },
  {
    match: ["pink", "dark", "discolor", "stain", "africa", "african", "cob", "ear", "mixed"],
    key: "discolored",
    label: "Discolored grain",
    confidence: 82,
    risk: "Medium",
    action: "Sell quickly or refer for review",
    detail: "Discoloration can reduce quality grade; avoid mixing with clean grain."
  }
];

function normalizeLabel(label: string): QualityKey {
  const value = label.toLowerCase().trim();

  if (value.includes("good") || value.includes("healthy") || value.includes("normal")) return "good";
  if (value.includes("broken") || value.includes("damage") || value.includes("defect")) return "broken";
  if (value.includes("impurity") || value.includes("dirty") || value.includes("foreign")) return "impurity";
  if (value.includes("discolor") || value.includes("stain") || value.includes("dark")) return "discolored";
  if (value.includes("mold") || value.includes("rotten") || value.includes("fung")) return "mold";

  return "discolored";
}

function fallbackPrediction(file: File | null) {
  const name = file?.name.toLowerCase() ?? "";
  const matched = rules.find((rule) => rule.match.some((term) => name.includes(term)));

  if (matched) {
    const { match, ...prediction } = matched;
    void match;
    return prediction;
  }

  return {
    key: "discolored",
    label: "Needs quality review",
    confidence: 71,
    risk: "Medium",
    action: "Do not mark as good; inspect or refer",
    detail: "The model API is not connected yet, so this sample should be reviewed before storage or sale."
  };
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
    return NextResponse.json({
      ...fallbackPrediction(file),
      source: "local-fallback"
    });
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
          fallback: fallbackPrediction(file)
        },
        { status: response.status }
      );
    }

    const result = (await response.json()) as ModelApiResponse;
    const key = normalizeLabel(result.label);
    const confidencePercent = result.confidence_percent ?? result.confidencePercent ?? result.confidence * 100;

    return NextResponse.json({
      key,
      label: result.label,
      confidence: Math.round(confidencePercent),
      confidenceRaw: result.confidence,
      confidencePercent,
      probabilities: result.probabilities ?? {},
      risk: result.risk,
      action: result.action,
      detail: result.recommendation ?? result.detail ?? "",
      source: "model-api"
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to call model API.",
        details: error instanceof Error ? error.message : "Unknown error",
        fallback: fallbackPrediction(file)
      },
      { status: 502 }
    );
  }
}
