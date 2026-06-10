import { NextResponse } from "next/server";

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

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("image");
  const name = file instanceof File ? file.name.toLowerCase() : "";
  const matched = rules.find((rule) => rule.match.some((term) => name.includes(term)));

  if (matched) {
    const { match, ...prediction } = matched;
    void match;
    return NextResponse.json(prediction);
  }

  return NextResponse.json(
    {
      key: "discolored",
      label: "Needs quality review",
      confidence: 71,
      risk: "Medium",
      action: "Do not mark as good; inspect or refer",
      detail: "The demo endpoint could not confidently verify this as clean maize grain, so it should be reviewed before storage or sale."
    }
  );
}
