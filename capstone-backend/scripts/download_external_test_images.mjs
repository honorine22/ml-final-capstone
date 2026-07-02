import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";

const DATASET_ID = "r6vvm5jkh6";
const DATASET_VERSION = 2;
const SAMPLE_COUNT = 12;
const API_ROOT = "https://data.mendeley.com/public-api";
const OUTPUT_ROOT = resolve("data/external_test/efficientmaize");

const folders = {
  good: "812d254e-d1da-4038-a210-6cad8e675321",
  bad_unresolved: "2e6c9d41-0d9a-4291-a729-db124abd55ea",
};

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: { Accept: "application/vnd.mendeley-public-dataset.1+json" },
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.json();
}

function evenlySpaced(files, count) {
  if (files.length <= count) return files;
  return Array.from({ length: count }, (_, index) => {
    const position = Math.round((index * (files.length - 1)) / (count - 1));
    return files[position];
  });
}

const manifest = [];

for (const [testLabel, folderId] of Object.entries(folders)) {
  const url = `${API_ROOT}/datasets/${DATASET_ID}/files?folder_id=${folderId}&version=${DATASET_VERSION}`;
  const files = await fetchJson(url);
  const selected = evenlySpaced(
    files.filter((file) => file.status === "COMPLETED" && file.content_details?.download_url),
    SAMPLE_COUNT,
  );

  const outputDir = resolve(OUTPUT_ROOT, testLabel);
  await mkdir(outputDir, { recursive: true });

  for (const [index, file] of selected.entries()) {
    const response = await fetch(file.content_details.download_url);
    if (!response.ok) throw new Error(`Could not download ${file.filename}: ${response.status}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    const extension = basename(file.filename).split(".").pop() || "jpg";
    const outputName = `${testLabel}_${String(index + 1).padStart(2, "0")}.${extension}`;
    await writeFile(resolve(outputDir, outputName), bytes);

    manifest.push({
      file: `${testLabel}/${outputName}`,
      source_file: file.filename,
      source_label: testLabel === "good" ? "good" : "bad",
      expected_app_label: testLabel === "good" ? "good" : "needs_manual_review",
      sha256: createHash("sha256").update(bytes).digest("hex"),
    });
  }
}

await mkdir(OUTPUT_ROOT, { recursive: true });
await writeFile(resolve(OUTPUT_ROOT, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Downloaded ${manifest.length} external test images to ${OUTPUT_ROOT}`);
