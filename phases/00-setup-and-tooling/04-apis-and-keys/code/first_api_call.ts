// Phase 0 · Lesson 04 — APIs and keys (TypeScript port).
// Reads GEMINI_API_KEY from env, parses a minimal .env file, then makes one
// generateContent call with global fetch. Set MOCK=1 to skip the network entirely.
// Refs: https://ai.google.dev/api/generate-content
//       https://nodejs.org/api/process.html#processenv
//       https://nodejs.org/api/globals.html#fetch (Node 18+ ships fetch)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";


type GeminiRequest = {
  contents: { parts: { text: string }[] }[];
  generationConfig: { maxOutputTokens: number };
};

type GeminiResponse = {
  candidates: { content: { parts: { text: string }[] } }[];
  usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number };
};

// .env loader. Same shape every framework follows; we skip a dep to stay
// portable. KEY=VALUE per line, # comments, optional surrounding quotes.
function loadDotenv(path: string): Record<string, string> {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    return {};
  }
  const out: Record<string, string> = {};
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

function mergeEnv(): NodeJS.ProcessEnv {
  // process.env wins so users can override the file without editing it.
  const fromFile = loadDotenv(resolve(process.cwd(), ".env"));
  return { ...fromFile, ...process.env };
}

// Fixture matches the real generateContent response shape, so the surrounding
// code is identical whether MOCK=1 or not.
const MOCK_RESPONSE: GeminiResponse = {
  candidates: [
    {
      content: {
        parts: [{ text: "A neural network learns patterns by adjusting connected weights." }],
      },
    },
  ],
  usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 12 },
};

async function callGemini(
  apiKey: string,
  model: string,
  request: GeminiRequest,
): Promise<GeminiResponse> {
  if (process.env.MOCK === "1" || apiKey === "mock") {
    return MOCK_RESPONSE;
  }

  const url = new URL(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
  );
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify(request),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`gemini ${resp.status}: ${body.slice(0, 200)}`);
  }
  return (await resp.json()) as GeminiResponse;
}

async function main(): Promise<number> {
  const env = mergeEnv();
  const model = (env.LLM_MODEL ?? "").trim() || "gemini-2.5-flash";
  const apiKey = env.GEMINI_API_KEY ?? "mock";
  const usingMock = process.env.MOCK === "1" || apiKey === "mock";

  process.stdout.write("=== API Calls ===\n\n");
  process.stdout.write(
    usingMock
      ? "Mode: MOCK (no network). Unset MOCK and export GEMINI_API_KEY for a live call.\n\n"
      : "Mode: LIVE.\n\n",
  );

  const request: GeminiRequest = {
    contents: [{ parts: [{ text: "What is a neural network in one sentence?" }] }],
    generationConfig: { maxOutputTokens: 256 },
  };

  try {
    const response = await callGemini(apiKey, model, request);
    const text = response.candidates[0]?.content.parts[0]?.text ?? "";
    const usage = response.usageMetadata ?? {};
    process.stdout.write(`response: ${text}\n`);
    process.stdout.write(
      `tokens: ${usage.promptTokenCount ?? "?"} in, ${usage.candidatesTokenCount ?? "?"} out\n`,
    );
    return 0;
  } catch (err) {
    process.stderr.write(`request failed: ${(err as Error).message}\n`);
    return 1;
  }
}

main().then((code) => process.exit(code));
