# Roadmap

**Goal:** a Hermes plugin that reads documents, knows which parts it could not read, and hands those to a
person — on the hosted API or entirely on your own machine. `1.0` is the finished article: any MiniCPM OCR
checkpoint, either deployment, installable by name from the catalog, with a tool surface that does not move
underneath the agent.

## Principles

- **The model is OpenBMB's; the plugin owns the loop.** Parsing is a commodity. Verify → re-read → decide →
  queue is the part that decides whether a number gets quoted or a human gets asked.
- **Every number in the README has a run behind it.** Vendor benchmarks stay the vendor's, with attribution.
- **Capabilities are declared, never assumed.** Blocks, PDFs and Jev availability are probed, and every payload
  says which one you actually got (`blocks_available`, job handles, error codes).
- **One toolset, two deployments.** The hosted `document-parse` API and any OpenAI-compatible server (llama.cpp,
  vLLM, SGLang, Ollama) are the same interface with different guarantees.
- **No silent success.** A failed call returns an error payload with a code and a next step, never an empty
  result that looks like an empty page.

## Milestones

### 0.5.0 — foundations · done (5 Nov 2026)

Four tools (`ocr_read`, `ocr_job`, `ocr_queue`, `ocr_verify`), dual backend, durable SQLite review queue,
documented limits and error taxonomy. 42 tests, ruff clean, `hermes plugins validate` 15/15.

### 0.6.0 — extraction and command line · 8 Nov

- `ocr_extract(source, fields)`: field-level key-information extraction with per-field confidence and ranked
  candidates, using the parser plus Jev as a scorer.
- `hermes ocr doctor | read | verify | queue`: the same core without an agent in the loop.
- Bundled skill: when to reach for OCR, which threshold, and what never to quote unverified.

**Exit:** extraction on ten documents with a per-field confidence table; `hermes ocr doctor` names the backend
it found on a clean machine.

### 0.7.0 — the loop · 10 Nov

- Crop-and-zoom re-read helper for a failed region (bbox in, tighter crop out, re-parse that crop).
- `hermes ocr calibrate`: score a labelled folder and emit thresholds per document type instead of one global
  number.
- Queue wired into the demo path: a low-confidence field ends in the queue, not in the answer.

**Exit:** the 90-second demo runs end to end twice, with the same numbers both times.

### 0.8.0 — submission build · 11 Nov

- Repo public, catalog entry (`subdir: minicpm_ocr`, full-SHA pin) and PR to `hermes-agent`.
- 90-second demo video, README with measured numbers and the failure table.
- Typeform filed the same evening.

**Exit:** PR open, form submitted with both links, before 5PM EST.

### 0.9.0 — Spark Open Build · 5 Dec

- MCP adapter over the same core, so the parse/verify pair is usable from Claude, Cursor or any MCP client —
  the integration the upstream repo does not ship.
- Published calibration table (printed / photographed / table / handwriting) and deployment docs for
  llama.cpp, vLLM, SGLang and Ollama.
- Case writeup: what the plugin does that a skill cannot, with the review-queue numbers.

**Exit:** an MCP client parses and verifies a document through the adapter; the docs let a stranger stand up the
local stack without asking a question.

### 1.0.0 — the final plugin · 13 Dec

- Stable tool schemas and semver; no breaking change without a major.
- Catalog-listed and installable by name; docs page generated from the pinned README.
- Queue export/import, retention policy, and a documented error taxonomy.
- CI green on every push: catalog validator, pytest, ruff.

**Exit:** `hermes plugins install minicpm-ocr` works on a clean machine, and the README's claims match that
build.

### 1.x — MiniCPM5 OCR

When OpenBMB ships the OCR checkpoint of the MiniCPM5 line (the S2 Phase-2 model is the likely slot), the plugin
should absorb it **without a rewrite**: a new model id, a capability probe, and one adapter test. The tool
surface already separates *what the model is* (`model`, `jev_model`) from *what the backend can do*
(`blocks`, jobs, Jev). That separation is the whole reason the version is a config change rather than a project.

**Exit:** the MiniCPM5 OCR model runs through the same four tools, with a measured table next to the 1.0 numbers
— and no change to the schemas the agent sees.

## Not planned

- Being an OCR engine, or fine-tuning one, inside this repo.
- Storing documents anywhere except the plugin's own data directory.
- Claiming on-device inference when the request is going to a hosted API.
- A desktop pane, watch folders or multi-user review before 1.0 — polish that competes with the loop.

## Assumptions and risks

- Model ids and free-API quotas can change; the config already exposes them.
- The local path is hardware-verified only when someone runs it on hardware; until then the README says so.
- Catalog review happens on a maintainer's schedule; an open PR is enough to submit, not to be listed.
