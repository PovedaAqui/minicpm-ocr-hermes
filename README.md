# MiniCPM-OCR for Hermes

**Read a document, and know which parts of it not to trust.**

A Hermes Agent plugin that turns [MiniCPM-OCR](https://github.com/OpenBMB/MiniCPM-OCR) into a toolset the
agent can reason about: every region comes back with a status, every claim can be scored against the image
it came from, and anything doubtful is re-read before it reaches the user.

Not affiliated with OpenBMB. MIT.

## Why this exists

An OCR call returns text. That is enough when the page is clean and the stakes are low, and it is not
enough when the answer is an invoice total, a contract date or a prescription line. The failure mode is
silent: the model produces a plausible string, the agent quotes it, nobody checks.

Two things fix that, and both are already in the model family:

- **MiniCPM-OCR 1.0** returns per-block `status` (`ok`, `failed`, `skipped_no_vlm`, `skipped_seal`).
- **MiniCPM-OCR Jev** scores any text against any image region, 0–1, and is model-agnostic — it grades
  another parser's output, or the agent's own transcription.

This plugin puts a stable interface on both, plus the loop around them: read → score → re-read the weak
region → accept, or hand it to a human.

## Status

`0.4.0` — four tools are wired (`ocr_read`, `ocr_job`, `ocr_queue`, `ocr_verify`), with both backend families
supported: the hosted `document-parse` API and any OpenAI-compatible local server, llama.cpp included. `ocr_extract`
lands next. 33 tests, ruff clean.

## Install

```bash
hermes plugins install /path/to/minicpm-ocr-hermes --subdir minicpm_ocr
hermes plugins enable minicpm-ocr
```

Requires Hermes `>=0.22.0` and Python `>=3.10`. No third-party Python dependencies.

## Configure

| Setting | What it is |
|---|---|
| `api_key` | ModelBest API key (secret, stored through Hermes' credential route — never in `config.yaml`) |
| `base_url` | Service root without `/v1`, e.g. `http://127.0.0.1:8000`. Empty = auto-discover |
| `threshold` | Jev score at or above which a region is accepted (default `0.6`) |

Set the key with `hermes config set plugins.entries.minicpm-ocr.settings.api_key …` or through the Desktop
plugin settings form. In code the plugin reads it with `agent.secret_scope.get_secret`.

## Tools

### `ocr_read(path, mode="summary"|"markdown"|"blocks", max_chars=20000)`

Parses one image (synchronously) or one PDF (as a job, polled to completion). Returns page count, region
count, block labels, failed regions with their bboxes, and a Markdown preview. Context-safe by default:
large output is truncated with a flag rather than flooding the turn.

### `ocr_job(job_id, action="status"|"cancel")`

Status of an asynchronous parse job: `pages_done` / `pages_total`, terminal state, page count.

### `ocr_verify(path, text, category="text"|"formula"|"table", threshold=0.6)`

Scores whether `text` actually matches the region at `path`. Use it before quoting a number, a date or a
name, and when comparing two candidate readings of the same crop.

### `ocr_queue(action="list"|"show"|"accept"|"reject"|"add"|"stats"|"export", …)`

The human gate. Every region a parse reports as `failed`, plus anything the agent flags, is stored in SQLite
under the plugin's data directory — not in the conversation, so it survives context compression, restarts and
`hermes plugins update`. `accept` / `reject` record a note and a timestamp; `export` writes the whole queue
as CSV for whoever does the reviewing.

## Backends

| Backend | Blocks + bboxes | PDFs | Notes |
|---|---|---|---|
| ModelBest API (`api.modelbest.ai`) | yes | yes, as jobs | `POST /v1/document-parse` + `POST /v1/document-parse/jobs`; Jev via Chat Completions |
| llama.cpp (`llama-server`) | **no** | **no** | OpenAI-compatible Chat Completions, port 8080 by default; needs `--mmproj`, one model per process |
| vLLM / SGLang | **no** | **no** | Same contract, port 8000 by default |
| Ollama | **no** | **no** | Same contract, port 11434; one model loaded per process too |

The difference is architectural, not a missing feature: per-region blocks are produced by the **layout stage**
in OpenBMB's SDK, which crops the page and sends each region to the model. A bare model server has no layout
stage, so it answers one image with one block of text. The tool payload always reports `blocks_available`, and
asking for a PDF on a local backend returns a clear error instead of a stack trace.

### llama.cpp

`llama-server` serves one model per process, so the parser and the judge need two of them — that is what
`jev_base_url` is for. Images only work if the server was started with a vision projector (`--mmproj`); without
it the model accepts text and rejects the image.

```bash
# names vary by quant — check the GGUF card for the exact files
llama-server -m MiniCPM-OCR-1.0-Q4_K_M.gguf --mmproj mmproj-minicpm-ocr-f16.gguf \
             -c 4096 --port 8080 --jinja
llama-server -m MiniCPM-OCR-Jev-Q4_K_M.gguf --mmproj mmproj-minicpm-ocr-jev-f16.gguf \
             -c 4096 --port 8081 --jinja

hermes config set plugins.entries.minicpm-ocr.settings.backend local
hermes config set plugins.entries.minicpm-ocr.settings.base_url http://127.0.0.1:8080
hermes config set plugins.entries.minicpm-ocr.settings.jev_base_url http://127.0.0.1:8081
hermes config set plugins.entries.minicpm-ocr.settings.model minicpm-ocr
```

A page consumes image tokens, so `-c 4096` is a floor rather than a luxury; `--image-max-tokens` is the knob that
bounds cost per image.

### Ollama

```bash
ollama pull minicpm-ocr
hermes config set plugins.entries.minicpm-ocr.settings.backend local
hermes config set plugins.entries.minicpm-ocr.settings.base_url http://127.0.0.1:11434
hermes config set plugins.entries.minicpm-ocr.settings.model minicpm-ocr:latest
```

`backend: auto` (the default) prefers an explicit `base_url`, then the hosted API when a key is present, then a
loopback server on :8000, :8080 or :11434.

## Measured, not claimed

Run on 2026-11-05 against the ModelBest API, one laptop, one key:

- Image parse (24 KB PNG, 3 blocks): **2.7 s**.
- PDF job (2 synthetic pages): submit `202`, terminal `Success`, **18.3 s** wall clock.
- `ocr_verify` on a correct line: **0.9974** and **0.9933**.
- `ocr_verify` on the same line with one digit changed (`1,284.50` → `1,248.50`): **0.0000**.
- Jev cold start exceeded 90 s once; warm calls are **1.3 s**. The client retries and defaults to a 240 s
  timeout rather than pretending the first call is fast.
- The local wire contract (`parse_page` over Chat Completions with `Text Recognition:`) was exercised against
  the hosted service, which speaks the same protocol: **4.0 s**, page text returned, zero blocks. It has not yet
  been run against a local vLLM/Ollama server — that is the next hardware-dependent check.

Vendor figures (benchmarks, throughput) belong to the model cards and are intentionally not repeated here.

## Development

```bash
python -m pytest      # 19 tests, no network
ruff check .
```

The test suite injects a fake transport, so nothing here needs a key or a network.

## Disclosure

- Sends the document you point it at to the configured backend. With the ModelBest backend that is a
  third-party API; with a local `base_url` nothing leaves the machine. The plugin does not upload anything
  you did not explicitly pass to a tool.
- No telemetry, no analytics, no self-updater.
- API keys are read through Hermes' secret route; the plugin never reads `.env`, `auth.json` or another
  tool's credential store.
- A failed call returns an error payload, never a silently empty result.

## Licence

MIT.
