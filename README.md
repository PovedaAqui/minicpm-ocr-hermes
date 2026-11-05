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

`0.1.0` — three tools are wired (`ocr_read`, `ocr_job`, `ocr_verify`). `ocr_extract` and `ocr_queue` land
next. The core client, limits, discovery and error taxonomy are done and tested.

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

## Backends

| Backend | Blocks + bboxes | Notes |
|---|---|---|
| ModelBest API (`api.modelbest.ai`) | yes | `POST /v1/document-parse`, jobs for PDFs, Jev via Chat Completions |
| vLLM / SGLang / llama.cpp / Ollama | **no** | OpenAI-compatible; page-level text only, because the layout stage lives in OpenBMB's SDK, not in the model |

The tool payload always reports `blocks_available` so the model never assumes a bbox it does not have.

## Measured, not claimed

Run on 2026-11-05 against the ModelBest API, one laptop, one key:

- Image parse (24 KB PNG, 3 blocks): **2.7 s**.
- PDF job (2 synthetic pages): submit `202`, terminal `Success`, **18.3 s** wall clock.
- `ocr_verify` on a correct line: **0.9974** and **0.9933**.
- `ocr_verify` on the same line with one digit changed (`1,284.50` → `1,248.50`): **0.0000**.
- Jev cold start exceeded 90 s once; warm calls are **1.3 s**. The client retries and defaults to a 240 s
  timeout rather than pretending the first call is fast.

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
