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

`0.6.0` — four tools are wired (`ocr_read`, `ocr_job`, `ocr_queue`, `ocr_verify`) and three deployment
surfaces are supported: the hosted ModelBest `document-parse` API, a local **document service** with the
full layout pipeline (official Docker image or the SDK Web Demo backend), and a bare OpenAI-compatible
chat server (llama.cpp, vLLM, SGLang, Ollama) for page-level text. 136 tests, ruff clean,
`hermes plugins validate` 15/15.

`ocr_extract` is the next tool; the review loop ships without it because extraction is optional and the
read/verify/queue path is the part that prevents a wrong number from being quoted.

## Install

```bash
# once listed in the Hermes plugin catalog
hermes plugins install minicpm-ocr

# or from a clone, before it is listed
cp -r minicpm_ocr ~/.hermes/plugins/minicpm-ocr
hermes plugins enable minicpm-ocr
```

The plugin lives in `minicpm_ocr/`; `plugin.yaml` sits there, which is why the catalog entry uses
`subdir: minicpm_ocr`.

Requires Hermes `>=0.22.0` and Python `>=3.10`. No third-party Python dependencies.

## Configure

| Setting | What it is |
|---|---|
| `backend` | `auto` \| `cloud` \| `local`. Which family to consider. Default `auto`. |
| `local_api` | `auto` \| `document` \| `chat`. `document` = full layout pipeline with blocks and PDFs; `chat` = bare model server, images only. Set this when `backend: local`. |
| `base_url` | Service root without `/v1`, e.g. `http://127.0.0.1:8966` (document) or `http://127.0.0.1:18499` (chat). Empty = auto-discover the default chat ports. |
| `api_key` | ModelBest API key. **Secret**: stored in `$HERMES_HOME/.env` as `MODELBEST_API_KEY`, never in `config.yaml`. |
| `parser_model` | Parser model id as the backend names it (`minicpm-ocr` on llama.cpp/Ollama, `MiniCPM-OCR-1.0` on ModelBest). |
| `jev_model` | Jev model id as the backend names it. |
| `jev_base_url` | Separate service root for Jev, without `/v1`. Empty = same as `base_url`. |
| `jev_api` | `auto` \| `rm` \| `chat`. `rm` = local Jev render+score service (`/v1/rm/score`); `chat` = hosted Jev Chat Completions. `auto` picks `rm` for a loopback URL, `chat` otherwise. |
| `threshold` | Jev score at or above which a region is accepted (default `0.6`). |

Backends are tried in order: explicit `base_url`, the hosted key, then the default local ports. A
loopback candidate that is not listening is skipped in about 0.5 s, so a dead local service returns
`backend_unavailable` instead of stalling through HTTP retries; if another candidate is up, it is used.
Set `backend` and `local_api` when you need one specific deployment. `ocr_verify` preflights a local
`jev_base_url` the same way.

Set the API key through the Desktop plugin form or by adding the variable to the profile's `.env`:

```bash
printf '\nMODELBEST_API_KEY=...\n' >> "${HERMES_HOME:-$HOME/.hermes}/.env"
```

The plugin reads it with `agent.secret_scope.get_secret("MODELBEST_API_KEY")`, so it follows the active
profile and never reads `.env` itself.

Non-secret settings:

```bash
hermes config set plugins.entries.minicpm-ocr.settings.backend local
hermes config set plugins.entries.minicpm-ocr.settings.local_api document
hermes config set plugins.entries.minicpm-ocr.settings.base_url http://127.0.0.1:8966
hermes config set plugins.entries.minicpm-ocr.settings.threshold 0.6
```

## Tools

### `ocr_read(path, mode="summary"|"markdown"|"blocks", max_chars=20000, wait=True, upright=True)`

Parses one image or PDF. Hosted images use the synchronous endpoint; PDFs and everything on a local
**document** service run as a job, polled for up to three minutes. Returns page count, region count, block
labels, failed regions with their bboxes, and a Markdown preview. Context-safe by default: large output is
truncated with a flag rather than flooding the turn. A job that outlives the wait returns its `job_id` and
status instead of failing — poll it with `ocr_job`.

A bare **chat** backend has no layout stage, so it returns page text only (`blocks_available: false`) and
rejects PDFs with a message that names the fix.

### `ocr_job(job_id, action="status"|"cancel")`

Status of an asynchronous parse job: `pages_done` / `pages_total`, terminal state, page count. Works against
the hosted API and a local document service.

### `ocr_verify(path, text, category="text"|"formula"|"table", threshold=0.6)`

Scores whether `text` actually matches the region at `path`. Hosted Jev speaks Chat Completions; a local
Jev service speaks `/v1/rm/score` and renders the candidate itself. A verdict other than `scored` is
returned as `scored: false` so the agent cannot mistake a transport failure for a low score. Use it before
quoting a number, a date or a name, and when comparing two candidate readings of the same crop.

### `ocr_queue(action="list"|"show"|"accept"|"reject"|"add"|"stats"|"export", …)`

The human gate. Every region a parse reports as `failed`, plus anything the agent flags, is stored in SQLite
under the plugin's data directory — not in the conversation, so it survives context compression, restarts and
`hermes plugins update`. `accept` / `reject` record a note and a timestamp; `export` writes the whole queue
as CSV for whoever does the reviewing.

## Backends

| Backend | Blocks + bboxes | PDFs | Jev | Notes |
|---|---|---|---|---|
| ModelBest API (`api.modelbest.ai`) | yes | yes, as jobs | hosted Chat Completions | `POST /v1/document-parse`, `/v1/document-parse/jobs` |
| Local document service (Docker / SDK Web Demo) | yes | yes, as jobs | separate local Jev service | `POST /api/document-parse/jobs`, `GET`/`DELETE` by id; layout runs in the service |
| llama.cpp / vLLM / SGLang / Ollama | **no** | **no** | separate local Jev service | OpenAI Chat Completions, page text only |

The difference is architectural: per-region blocks are produced by the **layout stage** in OpenBMB's SDK,
which crops the page and sends each region to the model. A bare model server has no layout stage, so it
answers one image with one block of text. The tool payload always reports `blocks_available`, and asking
for a PDF on a chat backend returns a clear error instead of a stack trace.

### Local, full pipeline (blocks + PDFs)

The official Docker image bundles the SDK, vLLM and the Web Demo backend. It is the shortest path when Docker
and the NVIDIA Container Toolkit are available; expose port `8000`, then:

```bash
hermes config set plugins.entries.minicpm-ocr.settings.backend local
hermes config set plugins.entries.minicpm-ocr.settings.local_api document
hermes config set plugins.entries.minicpm-ocr.settings.base_url http://127.0.0.1:8000
```

The image's `vllm` shim fixes the context at 40960 and 36 images per prompt. That fits a 32 GB card, not a
12 GB one: it OOMs during vision warmup. On 12 GB, bypass the shim inside the pipeline script with a
16k-context, one-image profile:

```bash
/opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/ocr --served-model-name minicpm-ocr-1.0 \
  --trust-remote-code --host 0.0.0.0 --port 18499 \
  --max-model-len 16384 --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"downsample_mode":"4x","max_slice_nums":9}' \
  --gpu-memory-utilization 0.55 --attention-backend FLASH_ATTN \
  --gdn-prefill-backend triton --trust-request-mm-kwargs --enforce-eager \
  --uvicorn-log-level warning
```

Then start the bundled backend with `OCR_LAYOUT_DEVICE=cpu`, `OCR_UPRIGHT_DEVICE=cpu`,
`PORT=8000 /opt/venv/bin/python -s server.py`. The plugin's default `upright: true` runs the CPU
orientation/deskew stage; on a 12 GB host that is much slower than `upright: false` on an already-upright
scan (see measurements).

Without Docker, run the SDK Web Demo backend against any recognition server. This is the combination
verified on an RTX 4070 (12 GB) with llama.cpp:

```bash
# 1. Recognition server: llama.cpp build a08ab75 or newer, Q8_0 weights + 4x projector.
llama-server \
  --model ./models/MiniCPM-OCR-1.0-gguf/MiniCPM-OCR-1_0-Q8_0.gguf \
  --mmproj ./models/MiniCPM-OCR-1.0-gguf/mmproj-model-f16-4x.gguf \
  --alias minicpm-ocr --host 127.0.0.1 --port 18499 \
  --ctx-size 8192 --n-gpu-layers 999 --parallel 1 --reasoning off

# 2. SDK environment (layout runs here; CPU layout leaves the GPU to recognition).
uv venv .minicpm-ocr --python 3.12 --seed
. .minicpm-ocr/bin/activate
pip install "minicpm-ocr[all]" "paddlex==3.4.2" "pypdfium2>=4" "paddlepaddle==3.2.1"
pip install fastapi uvicorn python-multipart

# 3. One parse to download the layout model into ./models/PP-DocLayoutV3_infer.
minicpmvocr parse --input sample.png --server-url http://127.0.0.1:18499 \
  --model minicpm-ocr --backend llamacpp --device cpu --workers 1 \
  --set upright.enabled=false --output /tmp/sample.md

# 4. Document service on :8966 (the page-level API the plugin talks to).
OCRWEB_PYTHON="$PWD/.minicpm-ocr/bin/python" \
OCR_ENGINE_TYPE=llamacpp OCR_SERVER_URL=http://127.0.0.1:18499 OCR_SERVED_NAME=minicpm-ocr \
OCR_LAYOUT_MODEL_DIR="$PWD/models/PP-DocLayoutV3_infer" \
OCR_LAYOUT_DEVICE=cpu OCR_LAYOUT_WORKERS=1 OCR_UPRIGHT_WORKERS=0 \
OCRWEB_DATA_DIR="$PWD/webdemo-data" PORT=8966 \
bash webdemo/backend/start_local_worker.sh
```

Then point the plugin at it (`backend: local`, `local_api: document`, `base_url: http://127.0.0.1:8966`).
`upright: true` on the tool call asks the service to detect and correct page orientation; set
`OCR_UPRIGHT_WORKERS` above zero and prepare `models/paddle_upright` if you want that path on.

### Local, page text only (llama.cpp / Ollama)

Good for crops and single pages; no blocks, no PDFs. With the server above:

```bash
hermes config set plugins.entries.minicpm-ocr.settings.backend local
hermes config set plugins.entries.minicpm-ocr.settings.local_api chat
hermes config set plugins.entries.minicpm-ocr.settings.base_url http://127.0.0.1:18499
hermes config set plugins.entries.minicpm-ocr.settings.parser_model minicpm-ocr
```

Ollama uses the same contract; register the GGUF model exactly as its deployment guide says, then set
`base_url` to `http://127.0.0.1:11434` (or your `OLLAMA_HOST`) and `parser_model` to the registered name.

### Local Jev

`ocr_verify` needs Jev's **render+score** service, not the raw scoring engine: it renders the candidate text
before scoring. The Docker image and the SDK script expose the same fields; start the service and point the
plugin at its port:

Jev needs its own llama.cpp build (`7eed6b768` or newer) and the GGUF pair from
`openbmb/MiniCPM-OCR-Jev-gguf`; start the engine on `:18501`, then the render+score service on
`:18500` with the renderer environment (Playwright Chromium + Pillow 12.3.0). A CJK font is required
for Chinese regions; without one Chromium draws boxes and the score is meaningless.

```bash
# engine (one model per process; this one is not a chat model)
llama-server -m MiniCPM-OCR-Jev-Q4_K_M.gguf --mmproj mmproj-model-f16.gguf \
  -ngl 99 -c 16384 --host 127.0.0.1 --port 18501 --alias minicpm-ocr-jev

# render+score service
cd MiniCPM-OCR/scripts/minicpm-ocr-jev
JEV_BACKEND_CONFIG=sdk/backends/llamacpp.yaml \
  PATH=/opt/venv-render/bin:$PATH bash sdk/start_render_score.sh 18500 http://127.0.0.1:18501
```

```bash
hermes config set plugins.entries.minicpm-ocr.settings.jev_base_url http://127.0.0.1:18500
hermes config set plugins.entries.minicpm-ocr.settings.jev_api rm
```

Without a local Jev service, `ocr_verify` returns `backend_unavailable`; it does not silently skip the check.

## Measured, not claimed

Run on 2026-11-06, one RTX 4070 12 GB, one i7-class CPU, Python 3.12:

| Path | What | Result |
|---|---|---|
| ModelBest API | 653 KB invoice PNG, 15 regions | **6–25 s** (warm–cold), 0 failed, 1 skipped |
| ModelBest API | 1-page PDF, 4 regions | **9–13 s** |
| ModelBest Jev | Jev example crop, correct transcription | **0.9952**, **1.9 s** |
| local Jev (Q4_K_M + render service) | same crop, correct transcription | **0.9948**, **1.0 s** (reference **0.99501**) |
| local Jev (Q4_K_M + render service) | same crop, reordered OCR output | **0.0020**, **0.8 s** |
| llama.cpp Q8 | single text crop (`Text Recognition:`) | **0.72 s** |
| llama.cpp + SDK document service | invoice, 15 regions, warm | **7.5–12.2 s** |
| llama.cpp + SDK document service | same invoice, first job (layout load) | **12.3 s** |
| llama.cpp + SDK document service | 1-page PDF, 4 regions | **9.0 s** |
| llama.cpp + SDK document service | 6-page PDF, 89 regions | **32.4 s** |
| Docker image (12 GB profile) | invoice, `upright: false` | **29–43 s** |
| Docker image (12 GB profile) | invoice, `upright: true` (CPU deskew) | **116 s** |
| Docker image (12 GB profile) | 1-page PDF | **41 s** |
| Docker OCR + local Jev | correct / reordered / Chinese | **0.9949 / 0.006 / 0.9427** |
| Docker image, cold start to `/health` | **111 s** |
| Docker image, 6-page PDF | **188 s**, job handle at the 180 s wait |
| Docker image, 2 images concurrently | **55 s** warm |
| Local Jev, 6 concurrent verifies | **1.0 s** total |

The local document service was exercised through `ocr_read` → `ocr_job` → `ocr_queue`, and the SDK CLI
wrote the same page/block JSON shape (`status`, `pages_done`, `pages_total`, `pages[]`, `markdown`) that the
hosted jobs API returns. Jev cold starts remain the slowest step on the hosted side; the client retries and
defaults to a 240 s timeout rather than pretending the first call is fast.

Vendor figures (benchmarks, throughput) belong to the model cards and are intentionally not repeated here.

## Development

```bash
python -m pytest      # 136 tests, no network, 98% line coverage
ruff check .
hermes plugins validate minicpm_ocr   # the same check catalog admission runs
```

The test suite injects a fake transport, so nothing here needs a key or a network.

`hermes plugins validate` passes all fifteen checks against Hermes `0.22.5` — manifest, config
schema, loadability, the isolated capability probe, declared-vs-registered tools/hooks/middleware, built-in
name collisions, the security scan and the no-core-override check. CI runs it on every push, alongside
pytest and ruff.

## Disclosure

- Sends the document you point it at to the configured backend. With the ModelBest backend that is a
  third-party API; with a local `base_url` nothing leaves the machine. The plugin does not upload anything
  you did not explicitly pass to a tool.
- URLs are `https` only, and the host must resolve to a public address: loopback, private ranges and the
  cloud metadata address are refused, on the first request and on every redirect.
- No telemetry, no analytics, no self-updater.
- API keys are read through Hermes' secret route (`get_secret`, declared as `env: MODELBEST_API_KEY`); the
  plugin never reads `.env`, `auth.json` or another tool's credential store.
- A failed call returns an error payload, never a silently empty result.

## Roadmap

The short version: `ocr_extract` and the CLI next, the demo build and catalog PR after that, the MCP adapter
and calibration table for Spark Open Build, `1.0` installable by name, and the MiniCPM5 OCR checkpoint
absorbed as a model-id change when OpenBMB ships it. Full milestones, exit criteria and non-goals:
[ROADMAP.md](ROADMAP.md).

## Licence

MIT — see [LICENSE](LICENSE). SPDX: `MIT`.
