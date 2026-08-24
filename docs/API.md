# HTTP API

> Status: **design, pre-implementation**. This is the contract the implementation is written
> against; OpenAPI is generated from the FastAPI models once the code exists.

Base URL: `http://127.0.0.1:8000`  ·  All endpoints are under `/v1`.

## Design rules

These are the conventions the whole surface follows, stated once:

1. **Resources and verbs, not RPC.** `POST /v1/ask` creates an answer, `GET /v1/documents/{id}`
   reads a document. No `/doQuery` style endpoints.
2. **Versioned from day one.** Everything sits under `/v1`. The path prefix is free now and
   expensive to retrofit later.
3. **Retrieval is separable from generation.** `POST /v1/search` returns exactly the chunks
   `/v1/ask` would have used, with the same scores, and needs no API key. This is the single
   most useful debugging affordance in the system: when an answer is wrong, it immediately
   separates "retrieval missed it" from "the model mishandled good context".
4. **Every answer is auditable.** No response containing an answer omits its citations, and
   every citation carries a character range into a real document.
5. **Diagnostics are part of the contract, not a debug flag.** Timings, scores, which gates
   fired and which provider ran are returned on every `/v1/ask`. They power the UI's
   "why this answer" panel and land verbatim in evaluation traces.
6. **Errors are RFC 9457 `application/problem+json`**, with a `trace_id` that matches the logs.
7. **Long work is a job, not a long request.** Evaluation runs return `202` and a run resource
   to poll, because a full eval is minutes of LLM calls.

## Endpoints at a glance

| Method | Path | Purpose | Needs LLM key |
|---|---|---|---|
| `GET` | `/v1/health` | liveness, index manifest, provider readiness | no |
| `POST` | `/v1/ask` | grounded answer with citations | yes |
| `POST` | `/v1/ask/stream` | same, streamed over SSE | yes |
| `POST` | `/v1/search` | retrieval only, no generation | no |
| `GET` | `/v1/documents` | list knowledge-base documents | no |
| `GET` | `/v1/documents/{doc_id}` | full document text | no |
| `GET` | `/v1/documents/{doc_id}/chunks` | chunks with offsets, for citation UIs | no |
| `POST` | `/v1/index/rebuild` | re-ingest the knowledge base | no |
| `POST` | `/v1/eval/runs` | start an evaluation run (202) | yes |
| `GET` | `/v1/eval/runs` | list past runs | no |
| `GET` | `/v1/eval/runs/{run_id}` | run status, metrics, report | no |

---

## `POST /v1/ask`

The primary endpoint.

**Request**

```json
{
  "question": "What are the responsibilities of a domain lead?",
  "top_k": 6,
  "strategy": "hybrid",
  "include_diagnostics": true,
  "verify_faithfulness": false
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `question` | string, 1–1000 chars | required | |
| `top_k` | int, 1–20 | config | chunks passed to the model |
| `strategy` | `hybrid` \| `dense` \| `lexical` | `hybrid` | exposed so the eval harness and a live demo can compare strategies without a restart |
| `include_diagnostics` | bool | `true` | |
| `verify_faithfulness` | bool | `false` | enables abstention gate 3 |

**Response `200`**

```json
{
  "question": "What are the responsibilities of a domain lead?",
  "answer": "Domain leads plan the learning roadmap for their domain, assign tasks to coordinators, conduct knowledge-sharing sessions, review technical projects, mentor coordinators, encourage hackathon participation, and coordinate cross-domain projects when required.",
  "answered": true,
  "abstained": false,
  "abstention_reason": null,
  "confidence": "high",
  "citations": [
    {
      "chunk_id": "leadership::c03",
      "doc_id": "leadership",
      "doc_title": "MLSC Leadership Structure",
      "source_file": "leadership.txt",
      "snippet": "Domain leads are responsible for: Planning the learning roadmap for their domain...",
      "char_range": [312, 620],
      "score": 0.87
    }
  ],
  "sources": ["leadership.txt"],
  "diagnostics": {
    "trace_id": "01J8XA...",
    "retrieval": {
      "strategy": "hybrid",
      "top_k": 6,
      "candidates_considered": 40,
      "top_score": 0.87,
      "score_margin": 0.31,
      "documents_represented": ["leadership", "domains"],
      "dense_ms": 4,
      "lexical_ms": 1,
      "fusion_ms": 1
    },
    "generation": {
      "provider": "gemini",
      "model": "gemini-2.5-flash",
      "prompt_version": "grounded-v1",
      "input_tokens": 912,
      "output_tokens": 96,
      "latency_ms": 740
    },
    "gates": {
      "retrieval_gate": "pass",
      "context_sufficiency": "pass",
      "citation_binding": "pass",
      "faithfulness_check": "skipped"
    },
    "total_ms": 751
  }
}
```

**Abstention is a `200`, not an error.** Refusing correctly is a successful outcome of the
system, and a 4xx would make it indistinguishable from a malformed request to a client:

```json
{
  "question": "Who is the current Technical Head of MLSC?",
  "answer": "The knowledge base describes the Technical Head role and its responsibilities, but does not name the person currently holding it.",
  "answered": false,
  "abstained": true,
  "abstention_reason": "insufficient_context",
  "confidence": "high",
  "citations": [],
  "sources": [],
  "diagnostics": { "gates": { "retrieval_gate": "pass", "context_sufficiency": "fail" }, "...": "..." }
}
```

`abstention_reason` is an enum, so clients can branch on it:
`no_relevant_context` (gate 1) · `insufficient_context` (gate 2) · `unfaithful_answer` (gate 3)
· `provider_unavailable`.

Note the shape of that refusal: it says what the KB *does* contain before saying what it does
not. "I cannot answer" is technically correct and useless; naming the gap is what makes the
refusal helpful, and the prompt is written to produce it.

---

## `POST /v1/ask/stream`

Identical request. Responds `text/event-stream` with named SSE events so a client can render
progressively and show its work:

```
event: retrieval    data: {"chunks":[...],"strategy":"hybrid"}
event: token        data: {"text":"Domain leads plan "}
event: token        data: {"text":"the learning roadmap "}
event: citations    data: {"citations":[...]}
event: done         data: {"answered":true,"diagnostics":{...}}
event: error        data: {"type":"...","title":"...","trace_id":"..."}
```

`retrieval` arriving before the first token means the UI can show the sources it is about to
use while the answer is still generating.

---

## `POST /v1/search`

Retrieval without generation. **No API key required** — this is the endpoint that makes the
system inspectable and the retrieval metrics reproducible offline.

```json
{ "query": "hackathon judging", "top_k": 5, "strategy": "hybrid", "explain": true }
```

```json
{
  "query": "hackathon judging",
  "strategy": "hybrid",
  "results": [
    {
      "chunk_id": "hackathons::c06",
      "doc_id": "hackathons",
      "doc_title": "MLSC Hackathons",
      "text": "Participants are generally evaluated based on factors such as innovation, technical implementation, problem relevance, feasibility, scalability and quality of presentation.",
      "char_range": [640, 810],
      "score": 0.91,
      "rank": 1,
      "explain": {
        "dense_rank": 1, "dense_score": 0.79,
        "lexical_rank": 2, "lexical_score": 6.41,
        "rrf_score": 0.0325,
        "matched_terms": ["hackathon", "judging", "evaluated"]
      }
    }
  ],
  "timings_ms": { "dense": 4, "lexical": 1, "fusion": 1, "total": 6 }
}
```

`explain` exposes each retriever's independent verdict. When hybrid beats dense-only on the
eval set, this is the object that shows *why* on any specific query.

---

## `GET /v1/health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "index": {
    "built_at": "2026-08-24T09:12:03Z",
    "documents": 6,
    "chunks": 41,
    "embedder": "BAAI/bge-small-en-v1.5",
    "dimension": 384,
    "chunker_version": "structural-v1",
    "stale": false
  },
  "generation": { "provider": "gemini", "model": "gemini-2.5-flash", "configured": true }
}
```

`index.stale` compares the manifest's per-document checksums against the files on disk, so a
knowledge base edited without re-indexing is visible rather than silently serving old content.
`generation.configured` reports whether a key is present without ever echoing it.

---

## `GET /v1/documents`, `/v1/documents/{doc_id}`, `/v1/documents/{doc_id}/chunks`

The knowledge base is a first-class, browsable resource. This exists so a citation in the UI is
clickable through to its source, and so a reviewer can confirm the system is reading the
documents it was given and nothing else.

`/chunks` returns each chunk with its `char_range`, letting a client highlight the exact
passage an answer rests on.

---

## `POST /v1/eval/runs`

```json
{ "dataset": "dev_set", "strategy": "hybrid", "metrics": ["retrieval", "generation", "abstention"], "top_k": 6 }
```

Returns `202 Accepted` with `Location: /v1/eval/runs/{run_id}` and `{"run_id": "...", "status": "queued"}`.

`GET /v1/eval/runs/{run_id}` returns progress while running, then the full result:

```json
{
  "run_id": "20260824-091203-hybrid",
  "status": "completed",
  "dataset": { "name": "dev_set", "questions": 30 },
  "config": { "strategy": "hybrid", "top_k": 6, "embedder": "BAAI/bge-small-en-v1.5",
              "provider": "gemini", "model": "gemini-2.5-flash", "prompt_version": "grounded-v1" },
  "metrics": {
    "retrieval": { "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0,
                   "doc_hit_rate": 0.0, "multi_doc_coverage": 0.0 },
    "generation": { "faithfulness": 0.0, "answer_relevancy": 0.0, "answer_correctness": 0.0 },
    "abstention": { "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "hallucination_rate": 0.0, "over_refusal_rate": 0.0 }
  },
  "report_path": "evaluation/runs/20260824-091203-hybrid/report.md"
}
```

Metric values above are **zeroed placeholders showing the response shape**, not results. Real
numbers land in `docs/EVALUATION.md` and `evaluation/reports/` once the harness runs.

The `config` block is repeated inside every run on purpose: a metric is meaningless without the
exact strategy, model and prompt version that produced it, and runs get compared weeks apart.

---

## Errors

`application/problem+json`, per RFC 9457:

```json
{
  "type": "https://mlsc-assistant/errors/index-not-built",
  "title": "Index not built",
  "status": 409,
  "detail": "No index found at data/index. Run `mlsc index` first.",
  "trace_id": "01J8XA..."
}
```

| Status | `type` | When |
|---|---|---|
| `400` | `invalid-request` | validation failure |
| `409` | `index-not-built` | no index on disk |
| `422` | `question-too-long` | over the configured limit |
| `429` | `provider-rate-limited` | upstream throttled us; `Retry-After` set |
| `503` | `provider-unavailable` | no key configured, or upstream is down |
| `504` | `provider-timeout` | generation exceeded the deadline |

`detail` always states the *action* that fixes it. Provider errors are surfaced as distinct
statuses rather than a generic `500`, because "you have not set a key" and "Gemini is down" and
"you are being throttled" need three different reactions from a caller.
