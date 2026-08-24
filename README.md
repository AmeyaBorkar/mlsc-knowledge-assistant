# MLSC Knowledge Assistant

A grounded, citation-first RAG assistant over the MLSC knowledge base. Every answer is traceable
to the passage it came from, and the system is built to **say "I don't know" when the knowledge
base does not contain the answer** rather than inventing something plausible.

Built for the MLSC AI/ML Domain Lead recruitment challenge.

> **Status: Phases 0-4 complete.** The full answering pipeline works: `mlsc ask` returns
> grounded, cited answers and refuses when the knowledge base does not have the fact.
> `mlsc index`, `mlsc search` and `mlsc eval` still run with no API key. The HTTP API, web UI
> and LLM-judged metrics are next. Real numbers in
> [`docs/EVALUATION.md`](docs/EVALUATION.md#results); what went wrong along the way is in
> [`docs/PLAN.md`](docs/PLAN.md).

---

## What it does

Ask a question in natural language; get an answer synthesised only from the six MLSC documents,
with the source document and exact passage for every claim.

```
$ mlsc ask "What are the responsibilities of a domain lead?"

Domain leads plan the learning roadmap for their domain, assign tasks to coordinators,
conduct knowledge-sharing sessions, review technical projects, mentor coordinators,
encourage hackathon participation, and coordinate cross-domain projects when required.

  Sources: leadership.txt
  [leadership::c03] "Domain leads are responsible for: Planning the learning roadmap..."
```

And when the knowledge base does not have it:

```
$ mlsc ask "Who is the current Technical Head of MLSC?"

The knowledge base describes the Technical Head role and its responsibilities, but does
not name the person currently holding it.

  Answered: no  ·  Reason: insufficient_context
```

That second case is the interesting one. Retrieval scores *high* on that question — the
leadership document discusses the Technical Head at length — so a similarity threshold alone
cannot catch it. See [abstention](#abstention-knowing-when-to-refuse).

**Measured across the 40-question dev set: hallucination rate 0.000, abstention F1 0.960,
all 12 unanswerable questions refused.** A threshold alone leaves a hallucination rate of
0.83; the second gate is what closes it.

---

## Design in one minute

```
question
   |
   +--> dense retrieval (bge-small, ONNX)  --+
   |                                          +--> RRF fusion --> per-doc diversification
   +--> lexical retrieval (BM25)  -----------+                              |
                                                                            v
                                                            [gate 1] retrieval threshold
                                                                            |
                                                     grounded prompt + structured output
                                                                            |
                                                          [gate 2] sufficient_context?
                                                                            |
                                                        citation binding + validation
                                                                            |
                                                   [gate 3] optional faithfulness verify
                                                                            v
                                                       answer + citations + diagnostics
```

Four things worth knowing up front:

**The corpus is tiny — 6 documents, 18 chunks — and that drove every decision.** No vector
database (an `18 x 384` NumPy dot product is faster than any network hop), but hybrid retrieval,
because rare exact terms like `Web3` and `Technical Head` are exactly where small dense embedders
blur and there is no redundancy in the corpus to recover from a miss.

**Retrieval runs with no API key.** Indexing, `/v1/search`, the CLI search command and every
retrieval metric work fully offline. Only answer generation needs a key. A reviewer can clone
this repo and reproduce half the evaluation with no secrets at all.

**Answering uses structured output, so refusing is a parsed field, not a string match.** The
model returns `{sufficient_context, answer, cited_chunk_ids, confidence}`, and the cited chunk
ids are validated against what was actually retrieved — a fabricated citation is caught
mechanically.

**Ports and adapters throughout.** Embedder, VectorStore and LLMProvider are interfaces;
swapping Gemini for Claude, or NumPy for Chroma, is a config change.

Full reasoning, including what each choice costs, is in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Abstention: knowing when to refuse

The brief asks the system to recognise when information is unavailable. This gets three
independent gates rather than a line in a prompt, because "not in the knowledge base" has
genuinely different causes:

| Gate | Runs | Cost | Catches |
|---|---|---|---|
| Retrieval threshold | before the LLM | free | nothing relevant exists at all — *"who won the IPL final?"* |
| Context sufficiency | during generation | one call | topic present, fact absent — *"who is the current Technical Head?"* |
| Faithfulness verify | after generation, optional | one call | an answer that drifted past its cited context |

The threshold for gate 1 is **calibrated by sweeping it across the evaluation set** and reading
the abstention precision/recall curve, not picked by hand.

Gate 1 is calibrated to **0.55** — the highest threshold that refuses no answerable question.
The measured curve is the strongest evidence in the project:

| Threshold | Hallucination | Over-refusal | Near-miss caught |
|---|---|---|---|
| **0.55** (shipped) | 0.83 | **0.00** | 0.00 |
| 0.75 | 0.25 | 0.39 | 0.67 |
| 0.80 | 0.00 | 0.71 | 1.00 |

Near-miss unanswerables score 0.71–0.78; answerable questions score 0.67–0.90. The
distributions overlap almost completely, so **catching "who is the current Technical Head?"
with a threshold means refusing 39% of real questions.** Gate 2 is a necessity, not a
preference — and Phase 7 has to show it actually closes the gap.

---

## Evaluation

Metrics are grouped by which part of the pipeline they blame, because a single headline number
cannot tell you whether to fix the chunker, the prompt or the threshold.

- **Retrieval** *(no LLM, deterministic, runs in CI)* — context precision@k, context recall@k,
  R-precision, average precision, MRR, nDCG@k, document recall, multi-document coverage.
  **Measured: recall@6 0.955, MRR 0.912, nDCG@6 0.891, multi-doc coverage 1.000** on a
  40-question dev set
- **Generation** *(LLM as judge, cached and seeded)* — faithfulness (claim-level), answer
  relevancy, answer correctness
- **Abstention** — precision, recall, F1, **hallucination rate**, over-refusal rate

That last group matters most here and is the one RAGAS does not really cover: **a system that
refuses every question scores a perfect 1.0 on faithfulness.** Only abstention metrics expose
that, and hallucination rate and over-refusal rate are always reported as a pair, since a
threshold can be moved to make either look excellent alone.

Metrics are implemented in-repo (~200 readable lines) with RAGAS available as an optional
cross-check. Reasoning for that, and full definitions, in
[`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Getting started

`mlsc index`, `mlsc search`, `mlsc eval` and `mlsc info` work without an API key. `mlsc ask`
needs one. `serve` arrives in Phase 5.

```bash
git clone https://github.com/AmeyaBorkar/mlsc-knowledge-assistant
cd mlsc-knowledge-assistant

python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

mlsc index                       # build the index — no API key needed
mlsc search "judging criteria"   # inspect retrieval — no API key needed

cp .env.example .env             # add a key for answer generation
mlsc ask "What technical domains exist in MLSC?"
mlsc serve                       # http://127.0.0.1:8000
```

A free Gemini key from [aistudio.google.com](https://aistudio.google.com/apikey) is the default.
Anthropic, OpenAI, Groq and Ollama are supported through the same interface — set
`MLSC_LLM__PROVIDER` and the matching key.

### CLI

| Command | Purpose | Needs a key |
|---|---|---|
| `mlsc index` | build the index from `data/knowledge_base/` | no |
| `mlsc search <query>` | retrieval only, with per-retriever scores | no |
| `mlsc ask <question>` | grounded answer with citations | yes |
| `mlsc serve` | run the API and web UI | yes |
| `mlsc eval` | run the evaluation harness | partly |
| `mlsc calibrate` | sweep the abstention threshold | yes |

### API

`POST /v1/ask` · `POST /v1/ask/stream` · `POST /v1/search` · `GET /v1/documents` ·
`GET /v1/health` · `POST /v1/eval/runs`

Full contract with request/response shapes in [`docs/API.md`](docs/API.md); interactive docs at
`/docs` once running.

---

## Repository layout

```
src/mlsc_assistant/
├── core/          domain models + ports (no I/O, no SDKs)
├── ingestion/     loading, structure-aware chunking, index building
├── embeddings/    Embedder adapters + cache
├── stores/        VectorStore adapters (numpy, chroma)
├── retrieval/     dense, lexical, fusion, diversification, reranking
├── generation/    prompts, answerer, verifier, provider adapters
├── evaluation/    datasets, metrics, judge, runner, reporting
├── api/           FastAPI app, routes, schemas, composition root
└── web/           single-page demo UI

data/knowledge_base/   the six source documents
evaluation/datasets/   evaluation sets (never imported by src/)
docs/                  architecture, API, evaluation, decisions, plan
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | system design, chunking, retrieval, abstention |
| [`docs/API.md`](docs/API.md) | HTTP contract, request/response shapes, error model |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | metric definitions, justification, ablations, results |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | why each choice was made, and what it costs |
| [`docs/PLAN.md`](docs/PLAN.md) | build order and risk assessment |

## A note on scope

The knowledge base is the sole source of truth for MLSC-specific information. No answers are
hard-coded anywhere: the only path from a question to an answer runs through retrieval and the
model, and a CI test asserts that no evaluation reference answer appears in the source tree.

## License

MIT
