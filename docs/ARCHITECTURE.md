# Architecture

> Status: **design, pre-implementation**. This document is the plan the code is written against.

## 1. What the system has to do

From the challenge brief, restated as testable obligations:

| # | Obligation | Where it is satisfied |
|---|---|---|
| R1 | Answer natural-language questions | `POST /v1/ask`, CLI `mlsc ask`, web UI |
| R2 | Answer *from the knowledge base only* | grounded prompt + citation binding + faithfulness gate |
| R3 | Return the source document(s) for every answer | `citations[]` carry `doc_id` + `char_range`, verified against retrieved context |
| R4 | Handle multi-document questions | hybrid retrieval + per-document diversification + multi-hop follow-up pass |
| R5 | Refuse when the KB does not contain the answer | three independent abstention gates (section 6) |
| R6 | Ship an evaluation component | `mlsc eval`, `POST /v1/eval/runs`, metrics in section 7 |

## 2. The single most important design constraint

**The corpus is tiny: 6 documents, ~6 KB, roughly 40 paragraphs.**

This is not an afterthought. It determines almost every decision below, and being explicit
about it is more honest than pretending we are building for a million documents:

- Similarity search over a `40 x 384` matrix is a sub-millisecond NumPy dot product. A vector
  database buys us nothing at this scale, so the **default store is NumPy**, and a Chroma
  adapter exists to prove the port is real rather than because we need it.
- With so few chunks, **one bad retrieval is ~2.5% of the corpus** and will visibly move any
  metric. Retrieval quality matters enormously; retrieval speed does not matter at all.
- Rare exact terms (`Web3`, `Technical Head`, `second-year coordinators`) are precisely where
  small dense embedders blur meanings together. **Lexical matching is not legacy baggage here,
  it is load-bearing.** Hence hybrid retrieval rather than dense-only.
- Paragraphs are short and pronoun-headed ("Each domain has domain leads..."). Embedded in
  isolation they lose their subject, so chunks are **prefixed with their document title at
  embed time** (section 4).

## 3. Shape: ports and adapters

The core pipeline is written against interfaces (`Protocol` classes in `core/ports.py`) and
never imports a provider SDK. Adapters are selected by configuration at startup.

```
                    +--------------------------------------------+
   HTTP / CLI / UI  |            application services            |
   -------------->  |   AskService | SearchService | EvalService  |
                    +----------------+---------------------------+
                                     | depends only on ports
        +----------------------------+----------------------------+
        v                            v                            v
  +-----------+              +---------------+            +--------------+
  | Embedder  |              |  VectorStore  |            | LLMProvider  |
  +-----------+              +---------------+            +--------------+
  | fastembed |  <- default  | numpy  <- def |            | gemini <- def|
  | sbert     |              | chroma        |            | anthropic    |
  | (api)     |              |               |            | openai       |
  +-----------+              +---------------+            | groq, ollama |
                                                          +--------------+
```

Why this matters beyond tidiness: it is what lets the **retrieval half of the system, and
every retrieval metric, run with no API key at all**. A reviewer can clone the repo, build the
index and reproduce the retrieval numbers offline. Only answer generation needs a secret.

### Package layout

```
src/mlsc_assistant/
├── core/
│   ├── models.py          Document, Chunk, ScoredChunk, Citation, Answer, AskResult
│   ├── ports.py           Embedder, VectorStore, LLMProvider, Chunker, Reranker (Protocols)
│   └── errors.py          typed domain errors, mapped to RFC 9457 responses at the edge
├── config.py              pydantic-settings over config.yaml + .env, nested MLSC_ overrides
├── ingestion/
│   ├── loader.py          data/knowledge_base/*.txt -> Document (title, body, checksum)
│   ├── chunker.py         structure-aware paragraph chunking (section 4)
│   └── pipeline.py        load -> chunk -> embed -> persist, and write the index manifest
├── embeddings/
│   ├── fastembed_embedder.py   BAAI/bge-small-en-v1.5 via ONNX runtime
│   ├── sbert_embedder.py       optional extra, same interface
│   └── cache.py                content-hash keyed, so re-indexing unchanged docs is free
├── stores/
│   ├── numpy_store.py     cosine over an in-memory matrix, persisted as .npz + manifest
│   └── chroma_store.py    optional extra
├── retrieval/
│   ├── dense.py           embedding search
│   ├── lexical.py         BM25 over lightly normalised tokens
│   ├── fusion.py          Reciprocal Rank Fusion
│   ├── diversify.py       per-document cap / MMR, the multi-document lever (section 5)
│   ├── rerank.py          Reranker port, no-op default, LLM rerank behind a flag
│   └── retriever.py       HybridRetriever, orchestrates the above and emits diagnostics
├── generation/
│   ├── prompts.py         versioned templates, prompt_version travels into every eval run
│   ├── answerer.py        context assembly -> structured answer -> citation binding
│   ├── verifier.py        post-hoc claim/faithfulness check (gate 3)
│   └── providers/         gemini.py, anthropic.py, openai.py, groq.py, ollama.py, registry.py
├── evaluation/
│   ├── dataset.py         eval-set schema and loaders (ours, plus whatever MLSC supplies)
│   ├── metrics/
│   │   ├── retrieval.py   precision@k, recall@k, MRR, nDCG@k, doc hit rate, multi-doc coverage
│   │   ├── generation.py  faithfulness, answer relevancy, answer correctness
│   │   └── abstention.py  abstention P/R/F1, hallucination rate on unanswerables
│   ├── judge.py           LLM-as-judge wrapper: structured verdicts, cached, seeded
│   ├── runner.py          orchestration, caching, per-question traces
│   └── report.py          JSON + Markdown report writer
├── api/
│   ├── app.py             application factory, lifespan (the index loads once, not per request)
│   ├── deps.py            composition root, the only place concrete adapters are chosen
│   ├── schemas.py         request/response DTOs, deliberately separate from domain models
│   └── routes/            ask.py, search.py, documents.py, eval.py, health.py
├── web/                   one static page: question box, answer, citations, diagnostics panel
└── cli.py                 typer: index, ask, search, serve, eval, calibrate
```

`core/models.py` is deliberately separate from `api/schemas.py`. Domain objects carry things
the wire should not (raw vectors, internal scores), and the public API contract should be free
to stay stable while internals move.

## 4. Ingestion and chunking

The documents are prose: a title on line 1, then one paragraph per non-empty line. The chunker
is therefore **structure-aware rather than character-count-blind**:

1. **Split on blank lines** into paragraphs, the author's own semantic units. This beats a
   fixed 512-token window, which would slice mid-sentence for no benefit at this size.
2. **Keep list blocks whole.** `domains.txt` contains a numbered list of the five domains and
   `leadership.txt` a bulleted list of lead responsibilities. Splitting either would make
   "what domains exist in MLSC" unanswerable from any single chunk. The chunker treats a run of
   list items, together with the sentence introducing it, as one atomic chunk.
3. **Merge short orphans** up to a floor of roughly 40 tokens, so one-line paragraphs do not
   become noisy, low-information vectors.
4. **Contextual header at embed time.** Each chunk is embedded as `"{doc_title} - {chunk_text}"`
   while the *stored* text stays clean for display and for citation offsets. This is the
   cheapest available fix for pronoun-headed paragraphs.
5. **Deterministic IDs**: `{doc_id}::c{index:02d}`, plus a SHA-256 of the chunk text. Stable IDs
   mean an evaluation set can reference specific chunks and stay valid across re-indexing.

Every index build writes `data/index/manifest.json`: embedder name and revision, vector
dimension, chunker version, per-document checksums, chunk count, build timestamp.
`GET /v1/health` returns it and every evaluation run records it.
**A metric without a manifest is not a result.**

## 5. Retrieval

```
query
  |--> dense: bge-small embedding -> cosine over the chunk matrix   -> ranked list A
  |--> lexical: BM25 over normalised tokens                         -> ranked list B
                              |
                    Reciprocal Rank Fusion (k = 60)
                              |
               per-document diversification (cap n per doc)
                              |
                    optional rerank (off by default)
                              |
                    top-k chunks + retrieval diagnostics
```

**Why RRF instead of a weighted score blend?** Cosine similarity and BM25 scores live on
incompatible, corpus-dependent scales, so any fixed `alpha * dense + (1 - alpha) * bm25` needs
re-tuning whenever the corpus changes. RRF consumes only *ranks*, so it is scale-free and has a
single interpretable constant. On a 40-chunk corpus there is nowhere near enough data to fit a
blending weight honestly.

**Multi-document questions (R4)** get two mechanisms:

- *Diversification.* After fusion, cap how many chunks any single document may contribute to
  the final context. Without it, a question like "how do domain leads relate to hackathons?"
  can fill every slot from `leadership.txt` and silently lose the hackathon half. The cap is a
  config value, and its effect is measured directly by the multi-document coverage metric.
- *Follow-up pass* (flag-gated). If the answerer reports insufficient context while retrieval
  scored well, decompose the question into sub-queries, retrieve for each and merge. Off by
  default because it doubles latency; switched on only if the evaluation shows it earning that cost.

Dense-only and lexical-only remain runnable as retrieval strategies, so the evaluation can
**demonstrate** that hybrid beats each component rather than asserting it.

## 6. Grounding and abstention

Refusing correctly is a first-class feature here, not a sentence in a prompt. Three independent
gates, each able to abstain on its own:

| Gate | When | Cost | What it catches |
|---|---|---|---|
| 1. Retrieval gate | before the LLM is called | free | questions with nothing lexically or semantically related in the corpus ("who won the IPL final?") |
| 2. Context sufficiency | during generation, as structured output | one call | content that *looks* related but does not contain the fact ("who is the current Technical Head?" — the KB describes the role and never names a person) |
| 3. Faithfulness verifier | after generation, flag-gated | one extra call | an answer that drifted beyond its cited context |

Gate 2 is the interesting one, and it is why generation uses **structured output** rather than
free text. The model must return:

```json
{
  "sufficient_context": true,
  "answer": "...",
  "cited_chunk_ids": ["domains::c02", "leadership::c01"],
  "confidence": "high"
}
```

Answering and self-assessing in one schema-constrained call makes the refusal decision a
*parsed field* rather than a string match on "I don't know". And `cited_chunk_ids` is then
**validated against the chunks actually retrieved**, so a fabricated citation is caught
mechanically instead of being trusted.

**The threshold for gate 1 is calibrated, never hand-picked.** `mlsc calibrate` sweeps it
across the dev set and reports the resulting abstention precision/recall curve, so we choose an
operating point deliberately and write down why. Guessing `0.5` and hoping is exactly what the
interview will probe.

## 7. Evaluation

The brief requires context precision, context recall, answer relevancy and faithfulness. We
report those, plus the family that actually matters for this problem.

**Retrieval — no LLM, fully deterministic, reproducible offline:**
precision@k, recall@k, MRR, nDCG@k, document-level hit rate, and multi-document coverage (the
fraction of multi-doc questions where *every* required document appears in the final context).

**Generation — LLM as judge, cached and seeded:**
faithfulness (is each claim entailed by the cited context), answer relevancy (embed the answer,
generate hypothetical questions from it, cosine against the real question — computed locally
with fastembed), and answer correctness against the reference answer.

**Abstention — the family the brief cares about and RAGAS does not really cover:**
abstention precision, abstention recall, **hallucination rate** (answered confidently where the
KB gives no support) and over-refusal rate (refused something that was answerable). A system
that refuses everything scores perfectly on faithfulness; only this family exposes that.

**Own the metrics, cross-check with RAGAS.** They are implemented in-repo, roughly 200 lines of
readable code, because the brief requires us to *justify* what we report and a metric you can
read is a metric you can defend. RAGAS runs as an optional extra to confirm our faithfulness
and relevancy track a published implementation; any divergence gets investigated and written
up rather than quietly dropped.

Every run writes `evaluation/runs/{run_id}/` with per-question traces (retrieved chunks and
scores, prompt version, judge verdicts) and a rolled-up `report.md`, so any headline number is
traceable back to a single question.

**On the evaluation set:** MLSC supplies theirs separately. We author our own dev set now
(`evaluation/datasets/dev_set.yaml`) covering all five question types from the brief, used only
for measurement and calibration, and `dataset.py` adapts their format when it arrives.

## 8. Explicitly not hard-coding answers

The brief forbids it, so it is enforced rather than promised:

- No question-to-answer mapping exists anywhere in `src/`. The only path from a question to an
  answer runs through retrieval and the LLM.
- The dev evaluation set lives under `evaluation/`, which `src/` never imports.
- A unit test asserts that no reference answer from any evaluation set appears verbatim in
  `src/`, and CI runs it on every push.

## 9. Deferred by design

Recorded so they read as decisions rather than oversights: query caching, multi-tenancy, auth
on the API, conversational multi-turn memory, incremental re-indexing, and a cross-encoder
reranker (a 40-chunk corpus cannot justify the latency).
