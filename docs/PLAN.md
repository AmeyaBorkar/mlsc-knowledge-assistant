# Implementation plan

Build order, chosen so that **every phase ends with something demonstrable** and the risky parts
happen early. The brief recommends about 6 hours of working time; the estimates below are a
budget, not a promise.

## Sequencing principle

Retrieval before generation, and evaluation before tuning.

Retrieval is the ceiling on everything downstream — no prompt recovers a fact that was never
fetched — and it is the half that needs no API key, so it can be built and measured immediately.
Evaluation comes before any tuning because tuning without measurement is guessing, and the guess
would then have to be defended in the interview.

---

### Phase 0 — Foundations *(~30 min)* — **done**

Config, domain models, ports, error types, CLI skeleton, CI.

- `core/models.py`, `core/ports.py`, `core/errors.py`
- `config.py` — pydantic-settings over `config.yaml` + `.env`
- `cli.py` skeleton with typer
- GitHub Actions: ruff, mypy, pytest

**Done when:** `mlsc --help` runs and CI is green on an empty test suite.

---

### Phase 1 — Ingestion and indexing *(~60 min)* — **done**

- `ingestion/loader.py` — read `data/knowledge_base/*.txt`, extract the title from line 1,
  checksum each document
- `ingestion/chunker.py` — structure-aware chunking (D9): paragraph splits, atomic list blocks,
  orphan merging, deterministic IDs, character offsets preserved for citations
- `embeddings/fastembed_embedder.py` + content-hash cache
- `stores/numpy_store.py` — persist as `.npz` plus `manifest.json`
- `ingestion/pipeline.py` — `mlsc index`

**Done when:** `mlsc index` produces `data/index/` and prints the chunk count, and unit tests
pin the chunk boundaries for `domains.txt` and `leadership.txt` — the two documents whose lists
must not be split.

**Risk retired:** fastembed 0.8.0 installs and runs cleanly on Python 3.13 / Windows. No
fallback to sentence-transformers needed.

**What this phase actually produced:** 6 documents to **18 chunks**, both list blocks intact.
Three findings worth carrying forward:

1. **bge-small has a high similarity floor.** Two entirely unrelated passages from this corpus
   (the domain list vs the hackathon judging paragraph) embed to cosine **0.65**. The intuition
   that "unrelated scores near zero" is false for this model, so the abstention threshold cannot
   be hand-picked — the placeholder 0.35 would never fire. Pinned by a test that fails if the
   floor moves.
2. **Merge policy had to be rewritten once.** A bidirectional rule that merged whenever *either*
   neighbour was short cascaded, because every paragraph in this corpus is individually under the
   floor — `about_mlsc.txt` collapsed into a single chunk. Replaced with a greedy accumulator that
   stops growing at the floor. Regression test added.
3. **Two latent bugs, both caught by tests rather than by reading.** `FileEmbeddingCache` defines
   `__len__`, so an empty cache was falsy and `cache or NullEmbeddingCache()` silently disabled
   caching on exactly the cold-start run that needed it. And a pinned `llm.model` in `config.yaml`
   survived a provider override through deep-merging, so `MLSC_LLM__PROVIDER=anthropic` would have
   called Anthropic with a Gemini model name.

---

### Phase 2 — Retrieval *(~60 min)* — **done**

- `retrieval/dense.py`, `retrieval/lexical.py` (BM25), `retrieval/fusion.py` (RRF),
  `retrieval/diversify.py` (per-document cap)
- `retrieval/retriever.py` — `HybridRetriever` with `strategy` switch and diagnostics
- `mlsc search "..."` — rich table output with per-retriever explain

**Done when:** `mlsc search "judging criteria"` returns the hackathon evaluation paragraph at
rank 1, and `--strategy dense|lexical|hybrid` visibly differ. Both hold.

**Still no API key needed at this point.**

#### What this phase measured

Six hand-written questions across the strategies. This is a probe, not an evaluation —
the numbers below come from questions I chose, which is exactly the bias Phase 3 exists to
remove. Reported anyway, because two of the findings are inconvenient.

**1. Hybrid does not uniformly beat dense here, and D1 is not yet proven.** Gold-chunk rank
across five answerable questions:

| Question | dense | lexical | hybrid |
|---|---|---|---|
| What technical domains exist in MLSC? | **1** | miss | 3 |
| Web3 | **1** | 2 | **1** |
| How are hackathon projects judged? | **1** | **1** | **1** |
| How many leads does each domain have? | 5 | **1** | 3 |
| What is expected of someone leading a domain? | **2** | 6 | 3 |

MRR: dense **0.74**, hybrid **0.60**. Dense wins this sample. Hybrid is *steadier* — it has
no rank-5 near-failure — but "more robust on five questions I wrote" is not evidence. If the
Phase 3 evaluation confirms this, the honest outcome is to change the default strategy and
report the negative result, not to keep hybrid because the design document says so.

**2. Fusion does rescue a real dense failure.** "How many leads does each domain have?" —
the answer sits in `leadership::c00` ("Each technical domain has two domain leads"), which
dense ranks **5th** because the phrasing is generic, and BM25 ranks **1st** on exact terms.
This is the case D1 was written for, and it is now an integration test.

**3. Fusion also degrades the brief's own example question.** For "What technical domains
exist in MLSC?", BM25 ranks the actual domain *list* 11th — the chunk is long and mostly
domain names, so query-term density is low — while promoting a short chunk that merely
mentions domains. RRF splits the difference and the list lands 3rd instead of 1st. Still
inside the context window, so the answer survives; the cost is precision and position.

**4. RRF's k=60 barely discriminates at this corpus size.** Across 18 candidates the whole
score range spans under 1.3x, so rank 1 and rank 11 are nearly indistinguishable after
fusion — which is *why* finding 3 happens. k=60 comes from TREC work over large candidate
pools. `rrf_k` and `candidate_k` are now on the ablation list rather than tuned by hand here.

**5. Abstention calibration data, which is the most useful output of this phase.** Best
cosine, query vs chunk:

| Question type | best cosine |
|---|---|
| off-domain ("who won the IPL final?") | **0.43** |
| answerable | 0.67 – 0.90 |
| near-miss unanswerable ("current Technical Head?") | **0.75** |
| near-miss unanswerable ("membership fee?") | **0.74** |

A threshold near 0.55 cleanly separates off-domain questions and nothing else. **The
near-miss unanswerables sit inside the answerable range**, so no threshold can catch them at
any operating point. That is the three-gate design's central claim, and it is now measured
rather than argued.

**6. BM25 title indexing: tested, kept.** Removing the document-title prefix from the BM25
index was a plausible fix for candidate flooding. Measured: it drops the domain-list chunk
out of the results entirely and helps nothing. Hypothesis rejected; `index_title` survives
as an ablation knob.

---

### Phase 3 — Evaluation harness, retrieval half *(~45 min)*

Deliberately before generation. Retrieval quality is measurable without an LLM, and doing it now
means phases 4–6 are tuned against numbers instead of vibes.

- `evaluation/dataset.py` + author `evaluation/datasets/dev_set.yaml` (~30 questions, at least a
  third unanswerable, gold chunk labels)
- `evaluation/metrics/retrieval.py` — precision@k, recall@k, MRR, nDCG@k, doc hit rate,
  multi-doc coverage
- `evaluation/runner.py` + `report.py`
- `mlsc eval --metrics retrieval`

**Done when:** a retrieval-only report renders, and the dense / lexical / hybrid ablation
produces real numbers. **This is the first point where D1 is proven rather than asserted** — if
hybrid does not win here, the design changes.

---

### Phase 4 — Generation *(~60 min)*

- `generation/providers/` — the `LLMProvider` port, a registry, and the Gemini adapter first
- `generation/prompts.py` — versioned grounded-answering template
- `generation/answerer.py` — context assembly, structured output (D5), citation binding and
  validation against retrieved chunks
- Abstention gates 1 and 2 (D6)
- `mlsc ask "..."`

**Done when:** `mlsc ask "What are the responsibilities of a domain lead?"` answers with
citations, and `mlsc ask "Who is the current Technical Head?"` refuses and explains that the KB
describes the role without naming anyone. **Those two commands are the core demo.**

---

### Phase 5 — HTTP API *(~45 min)*

- `api/app.py` (lifespan loads the index once), `api/deps.py` (composition root),
  `api/schemas.py`, `api/routes/*`
- RFC 9457 error handling, request tracing
- SSE streaming on `/v1/ask/stream`
- `mlsc serve`

**Done when:** the endpoints in `docs/API.md` respond as documented and `/docs` renders the
generated OpenAPI schema.

---

### Phase 6 — Web UI *(~30 min)*

One static page served by FastAPI: question box, streamed answer, citation cards that expand to
the source passage, and a collapsible diagnostics panel showing retrieved chunks with scores and
which gates fired.

Deliberately small — the brief says correctness over UI. But the diagnostics panel earns its
place: **being able to show, live, why the system refused a question is the strongest thing in
the demo.**

**Done when:** the demo can be driven end to end in a browser.

---

### Phase 7 — Full evaluation and the write-up *(~60 min)*

- `evaluation/metrics/generation.py` (faithfulness, relevancy, correctness) + `judge.py`
- `evaluation/metrics/abstention.py`
- `mlsc calibrate` — sweep the gate-1 threshold, choose the operating point deliberately
- Run every ablation from `docs/EVALUATION.md`
- Optional RAGAS cross-check
- Fill in the Results section: headline table, per-type breakdown, ablations, chosen operating
  point with rationale, and an honest list of remaining failures

**Done when:** `docs/EVALUATION.md` has real numbers and a paragraph on what the system still
gets wrong.

---

### Phase 8 — Polish *(remaining time)*

README run-through from a clean clone, docstrings, the no-hard-coded-answers CI test, and a
final pass to make sure the docs describe the system that actually exists.

---

## Where the time will actually go

Honest risk assessment, since estimates like the above are usually wrong in predictable ways:

| Risk | Mitigation |
|---|---|
| fastembed on Python 3.13 / Windows | retired in phase 1; sbert adapter is the fallback |
| Authoring 30 questions with gold chunk labels is slower than it looks | start with 15 covering all five types, grow to 30 in phase 7 |
| Gemini free-tier rate limits during eval runs | judge verdicts are cached by content hash, so re-runs are nearly free |
| Prompt iteration eating the clock | prompts are versioned and evaluated, so changes are accepted or rejected on numbers rather than re-read endlessly |
| Scope creep into the UI | phase 6 is capped at one page; anything more waits until phase 8 |

## Definition of done for the submission

- [ ] `git clone`, `pip install -e .`, `mlsc index`, `mlsc serve` works from clean
- [ ] Direct, multi-document, reasoning, unanswerable and ambiguous questions all behave correctly in a live demo
- [ ] Every answer carries citations that resolve to real passages
- [ ] `docs/EVALUATION.md` reports real numbers with ablations and a per-type breakdown
- [ ] Every design decision in `docs/DECISIONS.md` is one I can defend out loud
- [ ] No hard-coded answers anywhere, enforced by a test
