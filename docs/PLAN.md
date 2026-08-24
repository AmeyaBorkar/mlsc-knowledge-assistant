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

### Phase 3 — Evaluation harness, retrieval half *(~45 min)* — **done**

Deliberately before generation. Retrieval quality is measurable without an LLM, and doing it now
means phases 4–6 are tuned against numbers instead of vibes.

- `evaluation/dataset.py` + author `evaluation/datasets/dev_set.yaml` (~30 questions, at least a
  third unanswerable, gold chunk labels)
- `evaluation/metrics/retrieval.py` — precision@k, recall@k, MRR, nDCG@k, doc hit rate,
  multi-doc coverage
- `evaluation/runner.py` + `report.py`
- `mlsc eval --metrics retrieval`

**Done when:** a retrieval-only report renders, and the dense / lexical / hybrid ablation
produces real numbers. Both hold; full results in [EVALUATION.md](EVALUATION.md#results).

#### What this phase found

**The dev set is 40 questions** (28 answerable, 12 unanswerable, 8 of those near-miss),
covering all five types from the brief, with gold chunk labels validated against the live
index on every run so a stale label fails loudly instead of deflating recall invisibly.

**1. Phase 2's headline finding was caused by a bug, not by fusion being wrong.** Hybrid was
losing to dense because BM25 was voting on queries where it had no information — "What is
MLSC?" reduces to one term present in all 18 chunks, and Okapi's epsilon floor means it still
scores, by chunk length. Filtering non-discriminative query terms (D11) lifted hybrid recall
0.920 → 0.955, MRR 0.875 → 0.912, and restored the brief's own example question to rank 1.

**2. D1 is upheld only partially, and the write-up says so.** Hybrid wins MRR and nDCG, ties
R-Precision, and loses recall 0.955 vs 0.973. It stays the default on ranking quality, with the
recall gap — half a question out of 28 — stated plainly rather than buried.

**3. Two Phase 2 predictions were wrong.** `rrf_k` was predicted to be mis-scaled for an
18-chunk corpus, and `candidate_k` to be feeding noise into fusion. Sweeping both changes
nothing to three decimal places. Recorded as corrections.

**4. Contextual chunk headers are the biggest single win in the pipeline** — 9 to 11 points of
recall, 8 to 11 of nDCG, for the cost of a string prefix at embed time.

**5. Diversification is inert on this set.** Multi-document coverage is 1.000 with the
per-document cap at 3 and 1.000 with it disabled; at 1 or 2 it is actively harmful. It is kept
as a guard but has not earned its place on evidence, and the report says so.

**6. The abstention curve is the most valuable output.** Gate 1 was calibrated to **0.55** —
the highest threshold refusing no answerable question. Near-miss recall stays at **zero until
0.75**, where over-refusal is already **39%**; reaching full near-miss recall costs **71%**
over-refusal. So no threshold can catch "who is the current Technical Head?" without destroying
the system, and gate 1 alone leaves a hallucination rate of 0.83. The three-gate design is now
measured rather than argued — and Phase 7 has to show gate 2 actually closes that gap.

---

### Phase 4 — Generation *(~60 min)* — **done**

- `generation/providers/` — the `LLMProvider` port, a registry, and the Gemini adapter first
- `generation/prompts.py` — versioned grounded-answering template
- `generation/answerer.py` — context assembly, structured output (D5), citation binding and
  validation against retrieved chunks
- Abstention gates 1 and 2 (D6)
- `mlsc ask "..."`

**Done when:** `mlsc ask "What are the responsibilities of a domain lead?"` answers with
citations, and `mlsc ask "Who is the current Technical Head?"` refuses and explains that the KB
describes the role without naming anyone. Both do.

#### What this phase found

**1. Gate 2 closes the gap Phase 3 left open — completely.** Hallucination rate went from
**0.83 with gate 1 alone to 0.000** across the whole dev set, abstention F1 from 0.29 to 0.960,
and every one of the 9 near-miss unanswerables was refused. Gate 1 caught 2 of 12; gate 2
caught the other 10. One over-refusal (q25), on the question whose ambiguity the dataset notes
had flagged in advance.

**2. Gate 3 was implemented after all.** The plan scoped Phase 4 to gates 1 and 2, but leaving
`verify_faithfulness` as a documented-but-dead parameter was worse than the extra work. It is
off by default and costs a second call.

**3. The free tier is far more constrained than assumed, and it changed the model choice.**
`gemini-2.5-flash` stopped serving new keys mid-project (404, "no longer available to new
users"). Its successor `gemini-3.5-flash` allows **5 requests per minute and only 20 per day** —
a 40-question evaluation run cannot complete on it at all. `gemini-3.7-flash` took **55s** for a
two-token reply under load. The shipped default is `gemini-3.1-flash-lite`: ~2s per answer, and
it handled every hard abstention case correctly, including the adversarial one. Model selection
here was driven by measured quota and latency, not by preference.

**4. Two real bugs, both found by running the thing.**
`LLMConfig.api_key()` read `os.environ`, but pydantic-settings loads `.env` into the settings
*model* and never exports it — so a key pasted into `.env` was invisible to the provider, and
the unit tests missed it because they used `monkeypatch.setenv`. And `with_retry` parsed the
server's `retryDelay` only to decorate an error message while backing off on its own guess:
three blind retries total ~7s against a 60-second quota window. Both fixed; the second is D13.

**5. Thinking off is worth 30x.** Identical abstention verdict, 39s versus 1.18s (D14).

**6. A prompt edit silently no-opped.** A patch script without an `assert old in s` guard made
`str.replace` a no-op while still reporting success, and three rounds of "the model is ignoring
my instruction" followed. The model had never seen it.

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
