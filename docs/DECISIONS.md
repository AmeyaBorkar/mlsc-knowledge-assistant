# Decision log

Short records of the choices that shaped this system, each with the alternative that was
rejected and why. The interview will ask "why did you do it this way", and these are the
answers, written while the reasoning was fresh rather than reconstructed afterwards.

Format: **Context** (what forced a choice) · **Decision** · **Rejected** · **Cost** (what this
decision makes worse, because every one of them does something worse).

---

## D1 — Hybrid retrieval (BM25 + dense) fused with RRF

**Context.** 6 documents, 18 chunks. The vocabulary contains rare exact terms — `Web3`,
`Technical Head`, `code of conduct`, `second-year coordinators` — that a small dense embedder
tends to smear into neighbouring concepts. It also contains near-synonymous prose where lexical
matching alone would fail ("how are hackathon entries scored" vs "judging criteria").

**Decision.** Run BM25 and dense retrieval independently and fuse with Reciprocal Rank Fusion
(k=60). Keep dense-only and lexical-only runnable as strategies so the eval can prove the claim.

**Rejected.** *Dense-only* — the default reflex, and it loses exact rare terms in a corpus with
no redundancy to fall back on. *Weighted score blend* — cosine and BM25 live on incompatible,
corpus-dependent scales, and with 18 chunks there is not enough data to fit a blending weight
honestly; RRF uses ranks only, so it is scale-free with one interpretable constant.

**Cost.** Two indexes to keep in sync, and RRF discards score *magnitude* — a chunk that both
retrievers rank first with wildly different confidence looks the same as one they merely agree
on. The retrieval gate compensates by reading raw dense scores, not fused ranks.

---

## D2 — NumPy as the default vector store, Chroma as an adapter

**Context.** The entire corpus embeds to an `18 x 384` float matrix, about 28 KB.

**Decision.** Default store is an in-memory NumPy matrix persisted as `.npz`. `VectorStore`
remains a real port with a Chroma adapter behind an optional extra.

**Rejected.** *Chroma/FAISS/Qdrant as default* — introduces a service, a schema and a failure
mode to buy sub-millisecond search on a matrix that already searches in microseconds. Choosing a
vector database here would be resume-driven design, and the honest answer to "why did you add a
vector DB" would have been "because RAG tutorials have one".

**Cost.** Nothing scales past maybe 10^5 chunks. That is fine and the port exists precisely so
the answer to "what if the KB grows" is a config change, not a rewrite.

---

## D3 — fastembed (ONNX) over sentence-transformers

**Context.** Embeddings are needed for indexing, for retrieval and for the answer-relevancy
metric. The project should be runnable by a reviewer on Windows without a GPU or an API key.

**Decision.** `BAAI/bge-small-en-v1.5` via fastembed's ONNX runtime. ~150 MB, no PyTorch.

**Rejected.** *sentence-transformers* — same model, but drags in torch and transformers for
roughly 1 GB on disk and a 5–10 s cold start, buying nothing at this scale. A `sbert` adapter
exists as an optional extra so the port is demonstrably real. *API embeddings* — would make
indexing, retrieval and CI all require a network call and a secret, which forfeits the
best property of this design (see D4).

**Cost.** fastembed is a less familiar name than sentence-transformers, so it needs explaining.
The model is also English-only and 512-token capped — irrelevant for this corpus, relevant if the
KB ever grew multilingual.

---

## D4 — Retrieval works with no API key at all

**Context.** Generation needs a hosted LLM. Retrieval and every retrieval metric do not.

**Decision.** Treat this as an architectural invariant rather than a coincidence. Indexing,
`POST /v1/search`, the CLI `search` command and the whole retrieval metric suite run offline
with no secret. Only `/v1/ask` and LLM-judged metrics need a key.

**Rejected.** *Letting an LLM creep into retrieval* — query rewriting or LLM reranking on by
default would quietly break this. Both exist, both are flag-gated and off.

**Cost.** Some techniques that would help retrieval (HyDE, LLM query expansion) are held behind
flags rather than enabled by default. Worth it: a reviewer can clone the repo and reproduce half
the evaluation with zero setup, and CI can gate on retrieval quality without a secret.

---

## D5 — Structured output for answering, not free text

**Context.** Abstention needs to be a reliable, machine-readable decision. The naive approach —
prompt for "say I don't know" and then string-match the response — is brittle in both directions.

**Decision.** One schema-constrained call returns
`{sufficient_context, answer, cited_chunk_ids, confidence}`. The refusal decision is a parsed
boolean. `cited_chunk_ids` is then validated against the chunks actually retrieved, so a
fabricated citation is caught mechanically.

**Rejected.** *Free-text plus string matching* — fails when the model writes "the documents do
not appear to specify" and, worse, false-positives on a legitimate answer that happens to
contain the phrase. *A separate classifier call before answering* — doubles latency and cost to
answer a question the answering model is better positioned to judge, since it has already read
the context.

**Cost.** Ties us to providers with structured/JSON output. All five adapters support it; Ollama
support varies by model, so its adapter falls back to prompted JSON with a repair pass.

---

## D6 — Three abstention gates rather than one

**Context.** "Not in the knowledge base" has genuinely different causes: nothing relevant exists
at all, versus something relevant exists but does not contain the fact. The KB has a perfect
example of the second — the Technical Head role is described in detail, and no person is ever
named.

**Decision.** Gate 1 is a calibrated retrieval-score threshold (free, pre-LLM). Gate 2 is the
`sufficient_context` field (catches near-misses that gate 1 cannot). Gate 3 is an optional
post-hoc faithfulness check.

**Rejected.** *Threshold only* — cannot catch near-misses, which are the hard and interesting
case. *Model judgement only* — pays for an LLM call on questions with visibly nothing relevant,
and gives the model an opportunity to hallucinate where a threshold gives it none.

**Cost.** Three places to tune, and gates can disagree. `diagnostics.gates` reports each one
independently so a wrong refusal is attributable to a specific gate.

---

## D7 — Own the evaluation metrics; use RAGAS as a cross-check

**Context.** The brief names RAGAS and DeepEval, and requires that we understand and justify
what we report.

**Decision.** Implement retrieval, generation and abstention metrics in-repo (~200 readable
lines). Run RAGAS as an optional extra to cross-check faithfulness and answer relevancy.

**Rejected.** *RAGAS as the primary harness* — its judge prompts change between versions, it
pulls heavy transitive dependencies, it does not cover the abstention family this problem most
needs, and it would make the LLM-free retrieval metrics depend on an API key. Defending "RAGAS
said 0.87" is much harder than defending twenty lines you wrote.

**Cost.** More code to write and to be responsible for. Mitigated by the cross-check: agreement
between two independent implementations is stronger evidence than either alone, and disagreement
is a finding worth reporting.

---

## D8 — Provider-agnostic generation, Gemini as the default

**Context.** Free-tier access now, with the possibility of switching provider later; the brief
does not mandate a stack.

**Decision.** `LLMProvider` port with adapters for Gemini, Anthropic, OpenAI, Groq and Ollama,
resolved through a registry keyed by config. Gemini 2.5 Flash is the default: the most usable
free tier for repeated evaluation runs, with native structured output.

**Rejected.** *Coding directly against one SDK* — the cheapest thing today and the most
expensive thing in a month, and it would make the "swap the judge to a different provider" bias
mitigation impossible.

**Cost.** The port is a lowest-common-denominator interface, so provider-specific features
(prompt caching, extended thinking) are not exposed. Acceptable — none of them matter for a
900-token prompt.

---

## D9 — Structure-aware chunking, not fixed windows

**Context.** The documents are one-paragraph-per-line prose with a title on line 1. Two of them
contain lists (the five domains, the domain-lead responsibilities) that are semantically atomic.

**Decision.** Split on blank lines, keep list blocks whole together with their introducing
sentence, merge very short orphans up to a floor, and prefix the document title at embed time
while keeping stored text clean.

**Rejected.** *Fixed 512-token windows with overlap* — the standard default, and here it would
slice the domain list in half, making "what technical domains exist in MLSC" (the brief's own
example question) unanswerable from any single chunk. *Semantic/embedding-based chunking* —
sophisticated, and pointless when the author already marked the boundaries with blank lines.

**Cost.** The chunker is tailored to this corpus's shape and would need work for PDFs or
Markdown. Stated plainly: correctness on the corpus we actually have beats generality we do not
need. `Chunker` is a port, so a different strategy is an adapter.

---

## D10 — Abstention returns HTTP 200

**Context.** A refusal has to be distinguishable from an error by an API client.

**Decision.** Refusing is a successful outcome. `200` with `answered: false`, an
`abstention_reason` enum, and an explanation of what the KB *does* contain. HTTP errors are
reserved for actual failures.

**Rejected.** *`404` or `422` for "no answer"* — conflates a correctly working system with a
broken request, and would make client retry logic actively harmful.

**Cost.** Clients must read `answered` rather than only the status code. Documented in
`docs/API.md` and visible in the response shape.
