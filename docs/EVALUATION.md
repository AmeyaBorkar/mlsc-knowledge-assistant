# Evaluation methodology

> Status: **retrieval metrics implemented and run** (see Results at the end). Generation
> and LLM-judged abstention metrics arrive in Phase 7.

The brief asks for context precision, context recall, answer relevancy and faithfulness, and
says we must be able to justify the metrics we report. This document is that justification.

## 1. What each metric actually measures

A RAG system fails in three distinct places, and a metric is only useful if it isolates one of
them. Grouping them this way is the point:

```
question --> [ RETRIEVAL ] --> context --> [ GENERATION ] --> answer
                  |                              |
          did we fetch the right           did the model use only
          passages?                        what it was given?
                            \                  /
                             [ ABSTENTION ]
                    did the system know when it had nothing?
```

If answer correctness drops, only the split tells you whether to fix the chunker, the prompt or
the threshold. A single headline score cannot.

## 2. Retrieval metrics (no LLM, deterministic, key-free)

Computed against gold chunk/document labels in the evaluation set. No model judges these, so
they are exactly reproducible and can run in CI.

| Metric | Definition | Why it is here |
|---|---|---|
| **Context precision@k** | fraction of retrieved chunks that are relevant | Noise in the context window is what drags a model off-source. Low precision predicts low faithfulness. |
| **Context recall@k** | fraction of gold chunks that were retrieved | The hard ceiling on the whole system. A fact not retrieved cannot be answered, no matter how good the model is. |
| **MRR** | mean of `1 / rank of first relevant chunk` | Position matters: models attend unevenly across a context window, and a gold chunk at rank 6 is worth less than one at rank 1. |
| **nDCG@k** | rank-discounted gain over all relevant chunks | The rank-aware complement to recall, and the metric that moves when fusion reorders results without changing the set. |
| **Document hit rate** | did the correct *document* appear at all | Directly measures R3, "provide the relevant source document(s)", which is what a user actually sees. |
| **Multi-document coverage** | on multi-doc questions, fraction where *every* required document is in the final context | Directly measures R4. This is the metric the diversification step exists to move, and averaged precision/recall hide it completely. |

**Precision and recall are reported at several k** (3, 5, 6, 10). A single k invites tuning the
system to that k. The precision/recall tradeoff across k is also how `top_k` gets chosen: more
context raises recall and lowers precision, and the crossover is an empirical question.

**k matters unusually much here.** With 18 chunks, `k=10` is over half the entire corpus —
at that point "retrieval" barely narrows anything and precision collapses by construction. That
alone argues for a small k, and the sweep will show where.

## 3. Generation metrics (LLM as judge)

| Metric | How it is computed | Why this way |
|---|---|---|
| **Faithfulness** | decompose the answer into atomic claims; ask the judge whether each is entailed by the *cited* context; score = supported / total | Claim-level, not answer-level. A four-sentence answer with one invented sentence should not score 0 or 1, it should score 0.75, and only decomposition gives that resolution. Judging against *cited* rather than *all retrieved* context is stricter and matches what we show the user. |
| **Answer relevancy** | have the model generate N hypothetical questions the answer would answer; embed them with fastembed; mean cosine against the real question | Catches answers that are true and cited but evasive or off-target. Deliberately embedding-based rather than judged: it is cheap, deterministic given the generated questions, and the embedder is already loaded. |
| **Answer correctness** | judge compares the answer against the reference answer for factual agreement, scored on a rubric | The evaluation set ships reference answers; ignoring them would waste the strongest signal available. Kept separate from faithfulness because an answer can be perfectly grounded and still incomplete. |

### Guarding the judge

An LLM judge is a measurement instrument, and instruments need controls. Ours:

- **Structured verdicts.** The judge returns a schema (`{verdict, reason, evidence_span}`), never
  prose to be regex-matched. Every verdict carries its reason into the run trace, so a
  surprising score can be read rather than guessed at.
- **Cached and keyed** by `(question, answer, context, prompt_version)`, so re-running a report
  costs nothing and comparisons across runs are stable.
- **Temperature 0** and a pinned model id, both recorded in the run config.
- **Position/verbosity bias check.** On a sample, re-judge with claim order shuffled. If verdicts
  move, that is reported as instrument noise rather than a system change.
- **A human spot-check.** Roughly ten questions are graded by hand and compared with the judge.
  If they disagree materially, the judge prompt is the bug. This gets written up in the results.

The honest caveat, stated up front: the same model family generates and judges, which risks
self-preference bias. Mitigations are the human spot-check and the RAGAS cross-check below;
using a different provider for judging is a one-line config change if it proves necessary.

## 4. Abstention metrics (the ones the brief really cares about)

R5 — recognising when the answer is not in the knowledge base — is a listed requirement and an
explicit question type. None of the metrics above measure it properly, and one of them actively
rewards failing at it: **a system that refuses every question scores 1.0 on faithfulness.**

Treating "should this be answered?" as a binary classification:

| Metric | Definition |
|---|---|
| **Abstention precision** | of the questions we refused, how many were genuinely unanswerable |
| **Abstention recall** | of the genuinely unanswerable questions, how many we refused |
| **Abstention F1** | harmonic mean of the two |
| **Hallucination rate** | fraction of unanswerable questions answered confidently anyway — *the single worst failure mode this system has* |
| **Over-refusal rate** | fraction of answerable questions we refused — the cost of being too cautious |

The last two are reported as a pair, always. They trade off directly against each other, and a
threshold can be moved to make either one look excellent in isolation. `mlsc calibrate` sweeps
the retrieval-gate threshold and plots exactly this curve; the chosen operating point is stated
with its rationale rather than presented as a default.

## 5. Ablations

Design claims should be measured, not asserted. Each of these is one run:

| Ablation | Claim it tests |
|---|---|
| dense-only vs lexical-only vs hybrid | that fusion beats both components (section 5 of ARCHITECTURE) |
| with vs without contextual chunk headers | that prefixing the document title improves recall on pronoun-headed paragraphs |
| with vs without per-document diversification | that diversification is what moves multi-document coverage |
| `top_k` in {3, 5, 6, 10} | where the precision/recall crossover sits on an 18-chunk corpus |
| retrieval gate on vs off | how much of the abstention behaviour is the cheap gate rather than the model |
| `rrf_k` in {10, 20, 60} | whether the standard k=60 is too flat for 18 candidates (Phase 2 finding 4) |
| `candidate_k` in {6, 10, 15} | whether fusing over most of the corpus feeds RRF noise as rank signal |
| BM25 with vs without stemming | how much "hackathons" matching "hackathon" is actually worth |
| BM25 with vs without the title prefix | probed in Phase 2 and kept; confirm across the full set |

If an ablation shows a component does not help, **it gets removed and the negative result is
reported**. A hybrid retriever that loses to dense-only would be a finding worth stating, not a
finding worth hiding.

## 6. On RAGAS

RAGAS is the obvious framework choice and is listed in the brief, so not using it as the primary
harness needs a reason.

We implement the metrics ourselves and run RAGAS as a **cross-check**, because:

1. The brief requires justifying the metrics, and a metric whose implementation you can read in
   twenty lines is one you can defend under questioning. "RAGAS reported 0.87" is not an
   explanation of what 0.87 means.
2. The abstention family — the metrics this problem most needs — is not covered by RAGAS, so a
   custom layer was going to exist regardless.
3. RAGAS pins heavy transitive dependencies and its judge prompts change between versions, which
   is awkward for reproducibility on a small project.
4. Our retrieval metrics need no LLM at all; running them through a framework that assumes one
   would make CI depend on an API key for no benefit.

RAGAS remains an optional extra (`pip install -e ".[ragas]"`, `mlsc eval --with-ragas`). Where
our faithfulness and answer relevancy diverge from theirs, the divergence gets investigated and
written up. Two independent implementations agreeing is stronger evidence than either alone.

## 7. The evaluation set

MLSC supplies an evaluation set separately. Until it arrives we author our own dev set at
`evaluation/datasets/dev_set.yaml`, covering all five question types from the brief:

```yaml
- id: q07
  question: "How do MLSC hackathons connect to the technical domains?"
  type: multi_document            # direct | multi_document | reasoning | unanswerable | ambiguous
  answerable: true
  gold_documents: [hackathons, domains]
  gold_chunks: [hackathons::c05, domains::c02]
  reference_answer: "Hackathon projects may span several technical domains..."
  notes: "Requires joining the cross-domain project statement with the domain list."
```

Target composition: roughly 30 questions, with **at least a third unanswerable** — that class is
the hardest requirement and needs enough samples for its metrics to mean anything. Deliberately
included:

- *Near-miss unanswerables*, where the KB discusses the topic but not the fact. "Who is the
  current Technical Head?" is the sharpest one: the role is described in detail, the person
  never named. Retrieval will score high and only gate 2 can catch it. A test set of only
  obvious unanswerables ("who won the IPL?") would make the system look far better than it is.
- *Paraphrases of the same fact*, to test embedding robustness rather than keyword luck.
- *Reasoning questions* whose answer is stated nowhere verbatim and must be synthesised.

Gold chunk labels are assigned by reading the corpus, and are the one part of this project that
is genuinely subjective. The labelling rule is written down in the dataset header, applied
consistently, and any question where relevance is arguable is flagged with `notes`.

**This dev set is used only for measurement and calibration.** It is never imported by `src/`,
and a CI test asserts no reference answer appears verbatim in the source tree.

## 8. Reporting

Every run writes `evaluation/runs/{run_id}/`:

```
config.json      strategy, top_k, embedder, provider, model, prompt version, index manifest
per_question.jsonl   question, retrieved chunks + scores, answer, gates fired, judge verdicts
metrics.json     the rolled-up numbers
report.md        human-readable summary with per-type breakdowns
```

The report always breaks metrics down **by question type**, never only in aggregate. An overall
faithfulness of 0.9 hiding a hallucination rate of 0.4 on unanswerables is precisely the failure
this project is judged on, and only the breakdown reveals it.

---

## Results

**Retrieval only.** No LLM was involved in any number below, so all of it is exactly
reproducible offline and runs in CI. Generation and LLM-judged metrics arrive in Phase 7.

Dataset: `dev_set`, 40 questions (28 answerable and scored on retrieval, 12
unanswerable). Index: 18 chunks, `bge-small-en-v1.5`, chunker `structural-v1`, k = 6.

### Strategy comparison

| Strategy | Recall@6 | Precision@6 | R-Precision | MRR | nDCG@6 | Doc recall | Multi-doc coverage |
|---|---|---|---|---|---|---|---|
| dense | **0.973** | 0.232 | **0.741** | 0.858 | 0.866 | 1.000 | 1.000 |
| lexical | 0.795 | **0.250** | 0.616 | 0.711 | 0.700 | 0.893 | 0.714 |
| **hybrid** | 0.955 | 0.226 | **0.741** | **0.912** | **0.891** | 1.000 | 1.000 |

Read precision@6 with the labelling in mind: most questions have one gold chunk, so
precision@6 is capped at 0.167 for them. R-Precision is the comparable figure.

**Hybrid stays the default, but the honest reading is that it wins on ranking and loses
on recall.** It takes MRR (+0.054) and nDCG (+0.025) clearly, ties R-Precision, and gives
up 0.018 recall — half a question out of 28, comfortably inside noise at this sample size.
Ranking is worth paying for because the generator attends unevenly across its context
window, so a gold chunk at rank 1 is worth more than the same chunk at rank 5. Dense-only
remains a defensible choice and is one config flag away.

Per question, hybrid improves the first-relevant rank on five questions (q03 5→2, q11
3→1, q22 2→1, q26 2→1, q28 2→1) and loses ground on two (q25, q27). The single recall
loss is q25, where BM25 latches onto "students can learn and contribute to projects" as a
lexical decoy.

### By question type (hybrid)

| Type | Recall@6 | Precision@6 | MRR | All-docs hit |
|---|---|---|---|---|
| direct | 1.000 | 0.167 | 0.929 | 1.000 |
| multi_document | 0.958 | 0.361 | 1.000 | 1.000 |
| reasoning | 0.900 | 0.200 | 0.840 | 1.000 |
| ambiguous | 0.833 | 0.278 | 0.778 | 1.000 |

Ambiguous questions are the weakest class, which is the expected shape: terse, vaguely
worded queries give both retrievers less to work with. Multi-document questions do not
degrade — every one of them retrieves all its required documents.

### Ablations

| Variant | Recall@6 | R-Prec | MRR | nDCG@6 |
|---|---|---|---|---|
| hybrid (shipped) | 0.955 | 0.741 | 0.912 | 0.891 |
| — without contextual chunk headers | 0.848 | 0.661 | 0.809 | 0.783 |
| — without the low-IDF query filter | 0.920 | 0.723 | 0.875 | 0.849 |
| dense (shipped) | 0.973 | 0.741 | 0.858 | 0.866 |
| — without contextual chunk headers | 0.884 | 0.643 | 0.774 | 0.788 |

**Contextual chunk headers are the single largest win measured** — worth 9 to 11 points
of recall and 8 to 11 of nDCG. Prefixing each chunk with its document title before
embedding costs nothing and is the highest-value decision in the pipeline.

Knobs that turned out **not** to matter, contradicting predictions made in Phase 2:

| Ablation | Result |
|---|---|
| `rrf_k` ∈ {5, 10, 20, 30, 60} | identical recall; MRR moves 0.016 at k=5 and is flat above it |
| `candidate_k` ∈ {6, 8, 10, 15} | identical to three decimal places |
| per-document cap ∈ {3, 18} | identical — **the cap never fires on this set** |
| per-document cap ∈ {1, 2} | actively harmful: recall 0.750 and 0.902 |

Phase 2 predicted that RRF's k=60 was mis-scaled for an 18-chunk corpus and that
`candidate_k` was feeding noise into fusion. Both predictions were **wrong**. The real
problem was low-IDF query terms, and once those are filtered the fusion constant is
irrelevant. Recorded as a correction rather than quietly dropped.

Diversification is likewise **unproven**: multi-document coverage is 1.000 with the cap
at 3 and 1.000 with it disabled, so on this evaluation set the feature does nothing. It
is kept because it costs nothing and guards a real failure mode, but it has not earned
its place on evidence, and the honest statement is that it is inert here.

### Abstention gate 1 — the calibration curve

Threshold on best cosine, no LLM. This measures the **ceiling** of what a similarity
threshold can achieve alone.

| Threshold | Abst. P | Abst. R | Hallucination | Over-refusal | Near-miss R | Off-domain R |
|---|---|---|---|---|---|---|
| 0.450 | 1.00 | 0.17 | 0.83 | 0.00 | 0.00 | 0.67 |
| **0.550** | **1.00** | **0.17** | **0.83** | **0.00** | **0.00** | **0.67** |
| 0.650 | 0.60 | 0.25 | 0.75 | 0.07 | 0.00 | 1.00 |
| 0.700 | 0.50 | 0.25 | 0.75 | 0.11 | 0.00 | 1.00 |
| 0.750 | 0.45 | 0.75 | 0.25 | 0.39 | 0.67 | 1.00 |
| 0.800 | 0.38 | 1.00 | 0.00 | 0.71 | 1.00 | 1.00 |

**Committed operating point: 0.55** — the highest threshold that refuses no answerable
question. Gate 1 runs before the LLM and should remove obvious noise without ever harming
a real question, so zero over-refusal is the binding constraint rather than peak F1.

The important column is *Near-miss R*. It stays at **0.00 until 0.75**, by which point
**39% of answerable questions are being refused**; reaching 1.00 costs 71% over-refusal.
Near-miss unanswerables score 0.71–0.78 and answerable questions 0.67–0.90 — the
distributions overlap almost entirely.

So the three-gate design is not a stylistic preference. **No threshold can catch
"who is the current Technical Head?" without destroying the system**, and gate 1 alone
leaves a hallucination rate of 0.83. Whether gate 2 actually closes that gap is the
central question Phase 7 has to answer.

### What this does not yet measure

Faithfulness, answer relevancy, answer correctness, and the real abstention numbers all
need an LLM and arrive in Phase 7. Everything above is the retrieval ceiling: it says
what the generator will have to work with, not how well it uses it.
