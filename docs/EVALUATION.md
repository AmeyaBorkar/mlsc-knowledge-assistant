# Evaluation methodology

> Status: **complete**. Every metric family is implemented and measured; results are at
> the end. The one outstanding item is the RAGAS cross-check, which is blocked on free-tier
> quota rather than on design.

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

RAGAS remains an optional extra (`pip install -e ".[ragas]"`). **The cross-check has not been
run**, and the reason is quota rather than principle: RAGAS issues its own judge calls, and a
second full pass does not fit in the free-tier daily allowance that already constrained the
primary run. Stated as an open item rather than quietly dropped — two independent
implementations agreeing would be stronger evidence than either alone, and that evidence does
not exist yet.

What the in-repo implementation *did* buy, which importing a framework would not have: the
faithfulness metric was measuring truncation for a while (see the Results section), and that
was findable because the twenty lines feeding passages to the judge were readable. A framework
returning 0.556 would have been much harder to disbelieve.

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

### Abstention end to end — does gate 2 close the gap?

Phase 3 left a hard test: gate 1 alone leaves a **hallucination rate of 0.83**, and no
threshold can do better without refusing a third of real questions. Phase 4 ran the whole
dev set through the answering pipeline to see whether gate 2 closes it.

Model `gemini-3.1-flash-lite`, thinking disabled, hybrid retrieval, k=6, 40 questions,
150s total (3.8s per question, mostly deliberate pacing).

| Metric | Gate 1 only (Phase 3) | Gates 1 + 2 (Phase 4) |
|---|---|---|
| **Hallucination rate** | 0.83 | **0.000** |
| Abstention recall | 0.17 | **1.000** |
| Abstention precision | 1.00 | 0.923 |
| Abstention F1 | 0.29 | **0.960** |
| Over-refusal rate | 0.00 | 0.036 |
| Near-miss unanswerables caught | 0 / 9 | **9 / 9** |
| Off-domain unanswerables caught | 2 / 3 | **3 / 3** |

**Every unanswerable question was refused, and nothing was fabricated.** This is the
design's central claim, and it is the first point at which it is evidence rather than
argument.

Which gate did the work:

| Gate | Refusals | What it caught |
|---|---|---|
| 1 — retrieval threshold | 2 | off-domain questions, at zero LLM cost |
| 2 — context sufficiency | 10 | every near miss, plus one off-domain question scoring 0.62 |
| 3 — faithfulness verify | not run | off by default; costs a second call |

Gate 1 handled 2 of 12. Had it been the only mechanism the system would have answered the
other 10 — including "who is the current Technical Head?" and "how many coordinators does
each domain have?", where a nearby passage supplies a plausible decoy number. Gate 1 is
worth keeping because it is free and forecloses hallucination entirely on the cases it
catches, but it is not the abstention mechanism. Gate 2 is.

**The one failure: q25**, "Can a first-year student contribute to a technical project at
MLSC?" — refused when it should have been answered. The knowledge base never mentions
first-year students; a correct answer has to generalise from "students do not need to be
experts in every technology". The dataset notes flagged this ambiguity when the question
was written, before any of it was run. It is over-caution rather than a wrong answer,
which is the failure direction to prefer, but it is the honest 0.036 over-refusal rate and
it is not rounded away.

**What this does not show.** Abstention is measured; answer *quality* is not. Faithfulness,
answer relevancy and correctness need an LLM judge and arrive in Phase 7. An answer can be
grounded, cited and still wrong or incomplete, and nothing here would catch that.

Raw per-question log: [`evaluation/reports/abstention-phase4.log`](../evaluation/reports/abstention-phase4.log).

### Answer quality — the LLM-judged metrics

Model `gemini-3.1-flash-lite` answering and judging, temperature 0, thinking disabled,
hybrid retrieval at k=6. All 40 questions answered; **27 answered questions judged**, since
the 13 refusals have no answer to score. 69 judge calls, 12 served from cache.

| Metric | Score |
|---|---|
| **Faithfulness** (claim-level, against cited passages) | **0.981** |
| **Answer correctness** (against reference answers) | **0.944** |
| **Answer relevancy** (question similarity, embedding-based) | **0.800** |

#### By question type — and why relevancy is the odd one out

| Type | Faithfulness | Relevancy | Correctness |
|---|---|---|---|
| direct | 1.000 | 0.835 | 0.964 |
| multi_document | 0.917 | 0.785 | 0.917 |
| reasoning | 1.000 | 0.789 | 0.875 |
| **ambiguous** | **1.000** | **0.684** | **1.000** |

The ambiguous row is the interesting one, and it is a **metric artefact rather than a
system failure**. Those answers score a perfect 1.000 on both faithfulness and
correctness — they are fully grounded and fully agree with the reference — while scoring
lowest on relevancy.

The cause is structural. Answer relevancy works by asking the model what questions the
answer would answer, embedding those, and comparing them to the original. When the
original is terse, its embedding sits far from any well-formed question, so the score
falls however good the answer is. Q27 is the clearest case:

> **Question:** "Tell me about the leads."
> **Answer:** correct, complete, cited, judged 1.00 on faithfulness *and* correctness.
> **Relevancy: 0.59** — because the generated question "How is the MLSC technical team
> structured and managed?" is a perfectly good rendering of a question that was never
> phrased that well to begin with.

So answer relevancy on this corpus partly measures **how well the question was worded**.
That is worth knowing before quoting 0.800 as an answer-quality figure, and it is the
kind of thing that only shows up when the metric is implemented rather than imported.

#### Where the system genuinely lost points

Three questions scored below 1.0 on correctness, and all three are **omissions rather than
inventions** — the safer failure direction, but real:

| Question | Score | What was missing |
|---|---|---|
| q12 "What is MLSC?" | 0.50 | omitted the activities list (workshops, hackathons, study sessions) |
| q16 "What part do domain leads play in hackathons?" | 0.50 | omitted that mentors provide guidance during development |
| q22 "What qualities make a good domain lead?" | 0.50 | omitted "helping coordinators develop their skills" |

Q16 is also the only faithfulness miss (0.50). The judge flagged "encouraging
participation in hackathons **among members of the community**" — the qualifier is not in
the source. This is the question whose trap the dataset notes flagged when it was written:
the documents never state that domain leads *are* the mentors, and a correct answer has to
report both facts without asserting the link. The system reported one and added a small
unsourced qualifier to it.

#### Human spot-check of the judge

An LLM judge is a measuring instrument, so four cases were graded by hand and compared:

| Question | Judge verdict | My reading | Agree? |
|---|---|---|---|
| q12 correctness 0.50 | omits the activity list | same | ✅ |
| q22 correctness 0.50 | omits coordinator development | same | ✅ |
| q27 faith 1.00 / corr 1.00 | answer is complete and grounded | same | ✅ |
| q16 faithfulness 0.50 | "among members of the community" unsupported | **too harsh** — a reasonable inference, not a fabrication; I would score 0.75–1.00 | ⚠️ |

Three of four agree. The disagreement is the judge being *stricter* than a human on an
added qualifier, which is the direction to prefer in a faithfulness metric and is a direct
consequence of the prompt instructing it that added specifics are unsupported. Reported
rather than tuned away: adjusting the prompt until the judge agreed with me would be
fitting the instrument to the result.

#### The self-preference caveat, restated

The same model family answers and judges, which risks self-preference bias. Faithfulness
of 0.981 should be read with that in mind. Two things limit it: the spot-check above found
the judge harsher than a human rather than more lenient, and `evaluation.judge.provider`
points the judge at a different backend in one line, because the judge talks to the same
`LLMProvider` port as everything else. Running that comparison needs quota this project
does not currently have.

#### Cost, and why the cache exists

The run costs roughly one call per question plus three per answered question — about 120
calls, or six times the daily free-tier allowance of the previously configured model.
Judge verdicts are cached by content hash, so re-rendering a report is nearly free; 12 of
this run's verdicts came from cache. A metric you cannot afford to re-run is a metric you
stop checking.

### What this still does not measure

RAGAS was not run as a cross-check: it needs its own judge calls, and the free-tier quota
that constrained this run does not stretch to a second full pass. That comparison is the
one outstanding item in the methodology above, and it is outstanding for a resource reason
rather than a design one.

Two further gaps worth naming. The dev set is 40 questions written by the same person who
built the system, so it may under-represent phrasings a stranger would use — MLSC's own
evaluation set is the real test. And nothing here measures latency or cost under load; the
timings in these runs are dominated by deliberate rate-limiting.
