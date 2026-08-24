# Full evaluation — `20260824-110510-full-hybrid`

Dataset **dev_set** · 40 questions evaluated, 27 judged

```json
{
  "strategy": "hybrid",
  "top_k": 6,
  "embedder": "BAAI/bge-small-en-v1.5",
  "provider": "gemini",
  "model": "gemini-3.1-flash-lite",
  "judge_model": "gemini-3.1-flash-lite",
  "prompt_version": "grounded-v1",
  "abstention_threshold": 0.55,
  "families": [
    "retrieval",
    "generation",
    "abstention"
  ],
  "index": {
    "chunks": 18,
    "built_at": "2026-08-24T09:19:51.417749+00:00"
  }
}
```

## Generation quality

Scored on **answered questions only** — an abstention cites nothing and claims
nothing, so averaging refusals in would let a system that refuses everything
post a perfect faithfulness score.

| Metric | Score |
|---|---|
| Answer Correctness | 0.944 |
| Answer Relevancy | 0.800 |
| Faithfulness | 0.981 |

### By question type

| Type | answer correctness | answer relevancy | faithfulness |
|---|---|---|---|
| ambiguous | 1.000 | 0.684 | 1.000 |
| direct | 0.964 | 0.835 | 1.000 |
| multi_document | 0.917 | 0.785 | 0.917 |
| reasoning | 0.875 | 0.789 | 1.000 |

## Abstention

| Metric | Score |
|---|---|
| F1 | 0.960 |
| Hallucination Rate | 0.000 |
| Over Refusal Rate | 0.036 |
| Precision | 0.923 |
| Recall | 1.000 |

### By unanswerable type

| Kind | Caught | Recall |
|---|---|---|
| near_miss | 9/9 | 1.000 |
| off_domain | 3/3 | 1.000 |

## Retrieval (k = 6)

| Metric | Score |
|---|---|
| All Docs Hit Rate | 1.000 |
| Average Precision | 0.844 |
| Doc Recall | 1.000 |
| Mrr | 0.912 |
| Ndcg At K | 0.891 |
| Precision At K | 0.226 |
| R Precision | 0.741 |
| Recall At K | 0.955 |

## Judge

Model `gemini-3.1-flash-lite` · 69 calls · 12 cache hits.

The same model family generates and judges, which risks self-preference bias.
Mitigations: the human spot-check below, and `evaluation.judge.provider`, which
points the judge at a different backend in one line.

## Questions the system got wrong

| Question | Type | Problem | Text |
|---|---|---|---|
| `q12` | direct | answer_correctness=0.50, answer_relevancy=0.68 | What is MLSC? |
| `q16` | multi_document | answer_correctness=0.50, faithfulness=0.50 | What part do domain leads play in hackathons? |
| `q22` | reasoning | answer_correctness=0.50 | What qualities make someone a good domain lead? |
| `q25` | reasoning | refused an answerable question | Can a first-year student contribute to a technical project a |
| `q27` | ambiguous | answer_relevancy=0.59 | Tell me about the leads. |
| `q28` | ambiguous | answer_relevancy=0.60 | What do I need to know about behaving properly in the commun |
