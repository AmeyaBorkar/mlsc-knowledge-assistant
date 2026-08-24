# Retrieval evaluation — `20260824-095931-hybrid`

Dataset **dev_set** · 40 questions (28 scored on retrieval, 12 unanswerable)

Retrieval metrics only — no LLM was involved, so these numbers are exactly reproducible and require no API key.

## Configuration

```json
{
  "strategy": "hybrid",
  "top_k": 6,
  "candidate_k": 15,
  "rrf_k": 60,
  "max_chunks_per_document": 3,
  "bm25": {
    "k1": 1.5,
    "b": 0.75,
    "index_title": true
  },
  "embedder": "BAAI/bge-small-en-v1.5",
  "chunking": {
    "version": "structural-v1",
    "min_tokens": 40,
    "prepend_doc_title": true
  },
  "k_values": [
    3,
    5,
    6,
    10
  ],
  "index": {
    "built_at": "2026-08-24T09:19:51.417749+00:00",
    "chunks": 18,
    "documents": 6,
    "chunker_version": "structural-v1"
  }
}
```

## Headline (k = 6)

| Metric | Value |
|---|---|
| Recall@k | 0.955 |
| Precision@k | 0.226 |
| R-Precision | 0.741 |
| Avg Precision | 0.844 |
| MRR | 0.912 |
| nDCG@k | 0.891 |
| Doc recall | 1.000 |
| All-docs hit | 1.000 |
| Multi-doc coverage | 1.000 (7 questions) |

## Across k

| Metric | k=3 | k=5 | k=6 | k=10 |
|---|---|---|---|---|
| Recall@k | 0.866 | 0.955 | 0.955 | 1.000 |
| Precision@k | 0.393 | 0.271 | 0.226 | 0.146 |
| R-Precision | 0.741 | 0.741 | 0.741 | 0.741 |
| Avg Precision | 0.815 | 0.844 | 0.844 | 0.856 |
| MRR | 0.905 | 0.912 | 0.912 | 0.912 |
| nDCG@k | 0.852 | 0.891 | 0.891 | 0.909 |
| Doc recall | 0.964 | 0.982 | 1.000 | 1.000 |
| All-docs hit | 0.929 | 0.964 | 1.000 | 1.000 |

## By question type

Aggregates hide the cases that matter. Unanswerable questions are absent here by design: they have no gold passage, and are measured by abstention instead.

| Type | Recall@k | Precision@k | R-Precision | MRR | nDCG@k | All-docs hit |
|---|---|---|---|---|---|---|
| ambiguous | 0.833 | 0.278 | 0.333 | 0.778 | 0.687 | 1.000 |
| direct | 1.000 | 0.167 | 0.857 | 0.929 | 0.947 | 1.000 |
| multi_document | 0.958 | 0.361 | 0.708 | 1.000 | 0.913 | 1.000 |
| reasoning | 0.900 | 0.200 | 0.700 | 0.840 | 0.831 | 1.000 |

## Abstention gate 1 — threshold sweep

Gate 1 only: refuse when the best retrieval score falls below the threshold. No LLM involved, so this measures the *ceiling* of what a similarity threshold can achieve on its own.

| Threshold | Abst. P | Abst. R | F1 | Halluc. | Over-refuse | Near-miss R | Off-domain R |
|---|---|---|---|---|---|---|---|
| 0.300 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 0.350 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 0.400 | 1.00 | 0.08 | 0.15 | 0.92 | 0.00 | 0.00 | 0.33 |
| 0.450 | 1.00 | 0.17 | 0.29 | 0.83 | 0.00 | 0.00 | 0.67 |
| 0.500 | 1.00 | 0.17 | 0.29 | 0.83 | 0.00 | 0.00 | 0.67 |
| 0.550 | 1.00 | 0.17 | 0.29 | 0.83 | 0.00 | 0.00 | 0.67 |
| 0.600 | 0.67 | 0.17 | 0.27 | 0.83 | 0.04 | 0.00 | 0.67 |
| 0.650 | 0.60 | 0.25 | 0.35 | 0.75 | 0.07 | 0.00 | 1.00 |
| 0.700 | 0.50 | 0.25 | 0.33 | 0.75 | 0.11 | 0.00 | 1.00 |
| 0.750 | 0.45 | 0.75 | 0.56 | 0.25 | 0.39 | 0.67 | 1.00 |
| 0.800 | 0.38 | 1.00 | 0.55 | 0.00 | 0.71 | 1.00 | 1.00 |
| 0.850 | 0.34 | 1.00 | 0.51 | 0.00 | 0.82 | 1.00 | 1.00 |
| 0.900 | 0.31 | 1.00 | 0.47 | 0.00 | 0.96 | 1.00 | 1.00 |
| 0.950 | 0.30 | 1.00 | 0.46 | 0.00 | 1.00 | 1.00 | 1.00 |
| 1.000 | 0.30 | 1.00 | 0.46 | 0.00 | 1.00 | 1.00 | 1.00 |

**Best threshold with zero over-refusal:** `0.550` — catches 17% of unanswerable questions (67% of off-domain, 0% of near-miss) while refusing no answerable question.

## Weakest questions

| Question | Type | Recall | Gold | First relevant rank |
|---|---|---|---|---|
| `q25` Can a first-year student contribute to a technical p | reasoning | 0.50 | membership::c02, about_mlsc::c02 | 5 |
| `q26` What areas can I work on at MLSC? | ambiguous | 0.50 | domains::c01, about_mlsc::c01 | 1 |
| `q17` What is the path from joining MLSC as a member to ta | multi_document | 0.75 | membership::c00, membership::c01, membership::c02, leadership::c02 | 1 |
| `q01` What technical domains exist in MLSC? | direct | 1.00 | domains::c01 | 1 |
| `q02` What are the responsibilities of a domain lead? | direct | 1.00 | leadership::c01 | 1 |
