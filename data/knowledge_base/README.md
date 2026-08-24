# Knowledge base

The six documents in this directory are the MLSC knowledge base supplied with the
recruitment challenge, copied verbatim from `AboutMLSC.zip`.

They are the **sole source of truth** for MLSC-specific information. The assistant
answers from these files and nothing else, and refuses questions they do not cover.

Each document is checksummed at index time; `GET /v1/health` reports `index.stale`
if any file has changed since the index was built.
