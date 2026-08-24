"""Evaluation runs and index rebuilds — the two long operations.

Both return 202 and a resource to poll rather than blocking. A full evaluation is
minutes of retrieval and (in Phase 7) model calls; holding an HTTP connection open for
that is how requests time out at a proxy nobody remembers configuring.

The job store is an in-memory dict, which is honest about what this is: a single-process
demo API. Persisting runs across restarts would mean a database, and the *reports*
already persist to disk — which is where a run's actual value lives.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Response, status

from mlsc_assistant.api.deps import AppState, StateDep
from mlsc_assistant.api.schemas import EvalRunRequest, EvalRunSummary
from mlsc_assistant.core.errors import DocumentNotFoundError, MLSCError
from mlsc_assistant.core.models import RetrievalStrategy
from mlsc_assistant.evaluation.dataset import resolve_dataset, validate_against_index
from mlsc_assistant.evaluation.report import write_run
from mlsc_assistant.evaluation.runner import run_retrieval_evaluation

router = APIRouter(prefix="/eval", tags=["evaluation"])


def _execute(state: AppState, run_id: str, request: EvalRunRequest) -> None:
    """Run the evaluation and record the outcome. Never raises into the task runner."""
    record = state.eval_runs[run_id]
    record["status"] = "running"
    try:
        settings = state.settings
        dataset = resolve_dataset(
            settings.dataset_path, request.dataset or settings.evaluation.default_dataset
        )
        validate_against_index(
            dataset,
            known_chunk_ids=[c.chunk_id for c in state.store.all_chunks()],
            known_doc_ids={c.doc_id for c in state.store.all_chunks()},
        )
        run = run_retrieval_evaluation(
            dataset,
            state.retriever,
            settings,
            manifest=state.manifest,
            strategy=(RetrievalStrategy(request.strategy) if request.strategy else None),
            run_id=run_id,
        )
        report = write_run(settings.runs_path / run.run_id, run)
        record.update(
            status="completed",
            dataset=run.dataset,
            strategy=run.config["strategy"],
            metrics=run.metrics,
            report_path=str(report),
        )
    except MLSCError as exc:
        record.update(status="failed", error=exc.detail)
    except Exception as exc:  # a crashed job must be visible, not silently stuck
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")


@router.post("/runs", response_model=EvalRunSummary, status_code=status.HTTP_202_ACCEPTED)
def start_run(
    request: EvalRunRequest,
    state: StateDep,
    background: BackgroundTasks,
    response: Response,
) -> EvalRunSummary:
    run_id = f"api-{uuid.uuid4().hex[:12]}"
    record: dict[str, Any] = {"run_id": run_id, "status": "queued"}
    state.eval_runs[run_id] = record
    response.headers["Location"] = f"/v1/eval/runs/{run_id}"
    background.add_task(_execute, state, run_id, request)
    return EvalRunSummary(**record)


@router.get("/runs", response_model=list[EvalRunSummary])
def list_runs(state: StateDep) -> list[EvalRunSummary]:
    return [EvalRunSummary(**record) for record in state.eval_runs.values()]


@router.get("/runs/{run_id}", response_model=EvalRunSummary)
def get_run(run_id: str, state: StateDep) -> EvalRunSummary:
    record = state.eval_runs.get(run_id)
    if record is None:
        raise DocumentNotFoundError(
            f"No evaluation run {run_id!r}. Runs are held in memory and do not survive "
            "a restart; completed reports are on disk under evaluation/runs/."
        )
    return EvalRunSummary(**record)
