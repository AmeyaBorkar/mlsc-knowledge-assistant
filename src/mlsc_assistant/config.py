"""Configuration.

Layered, in increasing order of precedence:

1. defaults declared on the models below
2. ``config.yaml`` at the repository root
3. environment variables / ``.env``, prefixed ``MLSC_`` with ``__`` for nesting
   (``MLSC_RETRIEVAL__TOP_K=8``)

Secrets never live in ``config.yaml``. API keys are read from their conventional
environment variable names (``GOOGLE_API_KEY`` and friends) so that a key already
exported for another tool just works.

Note the two different jobs ``.env`` does here. pydantic-settings reads it to populate
``MLSC_``-prefixed *settings fields*, but provider SDKs read ``os.environ`` directly and
know nothing about this class. So ``.env`` is also loaded into the process environment
at import, without overriding anything already exported — otherwise a key pasted into
``.env`` would populate no field, reach no SDK, and silently look like no key at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

# override=False: an explicitly exported variable beats the file, which is the
# convention operators expect and what makes CI (no .env, real secrets) behave.
load_dotenv(REPO_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class KnowledgeBaseConfig(BaseModel):
    path: Path = Path("data/knowledge_base")
    glob: str = "*.txt"


class ChunkingConfig(BaseModel):
    strategy: Literal["structural", "fixed"] = "structural"
    version: str = "structural-v1"
    min_tokens: int = Field(40, ge=1)
    max_tokens: int = Field(320, ge=32)
    keep_lists_atomic: bool = True
    prepend_doc_title: bool = True

    @field_validator("max_tokens")
    @classmethod
    def _max_above_min(cls, v: int, info: Any) -> int:
        min_tokens = info.data.get("min_tokens")
        if min_tokens is not None and v <= min_tokens:
            raise ValueError("max_tokens must exceed min_tokens")
        return v


class EmbeddingConfig(BaseModel):
    backend: Literal["fastembed", "sbert"] = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    batch_size: int = 32
    cache_dir: Path = Path("data/cache/embeddings")
    models_dir: Path = Path("data/cache/models")
    """Where the ONNX model itself is cached. Kept inside the repo rather than the
    system temp directory so a cleared temp folder does not silently re-download."""


class StoreConfig(BaseModel):
    backend: Literal["numpy", "chroma"] = "numpy"
    path: Path = Path("data/index")


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75
    index_title: bool = True
    """Whether BM25 indexes the document-title prefix alongside the chunk text."""

    max_document_frequency: float = Field(0.5, gt=0.0, le=1.0)
    """Query terms appearing in more than this share of chunks are dropped as
    non-discriminative. At 1.0 the filter is disabled, which is the ablation."""


class RerankConfig(BaseModel):
    enabled: bool = False
    backend: Literal["llm", "cross_encoder"] = "llm"


class RetrievalConfig(BaseModel):
    strategy: Literal["hybrid", "dense", "lexical"] = "hybrid"
    top_k: int = Field(6, ge=1, le=50)
    candidate_k: int = Field(15, ge=1, le=200)
    rrf_k: int = Field(60, ge=1)
    max_chunks_per_document: int = Field(3, ge=1)
    bm25: BM25Config = BM25Config()
    rerank: RerankConfig = RerankConfig()

    @field_validator("candidate_k")
    @classmethod
    def _candidates_cover_top_k(cls, v: int, info: Any) -> int:
        top_k = info.data.get("top_k")
        if top_k is not None and v < top_k:
            raise ValueError("candidate_k must be >= top_k, or fusion has nothing to rank")
        return v


class AbstentionConfig(BaseModel):
    """Gate thresholds.

    ``min_dense_score`` is calibrated, not hand-picked. ``mlsc eval`` sweeps it and
    reports the abstention precision/recall curve; the committed 0.55 is the highest
    threshold that refuses no answerable question on the dev set.

    The measured curve settles a design question: near-miss unanswerables score 0.71 to
    0.78, inside the answerable range of 0.67 to 0.90. A threshold high enough to catch
    them refuses 39% of answerable questions, and one catching all of them refuses 71%.
    Gate 1 can remove off-domain noise and nothing else — gate 2 is a necessity rather
    than a preference. Full curve in docs/EVALUATION.md.
    """

    min_dense_score: float = Field(0.35, ge=0.0, le=1.0)
    min_score_margin: float = Field(0.0, ge=0.0)
    calibrated: bool = True
    """Set true only once the committed threshold came from a real sweep."""

    require_sufficient_context: bool = True
    verify_faithfulness: bool = False


class LLMConfig(BaseModel):
    provider: Literal["gemini", "anthropic", "openai", "groq", "ollama"] = "gemini"
    model: str | None = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_output_tokens: int = 800
    timeout_s: float = 30.0
    max_retries: int = 3
    models: dict[str, str] = Field(default_factory=dict)

    requests_per_minute: float | None = 15.0
    """Client-side pacing. The Gemini free tier enforces both a per-minute and a per-day
    quota; unpaced, an evaluation run exhausts the per-minute allowance in seconds and
    then spends itself in backoff. Set to null on a paid tier."""

    thinking_budget: int | None = 0
    """Gemini reasoning-token budget. 0 disables thinking, None leaves the model default.

    Measured in Phase 4: on a grounded abstention the verdict was identical with and
    without thinking, but took 39s versus 1.18s. Grounded extraction is not a reasoning
    task, and a 40-question evaluation run repeated per ablation makes that a 26-minute
    versus 1-minute difference. Kept configurable so the assumption stays testable.
    """

    def resolved_model(self) -> str:
        """Explicit model wins; otherwise fall back to the per-provider default."""
        if self.model:
            return self.model
        try:
            return self.models[self.provider]
        except KeyError as exc:  # pragma: no cover - guarded by config.yaml
            raise ValueError(f"No model configured for provider {self.provider!r}") from exc

    def api_key(self) -> str | None:
        env_var = {
            "gemini": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "ollama": None,
        }[self.provider]
        return None if env_var is None else os.environ.get(env_var) or None

    @property
    def requires_api_key(self) -> bool:
        return self.provider != "ollama"

    @property
    def is_configured(self) -> bool:
        """Whether generation can run. Never echoes the key itself."""
        return not self.requires_api_key or bool(self.api_key())


class GenerationConfig(BaseModel):
    prompt_version: str = "grounded-v1"
    max_context_chunks: int = 6
    include_doc_titles: bool = True


class JudgeConfig(BaseModel):
    provider: Literal["gemini", "anthropic", "openai", "groq", "ollama"] = "gemini"
    model: str | None = None
    temperature: float = 0.0
    cache: bool = True


class EvaluationConfig(BaseModel):
    dataset_dir: Path = Path("evaluation/datasets")
    runs_dir: Path = Path("evaluation/runs")
    default_dataset: str = "dev_set"
    k_values: list[int] = Field(default_factory=lambda: [3, 5, 6, 10])
    judge: JudgeConfig = JudgeConfig()
    relevancy_questions: int = 3


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])
    max_question_length: int = 1000
    serve_web_ui: bool = True


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLSC_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    knowledge_base: KnowledgeBaseConfig = KnowledgeBaseConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    store: StoreConfig = StoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    abstention: AbstentionConfig = AbstentionConfig()
    llm: LLMConfig = LLMConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    api: APIConfig = APIConfig()
    log_level: str = "INFO"

    repo_root: Path = REPO_ROOT

    # -- path helpers --------------------------------------------------------
    # Relative paths in config.yaml resolve against the repo root, not the current
    # working directory, so `mlsc index` behaves the same from any directory.

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.repo_root / path)

    @property
    def kb_path(self) -> Path:
        return self.resolve(self.knowledge_base.path)

    @property
    def index_path(self) -> Path:
        return self.resolve(self.store.path)

    @property
    def embedding_cache_path(self) -> Path:
        return self.resolve(self.embedding.cache_dir)

    @property
    def models_path(self) -> Path:
        return self.resolve(self.embedding.models_dir)

    @property
    def dataset_path(self) -> Path:
        return self.resolve(self.evaluation.dataset_dir)

    @property
    def runs_path(self) -> Path:
        return self.resolve(self.evaluation.runs_dir)

    # -- sources -------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - pydantic's signature
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence runs left to right. Environment beats .env beats YAML beats
        # defaults, so an operator can always override a committed value.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
        )


class _YamlSource(PydanticBaseSettingsSource):
    """Reads ``config.yaml`` as the lowest-precedence explicit source."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path | None = None) -> None:
        super().__init__(settings_cls)
        self.path = path or Path(os.environ.get("MLSC_CONFIG_FILE", DEFAULT_CONFIG_PATH))

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: ARG002
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        with self.path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise TypeError(f"{self.path} must contain a YAML mapping at the top level")
        return data


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Process-wide settings singleton.

    Cached because the API resolves it per request and re-reading YAML each time would
    be wasteful; ``reload=True`` exists for tests.
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings()
    return _settings
