"""Agent Orchestration — Model Registry.

Tracks every model the agent layer may use, loaded from models.yaml.
Provides:

  - ModelRecord           : Typed descriptor for one model entry.
  - AgentModelRegistry    : Reads models.yaml, checks local disk, and
                            surfaces only the models that are actually
                            installed and enabled.

Design decisions:
  - The agent does NOT load LLMs itself.  For the "rag" role the agent
    delegates entirely to Member 1's RAGPipeline.answer().  This registry
    is purely a catalog / health-check layer.
  - Placeholder models (enabled: false) are recorded so operators can
    see what is planned without crashing at import time.
  - Config path can be overridden via the AGENT_MODELS_YAML env-var or
    the constructor argument, enabling easy testing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Default config location ────────────────────────────────────────────────
_DEFAULT_YAML = Path(__file__).parent / "models.yaml"


class ModelRecord:
    """Typed, immutable descriptor for one entry in models.yaml."""

    __slots__ = (
        "role",
        "hf_repo_id",
        "local_path",
        "family",
        "context_window",
        "quantization",
        "enabled",
        "description",
        "name",
        "model_type",
        "capabilities",
        "supported_tasks",
        "runtime",
        "device_requirements",
        "priority",
        "fallback_for",
        "_installed",
    )

    def __init__(self, data: Dict[str, Any], project_root: Path) -> None:
        self.role: str = data["role"]
        self.hf_repo_id: str = data["hf_repo_id"]
        self.local_path: Path = project_root / data["local_path"]
        self.family: str = data.get("family", "unknown")
        self.context_window: int = int(data.get("context_window", 4096))
        self.quantization: str = data.get("quantization", "fp16")
        self.enabled: bool = bool(data.get("enabled", False))
        self.description: str = str(data.get("description", "")).strip()
        # These fields are intentionally data-driven: adding a model must not
        # require edits to the agent or planner.
        self.name: str = str(data.get("name", self.hf_repo_id))
        self.model_type: str = str(data.get("model_type", self.role))
        self.capabilities: List[str] = list(data.get("capabilities", [self.role]))
        self.supported_tasks: List[str] = list(data.get("supported_tasks", []))
        self.runtime: str = str(data.get("runtime", "transformers"))
        self.device_requirements: Dict[str, Any] = dict(data.get("device_requirements", {}))
        self.priority: int = int(data.get("priority", 100))
        self.fallback_for: List[str] = list(data.get("fallback_for", []))
        self._installed: Optional[bool] = None  # computed lazily

    # ------------------------------------------------------------------
    # Disk-presence check
    # ------------------------------------------------------------------

    def is_installed(self) -> bool:
        """Return True if config.json + at least one weight file exist on disk (or package entrypoint for processors)."""
        if self._installed is not None:
            return self._installed

        # Modular processor check (e.g. member3_ocr)
        if (self.local_path / "core" / "multimodal_processor.py").exists() or (self.local_path / "__init__.py").exists():
            if self.role in ("vision", "ocr") or self.family == "multimodal_ocr":
                self._installed = True
                return True

        # Diffusion model check (e.g. Stable Diffusion)
        if (self.local_path / "model_index.json").exists() and (self.local_path / "unet").exists():
            if self.role in ("image_generation", "diffusion") or "image_generation" in self.capabilities or self.family == "stable_diffusion":
                self._installed = True
                return True

        config_ok = (self.local_path / "config.json").exists()
        weights = (
            list(self.local_path.glob("*.safetensors"))
            + [f for f in self.local_path.glob("pytorch_model*.bin")
               if f.name != "training_args.bin"]
            + list(self.local_path.glob("*.index.json"))
            + (list((self.local_path / "onnx").glob("*.onnx"))
               if (self.local_path / "onnx").exists() else [])
        )
        self._installed = config_ok and len(weights) > 0
        return self._installed

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        """Human-readable status string for logging / CLI display."""
        if not self.enabled:
            return "disabled"
        if not self.is_installed():
            return "not_installed"
        return "ready"

    def __repr__(self) -> str:
        return (
            f"ModelRecord(role={self.role!r}, "
            f"hf_repo_id={self.hf_repo_id!r}, "
            f"status={self.status!r})"
        )


class AgentModelRegistry:
    """Config-driven catalog of all models available to the agent layer.

    Instantiation:
        registry = AgentModelRegistry()          # uses agent/models.yaml
        registry = AgentModelRegistry("path/to/models.yaml")

    Environment override:
        AGENT_MODELS_YAML=/path/to/models.yaml python -m agent ...
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        env_path = os.environ.get("AGENT_MODELS_YAML")
        resolved = Path(env_path) if env_path else (
            Path(config_path) if config_path else _DEFAULT_YAML
        )

        if not resolved.exists():
            raise FileNotFoundError(
                f"AgentModelRegistry: models.yaml not found at {resolved}. "
                "Set AGENT_MODELS_YAML or pass the path explicitly."
            )

        self._config_path = resolved
        self._project_root = _find_project_root()
        self._records: List[ModelRecord] = []
        self._embedding_info: Dict[str, Any] = {}
        self._load(resolved)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Parse YAML and populate _records."""
        try:
            import yaml  # pyyaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for AgentModelRegistry. "
                "Run: pip install pyyaml"
            ) from exc

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        for entry in raw.get("models", []):
            record = ModelRecord(entry, self._project_root)
            self._records.append(record)
            _status = record.status
            _icon = "[+]" if _status == "ready" else ("[-]" if _status == "disabled" else "[!]")
            logger.debug(
                "[AgentModelRegistry] %s %-8s %-10s %s",
                _icon,
                record.role,
                _status,
                record.hf_repo_id,
            )
            if _status == "not_installed":
                logger.warning(
                    "[AgentModelRegistry] Model %r is enabled but weights not found at %s. "
                    "Run download_llm_models.py to fetch it.",
                    record.hf_repo_id,
                    record.local_path,
                )

        self._embedding_info = raw.get("embedding", {})

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def all(self) -> List[ModelRecord]:
        """Return every model record (enabled and disabled)."""
        return list(self._records)

    def enabled(self) -> List[ModelRecord]:
        """Return records that are enabled in config."""
        return [r for r in self._records if r.enabled]

    def ready(self) -> List[ModelRecord]:
        """Return records that are enabled AND installed on disk."""
        return [r for r in self._records if r.status == "ready"]

    def by_role(self, role: str) -> List[ModelRecord]:
        """Return all records with the given role (enabled or not)."""
        return [r for r in self._records if r.role == role]

    def ready_for_role(self, role: str) -> List[ModelRecord]:
        """Return installed + enabled records for a given role."""
        return sorted(
            [r for r in self._records if r.role == role and r.status == "ready"],
            key=lambda r: r.priority,
        )

    def ready_for_capability(self, capability: str) -> List[ModelRecord]:
        """Return ready models advertising a capability, ordered by priority."""
        return sorted(
            [r for r in self._records if r.status == "ready" and capability in r.capabilities],
            key=lambda r: r.priority,
        )

    def primary_rag_model(self) -> Optional[ModelRecord]:
        """Return the first ready RAG-role model, or None if none installed."""
        candidates = self.ready_for_role("rag")
        return candidates[0] if candidates else None

    def get(self, hf_repo_id: str) -> Optional[ModelRecord]:
        """Retrieve a record by its full HF repo ID."""
        for r in self._records:
            if r.hf_repo_id == hf_repo_id:
                return r
        return None

    # ------------------------------------------------------------------
    # Health / audit
    # ------------------------------------------------------------------

    def health_report(self) -> Dict[str, Any]:
        """Return a structured health summary for logging or /status endpoints."""
        return {
            "config_path": str(self._config_path),
            "total_models": len(self._records),
            "ready": len(self.ready()),
            "disabled": len([r for r in self._records if not r.enabled]),
            "not_installed": len([r for r in self._records if r.status == "not_installed"]),
            "embedding_model": self._embedding_info.get("hf_repo_id", "unknown"),
            "models": [
                {
                    "role": r.role,
                    "hf_repo_id": r.hf_repo_id,
                    "status": r.status,
                    "local_path": str(r.local_path),
                    "model_type": r.model_type,
                    "capabilities": r.capabilities,
                    "supported_tasks": r.supported_tasks,
                    "runtime": r.runtime,
                    "quantization": r.quantization,
                    "device_requirements": r.device_requirements,
                    "priority": r.priority,
                    "fallback_for": r.fallback_for,
                }
                for r in self._records
            ],
        }

    def __repr__(self) -> str:
        ready = len(self.ready())
        total = len(self._records)
        return f"AgentModelRegistry(ready={ready}/{total}, config={self._config_path.name!r})"


# ── Helpers ───────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    """Walk upward from this file to locate the project root (has rag_engine/)."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "rag_engine").exists():
            return current
        current = current.parent
    # fallback: CWD
    return Path.cwd()


# ── Module-level singleton (lazy) ─────────────────────────────────────────

_registry: Optional[AgentModelRegistry] = None


def get_agent_registry(config_path: Optional[str | Path] = None) -> AgentModelRegistry:
    """Return the shared AgentModelRegistry singleton (created on first call)."""
    global _registry
    if _registry is None:
        _registry = AgentModelRegistry(config_path)
    return _registry


# ── Self-test / __main__ ──────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("Agent Model Registry — Self-Test")
    print("=" * 60)

    registry = AgentModelRegistry()
    print(repr(registry))
    print()

    report = registry.health_report()
    print(json.dumps(report, indent=2))
    print()

    primary = registry.primary_rag_model()
    if primary:
        print(f"Primary RAG model  : {primary.hf_repo_id}")
        print(f"  Local path       : {primary.local_path}")
        print(f"  Context window   : {primary.context_window:,} tokens")
        print(f"  Family           : {primary.family}")
        print(f"  Status           : {primary.status}")
    else:
        print("WARNING: No ready RAG model found.")

    print()
    print("All model statuses:")
    for r in registry.all():
        icon = "[+]" if r.status == "ready" else ("[-]" if r.status == "disabled" else "[!]")
        print(f"  {icon}  [{r.role:<10}] {r.hf_repo_id:<45}  {r.status}")
