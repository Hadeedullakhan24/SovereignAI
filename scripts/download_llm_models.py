"""Model Download, Integrity Verification, and Offline Loading Utility for Milestone 9 (Local LLMs).

Downloads open-weight Hugging Face LLM models to local disk once, enabling 100% air-gapped,
offline execution thereafter with strict `local_files_only=True` guarantees.

Supported open-weight models:
  - Qwen/Qwen2.5-1.5B-Instruct          -> models/llms/qwen2.5-1.5b-instruct/
  - HuggingFaceTB/SmolLM2-1.7B-Instruct -> models/llms/smollm2-1.7b-instruct/
  - microsoft/Phi-3.5-mini-instruct      -> models/llms/phi-3.5-mini-instruct/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Optional

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm_downloader")

TARGET_MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "folder": "qwen2.5-1.5b-instruct",
        "description": "Qwen 2.5 1.5B Instruct (Lightweight, High-Precision, 32k context)",
        "context_window": 32768,
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "folder": "smollm2-1.7b-instruct",
        "description": "SmolLM2 1.7B Instruct (Compact, Fast, 8k context)",
        "context_window": 8192,
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "folder": "phi-3.5-mini-instruct",
        "description": "Phi 3.5 Mini Instruct (3.8B, Reasoning-optimized)",
        "context_window": 131072,
    },
}


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_model_integrity(model_dir: Path) -> dict[str, Any]:
    """Verify that all required model assets exist on local disk and record checksums.
    
    Checks:
      1. config.json exists.
      2. Tokenizer assets exist (tokenizer.json or tokenizer_config.json or vocab.json).
      3. Transformers-compatible model weight files exist (.safetensors or pytorch_model*.bin).
      4. Computes SHA-256 checksums of core configuration and metadata files.
    """
    if not model_dir.exists():
        return {"valid": False, "error": f"Directory does not exist: {model_dir}"}

    # 1. Check config.json
    config_file = model_dir / "config.json"
    if not config_file.exists():
        return {"valid": False, "error": f"config.json missing in {model_dir}"}

    # 2. Check tokenizer files
    has_tokenizer = (
        (model_dir / "tokenizer.json").exists()
        or (model_dir / "tokenizer_config.json").exists()
        or (model_dir / "vocab.json").exists()
        or (model_dir / "vocab.txt").exists()
    )
    if not has_tokenizer:
        return {"valid": False, "error": f"Tokenizer files missing in {model_dir}"}

    # 3. Check Transformers-compatible model weights
    # Valid weights include *.safetensors (or sharded safetensors indexed via model.safetensors.index.json)
    # and pytorch_model*.bin (excluding training_args.bin). ONNX files are NOT valid for AutoModelForCausalLM.
    weights = [
        f for f in (list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model*.bin")))
        if f.name != "training_args.bin"
    ]
    if not weights:
        return {
            "valid": False,
            "error": f"No Transformers-compatible model weight files (.safetensors or pytorch_model*.bin) found in {model_dir}",
        }

    # 4. Compute checksums of key config files
    file_checksums: dict[str, str] = {}
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        p = model_dir / fname
        if p.exists():
            file_checksums[fname] = compute_file_sha256(p)

    return {
        "valid": True,
        "model_dir": str(model_dir.resolve()),
        "weights_count": len(weights),
        "file_checksums": file_checksums,
    }


def verify_offline_loading(model_dir: Path) -> dict[str, Any]:
    """Verify strictly offline model loading with local_files_only=True."""
    try:
        import gc
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        logger.info("Verifying offline tokenizer loading from %s (local_files_only=True)...", model_dir)
        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

        test_prompt = "Refinery safety SOP test verification."
        tokens = tokenizer.encode(test_prompt)
        if not tokens:
            return {"valid": False, "error": "Tokenizer produced empty token list"}

        logger.info("Verifying offline AutoModelForCausalLM loading from %s (local_files_only=True)...", model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        param_count = sum(p.numel() for p in model.parameters())

        # Release model from memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Offline verification successful. Model config type: %s, Vocab size: %d, Parameters: %d",
            config.model_type,
            len(tokenizer),
            param_count,
        )
        return {
            "valid": True,
            "model_type": config.model_type,
            "vocab_size": len(tokenizer),
            "parameters": param_count,
        }
    except Exception as e:
        logger.error("Offline loading verification failed: %s", e)
        return {"valid": False, "error": str(e)}


def download_model(
    repo_id: str,
    output_dir: Path,
    auth_token: Optional[str] = None,
    force: bool = False,
    verify_only: bool = False,
) -> bool:
    """Download model repository snapshot into destination directory and verify."""
    spec = TARGET_MODELS[repo_id]
    dest_dir = output_dir / spec["folder"]

    if verify_only:
        logger.info("Running verification only on %s...", dest_dir)
        integrity = verify_model_integrity(dest_dir)
        if not integrity["valid"]:
            logger.error("Integrity check failed: %s", integrity.get("error"))
            return False
        offline_check = verify_offline_loading(dest_dir)
        if not offline_check["valid"]:
            logger.error("Offline loading check failed: %s", offline_check.get("error"))
            return False
        logger.info("Verification passed for %s", repo_id)
        return True

    if dest_dir.exists() and not force:
        logger.info("Checking existing model at %s...", dest_dir)
        integrity = verify_model_integrity(dest_dir)
        if integrity["valid"]:
            logger.info("Model %s already exists and passed integrity verification. Skipping download.", repo_id)
            return True
        logger.warning("Existing model failed integrity check (%s). Re-downloading...", integrity.get("error"))

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading '%s' to '%s'...", repo_id, dest_dir)

    downloaded = False
    # Attempt 1: Hugging Face snapshot_download
    try:
        from huggingface_hub import snapshot_download

        logger.info("Attempting download via Hugging Face Hub...")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=auth_token,
            ignore_patterns=[
                "*.msgpack",
                "*.h5",
                "*.ot",
                "flax_model.msgpack",
                "tf_model.h5",
                "onnx/*",
                "runs/*",
                "*.tfevents*",
            ],
        )
        logger.info("Downloaded '%s' successfully from Hugging Face Hub to %s", repo_id, snapshot_path)
        downloaded = True
    except Exception as e:
        logger.warning("Hugging Face Hub download failed or unreachable (%s). Attempting ModelScope mirror fallback...", e)

    # Attempt 2: ModelScope mirror fallback
    if not downloaded:
        try:
            import importlib
            ms = importlib.import_module("modelscope")
            ms_snapshot_download = getattr(ms, "snapshot_download")

            logger.info("Downloading '%s' via ModelScope official mirror...", repo_id)
            snapshot_path = ms_snapshot_download(
                repo_id,
                local_dir=str(dest_dir),
            )
            logger.info("Downloaded '%s' successfully from ModelScope to %s", repo_id, snapshot_path)
            downloaded = True
        except Exception as e2:
            logger.error("Failed to download %s via both Hugging Face Hub and ModelScope: %s", repo_id, e2)
            return False

    # 1. Integrity Verification
    integrity = verify_model_integrity(dest_dir)
    if not integrity["valid"]:
        logger.error("Downloaded model %s failed integrity verification: %s", repo_id, integrity.get("error"))
        return False

    # 2. Offline Loading Verification
    offline_check = verify_offline_loading(dest_dir)
    if not offline_check["valid"]:
        logger.error("Downloaded model %s failed offline loading: %s", repo_id, offline_check.get("error"))
        return False

    # 3. Record model_integrity.json manifest
    integrity_record = {
        "repo_id": repo_id,
        "description": spec["description"],
        "context_window": spec.get("context_window", 4096),
        "integrity": integrity,
        "offline_verification": offline_check,
    }
    manifest_file = dest_dir / "model_integrity.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(integrity_record, f, indent=2)

    logger.info("Successfully downloaded, verified, and certified %s for 100%% offline execution.", repo_id)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download & Verify local LLM weights for MRPL Sovereign Workbench.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/llm"),
        help="Base directory to store models (default: models/llm).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "qwen", "smollm", "phi"] + list(TARGET_MODELS.keys()),
        help="Specific model to download or 'all'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already present.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing local models without downloading.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face authentication token.",
    )
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        targets = list(TARGET_MODELS.keys())
    elif args.model in ("qwen", "qwen2.5"):
        targets = ["Qwen/Qwen2.5-1.5B-Instruct"]
    elif args.model in ("smollm", "smollm2"):
        targets = ["HuggingFaceTB/SmolLM2-1.7B-Instruct"]
    elif args.model in ("phi", "phi-3.5"):
        targets = ["microsoft/Phi-3.5-mini-instruct"]
    else:
        targets = [args.model]

    success = True
    for repo_id in targets:
        print(f"\n{'='*70}")
        print(f"Target LLM: {repo_id}")
        print(f"{'='*70}")
        ok = download_model(
            repo_id=repo_id,
            output_dir=models_dir,
            auth_token=args.token,
            force=args.force,
            verify_only=args.verify_only,
        )
        if not ok:
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
