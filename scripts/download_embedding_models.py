"""Model Download and Integrity Verification Utility for Milestone 6.

Downloads Hugging Face embedding models to local disk once, enabling 100% air-gapped,
offline execution thereafter.
Supported models:
  - BAAI/bge-small-en-v1.5 -> models/embeddings/bge-small-en-v1.5/
  - BAAI/bge-base-en-v1.5  -> models/embeddings/bge-base-en-v1.5/
  - intfloat/e5-small-v2   -> models/embeddings/e5-small-v2/
  - intfloat/e5-base-v2    -> models/embeddings/e5-base-v2/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_downloader")

TARGET_MODELS: dict[str, dict[str, Any]] = {
    "BAAI/bge-small-en-v1.5": {
        "folder": "bge-small-en-v1.5",
        "dimension": 384,
        "description": "BGE Small English v1.5 (Fast, 384 dims)",
    },
    "BAAI/bge-base-en-v1.5": {
        "folder": "bge-base-en-v1.5",
        "dimension": 768,
        "description": "BGE Base English v1.5 (High accuracy, 768 dims)",
    },
    "intfloat/e5-small-v2": {
        "folder": "e5-small-v2",
        "dimension": 384,
        "description": "E5 Small v2 (Fast, 384 dims, prefix-based)",
    },
    "intfloat/e5-base-v2": {
        "folder": "e5-base-v2",
        "dimension": 768,
        "description": "E5 Base v2 (High accuracy, 768 dims, prefix-based)",
    },
}


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_model_integrity(model_dir: Path, expected_dimension: int) -> dict[str, Any]:
    """Verify that all required model assets exist, weights load offline, and dimensions match."""
    if not model_dir.exists():
        return {"valid": False, "error": f"Directory does not exist: {model_dir}"}

    # 1. Check config
    config_file = model_dir / "config.json"
    if not config_file.exists():
        return {"valid": False, "error": f"config.json missing in {model_dir}"}

    # 2. Check tokenizer files
    has_tokenizer = (
        (model_dir / "tokenizer.json").exists()
        or (model_dir / "vocab.txt").exists()
        or (model_dir / "tokenizer_config.json").exists()
    )
    if not has_tokenizer:
        return {"valid": False, "error": f"Tokenizer files missing in {model_dir}"}

    # 3. Check model weights
    weights_exist = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "pytorch_model.bin").exists()
    )
    if not weights_exist:
        return {"valid": False, "error": f"Model weights missing in {model_dir}"}

    # 4. Check offline load and dimension
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(model_dir), local_files_only=True, device="cpu")
        actual_dim = model.get_sentence_embedding_dimension()
        if actual_dim != expected_dimension:
            return {
                "valid": False,
                "error": f"Dimension mismatch: expected {expected_dimension}, got {actual_dim}",
            }
        test_vec = model.encode("Air-gapped verification test", normalize_embeddings=True)
        if len(test_vec) != expected_dimension:
            return {
                "valid": False,
                "error": f"Test vector length {len(test_vec)} != {expected_dimension}",
            }
    except Exception as e:
        return {"valid": False, "error": f"Offline loading test failed: {e}"}

    # 5. Compute checksums of key configuration and weight files
    file_checksums = {}
    for fname in ["config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin"]:
        p = model_dir / fname
        if p.exists():
            file_checksums[fname] = compute_file_sha256(p)

    return {
        "valid": True,
        "dimension": expected_dimension,
        "file_checksums": file_checksums,
    }


def download_model(
    repo_id: str,
    output_dir: Path,
    expected_dimension: int,
    force: bool = False,
) -> bool:
    """Download model repository snapshot into destination directory."""
    dest_dir = output_dir / TARGET_MODELS[repo_id]["folder"]

    if dest_dir.exists() and not force:
        logger.info("Checking existing model at %s...", dest_dir)
        check = verify_model_integrity(dest_dir, expected_dimension)
        if check["valid"]:
            logger.info("Model %s already exists and passed integrity verification. Skipping.", repo_id)
            return True
        logger.warning("Existing model failed integrity check (%s). Re-downloading...", check.get("error"))

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s...", repo_id, dest_dir)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax_model.msgpack", "tf_model.h5"],
        )
    except Exception as e:
        logger.error("Failed to download %s: %s", repo_id, e)
        return False

    # Verify downloaded files
    verification = verify_model_integrity(dest_dir, expected_dimension)
    if not verification["valid"]:
        logger.error("Downloaded model %s failed verification: %s", repo_id, verification.get("error"))
        return False

    # Record integrity metadata
    integrity_path = dest_dir / "model_integrity.json"
    with open(integrity_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)

    logger.info("Successfully downloaded and verified %s (dim=%d)", repo_id, expected_dimension)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download embedding models for Sovereign RAG Engine.")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/embeddings",
        help="Base directory to store models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "bge-small", "bge-base", "e5-small", "e5-base"] + list(TARGET_MODELS.keys()),
        help="Specific model to download or 'all'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already present.",
    )
    args = parser.parse_args()

    base_dir = Path(args.models_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        targets = list(TARGET_MODELS.keys())
    elif args.model in ("bge-small", "bge-small-en-v1.5"):
        targets = ["BAAI/bge-small-en-v1.5"]
    elif args.model in ("bge-base", "bge-base-en-v1.5"):
        targets = ["BAAI/bge-base-en-v1.5"]
    elif args.model in ("e5-small", "e5-small-v2"):
        targets = ["intfloat/e5-small-v2"]
    elif args.model in ("e5-base", "e5-base-v2"):
        targets = ["intfloat/e5-base-v2"]
    else:
        targets = [args.model]

    success = True
    for repo_id in targets:
        spec = TARGET_MODELS[repo_id]
        ok = download_model(
            repo_id=repo_id,
            output_dir=base_dir,
            expected_dimension=spec["dimension"],
            force=args.force,
        )
        if not ok:
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
