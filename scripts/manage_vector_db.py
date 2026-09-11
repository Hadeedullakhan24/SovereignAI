"""Enterprise Vector Database Administration CLI for MRPL Sovereign AI Workbench.

Usage:
    python scripts/manage_vector_db.py status
    python scripts/manage_vector_db.py health
    python scripts/manage_vector_db.py list-collections
    python scripts/manage_vector_db.py stats [--collection NAME]
    python scripts/manage_vector_db.py optimize [--collection NAME]
    python scripts/manage_vector_db.py vacuum [--collection NAME]
    python scripts/manage_vector_db.py backup [--snapshot-name NAME]
    python scripts/manage_vector_db.py restore --archive PATH [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.vector_store import (
    CollectionManager,
    CollectionVersionManager,
    QdrantVectorStore,
    SnapshotManager,
    StorageStatsCollector,
    VectorHealthMonitor,
    VectorOptimizer,
    VectorStoreConfig,
)


def _get_store(storage_path: str = "vector_db/qdrant") -> QdrantVectorStore:
    p = Path(storage_path)
    config = VectorStoreConfig(
        backend="qdrant",
        storage_path=p,
        journal_path=p / "journal",
        backup_path=p / "backups",
        telemetry_path=p / "telemetry",
    )
    return QdrantVectorStore(config)


def cmd_status(args: argparse.Namespace) -> int:
    """Print high-level status of the vector database."""
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — VECTOR DATABASE STATUS")
    print("=" * 78)

    store = _get_store(args.storage_path)
    collections = store.list_collections()

    total_points = 0
    print(f"\nStorage Path       : {Path(args.storage_path).resolve()}")
    print(f"Total Collections  : {len(collections)}")
    print("-" * 78)
    print(f"{'Collection Name':<35} {'Points':<10} {'Vectors':<10} {'Status':<15}")
    print("-" * 78)

    for col in collections:
        try:
            stats = store.get_collection_stats(col)
            points = getattr(stats, "points_count", getattr(stats, "total_points", 0))
            vectors = getattr(stats, "vector_count", getattr(stats, "total_vectors", points))
            total_points += points
            status_str = stats.status
            print(f"{col:<35} {points:<10} {vectors:<10} {status_str:<15}")
        except Exception as e:
            print(f"{col:<35} {'ERR':<10} {'ERR':<10} {str(e)[:20]:<15}")

    print("-" * 78)
    print(f"Total Indexed Chunks: {total_points}")

    # Version Manager status
    version_mgr = CollectionVersionManager(store)
    active_versions = version_mgr.get_all_active_versions()
    if active_versions:
        print("\nActive Domain Versions:")
        for domain, col_name in active_versions.items():
            print(f"  • {domain:<20} -> {col_name}")

    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Run comprehensive canary diagnostics and report health status."""
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — VECTOR DATABASE HEALTH CHECK")
    print("=" * 78)

    store = _get_store(args.storage_path)
    monitor = VectorHealthMonitor(store)

    report = monitor.run_diagnostics()

    print(f"\nStatus                 : {report.status.upper()}")
    print(f"Backend Type           : {report.backend_type}")
    print(f"Air-Gapped             : {'YES' if report.is_airgapped else 'NO'}")
    print(f"Canary Write Latency   : {report.canary_write_latency_ms:.2f} ms")
    print(f"Canary Read Latency    : {report.canary_read_latency_ms:.2f} ms")
    print(f"Total Collections      : {report.total_collections}")
    print(f"Total Vectors          : {report.total_vectors}")
    print(f"Free Disk Space        : {report.disk_free_bytes / (1024 * 1024 * 1024):.2f} GB")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  [!] {w}")

    if report.status != "HEALTHY":
        print("\n[WARNING] Health check reported degraded or non-healthy state.")
        return 1
    else:
        print("\n[SUCCESS] Vector database is operational and healthy.")
        return 0


def cmd_list_collections(args: argparse.Namespace) -> int:
    """List registered domain collections and versions."""
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — COLLECTION CATALOG")
    print("=" * 78)

    store = _get_store(args.storage_path)
    col_mgr = CollectionManager(store)
    version_mgr = CollectionVersionManager(store)

    catalog = col_mgr.get_default_catalog()
    active_versions = version_mgr.get_all_active_versions()
    deployed_collections = set(store.list_collections())

    print(f"\nTotal Catalog Domains: {len(catalog)}")
    print("-" * 78)
    print(f"{'Collection Name':<32} {'Dimension':<10} {'Distance':<10} {'Active':<8} {'Deployed'}")
    print("-" * 78)

    for config in catalog:
        dim_str = str(config.vector_size)
        dist_str = config.distance.value
        is_active = any(act == config.name for act in active_versions.values())
        active_str = "YES" if is_active else "NO"
        dep_str = "YES" if config.name in deployed_collections else "NO"
        print(f"{config.name:<32} {dim_str:<10} {dist_str:<10} {active_str:<8} {dep_str}")

    print("-" * 78)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Display storage and indexing statistics."""
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — STORAGE STATISTICS")
    print("=" * 78)

    store = _get_store(args.storage_path)
    collector = StorageStatsCollector(store)

    stats = collector.collect_stats()

    print(f"\nTotal Disk Footprint : {stats.total_disk_bytes / (1024 * 1024):.2f} MB")
    print(f"Total Collections    : {stats.total_collections}")
    print(f"Total Vectors        : {stats.total_vectors}")
    print(f"Avg Payload Size     : {stats.average_payload_size_bytes:.1f} bytes")

    if stats.collections_detail:
        print("\nCollection Breakdown:")
        print("-" * 78)
        print(f"{'Collection':<35} {'Points':<10} {'Vectors':<10} {'Segments'}")
        print("-" * 78)
        for col_name, cstats in stats.collections_detail.items():
            if args.collection and args.collection != col_name:
                continue
            print(f"{col_name:<35} {cstats.points_count:<10} {cstats.vector_count:<10} {cstats.segments_count}")
        print("-" * 78)

    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Trigger segment optimization and compaction."""
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — VECTOR OPTIMIZATION")
    print("=" * 78)

    store = _get_store(args.storage_path)
    optimizer = VectorOptimizer(store)

    if args.collection:
        collections = [args.collection]
    else:
        collections = store.list_collections()

    if not collections:
        print("No collections found to optimize.")
        return 0

    print(f"\nOptimizing {len(collections)} collection(s)...")
    t0 = time.perf_counter()

    for col in collections:
        print(f"\n  Optimizing '{col}'...")
        report = optimizer.optimize_all(col)
        print(f"    • Tasks Executed  : {len(report.tasks_executed)}")
        print(f"    • Reclaimed Bytes : {report.total_reclaimed_bytes} bytes")
        print(f"    • Duration        : {report.total_duration_ms:.2f} ms")

    elapsed = time.perf_counter() - t0
    print(f"\n[SUCCESS] Optimization completed in {elapsed:.2f}s.")
    return 0


def cmd_vacuum(args: argparse.Namespace) -> int:
    """Run segment merge and disk vacuum."""
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — STORAGE VACUUM")
    print("=" * 78)

    store = _get_store(args.storage_path)
    optimizer = VectorOptimizer(store)

    if args.collection:
        collections = [args.collection]
    else:
        collections = store.list_collections()

    for col in collections:
        print(f"Vacuuming '{col}'...")
        metric = optimizer.vacuum(col)
        print(f"  Reclaimed space: {metric.reclaimed_bytes} bytes ({metric.duration_ms:.2f} ms).")

    print("\n[SUCCESS] Vacuum completed.")
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    """Create a verified snapshot tar.gz archive of the vector database."""
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — VECTOR DATABASE BACKUP")
    print("=" * 78)

    store = _get_store(args.storage_path)
    snapshot_mgr = SnapshotManager(store)

    collections = [args.collection] if args.collection else store.list_collections()
    if not collections:
        print("No collections found to back up.")
        return 0

    for col in collections:
        print(f"\nCreating snapshot for collection '{col}'...")
        t0 = time.perf_counter()
        archive_path = snapshot_mgr.create_snapshot(col, snapshot_name=args.snapshot_name)
        elapsed = time.perf_counter() - t0

        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"  [SUCCESS] Backup archive created:")
        print(f"    • Archive Path : {archive_path}")
        print(f"    • Archive Size : {size_mb:.2f} MB")
        print(f"    • Duration     : {elapsed:.2f} s")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    """Restore a snapshot archive into the storage directory."""
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — VECTOR DATABASE RESTORE")
    print("=" * 78)

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"[ERROR] Archive file does not exist: {archive_path}")
        return 1

    collection_name = args.collection or archive_path.stem.split("_")[0]

    store = _get_store(args.storage_path)
    snapshot_mgr = SnapshotManager(store)

    if not args.force:
        confirm = input(f"Are you sure you want to restore '{archive_path.name}' into collection '{collection_name}'? [y/N]: ")
        if confirm.lower() != "y":
            print("Restore aborted by user.")
            return 0

    print(f"\nRestoring archive '{archive_path}' for collection '{collection_name}'...")
    t0 = time.perf_counter()
    snapshot_mgr.restore_snapshot(collection_name, archive_path)
    elapsed = time.perf_counter() - t0

    print(f"\n[SUCCESS] Vector collection '{collection_name}' restored successfully in {elapsed:.2f}s.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MRPL Sovereign RAG Engine — Vector Database Administration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--storage-path",
        default="vector_db/qdrant",
        help="Path to the Qdrant local storage directory.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # status
    subparsers.add_parser("status", help="Show vector store status and collection counts")

    # health
    subparsers.add_parser("health", help="Run canary test and diagnostics")

    # list-collections
    subparsers.add_parser("list-collections", help="List registered domain collections and active versions")

    # stats
    p_stats = subparsers.add_parser("stats", help="Display storage and telemetry statistics")
    p_stats.add_argument("--collection", default=None, help="Optional collection name to inspect")

    # optimize
    p_opt = subparsers.add_parser("optimize", help="Run segment optimization and compaction")
    p_opt.add_argument("--collection", default=None, help="Optional collection name to optimize")

    # vacuum
    p_vac = subparsers.add_parser("vacuum", help="Run vacuum and space reclamation")
    p_vac.add_argument("--collection", default=None, help="Optional collection name to vacuum")

    # backup
    p_bak = subparsers.add_parser("backup", help="Create snapshot archive")
    p_bak.add_argument("--collection", default=None, help="Optional specific collection to back up")
    p_bak.add_argument("--snapshot-name", default=None, help="Optional custom snapshot name")

    # restore
    p_res = subparsers.add_parser("restore", help="Restore database from snapshot archive")
    p_res.add_argument("--archive", required=True, help="Path to the tar.gz snapshot archive")
    p_res.add_argument("--collection", default=None, help="Target collection name for restoration")
    p_res.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "status": cmd_status,
        "health": cmd_health,
        "list-collections": cmd_list_collections,
        "stats": cmd_stats,
        "optimize": cmd_optimize,
        "vacuum": cmd_vacuum,
        "backup": cmd_backup,
        "restore": cmd_restore,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        return cmd_fn(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
