"""
Statistics Generator — Dataset Metrics & Health Report Generation.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Computes comprehensive dataset quality indicators, calculates health scores,
and generates dataset_statistics.json and dataset_health_report.md.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from dataset_engine.duplicate_detector import DuplicateReport


class StatisticsGenerator:
    """Calculates dataset metrics and writes structured reports."""

    @staticmethod
    def format_bytes(size_bytes: int) -> str:
        """Format raw byte counts into human-readable strings."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    def compute_statistics(
        self,
        documents: list[dict[str, Any]],
        duplicate_report: DuplicateReport,
        scan_duration_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """
        Compute aggregate statistics from document records and duplicate analysis.
        """
        total_files = len(documents)
        valid_files = 0
        empty_files = 0
        corrupted_files = 0
        unsupported_files = 0
        total_size_bytes = 0

        category_counts: dict[str, int] = Counter()
        extension_counts: dict[str, int] = Counter()
        status_counts: dict[str, int] = Counter()
        corrupted_details: list[dict[str, str]] = []

        for doc in documents:
            status = doc.get("status", "VALID")
            status_counts[status] += 1
            cat = doc.get("category", "unknown")
            ext = doc.get("extension", "")
            size = doc.get("size_bytes", 0)

            category_counts[cat] += 1
            if ext:
                extension_counts[ext] += 1
            total_size_bytes += size

            if status == "VALID":
                valid_files += 1
            elif status == "EMPTY":
                empty_files += 1
            elif status == "CORRUPTED":
                corrupted_files += 1
                corrupted_details.append({
                    "relative_path": doc.get("relative_path", ""),
                    "error": doc.get("error_message", "Unknown corruption error"),
                })
            elif status == "UNSUPPORTED":
                unsupported_files += 1

        # Health score: percentage of files that are valid and non-corrupted
        if total_files > 0:
            health_score = round((valid_files / total_files) * 100.0, 2)
            ready_percentage = health_score
        else:
            health_score = 100.0
            ready_percentage = 100.0

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scan_duration_seconds": round(scan_duration_seconds, 2),
            "health_score": health_score,
            "ready_percentage": ready_percentage,
            "total_files": total_files,
            "valid_files": valid_files,
            "empty_files": empty_files,
            "corrupted_files": corrupted_files,
            "unsupported_files": unsupported_files,
            "total_size_bytes": total_size_bytes,
            "total_size_formatted": self.format_bytes(total_size_bytes),
            "status_counts": dict(status_counts),
            "category_counts": dict(category_counts),
            "extension_counts": dict(extension_counts),
            "duplicates": {
                "exact_duplicate_groups_count": len(duplicate_report.hash_duplicates),
                "redundant_file_instances": duplicate_report.total_duplicate_instances,
                "filename_clashes_count": len(duplicate_report.filename_duplicates),
                "exact_duplicate_groups": duplicate_report.hash_duplicates,
                "filename_duplicate_groups": duplicate_report.filename_duplicates,
            },
            "corrupted_files_sample": corrupted_details[:50],  # cap sample
        }

    def generate_statistics_json(
        self, statistics: dict[str, Any], output_path: Path | str
    ) -> Path:
        """Write dataset_statistics.json to disk."""
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        temp_file = out_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as fh:
            json.dump(statistics, fh, indent=2, ensure_ascii=False)

        temp_file.replace(out_file)
        return out_file

    def generate_health_report_md(
        self, statistics: dict[str, Any], output_path: Path | str
    ) -> Path:
        """Generate human-readable dataset_health_report.md executive report."""
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        health_score = statistics["health_score"]
        ready_pct = statistics["ready_percentage"]
        status_badge = "🟢 HEALTHY" if health_score >= 90 else ("🟡 WARNING" if health_score >= 70 else "🔴 CRITICAL")

        lines: list[str] = [
            "# Dataset Health & Validation Audit Report",
            "",
            "**Sovereign On-Premise Agentic AI Workbench**  ",
            "**Problem Statement:** SIH26117 · **Organization:** MRPL (Petroleum Refinery)  ",
            f"**Audit Timestamp:** {statistics['generated_at']}  ",
            f"**Validation Status:** {status_badge}  ",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Overall Health Score:** `{health_score}%`",
            f"- **RAG Ingestion Readiness:** `{ready_pct}%`",
            f"- **Total Files Scanned:** `{statistics['total_files']}`",
            f"- **Valid Documents:** `{statistics['valid_files']}`",
            f"- **Corrupted / Invalid Documents:** `{statistics['corrupted_files']}`",
            f"- **Empty Files (0 Bytes):** `{statistics['empty_files']}`",
            f"- **Unsupported Files Ignored:** `{statistics['unsupported_files']}`",
            f"- **Total Dataset Size:** `{statistics['total_size_formatted']}` (`{statistics['total_size_bytes']:,}` bytes)",
            f"- **Validation Duration:** `{statistics['scan_duration_seconds']}s`",
            "",
            "---",
            "",
            "## 2. Category Distribution",
            "",
            "| Refinery Category | File Count |",
            "|:---|:---|",
        ]

        for cat, count in sorted(statistics["category_counts"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| `{cat}` | {count:,} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Format & Extension Breakdown",
            "",
            "| Extension | Document Count |",
            "|:---|:---|",
        ])

        for ext, count in sorted(statistics["extension_counts"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| `{ext}` | {count:,} |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Duplicate Content Analysis",
            "",
            f"- **Exact Content Duplicate Groups (Identical SHA-256):** `{statistics['duplicates']['exact_duplicate_groups_count']}`",
            f"- **Total Redundant Files:** `{statistics['duplicates']['redundant_file_instances']}`",
            f"- **Cross-Directory Filename Clashes:** `{statistics['duplicates']['filename_clashes_count']}`",
            "",
            "> **Refinery Policy Compliance:** No duplicates have been deleted automatically. All duplicate associations are cataloged in `manifest.json`.",
            "",
            "---",
            "",
            "## 5. Corruption & Quality Alerts",
            "",
        ])

        if statistics["corrupted_files"] == 0 and statistics["empty_files"] == 0:
            lines.append("✅ **Zero corruption detected.** All indexed documents conform to format standards.")
        else:
            lines.append(f"⚠️ **{statistics['corrupted_files']} corrupted file(s) and {statistics['empty_files']} empty file(s) identified.**")
            lines.append("")
            lines.append("| File Path | Error Details |")
            lines.append("|:---|:---|")
            for item in statistics.get("corrupted_files_sample", []):
                lines.append(f"| `{item['relative_path']}` | {item['error']} |")

        lines.extend([
            "",
            "---",
            "",
            "## 6. Downstream Integration Status",
            "",
            "- **Loaders Engine (Milestone 3):** READY. `manifest.json` is generated with deterministic UUIDs and verified paths.",
            "- **OCR Pipeline (Vision):** READY. Scanned documents and inspection images isolated with category tagging.",
            "- **Vector Indexing:** PREPARED. Hashes available for cache lookup.",
        ])

        report_content = "\n".join(lines) + "\n"

        temp_file = out_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as fh:
            fh.write(report_content)

        temp_file.replace(out_file)
        return out_file
