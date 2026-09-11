"""Dataset validation runner for Milestone 4: Cleaning & Normalization Engine.

Executes the pipeline on actual documents across:
- Manuals
- SOPs
- Inspection reports
- Maintenance documents
- Safety documents
- Engineering drawings (metadata only)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.loaders import LoaderFactory
from rag_engine.parsers import ParserFactory
from rag_engine.preprocessing import CleaningPipeline


def main() -> None:
    loader_factory = LoaderFactory()
    parser_factory = ParserFactory()
    pipeline = CleaningPipeline(strict_token_verification=False)

    categories = [
        "manuals",
        "safety_docs",
        "inspection_reports",
        "maintenance",
        "engineering_drawings",
        "emails",
    ]

    results: list[dict] = []

    for cat in categories:
        cat_dir = Path("datasets") / cat
        if not cat_dir.is_dir():
            continue

        sample_files = [f for f in sorted(cat_dir.glob("*.*")) if not f.name.startswith(".")][:3]

        for file_path in sample_files:
            try:
                loader = loader_factory.get_loader(file_path=file_path)
                doc = loader.load(file_path=file_path)
                parser, _ = parser_factory.get_parser(doc)
                parsed_doc = parser.parse(doc)

                chars_before = sum(len(s.content) for s in parsed_doc.sections)
                cleaned_doc = pipeline.clean(parsed_doc)
                chars_after = sum(len(s.content) for s in cleaned_doc.sections)

                stats = cleaned_doc.cleaning_statistics
                results.append({
                    "category": cat,
                    "file_name": file_path.name,
                    "status": cleaned_doc.cleaning_status,
                    "chars_before": chars_before,
                    "chars_after": chars_after,
                    "reduction_pct": round((chars_before - chars_after) / max(1, chars_before) * 100, 2),
                    "protected_tokens_count": len(cleaned_doc.protected_tokens),
                    "sample_protected_tokens": cleaned_doc.protected_tokens[:4],
                    "headers_removed": len(cleaned_doc.removed_headers),
                    "footers_removed": len(cleaned_doc.removed_footers),
                    "mapped_pages": len(cleaned_doc.page_map),
                    "exec_time_ms": stats.execution_time_ms if stats else 0.0,
                })
            except Exception as exc:
                results.append({
                    "category": cat,
                    "file_name": file_path.name,
                    "status": "ERROR",
                    "error": str(exc),
                })

    print("\n" + "=" * 90)
    print("MILESTONE 4: REAL DATASET CLEANING & NORMALIZATION VALIDATION REPORT")
    print("=" * 90)
    for r in results:
        if r.get("status") == "ERROR":
            print(f"[-] {r['category']}/{r['file_name']}: FAILED -> {r.get('error')}")
        else:
            tokens_str = ", ".join(r['sample_protected_tokens'])
            print(
                f"[+] {r['category']}/{r['file_name']:<30} | "
                f"Chars: {r['chars_before']:>5} -> {r['chars_after']:>5} (-{r['reduction_pct']}%) | "
                f"Tokens ({r['protected_tokens_count']}): [{tokens_str}] | "
                f"Time: {r['exec_time_ms']}ms"
            )

    report_path = Path("project_management/milestone_reports/cleaning_dataset_validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved structured validation report to: {report_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
