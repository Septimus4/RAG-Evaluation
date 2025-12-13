"""Generate an aggregated markdown report from the latest per-dataset evaluation artifacts.

Reads the newest summary_*.md per dataset in src/evaluation/results/ and derives the
corresponding rag_metrics_*.csv and ragas_metrics_*.csv (by timestamp).

Output:
- docs/reports/rag_evaluation_metrics.md
- REPORT_RAGAS_EVALUATION_METRICS.md (repo root)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("src/evaluation/results")
DOCS_REPORT_PATH = Path("docs/reports/rag_evaluation_metrics.md")
ROOT_REPORT_PATH = Path("REPORT_RAGAS_EVALUATION_METRICS.md")


@dataclass(frozen=True)
class DatasetRun:
    dataset_stem: str
    timestamp: str
    summary_path: Path
    rag_metrics_path: Path
    ragas_metrics_path: Path | None


def _latest_summaries(results_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (dataset_stem, timestamp, path) for latest summary per dataset."""
    candidates = []
    for p in results_dir.glob("summary_*.md"):
        m = re.match(r"summary_(.+?)_(\d{4}-\d{2}-\d{2}T.*)\.md$", p.name)
        if not m:
            continue
        candidates.append((m.group(1), m.group(2), p))

    latest: dict[str, tuple[str, Path]] = {}
    for ds, ts, path in candidates:
        if ds not in latest or ts > latest[ds][0]:
            latest[ds] = (ts, path)

    return [(ds, ts, path) for ds, (ts, path) in sorted(latest.items())]


def _run_for(ds: str, ts: str, results_dir: Path) -> DatasetRun:
    rag_metrics = results_dir / f"rag_metrics_{ts}.csv"
    ragas_metrics = results_dir / f"ragas_metrics_{ts}.csv"
    summary = results_dir / f"summary_{ds}_{ts}.md"

    if not rag_metrics.exists():
        raise FileNotFoundError(rag_metrics)

    return DatasetRun(
        dataset_stem=ds,
        timestamp=ts,
        summary_path=summary,
        rag_metrics_path=rag_metrics,
        ragas_metrics_path=ragas_metrics if ragas_metrics.exists() else None,
    )


def _describe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            rows.append((c, None, None, None))
        else:
            rows.append((c, float(s.mean()), float(s.min()), float(s.max())))
    out = pd.DataFrame(rows, columns=["metric", "mean", "min", "max"])
    return out


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.4f}"


def build_report(runs: list[DatasetRun]) -> str:
    retrieval_cols = [
        "recall@5",
        "precision@5",
        "mrr",
        "ndcg",
        "retrieval_latency_ms",
        "retrieval_fail_rate",
        "retrieved_context_count",
        "retrieval_diversity_ratio",
    ]
    context_cols = [
        "context_window_utilization_chars",
        "context_contamination_rate",
        "deduplication_rate",
    ]
    ragas_cols = [
        "answer_relevancy",
        "context_precision",
        "faithfulness",
    ]

    lines: list[str] = []
    lines.append("# RAGAS Evaluation — Key Metrics")
    lines.append("")
    lines.append("This report aggregates the latest evaluation run for each dataset.")
    lines.append("Artifacts are under `src/evaluation/results/`.")
    lines.append("")

    lines.append("## Datasets included")
    lines.append("")
    lines.append("| dataset | timestamp | rag_metrics | ragas_metrics |")
    lines.append("|---|---|---|---|")
    for r in runs:
        lines.append(
            f"| {r.dataset_stem} | {r.timestamp} | {r.rag_metrics_path.name} | "
            f"{(r.ragas_metrics_path.name if r.ragas_metrics_path else '(missing)')} |"
        )
    lines.append("")

    for r in runs:
        lines.append(f"## {r.dataset_stem}")
        lines.append("")

        rag_df = pd.read_csv(r.rag_metrics_path)
        retr = _describe(rag_df, retrieval_cols)
        ctx = _describe(rag_df, context_cols)

        lines.append("### Retrieval metrics")
        lines.append("")
        lines.append("| metric | mean | min | max |")
        lines.append("|---|---:|---:|---:|")
        for _, row in retr.iterrows():
            lines.append(f"| {row['metric']} | {_fmt(row['mean'])} | {_fmt(row['min'])} | {_fmt(row['max'])} |")
        lines.append("")

        lines.append("### Context metrics")
        lines.append("")
        lines.append("| metric | mean | min | max |")
        lines.append("|---|---:|---:|---:|")
        for _, row in ctx.iterrows():
            lines.append(f"| {row['metric']} | {_fmt(row['mean'])} | {_fmt(row['min'])} | {_fmt(row['max'])} |")
        lines.append("")

        lines.append("### RAGAS metrics")
        lines.append("")
        if r.ragas_metrics_path is None:
            lines.append("(missing `ragas_metrics_*.csv` for this dataset run)")
            lines.append("")
        else:
            ragas_df = pd.read_csv(r.ragas_metrics_path)
            desc = _describe(ragas_df, ragas_cols)
            lines.append("| metric | mean | min | max |")
            lines.append("|---|---:|---:|---:|")
            for _, row in desc.iterrows():
                lines.append(f"| {row['metric']} | {_fmt(row['mean'])} | {_fmt(row['min'])} | {_fmt(row['max'])} |")
            lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- With a small/limited corpus in `inputs/`, context-related metrics can be noisy and may under-represent real performance.")
    lines.append("- RAGAS may warn about generation counts (provider returns 1 generation). The evaluation continues with 1 and the scores remain usable.")

    return "\n".join(lines) + "\n"


def main() -> None:
    latest = _latest_summaries(RESULTS_DIR)
    if not latest:
        raise SystemExit("No summary_*.md files found in src/evaluation/results")

    runs = [_run_for(ds, ts, RESULTS_DIR) for ds, ts, _ in latest]

    report = build_report(runs)

    DOCS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_REPORT_PATH.write_text(report, encoding="utf-8")
    ROOT_REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Wrote {DOCS_REPORT_PATH}")
    print(f"Wrote {ROOT_REPORT_PATH}")


if __name__ == "__main__":
    main()
