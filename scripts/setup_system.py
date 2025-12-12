"""End-to-end setup script for the SportSee RAG system.

This script is intended for "one-command" reproducibility:
- (Optional) Load structured data into the SQL database from an Excel workbook.
- Build / verify the retriever index from the `inputs/` directory.
- Run a smoke-test query through the RAG pipeline.

Usage:
  python scripts/setup_system.py --input-dir inputs --smoke-question "Who led total points?"
  python scripts/setup_system.py --workbook inputs/regular\ NBA.xlsx --database-url sqlite:///./sportsee.db

Note: the TF-IDF retriever prototype is built in-memory each run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from observability.logfire_setup import configure_logfire, info, span
from data_pipeline.indexing import get_retriever_from_dir
from db.load_excel_to_db import load_workbook
from db.sql_tool import SQLTool
from rag.llm import LLMConfig
from rag.pipeline import run_rag_pipeline


def setup_database(*, workbook: Path, database_url: Optional[str]) -> None:
    configure_logfire()
    with span("setup.database", workbook=str(workbook), database_url=database_url or "(default)"):
        load_workbook(workbook=workbook, database_url=database_url, dry_run=False)
    info("setup.database.completed", workbook=str(workbook))


def setup_retriever(*, input_dir: Path) -> None:
    configure_logfire()
    with span("setup.retriever", input_dir=str(input_dir)):
        _ = get_retriever_from_dir(input_dir)
    info("setup.retriever.completed", input_dir=str(input_dir))


def smoke_test(
    *,
    question: str,
    input_dir: Path,
    database_url: Optional[str],
    model: str,
) -> str:
    configure_logfire()
    tool = SQLTool(database_url=database_url) if database_url else SQLTool()
    llm_config = LLMConfig(model=model)
    with span("setup.smoke_test", model=model, input_dir=str(input_dir)):
        answer = run_rag_pipeline(question, data_dir=input_dir, sql_tool=tool, llm_config=llm_config)
    info("setup.smoke_test.completed", model=model, used_sql=answer.used_sql)
    return answer.answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up the SportSee RAG system end-to-end.")
    parser.add_argument("--input-dir", type=Path, default=Path("inputs"), help="Directory containing raw files.")
    parser.add_argument("--workbook", type=Path, default=None, help="Optional Excel workbook to ingest into the DB.")
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL (e.g. sqlite:///./sportsee.db). If omitted, uses default settings.",
    )
    parser.add_argument(
        "--smoke-question",
        type=str,
        default="What does the starter note say?",
        help="Question used to smoke-test the pipeline.",
    )
    parser.add_argument("--model", type=str, default="mistral-small-latest", help="LLM model name.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    configure_logfire()

    if args.workbook is not None:
        if not args.workbook.exists():
            raise SystemExit(f"Workbook not found: {args.workbook}")
        setup_database(workbook=args.workbook, database_url=args.database_url)

    setup_retriever(input_dir=args.input_dir)

    answer = smoke_test(
        question=args.smoke_question,
        input_dir=args.input_dir,
        database_url=args.database_url,
        model=args.model,
    )
    print("\n--- Smoke test answer ---\n")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
