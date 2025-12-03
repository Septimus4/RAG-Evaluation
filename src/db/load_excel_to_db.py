"""Excel ingestion into the relational database."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import typer
from sqlalchemy.orm import Session

from db.connection import get_engine
from db.models import MatchRow, PlayerRow, ReportRow, StatRow
from db.schema import Base, Match, Player, Report, StatLine

try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None


app = typer.Typer(add_completion=False)


def validate_rows(df: pd.DataFrame, model) -> Tuple[list, list]:
    valid, errors = [], []
    for idx, row in df.iterrows():
        try:
            valid.append(model(**row.to_dict()))
        except Exception as exc:  # pragma: no cover
            errors.append((idx, str(exc)))
    return valid, errors


def insert_players(session: Session, rows: Iterable[PlayerRow]):
    for row in rows:
        session.merge(Player(**row.model_dump()))


def insert_matches(session: Session, rows: Iterable[MatchRow]):
    for row in rows:
        session.merge(Match(**row.model_dump()))


def insert_stats(session: Session, rows: Iterable[StatRow]):
    for row in rows:
        session.add(StatLine(**row.model_dump()))


def insert_reports(session: Session, rows: Iterable[ReportRow]):
    for row in rows:
        session.add(Report(**row.model_dump()))


@app.command()
def load(
    players_path: Path = typer.Option(..., exists=True, help="Excel file for players"),
    matches_path: Path = typer.Option(..., exists=True, help="Excel file for matches"),
    stats_path: Path = typer.Option(..., exists=True, help="Excel file for stats"),
    reports_path: Path = typer.Option(..., exists=True, help="Excel file for reports"),
    database_url: str = typer.Option(None, help="Database connection URL"),
    dry_run: bool = typer.Option(False, help="Validate only, do not write to DB"),
):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = Session(bind=engine)
    session = session_factory

    frames = {
        "players": (players_path, PlayerRow, insert_players),
        "matches": (matches_path, MatchRow, insert_matches),
        "stats": (stats_path, StatRow, insert_stats),
        "reports": (reports_path, ReportRow, insert_reports),
    }

    for name, (path, model, inserter) in frames.items():
        df = pd.read_excel(path)
        valid, errors = validate_rows(df, model)
        logging.info("%s: %s valid rows, %s errors", name, len(valid), len(errors))
        if errors:
            logging.warning("Errors while validating %s: %s", name, errors[:5])
        if dry_run:
            continue
        inserter(session, valid)
        if logfire:
            logfire.info("ingestion.table", table=name, valid=len(valid), errors=len(errors))
    if dry_run:
        typer.echo("Dry run completed; no data written.")
        session.rollback()
        return
    session.commit()
    typer.echo("Ingestion completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
