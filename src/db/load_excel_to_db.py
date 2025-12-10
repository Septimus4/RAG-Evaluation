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
from db.schema import Base, Match, Player, Report, StatLine, StatsFlat, StatsNBA, Team, AnalysisRow

try:  # pragma: no cover
    import logfire
    try:
        # Configure logfire with sensible defaults if not already configured
        # This prevents LogfireNotConfiguredWarning during ingestion.
        logfire.configure()
    except Exception:
        # If configuration fails (e.g., missing env), keep module available
        pass
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


def insert_stats_flat(session: Session, df):
    """Insert rows into stats_flat from a DataFrame with loose column names."""
    # Try flexible column resolution (English/French headers) and ignore non-string headers
    cols = {str(c).lower(): c for c in df.columns if isinstance(c, str)}
    def get_val(row, keys):
        for k in keys:
            if k in cols:
                return row[cols[k]]
        return None
    objects = []
    for _, row in df.iterrows():
        obj = StatsFlat(
            player_name=get_val(row, ["player", "joueur", "nom", "name"]),
            team=get_val(row, ["team", "equipe", "club"]),
            points=get_val(row, ["points", "pts"]),
            rebounds_def=get_val(row, ["rebounds_def", "rebonds_def", "def_reb", "reb_def"]),
            rebounds_off=get_val(row, ["rebounds_off", "rebonds_off", "off_reb", "reb_off"]),
            assists=get_val(row, ["assists", "passes", "ast"]),
        )
        objects.append(obj)
    for obj in objects:
        session.add(obj)
    session.commit()


def insert_stats_nba(session: Session, df: pd.DataFrame) -> None:
    """Insert rows into StatsNBA by mapping common English/French headers.
    Normalizes odd headers like a time value representing '3PM'.
    """
    import datetime as _dt
    def norm(h):
        if isinstance(h, _dt.time) and h.hour == 15:
            return "3pm"
        return str(h).strip().lower()
    cols = {norm(c): c for c in df.columns}
    colmap = {
        "player_name": ["player", "joueur", "nom", "name", "player_name"],
        "team": ["team", "équipe", "equipe", "club"],
        "position": ["pos", "position"],
        "games_played": ["gp", "games", "matches"],
        "minutes": ["min", "minutes"],
        "points": ["pts", "points"],
        "rebounds": ["reb", "rebonds", "totreb"],
        "assists": ["ast", "assist", "passes"],
        "steals": ["stl", "steals"],
        "blocks": ["blk", "blocks"],
        "turnovers": ["tov", "to", "turnovers"],
        "fg_made": ["fgm"],
        "fg_attempts": ["fga"],
        "fg_pct": ["fg%", "fg_pct"],
        "three_made": ["3pm", "3ptm", "3m"],
        "three_attempts": ["3pa", "3pta", "3a"],
        "three_pct": ["3p%", "3pt%", "3%"],
        "ft_made": ["ftm"],
        "ft_attempts": ["fta"],
        "ft_pct": ["ft%", "ft_pct"],
        "plus_minus": ["+/-", "plus_minus"],
        "offensive_rating": ["offrtg"],
        "defensive_rating": ["defrtg"],
        "usage_pct": ["usg%", "usage"],
    }
    def resolve(possible):
        for name in possible:
            key = name.lower()
            if key in cols:
                return cols[key]
        return None
    objects = []
    for _, row in df.iterrows():
        data = {}
        for field, candidates in colmap.items():
            col = resolve(candidates)
            if col is not None:
                val = row.get(col)
                if pd.notna(val):
                    data[field] = val
        if "player_name" in data:
            objects.append(StatsNBA(**data))
    if not objects:
        logging.error("StatsNBA mapping produced no rows; check 'Données NBA' headers.")
        return
    session.bulk_save_objects(objects)
    session.commit()


def insert_teams_from_equipe(session: Session, df: pd.DataFrame) -> None:
    """Map Equipe sheet with columns ['Code', "Nom complet de l'équipe"] into Team."""
    code_col = next((c for c in df.columns if str(c).strip().lower() == 'code'), None)
    name_col = next((c for c in df.columns if str(c).strip().lower().startswith('nom complet')), None)
    if code_col is None or name_col is None:
        logging.error("Equipe sheet missing required columns 'Code' and 'Nom complet de l'équipe'.")
        return
    # Fetch existing names to avoid UNIQUE conflicts on repeated runs
    try:
        existing = {row[0] for row in session.query(Team.name).all()}
    except Exception:
        existing = set()
    seen = set()
    inserted = 0
    total_rows = len(df)
    for _, r in df.iterrows():
        code = r.get(code_col)
        name = r.get(name_col)
        if pd.notna(code) and pd.notna(name):
            name_str = str(name)
            if name_str in existing or name_str in seen:
                continue
            session.add(Team(name=name_str, conference=None, division=None))
            seen.add(name_str)
            inserted += 1
    if inserted == 0:
        # If there were rows but all already existed, this is informational, not an error.
        if total_rows > 0 and existing:
            logging.info("Equipe sheet: all %s teams already present; nothing to insert.", total_rows)
            return
        else:
            logging.warning("Equipe sheet contained no valid team rows to insert.")
            return
    session.commit()


def insert_analysis_rows(session: Session, df: pd.DataFrame, sheet_name: str) -> None:
    """Insert narrative rows from Analyse sheets into AnalysisRow."""
    for _, r in df.iterrows():
        values = list(r.values)
        cols = (values + [None] * 8)[:8]
        session.add(AnalysisRow(sheet=sheet_name, col0=cols[0], col1=cols[1], col2=cols[2], col3=cols[3], col4=cols[4], col5=cols[5], col6=cols[6], col7=cols[7]))
    session.commit()


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


@app.command()
def load_workbook(
    workbook: Path = typer.Option(..., exists=True, help="Single Excel workbook containing sheets for players, matches, stats, reports"),
    players_sheet: str = typer.Option("players", help="Sheet name for players"),
    matches_sheet: str = typer.Option("matches", help="Sheet name for matches"),
    stats_sheet: str = typer.Option("stats", help="Sheet name for stats"),
    reports_sheet: str = typer.Option("reports", help="Sheet name for reports"),
    database_url: str = typer.Option(None, help="Database connection URL"),
    dry_run: bool = typer.Option(False, help="Validate only, do not write to DB"),
):
    """Load structured data from a single workbook with multiple sheets."""
    # Coerce Typer OptionInfo to None when called programmatically
    if not isinstance(database_url, str):
        database_url = None
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = Session(bind=engine)
    session = session_factory

    xls = pd.ExcelFile(workbook)
    sheets = {
        "players": (players_sheet, PlayerRow, insert_players),
        "matches": (matches_sheet, MatchRow, insert_matches),
        "stats": (stats_sheet, StatRow, insert_stats),
        "reports": (reports_sheet, ReportRow, insert_reports),
    }
    for name, (sheet, model, inserter) in sheets.items():
        try:
            if name == "stats":
                df = xls.parse(sheet_name=sheet, header=1)
            else:
                df = xls.parse(sheet_name=sheet)
        except Exception as exc:
            logging.warning("Sheet '%s' not found or failed to parse: %s", sheet, exc)
            continue
        if name == "stats":
            # Attempt to validate as structured; if it fails, fall back to flat stats
            valid, errors = validate_rows(df, model)
            if valid and not errors:
                logging.info("%s: %s valid rows, %s errors", name, len(valid), len(errors))
                if not dry_run:
                    inserter(session, valid)
            else:
                logging.info("Falling back to stats_flat ingestion for sheet '%s'", sheet)
                if not dry_run:
                    insert_stats_flat(session, df)
                if logfire:
                    logfire.info("ingestion.table_flat", table="stats_flat", valid=len(df), errors=len(errors))
            continue
        else:
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
    typer.echo("Workbook ingestion completed.")


def load_workbook_programmatic(
    workbook: Path,
    players_sheet: str = "players",
    matches_sheet: str = "matches",
    stats_sheet: str = "stats",
    reports_sheet: str = "reports",
    database_url: str | None = None,
    dry_run: bool = False,
):
    """Programmatic variant of workbook loader without Typer Option types."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = Session(bind=engine)
    session = session_factory
    xls = pd.ExcelFile(workbook)
    sheets = {
        "players": (players_sheet, PlayerRow, insert_players),
        "matches": (matches_sheet, MatchRow, insert_matches),
        "stats": (stats_sheet, StatRow, insert_stats),
        "reports": (reports_sheet, ReportRow, insert_reports),
    }
    for name, (sheet, model, inserter) in sheets.items():
        try:
            if name == "stats":
                df = xls.parse(sheet_name=sheet, header=1)
            else:
                df = xls.parse(sheet_name=sheet)
        except Exception as exc:
            logging.error("Sheet '%s' not found or failed to parse: %s", sheet, exc)
            continue

        if name == "stats":
            lower_cols = {str(c).lower() for c in df.columns if isinstance(c, str)}
            expected_flat = {"player", "team", "pts"}
            if expected_flat.issubset(lower_cols):
                logging.info("Using stats_flat ingestion for '%s'", sheet)
                if not dry_run:
                    insert_stats_flat(session, df)
                if logfire:
                    logfire.info("ingestion.table_flat", table="stats_flat", valid=len(df), errors=0)
            else:
                logging.info("Mapping '%s' into StatsNBA with wide header support", sheet)
                if not dry_run:
                    insert_stats_nba(session, df)
                if logfire:
                    logfire.info("ingestion.table", table="stats_nba", valid=len(df), errors=0)
            continue

        # Non-stats: check minimal required columns
        lower_cols = {str(c).lower() for c in df.columns if isinstance(c, str)}
        required_map = {
            "players": {"id", "name"},
            "matches": {"id", "date", "home_team", "away_team"},
            "reports": {"match_id", "comment"},
        }
        req = required_map.get(name, set())
        if req and not req.issubset(lower_cols):
            # Special-case known workbook sheets
            if sheet == "Equipe":
                logging.info("Ingesting teams from '%s'", sheet)
                if not dry_run:
                    insert_teams_from_equipe(session, df)
                if logfire:
                    logfire.info("ingestion.table", table="teams", valid=len(df), errors=0)
                continue
            if sheet in ("Analyse", "Analyse Vide"):
                logging.info("Ingesting analysis rows from '%s'", sheet)
                if not dry_run:
                    insert_analysis_rows(session, df, sheet)
                if logfire:
                    logfire.info("ingestion.table", table="analysis_rows", valid=len(df), errors=0)
                continue
            logging.error("Sheet '%s' missing required columns for '%s': need %s", sheet, name, sorted(req))
            continue
        valid, errors = validate_rows(df, model)
        if valid:
            logging.info("%s: %s valid rows, %s errors", name, len(valid), len(errors))
            if not dry_run:
                inserter(session, valid)
        else:
            logging.error("Sheet '%s' provided no valid rows for '%s'. Errors: %s", sheet, name, errors[:5])
        if logfire:
            logfire.info("ingestion.table", table=name, valid=len(valid), errors=len(errors))
    if dry_run:
        logging.info("Dry run completed; no data written.")
        session.rollback()
        return
    session.commit()
    logging.info("Workbook ingestion completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
