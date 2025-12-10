"""SQLAlchemy schema for structured SportSee data."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=False)
    name = Column(String, nullable=False)
    team = Column(String, nullable=True)
    position = Column(String, nullable=True)
    height_cm = Column(Float)
    weight_kg = Column(Float)
    age = Column(Integer)

    stats = relationship("StatLine", back_populates="player")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=False)
    date = Column(Date, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    competition = Column(String)
    venue = Column(String)
    home_score = Column(Integer)
    away_score = Column(Integer)

    stats = relationship("StatLine", back_populates="match")
    reports = relationship("Report", back_populates="match")


class StatLine(Base):
    __tablename__ = "stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    minutes = Column(Float)
    points = Column(Integer)
    fgm = Column(Integer)
    fga = Column(Integer)
    tpm = Column(Integer)
    tpa = Column(Integer)
    ftm = Column(Integer)
    fta = Column(Integer)
    rebounds_off = Column(Integer)
    rebounds_def = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)
    fouls = Column(Integer)

    player = relationship("Player", back_populates="stats")
    match = relationship("Match", back_populates="stats")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    title = Column(String)
    comment = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    match = relationship("Match", back_populates="reports")


class StatsFlat(Base):
    __tablename__ = "stats_flat"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_name = Column(String, nullable=True)
    team = Column(String, nullable=True)
    points = Column(Integer)
    rebounds_def = Column(Integer)
    rebounds_off = Column(Integer)
    assists = Column(Integer)

# New schema aligned to the workbook structure
class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    conference = Column(String)
    division = Column(String)

class StatsNBA(Base):
    __tablename__ = "stats_nba"
    id = Column(Integer, primary_key=True, autoincrement=True)
    # Core identifiers
    player_name = Column(String, index=True, nullable=False)
    team = Column(String, index=True)
    position = Column(String)
    # Volume stats
    games_played = Column(Integer)
    minutes = Column(Float)
    points = Column(Float)
    rebounds = Column(Float)
    assists = Column(Float)
    steals = Column(Float)
    blocks = Column(Float)
    turnovers = Column(Float)
    # Shooting splits
    fg_made = Column(Float)
    fg_attempts = Column(Float)
    fg_pct = Column(Float)
    three_made = Column(Float)
    three_attempts = Column(Float)
    three_pct = Column(Float)
    ft_made = Column(Float)
    ft_attempts = Column(Float)
    ft_pct = Column(Float)
    # Advanced rate stats (optional)
    plus_minus = Column(Float)
    offensive_rating = Column(Float)
    defensive_rating = Column(Float)
    usage_pct = Column(Float)
    # Free text notes if present
    notes = Column(Text)

class AnalysisRow(Base):
    __tablename__ = "analysis_rows"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sheet = Column(String, nullable=False)  # 'Analyse' or 'Analyse Vide'
    col0 = Column(Text)
    col1 = Column(Text)
    col2 = Column(Text)
    col3 = Column(Text)
    col4 = Column(Text)
    col5 = Column(Text)
    col6 = Column(Text)
    col7 = Column(Text)
