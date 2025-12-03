"""SQLAlchemy schema for structured SportSee data."""
from __future__ import annotations

from datetime import datetime

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
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="reports")
