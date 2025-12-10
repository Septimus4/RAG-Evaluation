"""Pydantic models mirroring the relational schema."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PlayerRow(BaseModel):
    id: int
    name: str
    team: Optional[str] = None
    position: Optional[str] = None
    height_cm: Optional[float] = Field(default=None, ge=0)
    weight_kg: Optional[float] = Field(default=None, ge=0)
    age: Optional[int] = Field(default=None, ge=0)


class MatchRow(BaseModel):
    id: int
    date: date
    home_team: str
    away_team: str
    competition: Optional[str]
    venue: Optional[str]
    home_score: Optional[int] = Field(default=None, ge=0)
    away_score: Optional[int] = Field(default=None, ge=0)


class StatRow(BaseModel):
    match_id: int
    player_id: int
    minutes: Optional[float] = Field(default=None, ge=0)
    points: Optional[int] = Field(default=None, ge=0)
    fgm: Optional[int] = Field(default=None, ge=0)
    fga: Optional[int] = Field(default=None, ge=0)
    tpm: Optional[int] = Field(default=None, ge=0)
    tpa: Optional[int] = Field(default=None, ge=0)
    ftm: Optional[int] = Field(default=None, ge=0)
    fta: Optional[int] = Field(default=None, ge=0)
    rebounds_off: Optional[int] = Field(default=None, ge=0)
    rebounds_def: Optional[int] = Field(default=None, ge=0)
    assists: Optional[int] = Field(default=None, ge=0)
    steals: Optional[int] = Field(default=None, ge=0)
    blocks: Optional[int] = Field(default=None, ge=0)
    turnovers: Optional[int] = Field(default=None, ge=0)
    fouls: Optional[int] = Field(default=None, ge=0)

    @field_validator("fga")
    @classmethod
    def attempts_non_zero(cls, v, info):  # pragma: no cover - sanity guard
        made = info.data.get("fgm")
        if v is not None and made is not None and made > v:
            raise ValueError("FGM cannot exceed FGA")
        return v


class ReportRow(BaseModel):
    match_id: int
    title: Optional[str] = None
    comment: str
