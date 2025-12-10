"""Natural-language to SQL helper for the SportSee schema."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from db.connection import get_engine
from rag.models import SQLQuery, SQLResult, SQLResultRow
try:  # pragma: no cover
    import logfire
except Exception:  # pragma: no cover
    logfire = None


SQL_FEW_SHOTS = """
-- Find the best three-point percentage over the last 5 matches
SELECT player_id, SUM(tpm) * 1.0 / NULLIF(SUM(tpa),0) AS three_pt_pct
FROM stats
JOIN matches ON stats.match_id = matches.id
GROUP BY player_id
ORDER BY matches.date DESC
LIMIT 5;

-- Compare team rebounds at home vs away
SELECT matches.home_team AS team, AVG(rebounds_off + rebounds_def) AS avg_reb
FROM stats
JOIN matches ON stats.match_id = matches.id
GROUP BY matches.home_team;
"""


class SQLTool:
    def __init__(self, database_url: str | None = None, row_limit: int = 50):
        self.engine = get_engine(database_url)
        self.row_limit = row_limit

    def _coerce_query(self, candidate: Any) -> SQLQuery:
        """Normalise legacy inputs into an ``SQLQuery`` instance."""
        if isinstance(candidate, SQLQuery):
            return candidate

        attrs = {
            "natural_query": getattr(candidate, "natural_query", ""),
            "limit": getattr(candidate, "limit", self.row_limit),
            "start_date": getattr(candidate, "start_date", None),
            "end_date": getattr(candidate, "end_date", None),
            "player": getattr(candidate, "player", None),
            "team": getattr(candidate, "team", None),
        }
        return SQLQuery(**attrs)

    def build_sql(self, query: SQLQuery | None = None, **legacy: Any) -> Tuple[str, Dict[str, Any]]:
        # Accept prior signature ``build_sql(tool=...)`` used in tests.
        if query is None:
            legacy_input = legacy.pop("tool", None) or legacy.pop("query", None)
            if legacy_input is None:
                legacy_input = SQLQuery(natural_query="")
            query = self._coerce_query(legacy_input)
        elif legacy:
            logging.getLogger(__name__).debug("Ignoring legacy kwargs in build_sql: %s", legacy)

        # Enforce hard limit regardless of incoming request.
        effective_limit = min(query.limit, self.row_limit)

        base = "SELECT player_id, points, rebounds_def, rebounds_off, assists FROM stats"
        conditions: List[str] = []
        params: Dict[str, Any] = {}
        if query.player or query.team:
            base += " JOIN players ON players.id = stats.player_id"
            if query.player:
                conditions.append("players.name = :player")
                params["player"] = query.player
            if query.team:
                conditions.append("players.team = :team")
                params["team"] = query.team
        if query.start_date:
            conditions.append("stats.match_id IN (SELECT id FROM matches WHERE date >= :start_date)")
            params["start_date"] = query.start_date
        if query.end_date:
            conditions.append("stats.match_id IN (SELECT id FROM matches WHERE date <= :end_date)")
            params["end_date"] = query.end_date
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        final_sql = f"{base}{where_clause} LIMIT {effective_limit}"
        return final_sql, params

    def run_query(self, natural_query: str) -> SQLResult:
        parsed = SQLQuery(natural_query=natural_query, limit=self.row_limit)
        sql, params = self.build_sql(parsed)
        start = time.perf_counter()
        rows = []
        try:
            with self.engine.begin() as conn:
                rows = list(conn.execute(text(sql), params).mappings())
        except SQLAlchemyError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            # Fallback to simplified stats_flat if relational tables not present
            try:
                fallback_sql = "SELECT player_name, team, points, rebounds_def, rebounds_off, assists FROM stats_flat LIMIT :limit"
                params2 = {"limit": self.row_limit}
                with self.engine.begin() as conn:
                    rows = list(conn.execute(text(fallback_sql), params2).mappings())
                sql = fallback_sql
            except Exception:
                logging.getLogger(__name__).warning("SQL query failed; returning empty result", exc_info=exc)
                if logfire:
                    logfire.info(
                        "sql_tool.run_query_error",
                        sql=sql,
                        latency_ms=latency_ms,
                        rows=0,
                        error=str(exc),
                    )
                return SQLResult(query=sql, rows=[], latency_ms=latency_ms)

        # If primary query returned no rows, attempt fallback to stats_flat as a secondary source
        if not rows:
            try:
                fallback_sql = "SELECT player_name, team, points, rebounds_def, rebounds_off, assists FROM stats_flat LIMIT :limit"
                params2 = {"limit": self.row_limit}
                with self.engine.begin() as conn:
                    rows = list(conn.execute(text(fallback_sql), params2).mappings())
                sql = fallback_sql
            except Exception:
                pass

        latency_ms = (time.perf_counter() - start) * 1000
        if logfire:
            logfire.info(
                "sql_tool.run_query",
                sql=sql,
                latency_ms=latency_ms,
                rows=len(rows),
            )
        result_rows = [SQLResultRow(columns=list(row.keys()), values=list(row.values())) for row in rows]
        return SQLResult(query=sql, rows=result_rows, latency_ms=latency_ms)

    def summarize(self, result: SQLResult) -> str:
        if not result.rows:
            return "No rows returned."
        sample = result.rows[0]
        return f"Sample row {dict(zip(sample.columns, sample.values))} (rows={len(result.rows)})"
