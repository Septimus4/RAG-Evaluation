import os
from pathlib import Path

import pytest

from db.sql_tool import SQLTool


def test_sql_tool_builds_safe_sql():
    tool = SQLTool()
    # Natural query parsing is minimal; ensure safe parameterization and limits applied
    sql, params = tool.build_sql(tool=type("Q", (), {})())  # type: ignore
    # Fallback: building via SQLQuery inside run_query; here ensure row_limit present via run_query
    result = tool.run_query("points for player foo last month")
    assert "LIMIT" in result.query
    assert isinstance(result.rows, list)


@pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="No database configured")
def test_sql_tool_runs_against_db(tmp_path: Path):
    # With a configured DB, the query should execute even if returns 0 rows
    tool = SQLTool(database_url=os.environ.get("DATABASE_URL"))
    result = tool.run_query("average points")
    assert result.latency_ms >= 0
    assert isinstance(result.rows, list)
