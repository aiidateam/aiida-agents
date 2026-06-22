"""Set up a local AiiDA profile for manual end-to-end agent testing.

Creates a brokerless sqlite profile with a bash code matching
core.arithmetic.add, so the full agent -> validator -> HITL -> submit
path can be exercised interactively (not via pytest).

Usage:
    verdi presto --profile-name agent-test   # create the profile first
    uv run python dev/setup_test_profile.py
    uv run python -c "from aiida import load_profile; load_profile('agent-test'); from aiida_agents.cli import main; main()"
"""

from __future__ import annotations

from aiida import load_profile, orm

PROFILE_NAME = "agent-test"


def main() -> None:
    load_profile(PROFILE_NAME)

    computer = orm.load_computer("localhost")

    existing = orm.QueryBuilder()
    existing.append(
        orm.InstalledCode,
        filters={"label": "bash"},
        project=["id"],
    )
    rows = existing.all()
    if rows:
        print(f"bash code already exists, pk={rows[0][0]}")
        return

    code = orm.InstalledCode(
        label="bash",
        computer=computer,
        filepath_executable="/bin/bash",
        default_calc_job_plugin="core.arithmetic.add",
    ).store()
    print(f"created bash code, pk={code.pk}")


if __name__ == "__main__":
    main()
