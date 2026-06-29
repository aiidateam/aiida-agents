"""Set up a local AiiDA profile for manual end-to-end agent testing.

Adds a bash code matching core.arithmetic.add to an existing sqlite
profile, so the full agent -> validate -> HITL -> submit path can be
exercised interactively (not via pytest).

The submit-only write path hands the process to the daemon, so the
profile needs a broker and a running daemon. `verdi presto --use-zmq`
provides both: it pins the ZMQ broker (no external services, skipping
RabbitMQ auto-detection) and starts the daemon.

Usage:
    verdi presto --profile-name agent-test --use-zmq   # broker + daemon
    uv run python dev/setup_test_profile.py
    uv run python -c "from aiida import load_profile; load_profile('agent-test'); from aiida_agents.cli import main; main()"
"""

from __future__ import annotations

from aiida import load_profile, orm
from aiida.common.exceptions import NotExistent

PROFILE_NAME = "agent-test"
CODE_LABEL = "bash"
COMPUTER_LABEL = "localhost"


def main() -> None:
    load_profile(PROFILE_NAME)

    # NodeCollection has no get_or_create (only Computer/Group do), so the
    # idempotent get-or-create for a code is load_code guarded by NotExistent.
    full_label = f"{CODE_LABEL}@{COMPUTER_LABEL}"
    try:
        code = orm.load_code(full_label)
    except NotExistent:
        code = orm.InstalledCode(
            label=CODE_LABEL,
            computer=orm.load_computer(COMPUTER_LABEL),
            filepath_executable="/bin/bash",
            default_calc_job_plugin="core.arithmetic.add",
        ).store()
        print(f"created {full_label}, pk={code.pk}")
    else:
        print(f"{full_label} already exists, pk={code.pk}")


if __name__ == "__main__":
    main()
