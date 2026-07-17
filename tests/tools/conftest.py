"""Fixtures for the tool tests."""

from __future__ import annotations

import pytest

from tests.fixtures import QueryArchive, build_query_archive


@pytest.fixture(scope="module")
def query_archive() -> QueryArchive:
    """A miniature of the fermi-surfaces archive (see ``tests.fixtures.archive``).

    Module-scoped and additive. The session-scoped ``aiida_profile`` is shared
    with every other test module, so this must not clean the storage -- that
    would destroy the session-scoped node fixtures the other modules rely on.
    Assertions therefore scope themselves to the group rather than counting the
    whole profile.
    """
    return build_query_archive()
