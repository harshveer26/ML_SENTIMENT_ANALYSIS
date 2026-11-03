"""Basic package scaffold checks."""

import sentiment


def test_package_has_version() -> None:
    """The installed package exposes its version."""
    assert sentiment.__version__ == "0.1.0"
