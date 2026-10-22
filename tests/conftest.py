"""Use a private project-local temp folder, avoiding shared Windows temp ACLs."""
from pathlib import Path
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if config.option.basetemp is None:
        folder = ROOT / ".cache"
        folder.mkdir(exist_ok=True)
        config._mma_temp = tempfile.TemporaryDirectory(prefix="pytest_", dir=folder)
        config.option.basetemp = config._mma_temp.name


def pytest_unconfigure(config):
    temporary = getattr(config, "_mma_temp", None)
    if temporary is not None:
        target = Path(temporary.name).resolve()
        # Verify the resolved target before recursive temporary cleanup on Windows.
        if target.parent != (ROOT / ".cache").resolve() or not target.name.startswith("pytest_"):
            raise RuntimeError("Unexpected pytest temporary cleanup target")
        temporary.cleanup()
