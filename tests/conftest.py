"""Shared pytest setup.

Puts the package and the script folders on ``sys.path`` so a test can import
any script by module name, and skips ``needs_cosma_data`` tests when the
COSMA results tree (``results/analysis``) is not available.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("diffcmb", "scripts", os.path.join("scripts", "paper"),
             os.path.join("scripts", "diagnostics")):
    _path = os.path.join(REPO, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

COSMA_DATA = os.path.join(REPO, "results", "analysis")


def pytest_collection_modifyitems(config, items):
    if os.path.isdir(COSMA_DATA):
        return
    skip = pytest.mark.skip(reason=f"needs COSMA data at {COSMA_DATA}")
    for item in items:
        if "needs_cosma_data" in item.keywords:
            item.add_marker(skip)
