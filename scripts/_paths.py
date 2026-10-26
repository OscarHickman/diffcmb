"""Put the package and every script folder on sys.path.

The pipeline scripts import one another by bare module name
(``import aggregate_coverage_ranks as acr``) across folders, so a script in a
subfolder runs

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
    import _paths  # noqa: E402,F401

and can then import any module in any folder. tests/conftest.py does the same.
"""
import os
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPTS)
# archive/ holds superseded scripts and slurm/ holds no Python; neither is imported.
_NOT_IMPORTED = {"archive", "slurm", "__pycache__"}
_SUBFOLDERS = sorted(
    os.path.join(SCRIPTS, name) for name in os.listdir(SCRIPTS)
    if os.path.isdir(os.path.join(SCRIPTS, name)) and name not in _NOT_IMPORTED)
for _path in reversed([os.path.join(REPO, "diffcmb"), SCRIPTS, *_SUBFOLDERS]):
    if _path not in sys.path:
        sys.path.insert(0, _path)
