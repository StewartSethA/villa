import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no external data")
    config.addinivalue_line("markers", "needs_vc3d: needs VC3D binaries (vc_tifxyz_selfcross); skipped without them")
