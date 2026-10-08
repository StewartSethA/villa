"""The fit launcher must import the INSTALLED vc_spiral (compiled extension), not spiral-fitting/vc_spiral source."""
import subprocess, sys
from pathlib import Path
from routeB import fit

ROOT = Path(__file__).resolve().parents[2]


def test_launcher_prefers_installed_vc_spiral(tmp_path):
    shadow = tmp_path / "sf" / "vc_spiral"; shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")           # a shadowing source package without the extension
    (tmp_path / "sf" / "fit_spiral.py").write_text("import vc_spiral.spiral_sampling as m, sys; print('FILE', m.__file__)")
    cmd = fit.fit_cmd(tmp_path / "sf", "ds", "cache"); cmd[0] = sys.executable
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    try:
        import vc_spiral.spiral_sampling  # noqa: F401
    except ImportError:
        assert "extension missing" in r.stderr or "No module" in r.stderr   # no extension here: must fail loudly, not silently
        return
    assert r.returncode == 0 and str(tmp_path) not in r.stdout, r.stderr[-500:]
