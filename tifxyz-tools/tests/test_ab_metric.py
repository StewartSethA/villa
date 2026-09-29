"""The published A/B table is recomputed from the shipped rows, so a changed row or a changed
metric definition shows up as a failing test rather than a silently different headline."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_metric_reproduces_the_published_table():
    out = subprocess.run([sys.executable, str(ROOT / "ab" / "metric.py"),
                          str(ROOT / "ab" / "ovn20260929" / "rows_reaudited.jsonl")],
                         capture_output=True, text=True, check=True).stdout
    assert "paired seeds 40" in out
    assert re.search(r"D: selfx0 40/40 +clean 17\.10 cm2 .* clean/cpu-h 0\.3642", out)
    assert re.search(r"A: selfx0 8/40 +clean 6\.85 cm2 .* clean/cpu-h 0\.0464", out)
    assert "D/A 7.85x  paired 95% CI [3.47, 48.37]" in out
