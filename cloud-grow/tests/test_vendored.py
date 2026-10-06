import hashlib
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_vendored_files_match_recorded_md5():
    doc = json.load(open(os.path.join(ROOT, "VENDORED.json")))
    for f, rec in doc["files"].items():
        got = hashlib.md5(open(os.path.join(ROOT, "cloud_grow", f), "rb").read()).hexdigest()
        assert got == rec["shipped_md5"], f"{f} was edited in place: re-vendor and update VENDORED.json"


def test_guard_policy_surface_the_runner_relies_on():
    from cloud_grow import growth_guard as GG
    for name in ("selfcross_check_incremental", "guard_round", "GuardState", "ZarrSampler", "ShadowContext", "lattice_area_cm2"):
        assert hasattr(GG, name), name
    f = GG.GuardPolicy.__dataclass_fields__
    for name in ("selfcross", "selfcross_fail_closed", "selfcross_min_cells", "selfcross_cut_interior", "selfcross_stop_inherited",
                 "selfcross_hairpin_abort_ratio"):
        assert name in f, name
    assert f["selfcross_fail_closed"].default is True


def test_no_fleet_imports_left_in_shipped_python():
    import re
    bad = []
    for r, _d, fs in os.walk(ROOT):
        if "tests" in r or ".git" in r:
            continue
        for fn in fs:
            if fn.endswith(".py"):
                for i, ln in enumerate(open(os.path.join(r, fn)), 1):
                    if re.search(r"from \.\.|from \.(db|alerts|workflow|config|stages|scrolls|provenance)\b|import vesuvius_pipeline|from vesuvius_pipeline", ln):
                        if "alerts" in ln and fn == "growth_guard.py":
                            continue          # `.alerts` is OUR cloud_grow/alerts.py
                        bad.append(f"{fn}:{i}: {ln.strip()}")
    assert not bad, bad
