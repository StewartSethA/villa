#!/usr/bin/env bash
# Build/refresh the ORPHAN branch `routeA-cloud-deploy` (no history) in a separate worktree from the main repo's files. Usage: build_branch.sh MAIN_REPO WORKTREE_DIR
# Single-source: every library file is COPIED from the main repo (src/vesuvius_pipeline/...); nothing is edited on the branch. No set -e (per-step checks).
MAIN="${1:?main repo}"; WT="${2:?worktree dir}"; BR=routeA-cloud-deploy
SRC=src/vesuvius_pipeline
cd "$MAIN" || exit 1
if [ ! -d "$WT" ]; then git worktree add --orphan -b "$BR" "$WT" || exit 1; fi
cd "$WT" || exit 1
mkdir -p "$SRC/routea_cloud" pins umbilicus tools tests patches
D=deploy/routeA_cloud
for f in __init__.py growth_guard.py resume_gate.py degeneracy.py selfcontact.py cloud_box.py cloud_import.py; do cp "$MAIN/$SRC/$f" "$SRC/$f" || echo "MISSING $f"; done
cp "$MAIN/$SRC"/routea_cloud/*.py "$SRC/routea_cloud/"
cp "$MAIN/$D/routeA_run.sh" "$MAIN/$D/README.md" "$MAIN/$D/BUILD_KIT.md" "$MAIN/$D/AGENT_GUIDE_FRAGMENT.md" "$MAIN/$D/.gitignore" . 2>/dev/null
cp "$MAIN/$D"/pins/* pins/
cp -r "$MAIN/$D"/umbilicus/* umbilicus/
cp "$MAIN/$D"/tools/pin_kit.py "$MAIN/$D"/tools/scrub_umbilicus.py tools/
cp "$MAIN/$D"/tools/build_branch.sh tools/
cp "$MAIN/$D"/patches/* patches/ 2>/dev/null
cp "$MAIN/$D"/tests/*.py tests/
cp "$MAIN/$D/pytest.ini" . 2>/dev/null
chmod +x routeA_run.sh
echo built
