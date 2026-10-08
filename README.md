# routeAB-deploy: Vesuvius Route A + Route B pushbuttons

Two one-command pipelines, identical on a local GPU/CPU box and on a rented machine. No credentials, no private hosts, no history.

| | what | command | needs |
|---|---|---|---|
| **Route A** | guarded surface **growth** from seeds with the production guard stack (VC3D tracer kit, self-collision guard, degeneracy resume gate) | `git clone -b routeAB-deploy --depth 1 https://github.com/StewartSethA/villa.git && cd villa && ./routeA_run.sh --scrolls PHerc0332,PHerc0211 --seeds 8 --hours 6` | CPU only; bash, tar, curl/wget/python3 |
| **Route B** | **spiral fit** of whole windings -> tiles -> flatten -> render -> **ink** -> export | `git clone -b routeAB-deploy --depth 1 https://github.com/StewartSethA/villa.git && cd villa && ./routeB_run.sh --scrolls PHerc0211,PHerc0125 --stripe-width 4500` | NVIDIA GPU (driver >= 525), g++ >= 13, curl/wget |

Route B options: `--stripe-width 4500|7500|13500|full`, `--smoke` (proof-sized run), `--stages fetch,fit,tiles,ink,export`, `--help`.
Both scripts download everything they need (pinned and hash/size-verified, resumable) into `./routeA_work/` or `./routeB_work/` (`ROUTEB_HOME`, `--workdir`).

* Route A details: `README_ROUTEA.md`. Route B details: `README_ROUTEB.md`.
* A chatbot/agent picking this up: **`AGENT_GUIDE.md`** (architecture, state, how to verify each stage, failure catalogue pointers).
* Something broke: **`TROUBLESHOOTING.md`**.
* Prove it works on a new machine: `./smoke_test.sh A|B|both` (small cases; prints wall time and GB downloaded).
* Shared plumbing (one implementation): `deploy_common/` (`bootstrap_env.sh` pinned uv + python + lock; `fetch_assets.py` verified resumable downloads; `branch_scan.py` secret/IP/size scan; run `python3 deploy_common/branch_scan.py .` before pushing any change).

Layout: `routeA_run.sh pins/ src/ umbilicus/ tests/ patches/ tools/` (Route A) | `routeB_run.sh routeB/ spiral-fitting/ lasagna/ vesuvius/ gpu_render/ ink/ models/` (Route B) | `deploy_common/` (both).
Provenance: library code is copied from Seth Stewart's ScrollPrizeTutorial working repo at the commits recorded in `routeB/BUILD_INFO.txt`; upstream pieces are Vesuvius Challenge `villa` (spiral-fitting, lasagna).
