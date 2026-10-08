"""Route A pushbutton deploy (routeA-cloud-deploy): ONE driver for the fleet and for a rented box.
  net.py        resumable HTTPS download with sha256, anonymous S3 listing + parallel sync (size + md5/etag verified), fail loud
  bootstrap.py  kit (sha256-pinned tarball + per-file md5 pins, self-check) and per-scroll inputs (surface prediction + normal grids from the public open-data bucket)
  seedprop.py   DB-free seed proposer on the prediction zarr (+ umbilicus radius bins)
  settings.py   the production guard settings snapshot -> GuardPolicy
  run.py        the orchestrator behind ./routeA_run.sh
Nothing here knows about a hub, a token, a private key or a database."""
