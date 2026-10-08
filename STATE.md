# STATE (v5.1 in progress, branch routeAB-deploy-v5, stopped on request)

## Done and tested (fake GPUs / local servers; nothing on real Blackwell)
- v5 (307bef4): link gate, tmux + dashboard, alarms, GPU smoke+speed, cu128 lock (verified install + vc_spiral build on pny), early prefetch, per-stripe staging, fill-idle, auto --gpu-speed.
- v5.1 (this commit): keep-alive S3 connections in deploy_common/fetch_assets.py (MEASURED on pny, 600 objects per run, n = 1:
  urllib one-connection-per-object 121 obj/s at 64 workers but 13 / 13 / 11 obj/s at 128 / 256 / 512; keep-alive 181 / 228 / 216 / 236 obj/s
  at 64 / 128 / 256 / 512) -> default ON (ROUTEB_S3_KEEPALIVE=0 disables), workers stay 128; tracks + nx/ny/grad_mag fetched CONCURRENTLY;
  z-chunk listing parallel; fetch log collapsed to one summary line per scroll per 30 s (routeB/fetchlog.py: GB done/total, MB/s, obj/s, ETA);
  planner constants from the real run (3.24 objects per slice per field, not 17.8/3; 180 obj/s default); `routeB_watch.sh --once` now ONE SCREEN
  (render_compact <= 40 lines x 100 cols) and `--brief` (no event stream), `--full` = old long output.
## Answer to (a)
Per-stripe staging DOES restrict lasagna objects to the stripe's z-chunks (manifest.lasagna_z_include), but on 48 GB cards each scroll is ONE
whole-height job (13,000 slices), so "first stripe" = the whole field (43-44 k objects per field): the observed 10000/43219 and 4000/44243.
## Half-done / next
- planner.startup_stripes() exists (first scroll(s) planned as k stripes when whole-scroll staging > 0.15 h) but make_plan(startup=False) by default
  and it is NOT wired into main()/Scheduler.add_scroll (needs sch.height_override from plan["facts"], flag --no-startup-stripes). UNTESTED.
- Not measured: 2 concurrent processes (GIL vs link); hub-side bench never completed (slow link); HTTP/2 not tried (stdlib has none).
- Dashboard tests for compact mode not written; golden snapshot regenerated on first run.
- Still open from v5: Route A kit 404; multi-GPU/OOM/hard-stop never on real GPUs; cu128 fits unvalidated numerically (D6).
