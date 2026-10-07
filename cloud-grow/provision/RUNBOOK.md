# RUNBOOK (one page): watching a cloud-grow fleet

The fleet runs itself: `deploy.sh PLAN --yes` provisions, watches, collects, imports, finishes and tears down. You (or a local coder) only READ `fleet_out/<name>.status.md` (refreshed every tick, default 10 min) and the importer summary. The watchdog (`watchdog.py`, plain Python) already does the only two automatic fixes: restart a dead runner (max 5 per box, never after STOP, never on tool md5 drift) and `down` at spend cap / deadline / more instances than planned.

## The loop (every 30-60 min, or when notified of exit != 0)
1. **Check**: open `fleet_out/<name>.status.md`. Top line: ok / TROUBLE / TRIPWIRE. Exit codes of deploy: 0 clean, 1 trouble seen, 2 refused before creating anything, 3 tripwire fired, 4 INSTANCES MAY REMAIN.
2. **Interpret** (fix the biggest group first, D13; never re-feed a failing stage, D14):
| finding | meaning | do |
|---|---|---|
| `failure rate x/y > 5 %` | segments failing | `ssh BOX`, read `why` strings in `/data/cloud-grow/run1/segments/*/rounds.jsonl`, group by string. Env fault (disk, RAM, data missing) -> fix on that box; tracer fault -> `deploy.sh PLAN --abort` and investigate |
| `selfx_unverified marker(s)` | self-crossing check did not run; surface not shippable | check `vc_tifxyz_selfcross` + `LD_LIBRARY_PATH` on the box (kit); importer will REFUSE these anyway |
| `tool md5 drift` | different binary than pinned: geometry may differ | do NOT keep growing: abort that box's rental, re-publish the kit, re-run smoke |
| `0 rounds in the last hour` | runner alive, no progress | `journalctl -u cloud-grow -n 100` on the box; data fetch incomplete? disk full? |
| `unreachable` 2 ticks | box gone (spot preemption) or network | check provider console; preempted box = lost disk, only uploaded tarballs survive |
| `BOOTSTRAP_FAILED` | bootstrap stopped | `/var/log/cloud-grow-bootstrap.log` on the box (pin gate / preflight / download sha) |
| `spend >= 80 % of cap` | model, not the bill | compare with the provider billing page; decide whether to let the cap end the run |
| TRIPWIRE | down already issued | confirm VERIFY output is empty; check console |
3. **Fix** only what the table says; anything else, escalate.
4. **Escalate** (to the human who owns the account) when: any exit 4; failure rate > 5 % on two boxes; REFUSED rate in the import summary above ~20 % or dominated by selfx reasons (stop scaling: the self-crossing rate is the measurement the smoke test exists for); spend model and provider bill differ by > 20 %.
5. **Always at the end**: provider console shows zero instances/disks/IPs for tag `cloud-grow=<name>`; the write key is deleted; the landing dir's `_bad/` is empty or explained.

## Ready prompt for a local Claude Code (optional)
```
You monitor a rented cloud-grow fleet. You may only: read fleet_out/<name>.status.md and .json, read the importer summary, and run
`./deploy.sh PLAN --dry-run`, `watchdog.py --plan PLAN --once`, and (ONLY if I say so) `./deploy.sh PLAN --abort`.
Never edit the plan, never print or request credentials, never run provider CLIs directly, never SSH without my say-so.
Every 30 min: read the status; follow cloud-grow/provision/RUNBOOK.md's table; report ONE line: ok / what is wrong / what you propose,
with numbers (n = boxes, segments; rounds/h; failure rate; $ spent model vs cap). Escalate to me per the RUNBOOK rules. Do not claim
anything is validated: all verified cm2 are box claims, unvalidated against human annotation.
```
