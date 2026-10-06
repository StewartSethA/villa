"""File-backed replacement for the hub database, for the grow loop only.

The production grow loop (`stages/grow.py`, `agent.py`) talks to the fleet's `pipeline.db` through a
sqlite3-shaped connection (on peers: `remotedb.RemoteDB`, an HTTP proxy that holds a hub token). A
rented box must not reach the hub and must never hold that token. This module is the SAME interface,
backed by one local sqlite3 file, restricted to exactly the tables and calls the grow loop uses:

    tables : segment, attempt, artifact, metric, pipeline_setting, stage_flag, seed
    calls  : connect, upsert_segment, record_metric, record_artifact, record_seed,
             set_paused, is_paused, set_pipeline_setting, start_attempt, finish_attempt
    reads  : `db.execute("SELECT ...")` with the production SQL (pipeline_setting, metric, artifact,
             attempt) works unchanged: rows are sqlite3.Row (index by name AND position).

Differences from production (stated, not hidden):
  * append-only metric/artifact tables, exactly like production; there is no UPDATE of a metric row;
  * no host, assignment, pipeline_def, lease or alert tables: scheduling is the caller's job;
  * `flatten_feedback_check` (growth_guard) reads `attempt` rows of stage 'flatten'; none are ever written
    here, so that shadow criterion is INERT on a cloud box (it reports checked=0).

`pipeline_db()` returns this module so vendored code that calls `pipeline_db().record_metric(...)` works.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS segment (
  seg TEXT PRIMARY KEY, scroll TEXT, route TEXT, fit_id TEXT, parent TEXT, created_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attempt (
  id INTEGER PRIMARY KEY AUTOINCREMENT, seg TEXT NOT NULL, stage TEXT NOT NULL, state TEXT NOT NULL,
  started_utc TEXT, ended_utc TEXT, host TEXT, pid INTEGER, tool TEXT, tool_md5 TEXT, cmd TEXT,
  params_json TEXT, progress REAL, detail TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS artifact (
  id INTEGER PRIMARY KEY AUTOINCREMENT, seg TEXT NOT NULL, stage TEXT NOT NULL, attempt_id INTEGER,
  kind TEXT NOT NULL, path TEXT NOT NULL, bytes INTEGER, mtime REAL, md5 TEXT, meta_json TEXT,
  recorded_utc TEXT NOT NULL, UNIQUE(seg, stage, kind, path, mtime));
CREATE INDEX IF NOT EXISTS ix_artifact_seg ON artifact(seg, stage, kind);
CREATE TABLE IF NOT EXISTS metric (
  id INTEGER PRIMARY KEY AUTOINCREMENT, seg TEXT NOT NULL, stage TEXT, attempt_id INTEGER,
  name TEXT NOT NULL, value REAL, text_value TEXT, control_name TEXT, control_value REAL,
  recorded_utc TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_metric_seg ON metric(seg, name);
CREATE TABLE IF NOT EXISTS pipeline_setting (
  key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_utc TEXT, updated_by TEXT);
CREATE TABLE IF NOT EXISTS stage_flag (
  seg TEXT NOT NULL, stage TEXT NOT NULL, paused INTEGER NOT NULL DEFAULT 0, paused_by TEXT,
  updated_utc TEXT, PRIMARY KEY (seg, stage));
CREATE TABLE IF NOT EXISTS seed (
  seg TEXT PRIMARY KEY, scroll TEXT NOT NULL, x INTEGER, y INTEGER, z INTEGER, score REAL, dist_l4 REAL,
  centre REAL, window_mean REAL, source TEXT, host TEXT, provenance_json TEXT, recorded_utc TEXT NOT NULL);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def connect(path: str | None = None) -> sqlite3.Connection:
    """Open (creating) the local state file. `path=None` or ':memory:' gives a throwaway in-memory store."""
    db = sqlite3.connect(path or ":memory:", timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def pipeline_db():
    return sys.modules[__name__]


def upsert_segment(db, seg, scroll=None, route=None, fit_id=None, parent=None):
    db.execute("INSERT INTO segment(seg,scroll,route,fit_id,parent,created_utc) VALUES(?,?,?,?,?,?) "
               "ON CONFLICT(seg) DO UPDATE SET scroll=COALESCE(excluded.scroll,scroll),"
               "route=COALESCE(excluded.route,route),fit_id=COALESCE(excluded.fit_id,fit_id),"
               "parent=COALESCE(excluded.parent,parent)", (seg, scroll, route, fit_id, parent, _now()))


def start_attempt(db, seg, stage, tool=None, host=None, cmd=None, params=None) -> int:
    cur = db.execute("INSERT INTO attempt(seg,stage,state,started_utc,host,pid,tool,cmd,params_json) "
                     "VALUES(?,?,?,?,?,?,?,?,?)",
                     (seg, stage, "running", _now(), host, os.getpid(), tool, cmd,
                      json.dumps(params) if params is not None else None))
    return int(cur.lastrowid)


def finish_attempt(db, attempt_id, state, detail=None, reason=None):
    db.execute("UPDATE attempt SET state=?, ended_utc=?, detail=COALESCE(?,detail), reason=COALESCE(?,reason) "
               "WHERE id=?", (state, _now(), detail, reason, attempt_id))


def record_metric(db, seg, name, value=None, text=None, stage=None, attempt_id=None,
                  control_name=None, control_value=None):
    db.execute("INSERT INTO metric(seg,stage,attempt_id,name,value,text_value,control_name,control_value,"
               "recorded_utc) VALUES(?,?,?,?,?,?,?,?,?)",
               (seg, stage, attempt_id, name, value, text, control_name, control_value, _now()))


def record_artifact(db, seg, stage, kind, path, attempt_id=None, meta=None, with_md5=False):
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    size = st.st_size
    if os.path.isdir(path):
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(path) for f in fs)
    db.execute("INSERT OR IGNORE INTO artifact(seg,stage,attempt_id,kind,path,bytes,mtime,md5,meta_json,recorded_utc) "
               "VALUES(?,?,?,?,?,?,?,?,?,?)",
               (seg, stage, attempt_id, kind, os.path.abspath(path), size, st.st_mtime, None,
                json.dumps(meta or {}), _now()))


def record_seed(db, scroll, seg, xyz, host=None, source=None, ct_value=None, score=None, dist_l4=None,
                window_mean=None, provenance=None):
    upsert_segment(db, seg, scroll=scroll, route="A")
    db.execute("INSERT OR IGNORE INTO seed(seg,scroll,x,y,z,score,dist_l4,centre,window_mean,source,host,"
               "provenance_json,recorded_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (seg, scroll, int(xyz[0]), int(xyz[1]), int(xyz[2]), score, dist_l4, ct_value, window_mean,
                source, host, json.dumps(provenance or {}), _now()))
    for n, v in zip(("seed_x", "seed_y", "seed_z"), xyz):
        record_metric(db, seg, n, float(v), stage="seed")


def seed_xyz_of(db, seg):
    r = db.execute("SELECT x,y,z FROM seed WHERE seg=?", (seg,)).fetchone()
    return (int(r["x"]), int(r["y"]), int(r["z"])) if r else None


def set_paused(db, seg, stage, paused, by=None):
    db.execute("INSERT INTO stage_flag(seg,stage,paused,paused_by,updated_utc) VALUES(?,?,?,?,?) "
               "ON CONFLICT(seg,stage) DO UPDATE SET paused=excluded.paused, paused_by=excluded.paused_by, "
               "updated_utc=excluded.updated_utc", (seg, stage, 1 if paused else 0, by or "cloud-grow", _now()))


def is_paused(db, seg, stage) -> bool:
    r = db.execute("SELECT paused FROM stage_flag WHERE seg=? AND stage=?", (seg, stage)).fetchone()
    return bool(r and r["paused"])


def set_pipeline_setting(db, key, value, by=None):
    db.execute("INSERT INTO pipeline_setting(key,value,updated_utc,updated_by) VALUES(?,?,?,?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_utc=excluded.updated_utc, "
               "updated_by=excluded.updated_by", (key, str(value), _now(), by or "cloud-grow"))


def latest_metric(db, seg, name):
    r = db.execute("SELECT value, text_value FROM metric WHERE seg=? AND name=? ORDER BY id DESC LIMIT 1",
                   (seg, name)).fetchone()
    return None if r is None else (r["value"] if r["value"] is not None else r["text_value"])
