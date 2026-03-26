-- Xybernetex AI OpenEngine — Database Schema
-- SQLite-compatible DDL (also compatible with PostgreSQL with minor type changes)
--
-- This file is the canonical schema definition for the OpenEngine persistence
-- layer.  All tables used by the engine are defined here.
--
-- To initialise a fresh SQLite database:
--   sqlite3 openengine.db < schemas/schema.sql
--
-- For PostgreSQL, replace:
--   INTEGER PRIMARY KEY AUTOINCREMENT  →  SERIAL PRIMARY KEY
--   TEXT                               →  TEXT  (compatible as-is)
--   REAL                               →  DOUBLE PRECISION
-- ---------------------------------------------------------------------------

-- ── Runs ──────────────────────────────────────────────────────────────────────
-- One row per goal execution episode.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT        PRIMARY KEY,          -- UUID or local_YYYYMMDD_...
    user_id         TEXT        NOT NULL,
    goal            TEXT        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'queued',
                                                      -- queued | running | done | error
    steps_taken     INTEGER     NOT NULL DEFAULT 0,
    max_steps       INTEGER     NOT NULL DEFAULT 20,
    created_at      TEXT        NOT NULL,             -- ISO-8601 UTC
    started_at      TEXT,
    completed_at    TEXT,
    error_message   TEXT,
    capability_manifest_path TEXT,
    model           TEXT
);

-- ── Ticks ─────────────────────────────────────────────────────────────────────
-- One row per step within a run (the "tick log").
CREATE TABLE IF NOT EXISTS ticks (
    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT        NOT NULL REFERENCES runs(run_id),
    step            INTEGER     NOT NULL,
    action_name     TEXT        NOT NULL,
    action_index    INTEGER     NOT NULL,
    state_vector    TEXT,                             -- JSON float array
    aspects_before  INTEGER     NOT NULL DEFAULT 0,
    aspects_after   INTEGER     NOT NULL DEFAULT 0,
    exec_success    INTEGER     NOT NULL DEFAULT 0,   -- 0/1 boolean
    exec_returncode INTEGER,
    reward          REAL,
    llm_prompt      TEXT,
    llm_response    TEXT,
    tool_name       TEXT,
    tool_params     TEXT,                             -- JSON
    tool_result     TEXT,                             -- JSON
    created_at      TEXT        NOT NULL
);

-- ── Artifacts ─────────────────────────────────────────────────────────────────
-- Structured outputs produced by WRITE_ARTIFACT actions.
CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT        NOT NULL REFERENCES runs(run_id),
    step            INTEGER     NOT NULL,
    artifact_type   TEXT        NOT NULL DEFAULT 'text',
                                                      -- text | json | markdown | html | code
    title           TEXT,
    content         TEXT        NOT NULL,
    created_at      TEXT        NOT NULL
);

-- ── Goal aspects ──────────────────────────────────────────────────────────────
-- The decomposed sub-goals identified during the PLAN action.
CREATE TABLE IF NOT EXISTS aspects (
    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT        NOT NULL REFERENCES runs(run_id),
    aspect_text     TEXT        NOT NULL,
    completed       INTEGER     NOT NULL DEFAULT 0,   -- 0/1 boolean
    completed_at_step INTEGER,
    created_at      TEXT        NOT NULL
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ticks_run_id   ON ticks(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run  ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_aspects_run    ON aspects(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_user      ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_status    ON runs(status);
