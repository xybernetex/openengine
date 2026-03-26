"""
working_memory.py
Xybernetex Agent Chassis — Working Memory

Full relational schema. The agent queries this database directly.
Artifacts are never compressed. Errors are never lost.
The compiler only touches reasoning narratives.

Tables:
  runs              — one per agent run
  steps             — one per agent loop iteration
  artifacts         — everything the agent produces (CODE, DOCUMENT, NARRATIVE, ANALYSIS, DIAGRAM, DECISION)
  executions        — execution results for CODE artifacts
  errors            — structured error log, queryable, tracks resolution
  goal_aspects      — agent's decomposition of its own goal, tracks completion
  loop_state        — per-topic action counts for loop detection
  context_compression — compressed reasoning narratives only
"""

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# ── DB path: prefer env var (outside Google Drive), fall back to local memory/ ─
# The local memory/ directory is inside Google Drive which corrupts the SQLite
# WAL journal during sync.  Set DATABASE_PATH (or DATABASE_URL) in .env to a
# path outside Drive, e.g.:  C:\xybernetex\working_memory.db
import os as _os
_db_env = (
    _os.getenv("DATABASE_PATH", "").strip()
    or _os.getenv("DATABASE_URL", "").strip().removeprefix("sqlite:///")
)
if _db_env:
    DB_PATH = Path(_db_env)
else:
    DB_PATH = Path(__file__).parent / "memory" / "working_memory.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Aspect Key Canonicalization ──────────────────────────────────────────────

# ── Prompt-Bleed Blocklist ───────────────────────────────────────────────────
# V2.3: The LLM reads aggressive system prompt directives ("CRITICAL",
# "FORBIDDEN", "override") and hallucinates them as goal aspects it needs
# to "solve". These phantom aspects stay in_progress forever and permanently
# block can_terminate(). Any aspect key that matches a blocklist fragment
# is silently dropped before it ever reaches the database.
# Individual WORDS that indicate prompt-bleed when found in any aspect key.
# These are split from compound keys so the word-level intersection works.
ASPECT_BLOCKLIST_WORDS = frozenset({
    # System prompt directive language
    "override", "obey", "forbidden", "critical", "directive", "mandate",
    "executive", "instruction", "compliance", "allowed",
    # Completion-status language the LLM parrots from warnings
    "deliverables", "missing", "incomplete", "manifest", "rewriting",
    "previously", "completed", "priority",
    # Meta/self-referential keys — not real goal aspects
    "must", "not", "do", "you", "are", "your", "only",
    "system", "prompt", "step", "current_state",
})

# Exact full-key matches for common hallucinated aspects that pass
# the word-level check (e.g. "planning" is a real word but not a goal aspect
# when used alone as a context state label).
ASPECT_BLOCKLIST_EXACT = frozenset({
    "step", "current_state", "planning", "deliverables_missing",
    "incomplete_manifest", "override_must_obey", "system_prompt",
})

# V2.3: Hard cap on total unique aspect keys per run.
# LLM decomposes most goals into 4-6 aspects; 10 is generous.
# After the cap, new keys are rejected but existing keys can still be updated.
ASPECT_CAP = 10


def _is_prompt_bleed(canonical_key: str) -> bool:
    """Return True if the key looks like it was derived from system prompt language."""
    # Exact match on known hallucinated keys
    if canonical_key in ASPECT_BLOCKLIST_EXACT:
        return True
    # Word-level intersection: any blocklisted word in the key → reject
    words = set(canonical_key.split("_"))
    return bool(words & ASPECT_BLOCKLIST_WORDS)


def canonicalize_aspect_key(name: str, existing_keys: list = None) -> str:
    """
    Normalize aspect names so that 'clarify requirements',
    'clarify_requirements', and 'Clarify Requirements' all resolve
    to the same canonical key: 'clarify_requirements'.

    Aggressively strips LLM hallucinated key=value syntax BEFORE
    normalization:  'option_analysis=complete' → 'option_analysis',
    'planning: in_progress' → 'planning'.

    When *existing_keys* is provided, performs word-overlap matching
    against existing aspects to prevent the LLM from inventing synonyms
    (e.g. 'cost' → 'financial_viability' → 'economic_cost' → orphan deadlock).
    Requires ≥50% word overlap to merge.
    """
    # Phase 0: strip LLM syntax hallucinations (Llama-3 emits 'key=value' or
    # 'key: value' as the aspect name itself — keep only the key portion)
    clean = name.strip().lower()
    for delim in ("=", ":", " - "):
        if delim in clean:
            clean = clean.split(delim)[0]

    s = clean.strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    if existing_keys:
        new_words = set(s.split("_"))
        best_key, best_score = None, 0.0
        for key in existing_keys:
            key_words = set(key.split("_"))
            if not key_words or not new_words:
                continue
            overlap = len(new_words & key_words)
            # Jaccard-ish: overlap / size of the smaller set
            score = overlap / min(len(new_words), len(key_words))
            if score > best_score:
                best_score = score
                best_key = key
        # ≥50% word overlap → merge into existing key (prevents orphans)
        if best_key and best_score >= 0.50:
            return best_key

    return s

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    goal            TEXT NOT NULL,
    user_id         TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    step_count      INTEGER DEFAULT 0,
    conclusion      TEXT,
    deliverable_manifest TEXT,
    capability_manifest TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    step_number     INTEGER NOT NULL,
    action_type     TEXT NOT NULL,
    label           TEXT NOT NULL,
    focus           TEXT NOT NULL,
    rationale       TEXT,
    reward          REAL,
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    step_id         TEXT NOT NULL REFERENCES steps(id),
    artifact_type   TEXT NOT NULL,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS executions (
    id              TEXT PRIMARY KEY,
    artifact_id     TEXT NOT NULL REFERENCES artifacts(id),
    stdout          TEXT,
    stderr          TEXT,
    returncode      INTEGER,
    success         INTEGER NOT NULL DEFAULT 0,
    timed_out       INTEGER NOT NULL DEFAULT 0,
    executed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    step_id         TEXT NOT NULL REFERENCES steps(id),
    artifact_id     TEXT,
    error_type      TEXT NOT NULL,
    error_text      TEXT NOT NULL,
    topic           TEXT NOT NULL,
    resolved        INTEGER NOT NULL DEFAULT 0,
    resolved_at_step INTEGER,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS goal_aspects (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    aspect          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    blocking_error_id TEXT,
    last_updated_step INTEGER,
    notes           TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(run_id, aspect)
);

CREATE TABLE IF NOT EXISTS loop_state (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    topic           TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    count           INTEGER NOT NULL DEFAULT 0,
    last_attempt_step INTEGER,
    last_artifact_id TEXT,
    UNIQUE(run_id, topic, action_type)
);

CREATE TABLE IF NOT EXISTS context_compression (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    after_step      INTEGER NOT NULL,
    summary         TEXT NOT NULL,
    raw_chars       INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);

-- Decision ticks — full telemetry for every planning decision
CREATE TABLE IF NOT EXISTS ticks (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    step_number     INTEGER NOT NULL,
    state_hash      TEXT NOT NULL,
    candidates      TEXT NOT NULL,   -- JSON array of candidate actions with predicted rewards
    chosen_action   TEXT NOT NULL,
    chosen_label    TEXT NOT NULL,
    override_applied INTEGER NOT NULL DEFAULT 0,
    override_reason TEXT,
    actual_reward   REAL,            -- filled in after execution
    judge_evaluation TEXT,           -- JSON: judge_evaluation block from API response
    created_at      TEXT NOT NULL
);

-- Decision cache — state_hash → best known action
CREATE TABLE IF NOT EXISTS decision_cache (
    state_hash      TEXT PRIMARY KEY,
    best_action     TEXT NOT NULL,
    best_label      TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.0,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    avg_reward      REAL NOT NULL DEFAULT 0.0,
    last_seen_step  INTEGER,
    last_updated    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_steps_run         ON steps(run_id, step_number);
CREATE INDEX IF NOT EXISTS idx_artifacts_run     ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type    ON artifacts(run_id, artifact_type);
CREATE INDEX IF NOT EXISTS idx_errors_run        ON errors(run_id, resolved);
CREATE INDEX IF NOT EXISTS idx_errors_topic      ON errors(run_id, topic);
CREATE INDEX IF NOT EXISTS idx_goal_aspects_run  ON goal_aspects(run_id);
CREATE INDEX IF NOT EXISTS idx_loop_run          ON loop_state(run_id, topic);
-- Atomic Evidence Units — Archivist output, normalized observations
CREATE TABLE IF NOT EXISTS aeu (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    step_id         TEXT REFERENCES steps(id),
    source          TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    raw_content     TEXT NOT NULL,
    structured      TEXT NOT NULL DEFAULT '{}',
    content_hash    TEXT NOT NULL,
    provenance      TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

-- Safety Monitor events
CREATE TABLE IF NOT EXISTS safety_events (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    tick_id         TEXT REFERENCES ticks(id),
    step_number     INTEGER NOT NULL,
    rule_id         TEXT NOT NULL,
    severity        TEXT NOT NULL,
    message         TEXT NOT NULL,
    evidence        TEXT NOT NULL DEFAULT '{}',
    action_taken    TEXT NOT NULL DEFAULT 'none',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticks_run         ON ticks(run_id, step_number);
CREATE INDEX IF NOT EXISTS idx_cache_hash        ON decision_cache(state_hash);
CREATE INDEX IF NOT EXISTS idx_aeu_run           ON aeu(run_id, content_type);
CREATE INDEX IF NOT EXISTS idx_aeu_hash          ON aeu(content_hash);
CREATE INDEX IF NOT EXISTS idx_safety_run        ON safety_events(run_id, severity);

-- Verification evidence — structured proof that something was tested
CREATE TABLE IF NOT EXISTS verification_evidence (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    step_id         TEXT NOT NULL REFERENCES steps(id),
    artifact_id     TEXT REFERENCES artifacts(id),
    tests_run       TEXT NOT NULL DEFAULT '[]',
    command_outputs  TEXT NOT NULL DEFAULT '[]',
    assertions_checked TEXT NOT NULL DEFAULT '[]',
    failures        TEXT NOT NULL DEFAULT '[]',
    success         INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_verification_run  ON verification_evidence(run_id);

-- Termination events — logged when DONE is proposed but blocked or overridden
CREATE TABLE IF NOT EXISTS termination_events (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    step_number     INTEGER NOT NULL,
    proposed_action TEXT NOT NULL DEFAULT 'DONE',
    allowed         INTEGER NOT NULL DEFAULT 0,
    blockers        TEXT NOT NULL DEFAULT '[]',
    goal_aspect_status TEXT NOT NULL DEFAULT '{}',
    artifact_inventory TEXT NOT NULL DEFAULT '[]',
    fallback_action TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_termination_run   ON termination_events(run_id);

-- Artifact relationship graph — Integrator output, typed edges between artifacts
CREATE TABLE IF NOT EXISTS artifact_edges (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(id),
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    rel_type    TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL,
    UNIQUE(run_id, source_id, target_id, rel_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_run     ON artifact_edges(run_id, rel_type);
CREATE INDEX IF NOT EXISTS idx_edges_target  ON artifact_edges(target_id);
"""

# Artifact types the agent can produce
ARTIFACT_TYPES = {
    "CODE":      "executable Python code",
    "DOCUMENT":  "structured prose document (spec, memo, report)",
    "NARRATIVE": "human-readable explanation or argument",
    "ANALYSIS":  "structured analysis (risk register, comparison, scorecard)",
    "DIAGRAM":   "structured text diagram (system, flow, relationship)",
    "DECISION":  "formal decision record with rationale and alternatives",
    "REPLAN":    "agent pivot — resets strategy when current approach is failing",
}


class WorkingMemory:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path                    # exposed for WorldEngine agents
        self.conn    = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate_schema()                    # add new columns to existing DBs

    def _migrate_schema(self):
        """
        Add new columns to pre-existing databases that predate the World Engine.
        SQLite does not support IF NOT EXISTS in ALTER TABLE; use try/except.
        """
        migrations = [
            "ALTER TABLE artifacts ADD COLUMN content_hash  TEXT",
            "ALTER TABLE artifacts ADD COLUMN is_duplicate  INTEGER DEFAULT 0",
            "ALTER TABLE artifacts ADD COLUMN duplicate_of  TEXT",
        ]
        for sql in migrations:
            try:
                self.conn.execute(sql)
            except Exception:
                pass          # column already exists
        self.conn.commit()

        # Migrate: add columns that may not exist in older DB files
        for _col_sql in [
            "ALTER TABLE runs ADD COLUMN deliverable_manifest TEXT",
            "ALTER TABLE runs ADD COLUMN user_id TEXT",
            "ALTER TABLE runs ADD COLUMN capability_manifest TEXT",
        ]:
            try:
                self.conn.execute(_col_sql)
                self.conn.commit()
            except Exception:
                pass  # Column already exists

    # ── Runs ──────────────────────────────────────────────────────────────────

    def create_run(
        self,
        goal: str,
        user_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO runs (id, goal, user_id, status, created_at) "
            "VALUES (?, ?, ?, 'running', ?)",
            (run_id, goal, user_id, _now())
        )
        self.conn.commit()
        return run_id

    def mark_complete(self, run_id: str, conclusion: str):
        self.conn.execute(
            "UPDATE runs SET status='complete', completed_at=?, conclusion=? WHERE id=?",
            (_now(), conclusion, run_id)
        )
        self.conn.commit()

    def mark_failed(self, run_id: str, reason: str):
        self.conn.execute(
            "UPDATE runs SET status='failed', completed_at=?, conclusion=? WHERE id=?",
            (_now(), reason, run_id)
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else {}

    # Capability Manifest

    def set_capability_manifest(self, run_id: str, manifest: dict):
        """Store the run-specific tool access manifest for this run."""
        self.conn.execute(
            "UPDATE runs SET capability_manifest=? WHERE id=?",
            (json.dumps(manifest), run_id)
        )
        self.conn.commit()

    def get_capability_manifest(self, run_id: str) -> dict | None:
        """Return the parsed capability manifest dict, or None if not yet set."""
        row = self.conn.execute(
            "SELECT capability_manifest FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row or not row["capability_manifest"]:
            return None
        try:
            return json.loads(row["capability_manifest"])
        except Exception:
            return None

    # ── Deliverable Manifest ───────────────────────────────────────────────────

    def set_deliverable_manifest(self, run_id: str, manifest: dict):
        """Store the agent's declared deliverable manifest for this run."""
        self.conn.execute(
            "UPDATE runs SET deliverable_manifest=? WHERE id=?",
            (json.dumps(manifest), run_id)
        )
        self.conn.commit()

    def get_deliverable_manifest(self, run_id: str) -> dict | None:
        """Return the parsed manifest dict, or None if not yet set."""
        row = self.conn.execute(
            "SELECT deliverable_manifest FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row or not row["deliverable_manifest"]:
            return None
        try:
            return json.loads(row["deliverable_manifest"])
        except Exception:
            return None

    def amend_manifest(self, run_id: str, new_deliverables: list):
        """
        Add new deliverables to the manifest. Never removes existing ones —
        the agent cannot shrink its commitment.
        new_deliverables: list of {"type": "...", "label": "..."}
        """
        manifest = self.get_deliverable_manifest(run_id) or {
            "deliverables": [], "done_when": ""
        }
        existing_keys = {
            (d.get("type", ""), d.get("label", "").lower())
            for d in manifest.get("deliverables", [])
        }
        added = 0
        for nd in new_deliverables:
            key = (nd.get("type", ""), nd.get("label", "").lower())
            if key not in existing_keys and nd.get("type") and nd.get("label"):
                manifest["deliverables"].append(nd)
                existing_keys.add(key)
                added += 1
        if added:
            self.set_deliverable_manifest(run_id, manifest)

    def check_deliverables_satisfied(self, run_id: str) -> tuple:
        """
        Check whether all declared deliverables have been produced as artifacts.
        Returns (satisfied: bool, missing: list[dict])
        If no manifest is set, returns (False, []) — caller falls back to aspect logic.

        Only WRITE_ARTIFACT steps count toward manifest completion.
        Matching requires BOTH:
          1. artifact_type == deliverable type (DOCUMENT/ANALYSIS/DECISION)
          2. Jaccard word overlap between artifact title and deliverable label >= 0.25
        This prevents "Risk Register" (DOCUMENT) from satisfying "GP Memo" (DOCUMENT).
        """
        manifest = self.get_deliverable_manifest(run_id)
        if not manifest or not manifest.get("deliverables"):
            return False, []

        # Pull type + title for all WRITE_ARTIFACT artifacts in this run
        rows = self.conn.execute(
            """SELECT a.artifact_type, a.title FROM artifacts a
               JOIN steps s ON a.step_id = s.id
               WHERE a.run_id=? AND s.action_type='WRITE_ARTIFACT'""",
            (run_id,)
        ).fetchall()
        produced = [(r[0], r[1] or "") for r in rows]

        _stop = frozenset({
            "the","a","an","is","are","was","were","be","been","of","in","on",
            "at","to","for","with","by","from","this","that","and","or","each",
            "all","any","key","across","per","its","their","s","t","3","1","2",
        })

        def _lw(text: str) -> set:
            return set(re.sub(r"[^\w\s]", " ", (text or "").lower()).split()) - _stop

        def _satisfied_by(deliverable: dict) -> bool:
            d_words = _lw(deliverable.get("label", ""))
            for _art_type, art_title in produced:
                # Type is intentionally not checked: the agent may produce an
                # ANALYSIS deliverable as a DOCUMENT artifact or vice-versa.
                # Label word overlap is the authoritative discriminator.
                a_words = _lw(art_title)
                if not d_words or not a_words:
                    continue
                union = len(d_words | a_words)
                if union == 0:
                    continue
                if len(d_words & a_words) / union >= 0.25:
                    return True
            return False

        missing = [d for d in manifest["deliverables"] if not _satisfied_by(d)]
        return len(missing) == 0, missing

    # ── Steps ─────────────────────────────────────────────────────────────────

    def record_step(self, run_id: str, step_number: int, action_type: str,
                    label: str, focus: str, rationale: str = "") -> str:
        step_id = f"step_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO steps
               (id, run_id, step_number, action_type, label, focus, rationale, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (step_id, run_id, step_number, action_type, label, focus, rationale, _now())
        )
        self.conn.execute(
            "UPDATE runs SET step_count = step_count + 1 WHERE id=?", (run_id,)
        )
        self.conn.commit()
        return step_id

    def complete_step(self, step_id: str, reward: float):
        self.conn.execute(
            "UPDATE steps SET reward=?, completed_at=? WHERE id=?",
            (reward, _now(), step_id)
        )
        self.conn.commit()

    def get_steps(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM steps WHERE run_id=? ORDER BY step_number", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Artifacts ─────────────────────────────────────────────────────────────

    def save_artifact(self, run_id: str, step_id: str, artifact_type: str,
                      title: str, content: str, metadata: dict = None) -> str:
        art_id = f"art_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO artifacts
               (id, run_id, step_id, artifact_type, title, content, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (art_id, run_id, step_id, artifact_type, title, content,
             json.dumps(metadata or {}), _now())
        )
        self.conn.commit()
        return art_id

    # ── Archivist integration ──────────────────────────────────────────────────

    def _archivist_annotate(self, artifact_id: str, content_hash: str,
                            is_duplicate: bool, duplicate_of: str | None):
        """Write Archivist dedup results back onto the artifact row."""
        self.conn.execute(
            """UPDATE artifacts
               SET content_hash=?, is_duplicate=?, duplicate_of=?
               WHERE id=?""",
            (content_hash, 1 if is_duplicate else 0, duplicate_of, artifact_id)
        )
        self.conn.commit()

    def _get_artifact_hashes(self, run_id: str,
                             exclude_id: str | None = None) -> list:
        """
        Returns [(id, content_hash, content)] for all artifacts in this run.
        Excludes exclude_id (the artifact just saved, to avoid self-comparison).
        Used by Archivist for near-duplicate detection.
        """
        if exclude_id:
            rows = self.conn.execute(
                """SELECT id, content_hash, content FROM artifacts
                   WHERE run_id=? AND id != ?
                   ORDER BY created_at""",
                (run_id, exclude_id)
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT id, content_hash, content FROM artifacts
                   WHERE run_id=?
                   ORDER BY created_at""",
                (run_id,)
            ).fetchall()
        return [(r["id"], r["content_hash"], r["content"]) for r in rows]

    # ── Integrator integration ─────────────────────────────────────────────────

    def _integrator_get_edges(self, run_id: str,
                              rel_type: str | None = None) -> list:
        """
        Read artifact_edges for this run.  Called from the main thread by
        Integrator.get_coverage_gaps() and format_coverage_for_prompt().
        """
        if rel_type:
            rows = self.conn.execute(
                """SELECT * FROM artifact_edges
                   WHERE run_id=? AND rel_type=?
                   ORDER BY confidence DESC""",
                (run_id, rel_type)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM artifact_edges WHERE run_id=? ORDER BY created_at",
                (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def save_execution(self, artifact_id: str, execution: dict) -> str:
        exec_id = f"exec_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO executions
               (id, artifact_id, stdout, stderr, returncode, success, timed_out, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (exec_id, artifact_id,
             execution.get('stdout', '')[:5000],
             execution.get('stderr', '')[:2000],
             execution.get('returncode', -1),
             1 if execution.get('success') else 0,
             1 if execution.get('timed_out') else 0,
             _now())
        )
        self.conn.commit()
        return exec_id

    def get_artifacts(self, run_id: str, artifact_type: str = None) -> list:
        # LEFT JOIN steps to surface step_number for Curator scoring display.
        # New Archivist columns (content_hash, is_duplicate, duplicate_of) are
        # included automatically via a.* once the migration has run.
        base = """
            SELECT a.*, COALESCE(s.step_number, 0) AS step_number
            FROM artifacts a
            LEFT JOIN steps s ON s.id = a.step_id
            WHERE a.run_id=?
        """
        if artifact_type:
            rows = self.conn.execute(
                base + " AND a.artifact_type=? ORDER BY a.created_at",
                (run_id, artifact_type)
            ).fetchall()
        else:
            rows = self.conn.execute(
                base + " ORDER BY a.created_at", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_artifact(self, run_id: str, artifact_type: str) -> dict | None:
        row = self.conn.execute(
            """SELECT a.*, e.stdout, e.stderr, e.returncode, e.success
               FROM artifacts a
               LEFT JOIN executions e ON e.artifact_id = a.id
               WHERE a.run_id=? AND a.artifact_type=?
               ORDER BY a.created_at DESC LIMIT 1""",
            (run_id, artifact_type)
        ).fetchone()
        return dict(row) if row else None

    def get_artifact_index(self, run_id: str) -> list:
        """Titles and types only for context dashboard."""
        rows = self.conn.execute(
            """SELECT a.artifact_type, a.title, a.created_at,
               e.success as exec_success, e.returncode
               FROM artifacts a
               LEFT JOIN executions e ON e.artifact_id = a.id
               WHERE a.run_id=? ORDER BY a.created_at""",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Errors ────────────────────────────────────────────────────────────────

    def log_error(self, run_id: str, step_id: str, error_type: str,
                  error_text: str, topic: str, artifact_id: str = None) -> str:
        err_id = f"err_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO errors
               (id, run_id, step_id, artifact_id, error_type, error_text, topic, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (err_id, run_id, step_id, artifact_id, error_type,
             error_text, topic, _now())
        )
        self.conn.commit()
        return err_id

    def resolve_errors_by_topic(self, run_id: str, topic: str, at_step: int):
        self.conn.execute(
            """UPDATE errors SET resolved=1, resolved_at_step=?
               WHERE run_id=? AND topic=? AND resolved=0""",
            (at_step, run_id, topic)
        )
        self.conn.commit()

    def get_open_errors(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM errors WHERE run_id=? AND resolved=0 ORDER BY created_at",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_errors(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM errors WHERE run_id=? ORDER BY created_at", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Goal Aspects ──────────────────────────────────────────────────────────

    def upsert_goal_aspect(self, run_id: str, aspect: str, status: str,
                           step_number: int, notes: str = "",
                           blocking_error_id: str = None):
        # Canonicalize to prevent drift between 'clarify requirements' and 'clarify_requirements'
        # Pass existing keys so fuzzy matching merges synonyms (e.g. 'cost' ↔ 'cost_analysis')
        existing_keys = [
            row["aspect"] for row in self.conn.execute(
                "SELECT DISTINCT aspect FROM goal_aspects WHERE run_id=?", (run_id,)
            ).fetchall()
        ]
        canonical = canonicalize_aspect_key(aspect, existing_keys=existing_keys)

        # V2.3 FIX A: Prompt-bleed blocklist — silently drop aspects derived
        # from system prompt language (e.g. 'override_must_obey', 'deliverables_missing')
        if _is_prompt_bleed(canonical):
            return  # silent drop — do not persist hallucinated directive aspects

        existing = self.conn.execute(
            "SELECT id FROM goal_aspects WHERE run_id=? AND aspect=?",
            (run_id, canonical)
        ).fetchone()
        if existing:
            self.conn.execute(
                """UPDATE goal_aspects
                   SET status=?, last_updated_step=?, notes=?, blocking_error_id=?
                   WHERE run_id=? AND aspect=?""",
                (status, step_number, notes, blocking_error_id, run_id, canonical)
            )
        else:
            # V2.3 FIX B: Aspect cap — reject new keys once we hit ASPECT_CAP.
            # Existing keys can always be updated (handled above).
            if len(existing_keys) >= ASPECT_CAP:
                return  # silent drop — too many aspects, likely hallucination
            self.conn.execute(
                """INSERT INTO goal_aspects
                   (id, run_id, aspect, status, last_updated_step,
                    notes, blocking_error_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"asp_{uuid.uuid4().hex[:8]}", run_id, canonical, status,
                 step_number, notes, blocking_error_id, _now())
            )
        self.conn.commit()

    def get_goal_aspects(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM goal_aspects WHERE run_id=? ORDER BY created_at",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def reset_aspects_for_replan(self, run_id: str, step_num: int):
        """
        Reset all in-progress aspects to pending when the agent takes a REPLAN action.
        Completed aspects are left untouched — work already done is not discarded.
        Blocked aspects are also reset to pending so the agent can attempt them fresh.
        """
        self.conn.execute(
            """UPDATE goal_aspects
               SET status='pending',
                   last_updated_step=?,
                   notes='Reset by REPLAN at step ' || ?
               WHERE run_id=? AND status IN ('in_progress', 'blocked')""",
            (step_num, step_num, run_id)
        )
        self.conn.commit()

    # ── Loop State ────────────────────────────────────────────────────────────

    def increment_loop(self, run_id: str, topic: str, action_type: str,
                       step_number: int, artifact_id: str = None):
        existing = self.conn.execute(
            "SELECT id FROM loop_state WHERE run_id=? AND topic=? AND action_type=?",
            (run_id, topic, action_type)
        ).fetchone()
        if existing:
            self.conn.execute(
                """UPDATE loop_state
                   SET count=count+1, last_attempt_step=?, last_artifact_id=?
                   WHERE run_id=? AND topic=? AND action_type=?""",
                (step_number, artifact_id, run_id, topic, action_type)
            )
        else:
            self.conn.execute(
                """INSERT INTO loop_state
                   (id, run_id, topic, action_type, count, last_attempt_step, last_artifact_id)
                   VALUES (?, ?, ?, ?, 1, ?, ?)""",
                (f"lp_{uuid.uuid4().hex[:8]}", run_id, topic,
                 action_type, step_number, artifact_id)
            )
        self.conn.commit()

    def get_loop_state(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM loop_state WHERE run_id=? ORDER BY count DESC",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Context Compression ───────────────────────────────────────────────────

    def save_compressed(self, run_id: str, after_step: int,
                        summary: str, raw_chars: int):
        cid = f"ctx_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO context_compression
               (id, run_id, after_step, summary, raw_chars, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cid, run_id, after_step, summary, raw_chars, _now())
        )
        self.conn.commit()

    def get_latest_compressed(self, run_id: str) -> str | None:
        row = self.conn.execute(
            """SELECT summary FROM context_compression
               WHERE run_id=? ORDER BY after_step DESC LIMIT 1""",
            (run_id,)
        ).fetchone()
        return row['summary'] if row else None

    def get_reasoning_history(self, run_id: str, max_chars: int = 2000) -> str:
        """Get recent reasoning narrative for compression input."""
        steps = self.get_steps(run_id)
        parts = []
        for step in steps:
            arts = self.conn.execute(
                """SELECT artifact_type, title, content FROM artifacts
                   WHERE step_id=? AND artifact_type NOT IN ('CODE')
                   ORDER BY created_at""",
                (step['id'],)
            ).fetchall()
            if arts:
                part = f"STEP {step['step_number']} [{step['action_type']}] {step['label']}\n"
                for a in arts:
                    part += f"  {a['title']}: {a['content'][:200]}\n"
                parts.append(part)
        full = "\n".join(parts)
        return full[-max_chars:] if len(full) > max_chars else full

    # ── Context Dashboard ─────────────────────────────────────────────────────

    def build_context(self, run_id: str, step_num: int) -> dict:
        """
        Assemble the full structured context the agent needs.
        Queries the database directly — no text parsing, no compression loss.
        """
        run            = self.get_run(run_id)
        steps          = self.get_steps(run_id)
        artifact_index = self.get_artifact_index(run_id)
        open_errors    = self.get_open_errors(run_id)
        goal_aspects   = self.get_goal_aspects(run_id)
        loop_state     = self.get_loop_state(run_id)
        compressed     = self.get_latest_compressed(run_id)
        latest_code    = self.get_latest_artifact(run_id, 'CODE')

        recent_steps = []
        for s in steps[-3:]:
            arts = self.conn.execute(
                "SELECT artifact_type, title FROM artifacts WHERE step_id=?",
                (s['id'],)
            ).fetchall()
            recent_steps.append({
                "step":      s['step_number'],
                "type":      s['action_type'],
                "label":     s['label'],
                "reward":    s['reward'],
                "artifacts": [dict(a) for a in arts],
            })

        # Recent artifact content — last 3 non-empty artifacts with full text
        # Used by QUERY_MEMORY, COMPARE_OPTIONS, and format_context_for_prompt
        all_artifacts = self.get_artifacts(run_id)
        recent_artifact_content = []
        for a in reversed(all_artifacts):
            if a.get('content') and len(a['content'].strip()) > 50:
                recent_artifact_content.append({
                    "artifact_type": a['artifact_type'],
                    "title":         a['title'],
                    "content":       a['content'],
                })
                if len(recent_artifact_content) >= 3:
                    break
        recent_artifact_content.reverse()  # chronological order

        manifest = self.get_deliverable_manifest(run_id)
        capability_manifest = self.get_capability_manifest(run_id)
        _, missing_deliverables = self.check_deliverables_satisfied(run_id)

        return {
            "goal":                    run.get('goal', ''),
            "step_number":             step_num,
            "goal_aspects":            goal_aspects,
            "artifact_index":          artifact_index,
            "open_errors":             open_errors,
            "loop_state":              loop_state,
            "latest_code":             latest_code,
            "recent_steps":            recent_steps,
            "recent_artifact_content": recent_artifact_content,
            "narrative":               compressed or "(no reasoning summary yet)",
            "capability_manifest":     capability_manifest,
            "manifest":                manifest,
            "missing_deliverables":    missing_deliverables,
        }

    def format_context_for_prompt(self, ctx: dict) -> str:
        """
        Format the context dashboard as structured text for the planner prompt.
        Clean sections. Not a prose blob.
        """
        lines = []

        # Goal aspects
        lines.append("GOAL ASPECTS:")
        if ctx['goal_aspects']:
            for a in ctx['goal_aspects']:
                marker = {'complete':'[DONE]','in_progress':'[WIP]',
                          'blocked':'[BLOCKED]','pending':'[TODO]'}.get(a['status'],'[?]')
                note = f"  — {a['notes'][:60]}" if a.get('notes') else ""
                lines.append(f"  {marker:<10} {a['aspect']}{note}")
        else:
            lines.append("  (not yet decomposed — decompose goal in first step)")
        lines.append("")

        # Deliverable manifest
        manifest = ctx.get("manifest")
        missing_deliverables = ctx.get("missing_deliverables", [])
        if manifest and manifest.get("deliverables"):
            lines.append("DELIVERABLE MANIFEST (your committed outputs):")
            missing_set = {(d.get("type"), d.get("label")) for d in missing_deliverables}
            for d in manifest["deliverables"]:
                status = "[MISSING]" if (d.get("type"), d.get("label")) in missing_set else "[DONE]  "
                lines.append(f"  {status} [{d.get('type','?')}] {d.get('label','')}")
            if missing_deliverables:
                lines.append(
                    f"  → {len(missing_deliverables)} deliverable(s) still required before DONE"
                )
            else:
                lines.append("  → ALL DELIVERABLES SATISFIED — DONE is valid now")
            lines.append("")

        capability_manifest = ctx.get("capability_manifest")
        if capability_manifest is not None:
            lines.append("AVAILABLE TOOLS:")
            tools = capability_manifest.get("tools", [])
            if tools:
                for tool in tools:
                    actions = tool.get("actions") or []
                    action_text = ", ".join(actions) if actions else "any action"
                    risk = tool.get("risk", "unknown")
                    access = "read-only" if tool.get("read_only") else "mutating"
                    lines.append(
                        f"  [{risk} / {access}] {tool.get('name', '?')} -> {action_text}"
                    )
                    if tool.get("description"):
                        lines.append(f"    {tool['description'][:160]}")
            else:
                lines.append("  (none configured for this run)")
            lines.append("")

        # Artifacts index
        lines.append("ARTIFACTS PRODUCED:")
        if ctx['artifact_index']:
            for a in ctx['artifact_index']:
                exec_tag = ""
                if a['artifact_type'] == 'CODE':
                    if a.get('exec_success') == 1:
                        exec_tag = " [EXECUTED: PASS]"
                    elif a.get('exec_success') == 0 and a.get('returncode') is not None:
                        exec_tag = " [EXECUTED: FAIL]"
                lines.append(f"  [{a['artifact_type']:<12}] {a['title']}{exec_tag}")
        else:
            lines.append("  (none yet)")
        lines.append("")

        # Unresolved errors — raw text, never summarized
        lines.append("UNRESOLVED ERRORS:")
        if ctx['open_errors']:
            for e in ctx['open_errors']:
                lines.append(f"  [{e['error_type']}] topic={e['topic']}")
                lines.append(f"  {e['error_text'][:400]}")
                lines.append("")
        else:
            lines.append("  (none)")
        lines.append("")

        # Loop detection
        lines.append("LOOP DETECTION:")
        active = [l for l in ctx['loop_state'] if l['count'] >= 2]
        if active:
            for l in active:
                lines.append(f"  WARNING: {l['action_type']} on '{l['topic']}' "
                             f"attempted {l['count']} times — consider different approach")
        else:
            lines.append("  (no loops detected)")
        lines.append("")

        # Latest code verbatim snippet
        if ctx['latest_code']:
            lc = ctx['latest_code']
            lines.append(f"LATEST CODE: {lc['title']}")
            if lc.get('stderr') and lc.get('success') != 1:
                lines.append(f"  ERROR: {(lc['stderr'] or '')[:500]}")
            lines.append("  FIRST 20 LINES:")
            for line in (lc['content'] or '').split('\n')[:20]:
                lines.append(f"    {line}")
            lines.append("")

        # Recent artifact content — full text for synthesis actions
        recent_artifacts = ctx.get("recent_artifact_content", [])
        if recent_artifacts:
            lines.append("RECENT ARTIFACT CONTENT:")
            for a in recent_artifacts:
                lines.append(f"[{a['artifact_type']}] {a['title']}:")
                lines.append(a['content'][:600])
                lines.append("---")
            lines.append("")

        # Narrative
        lines.append("REASONING NARRATIVE:")
        lines.append(ctx['narrative'][:600])

        return "\n".join(lines)

    # ── Full report data ──────────────────────────────────────────────────────

    def get_full_report_data(self, run_id: str) -> dict:
        return {
            "run":        self.get_run(run_id),
            "steps":      self.get_steps(run_id),
            "artifacts":  self.get_artifacts(run_id),
            "errors":     self.get_all_errors(run_id),
            "aspects":    self.get_goal_aspects(run_id),
            "loop_state": self.get_loop_state(run_id),
        }

    # ── State Hashing ────────────────────────────────────────────────────────

    def compute_state_hash(self, run_id: str) -> str:
        """
        Hash the decision-relevant system state.
        Includes a goal fingerprint so different goals never share cache entries.
        Excludes step_number so cache works across similar states in the same goal.
        """
        aspects   = self.get_goal_aspects(run_id)
        errors    = self.get_open_errors(run_id)
        artifacts = self.get_artifact_index(run_id)
        loops     = self.get_loop_state(run_id)

        # Include goal fingerprint to isolate cache per goal
        run = self.get_run(run_id)
        goal_fp = hashlib.sha256(
            (run.get('goal', '') if run else '').encode()
        ).hexdigest()[:8]

        state = {
            "goal_fp":     goal_fp,
            "aspects":     {a['aspect']: a['status'] for a in aspects},
            "open_errors": sorted(e['topic'] for e in errors),
            "artifact_types": sorted(a['artifact_type'] for a in artifacts),
            "active_loops": sorted(
                f"{l['topic']}:{l['action_type']}:{l['count']}"
                for l in loops if l['count'] >= 2
            ),
        }
        raw = json.dumps(state, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    # ── Tick Ledger ───────────────────────────────────────────────────────────

    def record_tick(self, run_id: str, step_number: int, state_hash: str,
                    candidates: list, chosen_action: str, chosen_label: str,
                    override_applied: bool = False,
                    override_reason: str = None) -> str:
        """Record full decision telemetry for this planning tick."""
        tick_id = f"tick_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO ticks
               (id, run_id, step_number, state_hash, candidates, chosen_action,
                chosen_label, override_applied, override_reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tick_id, run_id, step_number, state_hash,
             json.dumps(candidates), chosen_action, chosen_label,
             1 if override_applied else 0, override_reason, _now())
        )
        self.conn.commit()
        return tick_id

    def update_tick_reward(self, tick_id: str, actual_reward: float,
                            judge_evaluation: dict = None):
        """Fill in actual reward and optional judge evaluation after execution completes."""
        self.conn.execute(
            "UPDATE ticks SET actual_reward=?, judge_evaluation=? WHERE id=?",
            (actual_reward,
             json.dumps(judge_evaluation) if judge_evaluation else None,
             tick_id)
        )
        self.conn.commit()

    def get_ticks(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM ticks WHERE run_id=? ORDER BY step_number",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Decision Cache ────────────────────────────────────────────────────────

    def check_cache(self, state_hash: str) -> dict | None:
        """
        Look up cached decision for this state.
        Returns the cached entry if confidence is high enough.
        """
        row = self.conn.execute(
            "SELECT * FROM decision_cache WHERE state_hash=?",
            (state_hash,)
        ).fetchone()
        if not row:
            return None
        entry = dict(row)
        # Only trust cache if we've seen this state multiple times with good rewards
        if entry['hit_count'] >= 2 and entry['avg_reward'] >= 0.65:
            return entry
        return None

    def cache_decision(self, state_hash: str, action: str, label: str,
                       reward: float, step_number: int):
        """Update decision cache with outcome of this decision."""
        existing = self.conn.execute(
            "SELECT hit_count, avg_reward FROM decision_cache WHERE state_hash=?",
            (state_hash,)
        ).fetchone()

        if existing:
            new_count  = existing['hit_count'] + 1
            new_avg    = (existing['avg_reward'] * existing['hit_count'] + reward) / new_count
            confidence = min(new_avg * (new_count / (new_count + 2)), 0.99)
            self.conn.execute(
                """UPDATE decision_cache
                   SET hit_count=?, avg_reward=?, confidence=?,
                       last_seen_step=?, last_updated=?
                   WHERE state_hash=?""",
                (new_count, new_avg, confidence, step_number, _now(), state_hash)
            )
        else:
            self.conn.execute(
                """INSERT INTO decision_cache
                   (state_hash, best_action, best_label, confidence,
                    hit_count, avg_reward, last_seen_step, last_updated)
                   VALUES (?, ?, ?, ?, 1, ?, ?, ?)""",
                (state_hash, action, label, 0.0, reward, step_number, _now())
            )
        self.conn.commit()

    def get_cache_stats(self, run_id: str) -> dict:
        """Return cache hit statistics for this run."""
        ticks = self.get_ticks(run_id)
        total = len(ticks)
        cached_hits = sum(
            1 for t in ticks
            if self.check_cache(t['state_hash']) is not None
        )
        return {"total_ticks": total, "cache_hits": cached_hits,
                "hit_rate": cached_hits/total if total else 0}


    # ── Archivist ─────────────────────────────────────────────────────────────

    def archive(self, run_id: str, step_id: str, source: str,
                content_type: str, raw_content: str,
                structured: dict = None, provenance: dict = None) -> str:
        """
        Archivist: normalize and store an observation as an Atomic Evidence Unit.
        Never interprets content — only structures and hashes it.
        Deduplicates by content hash within a run.
        """
        content_hash = hashlib.sha256(raw_content.encode()).hexdigest()[:16]

        # Dedup — skip if identical content already stored in this run
        existing = self.conn.execute(
            "SELECT id FROM aeu WHERE run_id=? AND content_hash=?",
            (run_id, content_hash)
        ).fetchone()
        if existing:
            return existing['id']

        aeu_id = f"aeu_{uuid.uuid4().hex[:12]}"
        sql = (
            "INSERT INTO aeu "
            "(id, run_id, step_id, source, content_type, raw_content, "
            " structured, content_hash, provenance, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        self.conn.execute(sql, (
            aeu_id, run_id, step_id, source, content_type,
            raw_content[:8000],
            json.dumps(structured or {}),
            content_hash,
            json.dumps(provenance or {}),
            _now()
        ))
        self.conn.commit()
        return aeu_id

    def get_aeus(self, run_id: str, content_type: str = None) -> list:
        if content_type:
            rows = self.conn.execute(
                "SELECT * FROM aeu WHERE run_id=? AND content_type=? ORDER BY created_at",
                (run_id, content_type)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM aeu WHERE run_id=? ORDER BY created_at", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Safety Monitor ────────────────────────────────────────────────────────

    def safety_check(self, run_id: str, step_num: int,
                     tick_id: str, hops_data: list,
                     rewards: list) -> tuple:
        """
        Safety Monitor: evaluate system state after every tick.
        Returns (should_halt: bool, events: list[dict])
        Never modifies agent state — only observes and signals.
        """
        events   = []
        halt     = False
        steps    = self.get_steps(run_id)
        errors   = self.get_open_errors(run_id)
        aspects  = self.get_goal_aspects(run_id)
        loop_st  = self.get_loop_state(run_id)

        # SM-01: Reward free-fall
        if len(rewards) >= 4:
            recent = rewards[-4:]
            if all(r < 0.45 for r in recent):
                events.append(self._safety_event(
                    run_id, tick_id, step_num, "SM-01", "HALT",
                    f"Reward free-fall: last 4 steps all below 0.45 ({[f'{r:.2f}' for r in recent]})",
                    {"recent_rewards": recent}, "halted"
                ))
                halt = True

        # SM-02: Same error unresolved after 3 CODE attempts
        if not halt:
            for err in errors:
                topic = err['topic']
                row = self.conn.execute(
                    "SELECT count FROM loop_state WHERE run_id=? AND topic=? AND action_type='CODE'",
                    (run_id, topic)
                ).fetchone()
                n = row['count'] if row else 0
                if n >= 3:
                    events.append(self._safety_event(
                        run_id, tick_id, step_num, "SM-02", "HALT",
                        f"Error '{topic}' unresolved after {n} CODE attempts",
                        {"topic": topic, "code_attempts": n,
                         "error_text": err['error_text'][:200]}, "halted"
                    ))
                    halt = True
                    break

        # SM-03: Corrective overrides firing on 5 of last 5 ticks
        # "Gate" overrides (self-assessment, DONE blocks, artifact QA) are the system
        # working correctly and don't indicate the planner is ignoring constraints.
        # Only "corrective" overrides (rule violations, drift, convergence-forced
        # mid-run rewrites) count toward this check.
        if not halt:
            _gate_prefixes = (
                "self-assessment", "done blocked", "artifact qa",
                "self_assessment", "done_blocked",
            )
            ticks = self.get_ticks(run_id)
            recent_t = ticks[-5:]
            if len(recent_t) >= 5:  # only meaningful once we have a full 5-tick window
                corrective = [
                    t for t in recent_t
                    if t['override_applied']
                    and not (t.get('override_reason') or "").lower().startswith(_gate_prefixes)
                ]
                ov_count = len(corrective)
            else:
                ov_count = 0
            if ov_count >= 5:
                events.append(self._safety_event(
                    run_id, tick_id, step_num, "SM-03", "WARNING",
                    f"Corrective override fired {ov_count}/5 recent ticks — planner ignoring constraints",
                    {"override_count": ov_count}, "flagged"
                ))

        # SM-04: No goal aspects defined by step 4
        if not halt and len(steps) >= 4 and not aspects:
            events.append(self._safety_event(
                run_id, tick_id, step_num, "SM-04", "WARNING",
                f"No goal aspects defined by step {step_num} — agent has no structured plan",
                {"step_number": step_num}, "flagged"
            ))

        # SM-05: All aspects blocked
        if not halt and aspects:
            if len(aspects) >= 2 and all(a['status'] in ('blocked', 'failed') for a in aspects):
                events.append(self._safety_event(
                    run_id, tick_id, step_num, "SM-05", "HALT",
                    f"All {len(aspects)} goal aspects are blocked — no viable path forward",
                    {"aspects": {a['aspect']: a['status'] for a in aspects}}, "halted"
                ))
                halt = True

        # SM-06: 10+ steps with no successful code execution — only when CODE was attempted
        # (Non-code goals will never have execution_success AEUs — don't false-positive on them)
        if not halt and step_num >= 10:
            aeus = self.get_aeus(run_id, 'execution_success')
            code_artifacts = self.get_artifacts(run_id, 'CODE')
            if not aeus and code_artifacts:
                events.append(self._safety_event(
                    run_id, tick_id, step_num, "SM-06", "WARNING",
                    f"Step {step_num}: {len(code_artifacts)} CODE artifact(s) produced but none executed successfully",
                    {"step_number": step_num, "code_attempts": len(code_artifacts)}, "flagged"
                ))

        # Persist all events
        for ev in events:
            sql = (
                "INSERT INTO safety_events "
                "(id, run_id, tick_id, step_number, rule_id, severity, "
                " message, evidence, action_taken, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            self.conn.execute(sql, (
                ev['id'], run_id, ev['tick_id'], step_num,
                ev['rule_id'], ev['severity'], ev['message'],
                json.dumps(ev['evidence']), ev['action_taken'], _now()
            ))
        if events:
            self.conn.commit()

        return halt, events

    def _safety_event(self, run_id: str, tick_id: str, step_num: int,
                      rule_id: str, severity: str, message: str,
                      evidence: dict, action: str) -> dict:
        return {
            "id":          f"sev_{uuid.uuid4().hex[:12]}",
            "run_id":      run_id,
            "tick_id":     tick_id,
            "step_number": step_num,
            "rule_id":     rule_id,
            "severity":    severity,
            "message":     message,
            "evidence":    evidence,
            "action_taken": action,
        }

    def get_safety_events(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM safety_events WHERE run_id=? ORDER BY created_at",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Verification Evidence ────────────────────────────────────────────────

    def add_verification_evidence(self, run_id: str, step_id: str,
                                   artifact_id: str = None,
                                   tests_run: list = None,
                                   command_outputs: list = None,
                                   assertions_checked: list = None,
                                   failures: list = None,
                                   success: bool = False) -> str:
        """Store structured verification evidence — not prose claims."""
        ve_id = f"ve_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO verification_evidence
               (id, run_id, step_id, artifact_id, tests_run, command_outputs,
                assertions_checked, failures, success, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ve_id, run_id, step_id, artifact_id,
             json.dumps(tests_run or []),
             json.dumps(command_outputs or []),
             json.dumps(assertions_checked or []),
             json.dumps(failures or []),
             1 if success else 0,
             _now())
        )
        self.conn.commit()
        return ve_id

    def has_verification_evidence(self, run_id: str) -> bool:
        """Check if any structured verification evidence exists for this run."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM verification_evidence WHERE run_id=?",
            (run_id,)
        ).fetchone()
        return row['cnt'] > 0 if row else False

    def has_successful_verification(self, run_id: str) -> bool:
        """Check if there is at least one successful verification."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM verification_evidence WHERE run_id=? AND success=1",
            (run_id,)
        ).fetchone()
        return row['cnt'] > 0 if row else False

    def get_verification_evidence(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM verification_evidence WHERE run_id=? ORDER BY created_at",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Termination Gate ─────────────────────────────────────────────────────

    def can_terminate(self, run_id: str, goal: str,
                       near_limit: bool = False) -> tuple:
        """
        Check whether DONE is permissible.
        Returns (allowed: bool, blockers: list[str]).

        For technical implementation tasks, DONE requires:
          - At least one CODE artifact exists
          - At least one verification evidence record exists
          - No required goal aspects are pending or in_progress
          - Conclusion is not a bare restatement of the goal

        For non-technical tasks, DONE requires:
          - No goal aspects are pending (if any were defined)

        near_limit=True relaxes the risk_assessment requirement — it is
        desirable but not blocking when the step budget is nearly exhausted.
        """
        blockers = []
        is_technical = _goal_is_technical(goal)

        artifacts = self.get_artifacts(run_id)
        aspects = self.get_goal_aspects(run_id)

        if is_technical:
            # Must have at least one CODE artifact
            code_artifacts = [a for a in artifacts if a['artifact_type'] == 'CODE']
            if not code_artifacts:
                blockers.append("missing_code_artifact")

            # Must have structured verification evidence (not just prose "verified")
            if not self.has_verification_evidence(run_id):
                # Fall back: accept a successful execution AEU as weak evidence
                exec_aeus = self.get_aeus(run_id, 'execution_success')
                if not exec_aeus:
                    blockers.append("missing_verification_evidence")

            # Must have a risk assessment artifact or risk content
            # Waived when near step limit — nice-to-have, not blocking
            if not near_limit:
                risk_artifacts = [
                    a for a in artifacts
                    if a['artifact_type'] == 'ANALYSIS'
                    and any(kw in a.get('title', '').lower()
                            for kw in ('risk', 'production', 'concern', 'issue'))
                ]
                risk_in_content = any(
                    'risk' in a.get('content', '').lower()[:500]
                    for a in artifacts
                    if a['artifact_type'] in ('ANALYSIS', 'DOCUMENT', 'DECISION')
                )
                if not risk_artifacts and not risk_in_content:
                    blockers.append("missing_risk_assessment")

        # All modes: no incomplete goal aspects
        # When near_limit, only hard-block on aspects that are genuinely pending
        # (in_progress is acceptable near the limit — the work was attempted)
        #
        # V3.0: Operational/action-tracking aspects are NEVER termination blockers.
        # These are created by the aspect_map in cf_agent.py to track which action
        # types have been used — they're not goal decomposition aspects.
        _OPERATIONAL_ASPECTS = frozenset({
            "planning", "web_research", "option_analysis", "comparison",
            "assumption_testing", "knowledge_synthesis", "artifact_production",
            "code_execution", "reflection", "options_generation",
            "artifact_quality",
        })
        block_statuses = ('pending',) if near_limit else ('pending', 'in_progress')
        incomplete = [
            a for a in aspects
            if a['status'] in block_statuses
            and a['aspect'] not in _OPERATIONAL_ASPECTS
        ]
        if incomplete:
            blockers.append(
                f"incomplete_goal_aspects:{','.join(a['aspect'] for a in incomplete[:3])}"
            )

        return (len(blockers) == 0, blockers)

    def log_termination_event(self, run_id: str, step_number: int,
                               allowed: bool, blockers: list,
                               fallback_action: str = None) -> str:
        """Log every DONE proposal — whether allowed or blocked."""
        te_id = f"te_{uuid.uuid4().hex[:12]}"
        aspects = self.get_goal_aspects(run_id)
        artifacts = self.get_artifact_index(run_id)

        self.conn.execute(
            """INSERT INTO termination_events
               (id, run_id, step_number, proposed_action, allowed, blockers,
                goal_aspect_status, artifact_inventory, fallback_action, created_at)
               VALUES (?, ?, ?, 'DONE', ?, ?, ?, ?, ?, ?)""",
            (te_id, run_id, step_number,
             1 if allowed else 0,
             json.dumps(blockers),
             json.dumps({a['aspect']: a['status'] for a in aspects}),
             json.dumps([
                 {"type": a['artifact_type'], "title": a['title']}
                 for a in artifacts
             ]),
             fallback_action,
             _now())
        )
        self.conn.commit()
        return te_id

    def get_termination_events(self, run_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM termination_events WHERE run_id=? ORDER BY created_at",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()


def _goal_is_technical(goal: str) -> bool:
    """Detect whether a goal requires code/implementation deliverables."""
    keywords = [
        "implement", "write", "build", "code", "create", "run", "execute",
        "develop", "program", "script", "function", "class", "module",
        "debug", "fix", "refactor", "test", "deploy", "sqlite", "api",
        "endpoint", "database", "server", "queue", "pipeline", "parser",
    ]
    goal_lower = goal.lower()
    return sum(1 for k in keywords if k in goal_lower) >= 2


def _now() -> str:
    return datetime.now().isoformat()
