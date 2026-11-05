# fio/storage/staging.py
# ─────────────────────────────────────────────────────────────────────────────
# Run Chroma's SQLite store on node-LOCAL disk, then copy the finished DB
# directory back to NFS with a whole-file copy.
#
# Why: SQLite's locking + journaling are unreliable on NFS, which produces
#   sqlite3.DatabaseError: database disk image is malformed
# during Chroma upsert/purge_log AND during reads. By keeping ALL live SQLite
# I/O on local disk and only ever doing a directory copy to/from NFS, that
# corruption cannot happen.
#
# Works the same whether or not you're under SLURM:
#   • SLURM:     root = $TMPDIR or /tmp/chroma_stage_<user>_<jobid>
#   • plain CPU: root = /tmp/chroma_stage_<user>   (no job id available)
# Per-target tags (below) keep different configs / parallel runs isolated in
# BOTH cases, so a missing SLURM_JOB_ID is fine.
#
# ── BUILD phase (writes, then publishes to NFS) ──────────────────────────────
#   from fio.storage.staging import StagedChromaDB
#   stager   = StagedChromaDB(local_root=os.environ.get("LOCAL_DB_ROOT"))
#   opt_live = stager.register("chroma_fio_opt", OPT_DB_DIR)   # local dir to write
#   # ... build using opt_live as the Chroma db_dir ...
#   stager.sync_all()          # publish local -> NFS once the build is finished
#
# ── RETRIEVAL / QA phase (reads only, never writes back) ─────────────────────
#   from fio.storage.staging import StagedChromaDB
#   stager  = StagedChromaDB(local_root=os.environ.get("LOCAL_DB_ROOT"),
#                            sync_on_exit=False)              # nothing to sync
#   opt_live = stager.stage_in("chroma_fio_opt", OPT_DB_DIR)  # NFS -> local copy
#   opt_db   = ChromaFIOStorage(db_dir=opt_live)              # query the local copy
#   # ... retrieve from memory ...
#   # local copy is auto-removed at exit (or call stager.cleanup_staged_in())
# ─────────────────────────────────────────────────────────────────────────────
#
# FIXES (vs. original):
#   1. Per-config local dirs. The live dir name includes a tag derived from the
#      NFS target path, so different ablation configs (and parallel runs on a
#      shared node) can NEVER share /tmp/.../chroma_fio_opt. Sharing that path
#      caused:  sqlite3.OperationalError: table embeddings_queue already exists.
#   2. Hardened cleanup. If a stale chroma.sqlite3 survives rmtree, we FAIL
#      LOUDLY instead of silently building on a corrupt DB.
#   3. Sync gated on success. atexit/signal auto-sync no longer publishes a
#      partial/corrupt DB onto NFS on a failed/crashed run. Opt in with
#      sync_partial_on_exit=True for partial-progress saves.
#   4. _sync_one refuses to clobber NFS with an empty/invalid local dir, and
#      _safe_sync no-ops when nothing was registered (clean for read-only use).
#   5. stage_in copies are tracked and auto-removed at exit.
# ─────────────────────────────────────────────────────────────────────────────

import atexit
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile


def _tag_from_nfs(nfs_dir: str) -> str:
    """A short, filesystem-safe, per-target tag.

    Uses the parent build-dir name (which already encodes the config, e.g.
    'BUILD_7day_ablation_proto_U_full__max_prototypes=400') so the local
    staging path is unique per config and human-readable for debugging.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(nfs_dir))) or "build"
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", parent)


class StagedChromaDB:
    def __init__(
        self,
        local_root: str | None = None,
        sync_on_exit: bool = True,
        sync_partial_on_exit: bool = False,
    ):
        self.local_root = self._pick_local_root(local_root)
        self._map: dict[str, tuple[str, str]] = {}   # name -> (local_dir, nfs_dir)
        self._staged_in: list[str] = []              # read-only copies to clean up
        self._synced = False
        self._build_ok = False                       # set True by sync_all()/mark_success()
        self.sync_partial_on_exit = sync_partial_on_exit

        # Always clean up read-only stage_in copies at exit (cheap, registered
        # first so it runs LAST — after any sync).
        atexit.register(self._cleanup_staged_in)

        if sync_on_exit:
            # Sync on normal interpreter exit ...
            atexit.register(self._safe_sync)
            # ... and on SLURM timeout / Ctrl-C, so a finished build isn't lost
            # (only if marked successful, or opt-in for partials).
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    prev = signal.getsignal(sig)
                    signal.signal(sig, self._make_handler(prev))
                except (ValueError, OSError):
                    pass  # e.g. not in the main thread — atexit still covers us

    # ── local root selection ────────────────────────────────────────────────
    def _pick_local_root(self, hint: str | None) -> str:
        user = os.environ.get("USER", "user")
        job  = os.environ.get("SLURM_JOB_ID", "")     # empty when not under SLURM
        sub  = f"chroma_stage_{user}" + (f"_{job}" if job else "")

        candidates = []
        if hint:
            candidates.append(hint)
        if os.environ.get("TMPDIR"):                      # SLURM per-job scratch
            candidates.append(os.path.join(os.environ["TMPDIR"], sub))
        candidates.append(os.path.join("/tmp", sub))
        candidates.append(os.path.join("/dev/shm", sub))   # RAM-backed, fast
        candidates.append(os.path.join(tempfile.gettempdir(), sub))

        for base in candidates:
            if not base:
                continue
            try:
                os.makedirs(base, exist_ok=True)
                probe = os.path.join(base, ".wtest")
                with open(probe, "w") as fh:
                    fh.write("ok")
                os.remove(probe)
            except OSError:
                continue
            if self._is_nfs(base):
                print(f"[StagedChromaDB] {base} is on NFS — skipping")
                continue
            print(f"[StagedChromaDB] local staging root → {base}")
            return base

        raise RuntimeError(
            "No writable node-LOCAL (non-NFS) staging directory found.\n"
            "SQLite/Chroma cannot run safely directly on NFS. Set LOCAL_DB_ROOT "
            "to a node-local path, e.g.:  LOCAL_DB_ROOT=/tmp/$USER python scripts/build_memory.py"
        )

    @staticmethod
    def _is_nfs(path: str) -> bool:
        try:
            out = subprocess.run(
                ["stat", "-f", "-c", "%T", path],
                capture_output=True, text=True, timeout=5,
            )
            return "nfs" in out.stdout.lower()
        except Exception:
            return False   # if we can't tell, don't block

    # ── registration (BUILD phase: writes, synced back) ───────────────────────
    def register(self, name: str, nfs_dir: str | None) -> str | None:
        """Map a logical DB name to a fresh, per-config local dir; remember its
        NFS target. Returns the LOCAL dir to hand to Chroma, or None if nfs_dir
        is falsy (e.g. STORE_RAW=False).

        The local dir is unique per NFS target, so distinct ablation configs and
        parallel runs on the same node never collide on one chroma.sqlite3."""
        if not nfs_dir:
            return None

        # Per-config local dir: name + tag derived from the NFS target.
        local_dir = os.path.join(self.local_root, f"{name}__{_tag_from_nfs(nfs_dir)}")

        # Clean slate per run ...
        shutil.rmtree(local_dir, ignore_errors=True)
        os.makedirs(local_dir, exist_ok=True)

        # ... but DON'T silently build on top of a DB that rmtree couldn't remove
        # (e.g. still held open by a leftover/zombie process). That stale file is
        # exactly what triggers "table embeddings_queue already exists".
        leftover = os.path.join(local_dir, "chroma.sqlite3")
        if os.path.exists(leftover):
            raise RuntimeError(
                f"[StagedChromaDB] a stale Chroma DB survived cleanup at {local_dir}. "
                f"Another process may still hold it open. Kill leftover jobs and/or "
                f"remove the path manually, then retry:\n    rm -rf {local_dir}"
            )

        self._map[name] = (local_dir, nfs_dir)
        print(f"[StagedChromaDB] {name}: live(local)={local_dir}  final(NFS)={nfs_dir}")
        return local_dir

    # ── read-only: copy NFS -> local (RETRIEVAL phase, no sync-back) ──────────
    def stage_in(self, name: str, nfs_dir: str | None) -> str | None:
        """For READ-ONLY consumers (retrieval / QA). Copies an existing NFS
        Chroma dir to node-local disk and returns the LOCAL path to query
        against. Nothing is written back. Makes concurrent readers safe — each
        gets its own copy. The copy is auto-removed at exit.
        Returns None if nfs_dir is falsy; raises if the source doesn't exist."""
        if not nfs_dir:
            return None
        if not os.path.isdir(nfs_dir):
            raise FileNotFoundError(
                f"[StagedChromaDB] stage_in source not found on NFS: {nfs_dir}"
            )
        # Unique per source + pid so concurrent QA readers can't clobber each other.
        local_dir = os.path.join(
            self.local_root, f"{name}__{_tag_from_nfs(nfs_dir)}__ro{os.getpid()}"
        )
        shutil.rmtree(local_dir, ignore_errors=True)
        shutil.copytree(nfs_dir, local_dir)
        self._staged_in.append(local_dir)
        # NOT added to self._map -> never synced back (read-only)
        print(f"[StagedChromaDB] staged-in {name}: {nfs_dir} -> {local_dir} (read-only)")
        return local_dir

    # ── success signalling ────────────────────────────────────────────────────
    def mark_success(self):
        """Call when the build finished cleanly. Enables auto-sync on exit."""
        self._build_ok = True

    # ── sync local -> NFS ─────────────────────────────────────────────────────
    def sync_all(self):
        # An explicit sync_all() call is itself the success signal.
        self._build_ok = True
        for name, (local_dir, nfs_dir) in self._map.items():
            self._sync_one(name, local_dir, nfs_dir)
        self._synced = True

    def _sync_one(self, name: str, local_dir: str, nfs_dir: str):
        if not os.path.isdir(local_dir):
            return

        # Never clobber a good NFS build with an empty/invalid local dir.
        if not os.path.exists(os.path.join(local_dir, "chroma.sqlite3")):
            print(
                f"[StagedChromaDB] {name}: no chroma.sqlite3 in {local_dir} — "
                f"skipping sync to protect {nfs_dir}"
            )
            return

        parent = os.path.dirname(os.path.abspath(nfs_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        staging = nfs_dir + ".syncing"
        shutil.rmtree(staging, ignore_errors=True)
        # Whole-directory copy onto NFS — no live SQLite locking/journaling here.
        shutil.copytree(local_dir, staging)

        backup = nfs_dir + ".old"
        if os.path.exists(nfs_dir):
            shutil.rmtree(backup, ignore_errors=True)
            os.rename(nfs_dir, backup)          # keep previous until swap done
        os.rename(staging, nfs_dir)             # near-atomic publish
        shutil.rmtree(backup, ignore_errors=True)
        print(f"[StagedChromaDB] synced {name}: {local_dir} -> {nfs_dir}")

    def cleanup_local(self):
        for _, (local_dir, _) in self._map.items():
            shutil.rmtree(local_dir, ignore_errors=True)

    def cleanup_staged_in(self):
        self._cleanup_staged_in()

    # ── internal: exit / signal handling ─────────────────────────────────────
    def _cleanup_staged_in(self):
        for d in self._staged_in:
            shutil.rmtree(d, ignore_errors=True)
        self._staged_in.clear()

    def _safe_sync(self):
        if self._synced:
            return
        if not self._map:
            return  # nothing registered for write-back (read-only / stage_in only)
        # Do NOT publish a partial/corrupt DB onto NFS on a failed or crashed run.
        # register() always rebuilds from scratch, so partial progress is useless
        # anyway and only risks overwriting a good prior BUILD_*.
        if not self._build_ok and not self.sync_partial_on_exit:
            print(
                "[StagedChromaDB] build did not complete (no mark_success/sync_all) "
                "— NOT syncing partial DB to NFS. "
                "Set sync_partial_on_exit=True to override."
            )
            return
        try:
            self.sync_all()
        except Exception as e:
            print(f"[StagedChromaDB] exit sync failed: {e}", file=sys.stderr)

    def _make_handler(self, prev_handler):
        def handler(signum, frame):
            print(f"[StagedChromaDB] signal {signum} — attempting sync before exit")
            self._safe_sync()
            if callable(prev_handler):
                prev_handler(signum, frame)
            else:
                sys.exit(128 + signum)
        return handler