import argparse

from egorag.agents.RagAgent import RagAgent
from egorag.database.Chroma import Chroma
from egorag.utils.gen_event import gen_event

# added by me for logging
from experiment_logger import start_experiment


# ── Default DB location (NFS) ────────────────────────────────────────────────
# Final (published) DB location. The DB ends up at <db_dir>/<db_name>.
#   staging ON  (default)     : live SQLite writes happen on node-local disk,
#                               then the finished DB dir is copied here.
#   staging OFF (--no_staging): the DB is written straight into this folder.
NFS_CHROMA_DB_DIR = "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/chroma_db"


# Set Args
def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize RagAgent with database and JSON path."
    )
    parser.add_argument("--db_name", required=True, help="Name of the Chroma database / collection")
    parser.add_argument(
        "--json_path", required=True, help="Path to the JSON file for database creation"
    )
    parser.add_argument(
        "--diary_dir", required=False, default="events_diary",
        help="Path to the diary directory (consumed by gen_event)",
    )

    # ── participant id + video tree ──────────────────────────────────────────
    # create_database_from_json_NEW builds the fallback video path as:
    #     os.path.join(video_base_dir, agent_name, DAYx, f"{DAYx}_{agent_name}_{start}.mp4")
    # so video_base_dir is the PARENT of the participant folder.
    # e.g. TASHA -> data/EgoLife/A3_TASHA/DAYx/...
    parser.add_argument(
        "--agent_name", default="A1_JAKE",
        help="Participant id: A1_JAKE, A2_ALICE, A3_TASHA, A4_LUCIA, A5_KATRINA, A6_SHURE",
    )
    parser.add_argument(
        "--video_base_dir", default="data/EgoLife",
        help="Parent folder of the participant video folders "
             "(relative paths resolve against the folder you run from).",
    )

    # ── DB location + staging ────────────────────────────────────────────────
    parser.add_argument(
        "--db_dir", default=NFS_CHROMA_DB_DIR,
        help="Base dir holding the Chroma DB. DB ends up at <db_dir>/<db_name>.",
    )
    parser.add_argument(
        "--no_staging", action="store_true",
        help="Write the DB straight into --db_dir instead of building on "
             "node-local disk and copying to NFS. Use on machines without NFS.",
    )
    parser.add_argument(
        "--local_root", default=None,
        help="Node-local staging root (defaults: $LOCAL_DB_ROOT / $TMPDIR / /tmp / /dev/shm). "
             "Ignored with --no_staging.",
    )
    return parser.parse_args()


# Main function
def main(args):
    # staging ON by default; --no_staging turns it off
    staging = not args.no_staging
    print(f"[create_database] agent={args.agent_name}  db={args.db_dir}/{args.db_name}  "
          f"staging={'ON' if staging else 'OFF'}")

    # Initialize the Chroma database with the provided name.
    # db_dir is the FINAL location in both modes.
    db_t = Chroma(
        name=args.db_name,
        db_dir=args.db_dir,
        staging=staging,
        mode="build",
        local_root=args.local_root if staging else None,
    )

    # Initialize the RagAgent
    agent = RagAgent(
        database_t=db_t,
        name=args.agent_name,
        video_base_dir=args.video_base_dir,
    )

    # Create the database from the provided JSON file
    # agent.create_database_from_json(args.json_path)
    agent.create_database_from_json_NEW(json_path=args.json_path)

    # added by me
    # With staging, publish the node-local build to its final (NFS) location
    # BEFORE gen_event reads it back. gen_event receives db_dir + staging so
    # get_event_min opens the SAME final location (staging: copy -> local -> read;
    # no staging: read in place).
    if staging:
        db_t.sync()

    gen_event(args.db_name, args.diary_dir, db_dir=args.db_dir, staging=staging)

    # Optional: View the content of the database (first 10 entries)
    agent.database_t.view_database(n=10)


# Entry point for the script
if __name__ == "__main__":
    args = parse_args()

    # logging setup — name built from the run's own arguments
    logger = start_experiment(f"CreateDB_{args.agent_name}_{args.db_name}")
    main(args)
    logger.info("✅ Database build finished")