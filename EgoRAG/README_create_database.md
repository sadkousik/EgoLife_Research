# create_database.py

Builds the EgoRAG Chroma database for one participant from a caption JSON file. It then generates the event diary and prints the first 10 entries.

## What it does

1. Creates the Chroma DB `<db_dir>/<db_name>`.
2. Fills it from `--json_path` (`RagAgent.create_database_from_json_NEW`).
3. With staging on, copies the finished DB from local disk to `--db_dir` (`db_t.sync()`).
4. Generates the event diary into `--diary_dir` (`gen_event`).
5. Prints the first 10 DB entries.

## Usage

On the cluster (staging on, the default):

```bash
python create_database.py \
    --db_name tasha_db \
    --json_path <captions.json> \
    --agent_name A3_TASHA
```

On a machine without NFS (staging off):

```bash
python create_database.py \
    --db_name tasha_db \
    --json_path <captions.json> \
    --agent_name A3_TASHA \
    --db_dir ./chroma_db \
    --no_staging
```

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--db_name` | required | Name of the DB / collection |
| `--json_path` | required | Caption JSON to build from |
| `--agent_name` | `A1_JAKE` | Participant: `A1_JAKE`, `A2_ALICE`, `A3_TASHA`, `A4_LUCIA`, `A5_KATRINA`, `A6_SHURE` |
| `--video_base_dir` | `data/EgoLife` | Parent folder of the participant video folders |
| `--diary_dir` | `events_diary` | Output folder for the event diary |
| `--db_dir` | `/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/chroma_db` | Final DB location |
| `--no_staging` | off | Write straight into `--db_dir`; no local build, no copy |
| `--local_root` | `$LOCAL_DB_ROOT`, `$TMPDIR`, `/tmp` or `/dev/shm` | Local folder used while staging; ignored with `--no_staging` |

`--video_base_dir` is the parent of the participant folder, so videos are expected at:

```
<video_base_dir>/<agent_name>/DAYx/DAYx_<agent_name>_<start>.mp4
```

A relative path resolves against the folder you run from.

## Staging

Writing ChromaDB's SQLite files directly on NFS is slow and can corrupt the database. With staging on:

```
build on local disk  →  sync() copies to <db_dir>/<db_name>  →  gen_event reads the NFS copy
```

Use `--no_staging` only when `--db_dir` is on a local disk.

## Requirements

- The **modified** `egorag` package, which adds:
  - the extra `Chroma(...)` arguments: `db_dir`, `staging`, `mode`, `local_root`;
  - `Chroma.sync()`;
  - `RagAgent.create_database_from_json_NEW`;
  - `gen_event(..., db_dir, staging)`.

  The script does not run against the upstream EgoRAG package.
- `experiment_logger.py` in the same folder as this script. Each run is logged as `CreateDB_<agent_name>_<db_name>`.