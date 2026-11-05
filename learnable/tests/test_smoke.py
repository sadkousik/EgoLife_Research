"""End-to-end smoke test on synthetic FIOs (CPU, offline, ~30 s).

    cd <repo root>
    python learnable/tests/test_smoke.py            # uses a stub encoder
    REAL_ENCODER=1 python learnable/tests/test_smoke.py   # uses all-MiniLM-L6-v2

Checks:
  1. LearnableFIOOptimizer without a model keeps exactly what FIOOptimizer keeps.
  2. An untrained model reproduces the heuristic keep set (init == full model).
  3. trace -> train -> checkpoint -> build; a hard budget B holds via eviction.
  4. online learning changes parameters during a build.
  5. learnable/scripts/build_memory.py writes the Chroma opt store (if chromadb is installed).
"""

import contextlib
import copy
import io
import json
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if not os.environ.get("REAL_ENCODER"):
    sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(ROOT))

from learnable.adaimprint.fio_adapter import FULL_OPT_KWARGS, FIOOptimizer, LearnableFIOOptimizer  # noqa: E402
from learnable.adaimprint.model import LearnableImportance, LearnableImportanceConfig  # noqa: E402
from learnable.scripts import record_trace as trace_mod, train as train_mod  # noqa: E402
import torch  # noqa: E402


def make_wearer(root: Path, seed: int):
    rng = random.Random(seed)
    acts = ["pick up", "look at", "put down", "open", "cut", "talk to", "wash", "type on"]
    objs = ["phone", "cup", "knife", "laptop", "door", "bread", "box", "flower", "plate"]
    (root / "cache").mkdir(parents=True)
    caps, stream = [], []
    for day in (1, 2, 3):
        for k in range(100):
            st = f"{10 + k // 20:02d}{(k % 20) * 3:02d}0000"
            caps.append({"date": f"DAY{day}", "start_time": st, "end_time": st, "text": "", "video_path": ""})
            fios = [{"id": f"DAY{day}-{st}-{n}", "date": f"DAY{day}", "start_time": st, "end_time": st,
                     "persons": rng.sample(["Jake", "Alice", "Tasha"], rng.randint(0, 2)),
                     "action": rng.choice(acts), "object": rng.choice(objs),
                     "tool": rng.choice(["", "knife"]), "app": "", "source_location": "",
                     "target_location": rng.choice(["", "table"]),
                     "speech_content": rng.choice(["", "", "ok"]), "attributes": "{}"}
                    for n in range(rng.randint(0, 4))]
            (root / "cache" / f"DAY{day}-{st}.json").write_text(json.dumps(fios))
            stream.append(fios)
    (root / "captions.json").write_text(json.dumps(caps))
    return stream


def build(opt, stream):
    for fios in stream:
        res = opt.process(copy.deepcopy(fios))
    return res["kept"]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="learnable_smoke_"))
    streams = {w: make_wearer(tmp / w, s) for w, s in (("A", 1), ("B", 2), ("C", 3))}
    kw = dict(FULL_OPT_KWARGS)

    # 1. adapter without a model == production optimizer
    with contextlib.redirect_stdout(io.StringIO()):
        ref = build(FIOOptimizer(**kw), streams["A"])
    ada = build(LearnableFIOOptimizer(quiet=True, **kw), streams["A"])
    assert [f["id"] for f in ref] == [f["id"] for f in ada], "adapter changed keep decisions"
    print(f"[1] adapter == FIOOptimizer ({len(ref)} entries)")

    # 2. untrained model reproduces the heuristic
    model = LearnableImportance(LearnableImportanceConfig(n_protos=kw["max_prototypes"],
                                                          tau=kw["tau_threshold"],
                                                          lambda_init=kw["lambda_decay"]))
    init = build(LearnableFIOOptimizer(model=model, quiet=True, **kw), streams["A"])
    assert [f["id"] for f in init] == [f["id"] for f in ref], "init model differs from heuristic"
    worst = max(abs(a["importance_score"] - b["importance_score"]) for a, b in zip(init, ref))
    assert worst < 1e-5, worst
    print(f"[2] untrained model == heuristic (max |Δw| = {worst:.1e})")

    # 3. trace -> train -> build with budget
    for w in streams:
        trace_mod.main(["--captions", str(tmp / w / "captions.json"), "--cache-dir", str(tmp / w / "cache"),
                        "--out", str(tmp / "traces" / f"{w}.pt"), "--wearer", w])
    tA = torch.load(tmp / "traces" / "A.pt", weights_only=False)
    below = int(((tA["cand"] >= 0) & (tA["cand_sim"] < kw["compression_sim_threshold"])).sum())
    assert below > 0, "trace recorded no candidates below the heuristic consolidation threshold"
    print(f"[3a] trace: {below} candidates in [floor, {kw['compression_sim_threshold']}) "
          f"available to the learned consolidation gate")
    train_mod.main(["--train", str(tmp / "traces/A.pt"), str(tmp / "traces/B.pt"),
                    "--val", str(tmp / "traces/C.pt"), "--out", str(tmp / "ckpt"), "--epochs", "1"])
    budget = 40
    opt = LearnableFIOOptimizer.from_checkpoint(str(tmp / "ckpt/last.pt"), memory_budget=budget,
                                                quiet=True, record_trace=False, **kw)
    build(opt, streams["C"])
    mem = opt.final_memory()
    rep = opt.memory_report()
    assert len(mem) <= budget, len(mem)
    assert all(not k.startswith("_") for f in mem for k in f), "internal fields leaked"
    print(f"[3] budgeted build: {len(mem)} live entries (B={budget}), {rep}")

    # 4. online learning updates the controller during the build
    opt = LearnableFIOOptimizer.from_checkpoint(str(tmp / "ckpt/last.pt"), online_lr=1e-3,
                                                quiet=True, record_trace=False, **kw)
    before = {k: v.clone() for k, v in opt.lmodel.state_dict().items()}
    build(opt, streams["C"])
    changed = sum(int(not torch.equal(before[k], v)) for k, v in opt.lmodel.state_dict().items())
    assert changed > 0, "online learning did not change any parameter"
    print(f"[4] online learning: {changed} tensors updated, {len(opt.final_memory())} entries")
    # 5. standalone build writes the Chroma opt store (needs chromadb)
    try:
        import chromadb  # noqa: F401
    except ImportError:
        print("[5] skipped: chromadb not installed")
    else:
        from learnable.scripts import build_memory as build_fio
        out = tmp / "build"
        with contextlib.redirect_stdout(io.StringIO()):
            build_fio.main(["--ckpt", str(tmp / "ckpt/last.pt"), "--captions", str(tmp / "C/captions.json"),
                            "--cache-dir", str(tmp / "C/cache"), "--out", str(out), "--budget", "40",
                            "--doc-fix"])
        cfg = json.loads((out / "build_config.json").read_text())
        col = chromadb.PersistentClient(path=str(out / "chroma_fio_opt")).get_collection(
            "collection_day1_to_day7")
        assert col.count() == cfg["n_entries"] <= 40, (col.count(), cfg["n_entries"])
        print(f"[5] build_fio: {col.count()} entries in chroma_fio_opt")
    print(f"OK  ({tmp})")


if __name__ == "__main__":
    main()
