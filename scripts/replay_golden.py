"""Replay a recorded golden I/O set through a checkpoint and compare its actions with the recorded ones.

The golden set is written by a simulator client with ``--args.record_dir`` (one pickle per episode,
see experiments/2_libero/run_libero_client.py). Each recorded query holds the raw observation batch,
the request options (including ``request_seed``) and the action chunk the reference model returned.

Replaying is open-loop: every model sees exactly the same inputs, so per-query differences are
comparable across models. Closed-loop success rate still has to be measured with the eval script.

Usage (pointact env, repo root):
    pixi run python scripts/replay_golden.py --args.golden_dir <dir> --args.pretrained_path <ckpt> \
        --args.out_dir <dir>

Noise floor: replay the reference checkpoint itself. ``seed_offset 0`` should reproduce the golden
actions (any residual is GPU nondeterminism); ``seed_offset 1`` changes the sampling noise and shows
how much a model differs from itself.
"""

import csv
import dataclasses
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import tqdm
import tyro

from pointact.utils.torch_utils import set_seed
from scripts.run_server import Policy, ServerArgs


@dataclasses.dataclass
class ReplayArgs:
    golden_dir: str
    pretrained_path: str
    out_dir: str
    num_denoise_steps: int = 10
    seed_offset: int = 0  # added to each recorded request_seed
    max_episodes: int | None = None  # replay only the first N episode files


METRICS = ["pos_err_cm", "rot_err_deg", "grip_agree", "pos_err_cm_chunk", "rot_err_deg_chunk"]


def compare_chunks(ref: np.ndarray, new: np.ndarray, n_exec: int) -> dict:
    """Compare two action chunks of shape (T, 8): xyz, quat (xyzw), gripper.

    Metrics without suffix use the first ``n_exec`` steps (the part the robot executes before
    replanning); ``*_chunk`` metrics use every step both chunks have.
    """
    n = min(len(ref), len(new))
    ref, new = ref[:n], new[:n]
    pos_err = np.linalg.norm(ref[:, :3] - new[:, :3], axis=-1) * 100
    q_ref = ref[:, 3:7] / np.linalg.norm(ref[:, 3:7], axis=-1, keepdims=True)
    q_new = new[:, 3:7] / np.linalg.norm(new[:, 3:7], axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q_ref * q_new, axis=-1)), 0.0, 1.0)
    rot_err = np.degrees(2 * np.arccos(dot))
    grip_agree = (ref[:, 7] > 0.5) == (new[:, 7] > 0.5)
    k = min(n_exec, n)
    return {
        "pos_err_cm": float(pos_err[:k].mean()),
        "rot_err_deg": float(rot_err[:k].mean()),
        "grip_agree": float(grip_agree[:k].mean()),
        "pos_err_cm_chunk": float(pos_err.mean()),
        "rot_err_deg_chunk": float(rot_err.mean()),
    }


def summarize(rows: list[dict]) -> dict:
    out = {"num_queries": len(rows)}
    for m in METRICS:
        vals = np.array([r[m] for r in rows])
        out[m] = {"mean": float(vals.mean()), "p95": float(np.percentile(vals, 95)), "max": float(vals.max())}
    return out


def main(args: ReplayArgs) -> None:
    episode_files = sorted(Path(args.golden_dir).glob("*.pkl"))
    if args.max_episodes is not None:
        episode_files = episode_files[: args.max_episodes]
    assert episode_files, f"No recorded episodes in {args.golden_dir}"

    set_seed(0)
    policy = Policy(ServerArgs(pretrained_path=args.pretrained_path, num_denoise_steps=args.num_denoise_steps))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, new_actions = [], {}
    for episode_file in tqdm.tqdm(episode_files):
        with open(episode_file, "rb") as f:
            record = pickle.load(f)
        meta = record["meta"]
        n_exec = meta["client_args"]["replan_steps"]
        policy.reset()
        for query_idx, query in enumerate(record["queries"]):
            options = dict(query["options"])
            options["request_seed"] = options["request_seed"] + args.seed_offset
            batch = dict(query["batch"])  # get_action converts images in place
            action = np.asarray(policy.get_action(batch, options)["action"])
            new_actions[f"{episode_file.stem}/{query_idx}"] = action
            rows.append({
                "episode": episode_file.stem,
                "task_id": meta["task_id"],
                "episode_idx": meta["episode_idx"],
                "success": meta["success"],
                "query_idx": query_idx,
                "env_step": query["env_step"],
                **compare_chunks(query["action"][0], action[0], n_exec),
            })

    by_task, by_outcome = defaultdict(list), defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
        by_outcome["success" if r["success"] else "failure"].append(r)
    summary = {
        "config": dataclasses.asdict(args),
        "num_episodes": len(episode_files),
        "golden_success_rate": float(np.mean([r["success"] for r in rows if r["query_idx"] == 0])),
        "overall": summarize(rows),
        "by_outcome": {k: summarize(v) for k, v in sorted(by_outcome.items())},
        "by_task": {str(k): summarize(v) for k, v in sorted(by_task.items())},
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_query.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "actions.pkl", "wb") as f:
        pickle.dump(new_actions, f, protocol=pickle.HIGHEST_PROTOCOL)

    overall = summary["overall"]
    print(f"Replayed {len(rows)} queries from {len(episode_files)} episodes -> {out_dir}")
    for m in METRICS:
        print(f"  {m:18s} mean {overall[m]['mean']:.4f}  p95 {overall[m]['p95']:.4f}  max {overall[m]['max']:.4f}")


if __name__ == "__main__":
    tyro.cli(main)
