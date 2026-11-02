# Libero Benchmark

This directory contains installation, data preparation, training and evaluating scripts on the Libero benchmark.

## Install Libero

Follow the instructions [here](https://github.com/Lifelong-Robot-Learning/LIBERO) to install LIBERO simulator.

Note: make sure you use the same Mujoco version (3.3.6). Otherwise, the object placements for some tasks would be different from training data distributions.

## Dataset Preparation

The Libero benchmark consists of four main task suites: Spatial, Object, Goal, and Long-horizon.

You can download our preprocessed data:
```bash
hf download --repo-type dataset cshizhe/Libero \
    --local-dir robot_data/libero/lerobot_point_lmdb
```

Generate state/action statistics:
```bash
for tasksuite in spatial object goal 10
do
python data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs robot_data/libero/lerobot_point_lmdb/libero_${tasksuite}_no_noops \
    --output_file robot_data/libero/lerobot_point_lmdb/libero_${tasksuite}_no_noops/robot_state_action_stats/rot6d.json \
    --state_rotation_slice 3 7 \
    --action_rotation_slice 3 7 \
    --rotation_type quat \
    --target_rotation_type rot6d
done
```


## Training

We support EO1, EO1-Point, QwenGR00T, QwenGR00T-Point, Pi0, and PointAct.
We train all models with an effective batch size of 128 using 1 or 2 H100 GPUs.

```bash
# EO1
bash experiments/2_libero/train_eo1.sh
# EO1-Point
bash experiments/2_libero/train_eo1_point.sh
# QwenGR00T
bash experiments/2_libero/train_vla2.sh
# QwenGR00T-Point
bash experiments/2_libero/train_vla2_point.sh
# Pi0
bash experiments/10_rlbench/train_pi0.sh
# PointACT
bash experiments/10_rlbench/train_pointact_concerto.sh
bash experiments/10_rlbench/train_pointact_utonia.sh
```

## Evaluation

```bash
bash experiments/2_libero/eval_libero.sh <task-suite> <expr-dir> <ckpt-step> <rotation-type> <args>
# for example
bash experiments/2_libero/eval_libero.sh spatial $SCRATCH/datasets/PointAct_exprs/libero/vla2_point/data-libero-spatial-point-frontview-rot6d-image.frontview_ck16_lr5e-5_gpu1_bs128_epoch50-freeze.vlm 21300 rot6d " --args.use_depth --args.verbose --args.delta_action"
```

## Golden I/O recording and replay

To check how much a changed model differs from a reference model, record every policy query of a
reference evaluation (raw observations, request options, returned action chunk and episode outcome),
then replay the same inputs through another checkpoint and compare the actions.

```bash
# 1. Record during a normal evaluation of the reference checkpoint (one .pkl per episode).
#    --args.record_dir also sends a per-query seed, so the server output is reproducible.
bash experiments/2_libero/eval_libero.sh spatial <ref-expr-dir> <ckpt-step> rot6d \
    "--args.use_depth --args.delta_action --args.record_dir <golden-dir>"

# 2. Replay the recorded inputs through any checkpoint (open-loop, pointact env).
pixi run python scripts/replay_golden.py --args.golden_dir <golden-dir> \
    --args.pretrained_path <expr-dir>/checkpoint-<step> --args.out_dir <out-dir>
```

`replay_golden.py` writes `summary.json` (overall, per task and per golden episode outcome),
`per_query.csv` and `actions.pkl`. Metrics use the executed part of each chunk (`replan_steps`) and the
full chunk (`*_chunk`): position error (cm), rotation error (deg) and gripper open/close agreement.
Replay the reference checkpoint itself first: `--args.seed_offset 0` should reproduce the golden
actions, and `--args.seed_offset 1` gives the noise floor from sampling. Open-loop differences are a
proxy; closed-loop success rate from `eval_libero.sh` remains the final check.
