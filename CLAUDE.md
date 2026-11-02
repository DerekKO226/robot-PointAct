# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **PointAct** (RSS 2026), a 3D-aware vision-language-action (VLA) policy: a pretrained Qwen2.5-VL backbone for semantics plus a separate point-action expert (Point Transformer V3, Concerto or Utonia weights) that decodes actions from multi-scale point-cloud features. The repo also contains the baselines from the paper: EO1, EO1-Point, QwenGR00T (`vla_dual`), QwenGR00T-Point, Pi0, Pi0.5.

## Environments

There are always at least two environments, and they must not be mixed:
- `pointact`: training, data prep, checkpoint loading and the policy server. **Managed by pixi**: config in `[tool.pixi.*]` of `pyproject.toml`, lock in `pixi.lock`, env in `.pixi/envs/default`. Run things with `pixi run <cmd>`; add dependencies by editing `pyproject.toml` (or `pixi add`), then `pixi install` and commit `pixi.lock`. Pinned: `torch==2.7.0` (+cu128, from the PyTorch index), `transformers==5.5.4`, `numpy==1.26.4`, `lerobot<=0.3.4`.
- One env per simulator. `libero` is a separate pixi environment (`[tool.pixi.feature.libero]`, CPU torch, MuJoCo with EGL): `pixi install -e libero && pixi run -e libero setup-libero` (downloads LIBERO at a pinned commit into `.pixi/build-src/LIBERO` and writes `.pixi/libero-config/config.yaml`). `rlbench` and `robocasa365` are still conda envs. Never install simulator packages into `pointact`.

CUDA extensions (spconv, torch-scatter, flash-attn): on linux-64 pixi installs prebuilt wheels; on linux-aarch64 (this dev machine is a DGX Spark, GB10 = sm_121) they are compiled by `pixi run build-cuda-ext` (`scripts/build_cuda_ext.sh`) and are not tracked in `pixi.lock`. spconv uses cumm 0.8.2 (the first version supporting sm_120), installed with `--no-deps` because spconv 2.3.8 pins `cumm<0.8`. `pixi run check-env` smoke-tests everything on the GPU.

GB10 (sm_121) needs NVRTC >= 12.9. With 12.8, torch JIT kernels (such as an integer `prod()` in Qwen's `get_image_features`) fail with `nvrtc: error: invalid value for --gpu-architecture`. The aarch64 torch wheel bundles its own CUDA 12.8 libraries in `torch/lib`, so on linux-aarch64 pixi installs `nvidia-cuda-nvrtc-cu12==12.9.86` from PyPI and the activation env puts it first on `LD_LIBRARY_PATH`. **Run python through `pixi run` / `pixi shell`, not `.pixi/envs/default/bin/python` directly**, otherwise the 12.8 NVRTC gets loaded. The conda toolkit is also `cuda-version = 12.9.*` on aarch64. cumm links `libnvrtc-builtins.so.12.X` directly, so after any change to the conda CUDA version, rerun `pixi run build-cuda-ext spconv`.

Training scripts set `TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1`, so pretrained models (Qwen2.5-VL-3B, Concerto/Utonia `.pth`, `lerobot/pi0_base`) must already be downloaded. Paths in scripts assume `$SCRATCH/datasets/...` and data under `robot_data/...` relative to the repo root.

## Common commands

All scripts are run from the repo root (they set `PYTHONPATH=$(pwd)`), inside the pixi env (`pixi run ...` or `pixi shell`).

```bash
pixi install && pixi run build-cuda-ext   # set up env (build step only needed on aarch64)
pixi run check-env

# Lint / format (ruff config in pyproject.toml, line length 110)
pixi run lint
pixi run fmt

# State/action normalization stats (required before training; referenced by state_action_norm_file in data configs)
python data_prep/prepare_robot_state_action_stats.py --dataset_dirs <dir> --output_file <dir>/robot_state_action_stats/rot6d.json \
    --state_rotation_slice 3 7 --action_rotation_slice 3 7 --rotation_type quat --target_rotation_type rot6d

# Train (shell scripts wrap `accelerate launch scripts/train.py`; edit GPUS / batch size / paths inside)
bash experiments/2_libero/train_pointact_concerto.sh experiments/2_libero/data_configs/data-libero-spatial-point-frontview.yaml
bash experiments/10_rlbench/train_pointact_clf_concerto.sh

# Evaluate (starts run_server.py in the pointact env, then the simulator client in the sim env)
bash experiments/2_libero/eval_libero.sh <spatial|object|goal|10> <expr-dir> <ckpt-step> <rot6d|euler> "--args.use_depth --args.delta_action"
bash experiments/10_rlbench/eval_hybridvla_10tasks.sh <expr-dir> <ckpt-step> <seed> <rot-type> "--args.num_episodes 100"
```

Eval scripts run the server with `pixi run python`. The LIBERO client runs with `pixi run -e libero python`, and RLBench uses `$HOME/miniconda3/envs/rlbench/bin/python`. Override them with `SERVER_PYTHON` / `CLIENT_PYTHON`. On linux-aarch64, torchcodec has no wheel, so LeRobot video decoding falls back to pyav (`lerobot_data_video_backend` defaults to `None`). There is no test suite in the repo (pytest is only listed as an optional dependency).

## Architecture

### Training pipeline
- `scripts/train.py` is the single entry point for the Qwen-based models. `--model_class <Name>` is looked up in `scripts/train_registry.py` (`TRAIN_REGISTRY`), which maps it to a `TrainRecipe`: dotted paths for model class, config class, processor class, data-module function and trainer, plus a function that maps CLI args to config kwargs. **Adding a new model means adding a recipe here and an entry in `MODEL_MAP` in `scripts/run_server.py`.**
- Pi0/Pi0.5 use a separate entry point, `scripts/train_pi.py` (with `pointact/data/dataset_pi.py`).
- CLI args are the dataclass `pointact/train/pipeline_config.py::TrainPipelineConfig` (HF `TrainingArguments` + model/data fields such as `ptv3_*`, `chunk_size`, separate `merger_lr` / `vision_lr`, and `freeze_*` flags). `pointact/train/trainer.py::VLATrainer` is the HF Trainer subclass.
- A fresh model is built from `--vlm-name-or-path` (Qwen2.5-VL loaded via the vendored `pointact/model/backbone/qwen2_5_vl`) and passed into the model as `vlm_backbone`. With `--model-name-or-path`, it resumes from a saved checkpoint instead.

### Data
- A data config YAML (`experiments/*/data_configs/*.yaml`) lists `lerobot_datasets` (and optionally `mm_datasets`). Each entry's `class_name` is resolved through `pointact/data/robot/registry.py` (`@register_robot_dataset`): `LeRobotDataset` (2D, `data_2d.py`) or `LeRobotPointCloudDataset` (3D, `data_3d.py`, reads point clouds from LMDB under `point_cloud_dirname`, crops to `points_workspace`, subsamples to `max_npoints`, and applies rotation augmentation).
- Per-entry keys control the important behaviour: `converted_rot_type` (actions/states are converted from quaternion to `rot6d` or `euler`), `is_delta_action`, `video_key_ids_for_vlm` (which cameras go to the VLM; `[]` means point cloud only), and `state_action_norm_file`.
- `pointact/data/dataset.py` wraps these into `SupervisedDataset` with two prompt styles: `MonolithicPrompt` (EO1: actions are tokens inside the VLM sequence) vs `DualPrompt` (vla_dual/PointAct: VLM gives context, a separate action head decodes). Recipes choose `create_monolithic_prompt_data_module` or `create_dual_prompt_data_module`.

### Models (`pointact/model/`)
- `vla_pointact/`: PointAct. `VLAEncDec3DBaseModel` has four variants: `{Classification, Regression}` × `{plain, WithAction}`. Classification uses binned position heads (`ptv3_clf_head_pos_bins`); regression uses a heatmap with `regression_head_heatmap_temp`.
- `ptv3/concerto` and `ptv3/utonia` are parallel vendored PTv3 implementations, selected with `--ptv3_backend`. `model_ca.py` / `model_ca_action.py` add point cross-attention variants (`--ptv3_apply_point_ca`). Initial weights come from `--ptv3_init_ckpt_file`.
- `action_head/`: shared flow-matching DiT action head used by the dual-system models.
- Each model package has its own `processing_*.py`. The processor handles both training-time tokenization and inference: `processor.select_action(model, batch, pred_rot_type=..., points_workspace=...)` is what the server calls.

### Server-client evaluation
- `scripts/run_server.py` reads `architectures[0]` from the checkpoint's `config.json`, picks `(model_class, processor_class)` from `MODEL_MAP`, and serves it over ZMQ + msgpack (`pointact/utils/server_client.py`: `PolicyServer` / `PolicyClient`).
- Simulator clients (`experiments/*/run_*_client.py`) use `pointact/robot_envs/<sim>_utils/` to build observations (RGB-D → point clouds) and send them with options such as `pred_rot_type` and `points_workspace`. The client's rotation type and workspace must match what the model was trained with.

### Data conversion
`data_prep/` converts raw simulator data to LeRobot v2.1 format with point clouds in LMDB (`rlbench_to_lerobot/`, `robocasa365_to_lerobot/`). Preprocessed LIBERO and RLBench data can be downloaded from the Hugging Face `cshizhe/*` datasets (see the per-benchmark READMEs).

## Notes
- The top-level README links to `experiments/13_robocasa365`, but that directory is not in the repo. Only the data prep and `robot_envs/robocasa365_utils` exist.
- Output dir and run name encode the hyperparameters (see the training scripts). Checkpoints are saved as `<output_dir>/checkpoint-<step>`, which is the layout the eval scripts expect.
