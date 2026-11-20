# PointAct

Official implementation of [PointACT: Vision-Language-Action Models with Multi-Scale Point-Action Interaction](https://cshizhe.github.io/projects/pointact.html).

<p align="center">
  <a href="https://cshizhe.github.io/projects/pointact.html">
    <img src="https://img.shields.io/badge/Project-PointAct_Webpage-0F172A?style=for-the-badge&logo=googlechrome&logoColor=white" alt="PointAct project webpage">
  </a>
  <a href="https://arxiv.org/abs/2605.21414">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="PointAct paper">
  </a>
  <a href="https://huggingface.co/cshizhe">
    <img src="https://img.shields.io/badge/Models_&_Data-Hugging_Face-F59E0B?style=for-the-badge&logo=huggingface&logoColor=white" alt="PointAct models and data">
  </a>
</p>

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://cshizhe.github.io/projects/resources/pointact/teaser_a.png" alt="Monolithic 3D-aware VLA teaser">
      <br>
      <sub><b>Monolithic 3D-aware VLA</b></sub>
    </td>
    <td align="center" width="33%">
      <img src="https://cshizhe.github.io/projects/resources/pointact/teaser_b.png" alt="Dual-system 3D-aware VLA teaser">
      <br>
      <sub><b>Dual-system 3D-aware VLA</b></sub>
    </td>
    <td align="center" width="33%">
      <img src="https://cshizhe.github.io/projects/resources/pointact/teaser_c.png" alt="PointAct teaser">
      <br>
      <sub><b>PointAct</b></sub>
    </td>
  </tr>
</table>

PointAct is a 3D-aware vision-language-action policy for robot manipulation. It keeps a pretrained vision-language backbone for semantic understanding and adds a dedicated point-action expert so that multi-scale 3D geometry can directly shape action decoding.

> [!IMPORTANT]
> **Tested platform: NVIDIA DGX Spark only.** This fork has been verified end to end on a single NVIDIA DGX Spark: environment setup, training, policy server and closed-loop LIBERO evaluation. **x86_64 (`linux-64`) has not been tested.** `pixi.lock` includes `linux-64`, but those environments have never been run, and results there (including success rates) may differ. See [Tested Platform](#tested-platform).

## Tested Platform

| Component | Version |
| --- | --- |
| System | NVIDIA DGX Spark, GB10 GPU (compute capability 12.1 / sm_121), **aarch64** (`linux-aarch64`) |
| OS / driver | Ubuntu 24.04.3 LTS, kernel 6.14.0-1013-nvidia, NVIDIA driver 580.95.05 |
| PyTorch | 2.7.0+cu128, with NVRTC 12.9.86 (`nvidia-cuda-nvrtc-cu12`) loaded first |
| Libraries | transformers 5.5.4, spconv 2.3.8 + cumm 0.8.2, flash-attn 2.8.3, torch-scatter 2.1.2 |
| Simulator | MuJoCo 3.3.6 (pixi `libero` environment) |

Why results on x86_64 may differ:
- On aarch64, the CUDA extensions (spconv, flash-attn, torch-scatter) are compiled from source by `pixi run build-cuda-ext`. On `linux-64`, pixi installs prebuilt wheels instead, so the CUDA kernels are different builds.
- GPU architecture, attention kernels and MuJoCo builds all differ between platforms. Small numerical differences can add up over a closed-loop rollout and change the success rate.
- The prebuilt-wheel installation path on `linux-64` has not been run, so it may also need fixes to install.

If you run on another platform, re-evaluate the [released checkpoint](#released-checkpoints) and compare against the numbers reported here before relying on them.

## Installation

Please follow the main setup guide in [INSTALLATION.md](INSTALLATION.md). The recommended workflow is:

1. Create the core `pointact` environment for training, checkpoint loading, preprocessing, and the policy server.
2. Create separate simulator environments.
3. Use the server-client evaluation pipeline since the policy and simulator usually need different environments.

### Using pixi

The core `pointact` environment is managed by [pixi](https://pixi.sh). Its configuration is in the `[tool.pixi.*]` sections of `pyproject.toml`, and `pixi.lock` pins every package for `linux-64` and `linux-aarch64` (torch 2.7.0 + CUDA 12.8). The LIBERO simulator has its own pixi environment, `libero`. The other simulators (RLBench, RoboCasa365) still use conda, as described in each benchmark's README.

**1. Set up the environment (once)**

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # install pixi; make sure ~/.pixi/bin is on your PATH

pixi install                 # create the environment in .pixi/envs/default
pixi run build-cuda-ext      # linux-aarch64 only (e.g. DGX Spark): compile spconv, torch-scatter, flash-attn (~30 min)
pixi run check-env           # check imports and run each CUDA extension on the GPU
```

On linux-64, `pixi install` already installs prebuilt spconv / torch-scatter / flash-attn, so skip `build-cuda-ext`.
`build-cuda-ext` targets compute capability 12.0 by default. Use `CUDA_ARCH=9.0 pixi run build-cuda-ext` for other GPUs, or build a single package with `pixi run build-cuda-ext spconv|torch-scatter|flash-attn`.

**2. Run commands inside the environment**

```bash
# prefix any command with `pixi run`
pixi run bash experiments/2_libero/train_pointact_concerto.sh experiments/2_libero/data_configs/data-libero-spatial-point-frontview.yaml
pixi run python data_prep/prepare_robot_state_action_stats.py --help

# or open a shell with the environment activated
pixi shell
```

**3. Evaluation**

Set up the LIBERO simulator environment once (CPU-only torch, MuJoCo rendering with EGL):

```bash
pixi install -e libero
pixi run -e libero setup-libero   # downloads LIBERO at a pinned commit into .pixi/build-src/LIBERO and writes its config
```

The eval scripts start the policy server with `pixi run python`. The LIBERO client runs with `pixi run -e libero python`, and the RLBench client with the conda environment (`$HOME/miniconda3/envs/rlbench/bin/python`). Override either with `SERVER_PYTHON` / `CLIENT_PYTHON`:

```bash
bash experiments/2_libero/eval_libero.sh spatial <expr-dir> <ckpt-step> rot6d "--args.use_depth --args.delta_action"
```

**4. Managing packages**

```bash
pixi add --pypi <package>    # add a PyPI dependency (written to pyproject.toml)
pixi add <package>           # add a conda-forge dependency
pixi install                 # sync the environment after editing pyproject.toml by hand
```

Commit both `pyproject.toml` and `pixi.lock` after changing dependencies. Do not use `pip install` for regular dependencies, since pixi will not track them.

**Available tasks:** `check-env`, `build-cuda-ext`, `lint` (`ruff check .`), `fmt` (`ruff format .`), and `setup-libero` (in the `libero` environment).

**Notes**
- On aarch64, the source-built CUDA extensions are not recorded in `pixi.lock`. `pixi install` keeps them, but after recreating the environment (e.g. `rm -rf .pixi`) run `pixi run build-cuda-ext` again.
- DGX Spark / GB10 (sm_121) needs NVRTC 12.9 or newer, but the aarch64 torch wheel bundles NVRTC 12.8. On linux-aarch64, pixi installs NVRTC 12.9 and puts it first on `LD_LIBRARY_PATH` when the environment is activated. Always start Python through `pixi run` or `pixi shell`. Running `.pixi/envs/default/bin/python` directly fails during training with `nvrtc: error: invalid value for --gpu-architecture`.
- `torchcodec` has no linux-aarch64 wheel, so LeRobot decodes videos with pyav there.
- `$SCRATCH` (pretrained weights and experiment outputs) defaults to `~/scratch` inside the pixi environment.
- Training scripts run offline (`TRANSFORMERS_OFFLINE=1`), so download the pretrained backbones first (see [INSTALLATION.md](INSTALLATION.md#pretrained-backbones)), e.g. `pixi run hf download --repo-type model Qwen/Qwen2.5-VL-3B-Instruct`.

## Supported Benchmarks

This repository currently supports three simulators:

| Benchmark | Simulator | Experiment Path |
| --- | --- | --- |
| [LIBERO](experiments/2_libero/README.md) | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | [`experiments/2_libero`](experiments/2_libero) |
| [RLBench](experiments/10_rlbench/README.md) | [RLBench](https://github.com/stepjam/RLBench) | [`experiments/10_rlbench`](experiments/10_rlbench) |
| [RoboCASA365](experiments/13_robocasa365/README.md) | [RoboCASA](https://github.com/robocasa/robocasa) | [`experiments/13_robocasa365`](experiments/13_robocasa365) |


## Released Checkpoints

Checkpoints reproduced in this fork on a single NVIDIA DGX Spark (GB10). These are not official releases from the PointAct authors. The success rates were measured on that platform only (see [Tested Platform](#tested-platform)).

The weights are built with Qwen and are for **non-commercial use only**. They are a derivative of [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), which is under the Qwen RESEARCH LICENSE AGREEMENT, and [Concerto](https://huggingface.co/Pointcept/Concerto), which is under CC BY-NC 4.0. See the model card for details.

| Model | Benchmark | Success | Weights |
| --- | --- | --- | --- |
| PointAct-Concerto (regression, 20k steps, bs32) | LIBERO-Spatial | 95.4% (477/500) | [kodaiyen/pointact-concerto-libero-spatial](https://huggingface.co/kodaiyen/pointact-concerto-libero-spatial) |

```bash
# The eval scripts expect <expr-dir>/checkpoint-<step>
pixi run hf download kodaiyen/pointact-concerto-libero-spatial \
    --local-dir exprs/pointact-concerto-libero-spatial/checkpoint-20000
bash experiments/2_libero/eval_libero.sh spatial exprs/pointact-concerto-libero-spatial 20000 rot6d \
    "--args.use_depth --args.delta_action"
```


## Supported Models

This repository includes PointAct and several comparison VLA policies used in our experiments:

| Model | Directory | Notes |
| --- | --- | --- |
| `PointAct` | [pointact/model/vla_pointact](pointact/model/vla_pointact) | Supports both [Concerto](https://arxiv.org/abs/2510.23607) and [Utonia](https://arxiv.org/abs/2603.03283) Point Transformer backbones |
| `EO1` | [pointact/model/eo1](pointact/model/eo1) | Monolithic VLA baseline |
| `EO1-Point` | [pointact/model/eo1](pointact/model/eo1) | EO-1 variant with point-cloud input |
| `QwenGR00T` | [pointact/model/vla_dual](pointact/model/vla_dual) | Dual-system VLA baseline |
| `QwenGR00T-Point` | [pointact/model/vla_dual](pointact/model/vla_dual) | VLA-Dual variant with point-cloud input |
| `Pi0` | [pointact/model/pi0](pointact/model/pi0) | PI0 baseline |
| `Pi0.5` | [pointact/model/pi05](pointact/model/pi05) | PI0.5 baseline |


## Acknowledgements

This codebase builds on several excellent open-source projects, especially [EO-1](http://eo-robotics.ai/eo-1), [GR00T](https://github.com/NVIDIA/Isaac-GR00T), and [LeRobot](https://github.com/huggingface/lerobot). We thank the authors and maintainers of these libraries for making their work available to the community.

## Citation

If you find PointAct useful in your research, or if you use this code, please cite:

```bibtex
@InProceedings{Chen_2026_PointACT,
    author    = {Chen, Shizhe and Pacaud, Paul and Schmid, Cordelia},
    title     = {PointACT: Vision-Language-Action Models with Multi-Scale Point-Action Interaction},
    booktitle = {Robotics: Science and Systems (RSS)},
    year      = {2026}
}
```
