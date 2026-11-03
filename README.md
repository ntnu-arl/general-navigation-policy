# General Navigation Policy

Vision-based multirotor navigation with [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [skrl](https://skrl.readthedocs.io/).

Runs in Docker on Isaac Sim **6.1** + Isaac Lab **`v3.0.0-EA`**.

## Prerequisites

- Linux + NVIDIA GPU
- [Docker](https://docs.docker.com/engine/install/) ≥ 26 and Compose ≥ 2.25
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NGC access (`docker login nvcr.io` for `nvcr.io/nvidia/isaac-sim:6.1.0`)

## Setup

```bash
mkdir -p gnp_release && cd gnp_release
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout v3.0.0-EA && cd ..
git clone https://github.com/ntnu-arl/general-navigation-policy.git
cd general-navigation-policy
```

```text
gnp_release/
├── IsaacLab/
└── general-navigation-policy/
```

```bash
./docker/run.sh start    # builds Lab base + GNP image (first run is slow)
./docker/run.sh enter
```

GNP and skrl are installed in the image. The repo is bind-mounted at `/workspace/general_navigation_policy`, so host edits apply immediately.

### Pretrained assets

Shipped under `pretrained/`:

| File | Description |
|------|-------------|
| `arbitrary_sensor_setup_pretrained.pt` | Seed-47 spherical-range CNN-GRU policy (Lab 3 / skrl 2) |
| `agent.yaml` | Matching skrl agent config |
| `example_run_30s.mp4` | Example follow-cam rollout with camera frustum overlays |

Use the local checkpoint path below (this is **not** Isaac Lab's Nucleus `--use_pretrained_checkpoint`).

### GPU / container isolation

The long-lived train container is started **without X11** (no host
`DISPLAY` / `/tmp/.X11-unix` mount). `NVIDIA_DRIVER_CAPABILITIES=all` matches
`docker/compose.gnp.yaml` so RTX/Kit graphics stay available for headless
capture; the EPERM failures came from binding host X11 into a multi-day train
container, not from these capabilities.

- Keep training in `isaac-lab-base` with `HEADLESS=1` and `--device cuda:N`.
- For visualization / one-off GPU jobs, use a **fresh** sidecar (does not restart training):

```bash
./docker/run.sh viz --gpus device=2
# drops into bash (not Isaac Streaming). Then run play/eval with --device cuda:0
# (mapped GPU appears as cuda:0 inside the sidecar).
```

- Opt into X11 on the train container only when you need a Kit window:
  `GNP_ENABLE_X11=1 ./docker/run.sh start` (recreate required; stops training).
- If `nvidia-smi` / new CUDA fails inside `isaac-lab-base` but host GPUs are fine,
  recreate when training can stop: `./docker/run.sh stop && ./docker/run.sh start`.

## Train / play / evaluate

Inside the container:

```bash
cd /workspace/general_navigation_policy

HEADLESS=1 isaaclab -p scripts/train_skrl.py \
  --task Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-v0 --agent spherical_range_cnn_gru \
  --device cuda:0

HEADLESS=1 isaaclab -p scripts/play_skrl.py \
  --task Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-Play-v0 \
  --model spherical_range_cnn_gru \
  --checkpoint pretrained/arbitrary_sensor_setup_pretrained.pt

HEADLESS=1 isaaclab -p scripts/evaluate_skrl.py \
  --task Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-Play-v0 \
  --model spherical_range_cnn_gru \
  --checkpoint pretrained/arbitrary_sensor_setup_pretrained.pt
```

Isaac Lab 3.0 removed ``--headless``; use ``HEADLESS=1`` (or omit a Kit visualizer). Do **not** set `CUDA_VISIBLE_DEVICES` with Kit — use `--device cuda:N` instead.

| Task | Description |
|------|-------------|
| `Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-v0` | Arbitrary multi-camera sensor setup + spherical-range CNN-GRU |
| `Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-Play-v0` | Play / eval variant |

`exit` the shell, then on the host: `./docker/run.sh stop`.

If you change `setup.py` / package dependencies, rebuild: `./docker/run.sh build`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
