#!/usr/bin/env bash
# BSD 3-Clause License — Copyright (c) 2026, NTNU Autonomous Robots Lab
# Wrapper around Isaac Lab's docker/container.py that builds a GNP image on top
# of isaac-lab-base and bind-mounts this repo.
#
# Usage:
#   ./docker/run.sh start
#   ./docker/run.sh enter
#   ./docker/run.sh stop
#   ./docker/run.sh build
#   ./docker/run.sh viz [--gpus device=N] [command...]   # ephemeral GPU sidecar
#
# X11: disabled by default for the long-lived train container (avoids host
# display/NVIDIA reprobes breaking the container device allowlist). Opt in with:
#   GNP_ENABLE_X11=1 ./docker/run.sh start
#
# Layout:
#   gnp_release/
#   ├── IsaacLab/                 # git checkout v3.0.0-EA
#   └── general_navigation_policy/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${GNP_ROOT}/../IsaacLab" 2>/dev/null && pwd || true)"
COMPOSE_FILE="${SCRIPT_DIR}/compose.gnp.yaml"
LAB_BASE_IMAGE="isaac-lab-base"
GNP_IMAGE="isaac-lab-gnp"
CONTAINER_PY=""

if [[ -z "${ISAACLAB_ROOT}" || ! -f "${ISAACLAB_ROOT}/docker/container.py" ]]; then
  echo "[ERROR] Sibling Isaac Lab not found at: ${GNP_ROOT}/../IsaacLab"
  echo "        Clone official Isaac Lab next to this repo and check out v3.0.0-EA:"
  echo "          cd $(dirname "${GNP_ROOT}")"
  echo "          git clone https://github.com/isaac-sim/IsaacLab.git"
  echo "          cd IsaacLab && git checkout v3.0.0-EA"
  exit 1
fi

CONTAINER_PY="${ISAACLAB_ROOT}/docker/container.py"
CONTAINER_CFG="${ISAACLAB_ROOT}/docker/.container.cfg"

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "[ERROR] Missing compose override: ${COMPOSE_FILE}"
  exit 1
fi

EXPECTED_TAG="v3.0.0-EA"
if command -v git >/dev/null 2>&1; then
  LAB_DESC="$(git -C "${ISAACLAB_ROOT}" describe --tags --exact-match 2>/dev/null || true)"
  if [[ -n "${LAB_DESC}" && "${LAB_DESC}" != "${EXPECTED_TAG}" ]]; then
    echo "[WARN] Isaac Lab is at '${LAB_DESC}', recommended pin is '${EXPECTED_TAG}'."
  elif [[ -z "${LAB_DESC}" ]]; then
    LAB_HEAD="$(git -C "${ISAACLAB_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "[INFO] Isaac Lab HEAD=${LAB_HEAD} (expected tag ${EXPECTED_TAG})."
  fi
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <start|enter|stop|build|config|copy|viz> [args...]"
  exit 1
fi

CMD="$1"
shift || true

ensure_lab_base() {
  if docker image inspect "${LAB_BASE_IMAGE}" >/dev/null 2>&1; then
    echo "[INFO] Found base image: ${LAB_BASE_IMAGE}"
    return 0
  fi
  echo "[INFO] Building Isaac Lab base image '${LAB_BASE_IMAGE}' (first time; can take a while)..."
  python3 "${CONTAINER_PY}" build
}

# Keep the long-lived train container off the host X11/display path unless opted in.
# Isaac Lab's container.py merges x11.yaml when X11_FORWARDING_ENABLED=1; that
# bind-mounts /tmp/.X11-unix and sets DISPLAY, which has correlated with stale
# nvidia-container device filters (new CUDA opens → EPERM).
configure_x11_policy() {
  local want="${GNP_ENABLE_X11:-0}"
  mkdir -p "$(dirname "${CONTAINER_CFG}")"
  if [[ ! -f "${CONTAINER_CFG}" ]]; then
    printf '[X11]\nx11_forwarding_enabled = %s\n' "${want}" > "${CONTAINER_CFG}"
  else
    python3 - "${CONTAINER_CFG}" "${want}" <<'PY'
import configparser, sys
path, want = sys.argv[1], sys.argv[2]
cfg = configparser.ConfigParser()
cfg.read(path)
if not cfg.has_section("X11"):
    cfg.add_section("X11")
prev = cfg.get("X11", "x11_forwarding_enabled", fallback=None)
if prev != want:
    cfg.set("X11", "x11_forwarding_enabled", want)
    with open(path, "w", encoding="utf-8") as f:
        cfg.write(f)
    print(f"[INFO] Set X11_FORWARDING_ENABLED={want} in {path}")
else:
    print(f"[INFO] X11_FORWARDING_ENABLED={want} (unchanged)")
PY
  fi
  if [[ "${want}" == "1" ]]; then
    echo "[WARN] X11 enabled for isaac-lab-base. Prefer GNP_ENABLE_X11=0 for long training runs."
  else
    echo "[INFO] X11 forwarding disabled for train container (set GNP_ENABLE_X11=1 to enable)."
  fi
}

# Ephemeral container with a fresh NVIDIA device allowlist. Use this for viz /
# one-off GPU work while training keeps the long-lived container.
run_viz_sidecar() {
  local gpu_arg="--gpus"
  local gpu_val="device=${GNP_VIZ_GPU:-2}"
  local -a docker_args=()
  # Empty by default: --entrypoint bash with no args gives an interactive shell.
  # Do NOT pass a trailing "bash" — that becomes `bash bash` and fails with
  # "/usr/bin/bash: cannot execute binary file".
  local -a cmd=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpus)
        shift
        gpu_val="${1:?--gpus requires a value (e.g. device=2)}"
        shift
        ;;
      --)
        shift
        cmd=("$@")
        break
        ;;
      *)
        cmd=("$@")
        break
        ;;
    esac
  done

  if ! docker image inspect "${GNP_IMAGE}" >/dev/null 2>&1; then
    echo "[ERROR] Image '${GNP_IMAGE}' not found. Run: ./docker/run.sh start"
    exit 1
  fi

  echo "[INFO] Starting ephemeral viz sidecar (${gpu_arg} ${gpu_val})"
  echo "[INFO] This does not touch the long-lived isaac-lab-base train container."

  # Mirror the GNP bind + Kit caches from the running train container when present;
  # otherwise mount the repo alone (enough for scripts + outputs on the bind).
  # Image ENTRYPOINT is /isaac-sim/runheadless.sh (Isaac Streaming). Override so
  # we get a shell / script runner like the compose train container (entrypoint: bash).
  docker_args+=(
    --rm -it
    --entrypoint bash
    --network host
    -w /workspace/general_navigation_policy
    "${gpu_arg}" "${gpu_val}"
    -e ACCEPT_EULA=Y
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -e ISAACSIM_PATH=/workspace/isaaclab/_isaac_sim
    -v "${GNP_ROOT}:/workspace/general_navigation_policy"
  )

  if docker inspect isaac-lab-base >/dev/null 2>&1; then
    # Reuse volumes/binds from the train container, but never X11/display mounts.
    while IFS='|' read -r mtype src dst; do
      [[ -z "${mtype}" ]] && continue
      case "${src}" in
        /tmp/.X11-unix|/etc/localtime) continue ;;
        /tmp/tmp.*) continue ;;
      esac
      case "${dst}" in
        /tmp/.X11-unix|/etc/localtime|/workspace/general_navigation_policy) continue ;;
      esac
      docker_args+=(-v "${src}:${dst}")
    done < <(docker inspect isaac-lab-base --format '{{range .Mounts}}{{printf "%s|%s|%s\n" .Type .Source .Destination}}{{end}}')
  else
    echo "[WARN] isaac-lab-base not running; sidecar mounts only the GNP repo bind."
  fi

  if ((${#cmd[@]})); then
    exec docker run "${docker_args[@]}" "${GNP_IMAGE}" "${cmd[@]}"
  else
    exec docker run "${docker_args[@]}" "${GNP_IMAGE}"
  fi
}

echo "[INFO] Isaac Lab: ${ISAACLAB_ROOT}"
echo "[INFO] GNP mount: ${GNP_ROOT} -> /workspace/general_navigation_policy"
export GNP_HOST_PATH="${GNP_ROOT}"

# Container UID often differs from the host mount owner; keep world-writable
# outputs/ and logs/ trees so train/viz artifacts land on the bind (host-visible).
ensure_host_outputs_dir() {
  mkdir -p "${GNP_ROOT}/outputs/analysis" "${GNP_ROOT}/logs"
  chmod 777 "${GNP_ROOT}/outputs" "${GNP_ROOT}/outputs/analysis" "${GNP_ROOT}/logs" 2>/dev/null || true
}

case "${CMD}" in
  start|build)
    ensure_host_outputs_dir
    ensure_lab_base
    configure_x11_policy
    ;;
  enter|viz)
    ensure_host_outputs_dir
    if [[ "${CMD}" == "viz" ]]; then
      run_viz_sidecar "$@"
    fi
    ;;
esac

exec python3 "${CONTAINER_PY}" "${CMD}" "$@" --files "${COMPOSE_FILE}"
