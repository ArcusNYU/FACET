"""
launch_test.py — inference / evaluation entry point.

Launcher selection (`accel.launcher` in test.yaml):
    auto        -> python  if len(gpu_ids) <= 1  else  accelerate   (default)
    python      -> always `python test.py ...`           (single process)
    accelerate  -> always `accelerate launch ...`        (DDP, N processes)

GPU visibility is pinned via CUDA_VISIBLE_DEVICES = accel.gpu_ids.

Usage:
    python launch_test.py # default using ./test.yaml
    python launch_test.py --test_yaml runs/<run>/test.yaml
    python launch_test.py --launcher accelerate
    python launch_test.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FACET test launcher.")
    p.add_argument(
        "--test_yaml",
        type=str,
        default=str(_PROJECT_ROOT / "test.yaml"),
        help="Path to test.yaml.",
    )
    p.add_argument(
        "--launcher",
        choices=("auto", "python", "accelerate"),
        default=None,
        help="Override accel.launcher from test.yaml.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assembled command + env and exit without launching.",
    )
    return p.parse_args()


def _read_accel(test_yaml: Path) -> dict:
    """Pull the `accel:` block out of test.yaml (raw, no dataclass merge)."""
    if not test_yaml.is_file():
        sys.exit(f"[launch_test] test.yaml not found: {test_yaml}")
    with open(test_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("accel", {}) or {}


def _normalise_gpu_ids(raw_ids) -> List[int]:
    """accel.gpu_ids may be a list, a single int, or missing -> always a list."""
    if raw_ids is None:
        return [0]
    if isinstance(raw_ids, int):
        return [raw_ids]
    return [int(g) for g in raw_ids]


def _resolve_launcher(name: str, n_gpus: int) -> str:
    if name == "auto":
        return "python" if n_gpus <= 1 else "accelerate"
    return name


def _accelerate_argv() -> List[str]:
    """`accelerate launch` prefix, robust to it not being on PATH."""
    exe = shutil.which("accelerate")
    if exe:
        return [exe, "launch"]
    return [sys.executable, "-m", "accelerate.commands.launch"]


def build_command(launcher: str, gpu_ids: List[int], precision: str,
                  main_process_port: int, tail: List[str]) -> List[str]:
    n = len(gpu_ids)
    test_py = str(_PROJECT_ROOT / "test.py")

    if launcher == "python":
        return [sys.executable, test_py, *tail]

    # accelerate (DDP / multi-process)
    cmd = [
        *_accelerate_argv(),
        "--num_machines", "1",
        "--num_processes", str(max(n, 1)),
        "--main_process_port", str(main_process_port),
        "--mixed_precision", precision,
        "--dynamo_backend", "no",
    ]
    if n > 1:
        cmd.append("--multi_gpu")
    cmd += [test_py, *tail]
    return cmd


def main() -> None:
    args = parse_args()
    test_yaml = Path(args.test_yaml).resolve()

    accel = _read_accel(test_yaml)
    gpu_ids = _normalise_gpu_ids(accel.get("gpu_ids"))
    precision = str(accel.get("precision", "bf16"))
    main_process_port = int(accel.get("main_process_port", 29500))
    launcher_cfg = args.launcher or str(accel.get("launcher", "auto"))
    launcher = _resolve_launcher(launcher_cfg, len(gpu_ids))

    tail = ["--test_yaml", str(Path(args.test_yaml).resolve())]
    cmd = build_command(launcher, gpu_ids, precision, main_process_port, tail)

    # Pin device visibility; child sees the requested GPUs remapped to 0..N-1.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    print(f"[launch_test] test_yaml         = {test_yaml}")
    print(f"[launch_test] accel.launcher    = {launcher_cfg} -> {launcher}")
    print(f"[launch_test] accel.gpu_ids     = {gpu_ids}  (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
    print(f"[launch_test] accel.precision   = {precision}")
    if launcher == "accelerate":
        print(f"[launch_test] main_process_port = {main_process_port}")
    print(f"[launch_test] command           = {' '.join(cmd)}")

    if args.dry_run:
        print("[launch_test] --dry-run set; not executing.")
        return

    proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
