"""Modal app for training-job dispatch on H100s.

Image: nvidia/cuda 12.4 + Python 3.11 + the pinned local venv via
requirements-modal.txt (--no-deps; vLLM 0.17 has tight declared bounds, see
DEPENDENCIES.md). Codebase mounted at /repo via add_local_dir(copy=False) as
the last layer so code-only edits don't bust the deps cache.

Volume: gr-modal-pilot mounted at /output (checkpoints + logs persist).
Secrets: gr-pilot-keys (WANDB_API_KEY, OPENAI_API_KEY).

Two function entry points:

  - `train_one(params, sweep_name)` — 1 run / container. Output goes to
    /output/<sweep>/<run_name>/. vllm_spawn=True so the worker runs vLLM
    in-process (no separate server inside the container).

  - `train_many(params_list, sweep_name)` — N runs / container with CUDA MPS.
    For small models (SmolLM2-135M etc.) the H100 is enormously underused at
    `vllm_gpu_memory=0.05`; packing seeds together lets one paid container run
    a whole hyperparam point's seeds concurrently. Callers should scale
    `vllm_gpu_memory` per item so the sum fits the device. Blast radius is the
    pack: a single hang/SIGTERM at the container timeout kills all siblings.

Use `_group_runs(...)` to plan packs (default groups all params except
seed/run_name; configurable). sweep.py's `--backend modal` uses this to feed
train_many automatically; this module's `launch_modal_*` entrypoints still
exist for one-shot bare launches.

Sync back (from rl/): `modal volume get gr-modal-pilot / output/`.
"""
from __future__ import annotations

import os

import modal

# Mount whichever checkout this file lives in (repo root = tools/'s parent), so a
# git worktree mounts its own code rather than a hardcoded path.
REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_REMOTE = "/repo"
OUTPUT_REMOTE = "/output"

app = modal.App("gr-pilot")

# Outputs (checkpoints, train.log, routing_eval.jsonl) persist on this volume.
vol = modal.Volume.from_name("gr-modal-pilot", create_if_missing=True)

# API keys injected into every container.
#   gr-pilot-keys — historical: created by jnward 2026-05-23; OPENAI_API_KEY
#     for the topic env's gpt-5-nano judge retain reward. (Originally intended
#     to carry WANDB_API_KEY too but only OPENAI_API_KEY is present today.)
#   wandb-key     — WANDB_API_KEY for whichever user is running the sweep.
#     Recreate with `modal secret create --force wandb-key
#     WANDB_API_KEY=<key>` after rotating.
# GRAFT_MODAL_SECRETS overrides the list (comma-separated names; "" = none) for
# a Modal environment that lacks these secrets — pair "" with no_wandb runs. The
# value is baked into the image env below so the container, which re-imports
# this module, builds the same dependency list as the launcher.
SECRET_NAMES = os.environ.get("GRAFT_MODAL_SECRETS", "gr-pilot-keys,wandb-key")
secrets = [modal.Secret.from_name(n.strip()) for n in SECRET_NAMES.split(",") if n.strip()]

# Image: build deps from pyproject.toml + uv.lock so it matches the local venv.
# vLLM 0.17.0 has broken dep metadata (declared bounds too tight for torch 2.10 /
# transformers 5.2 — see DEPENDENCIES.md); uv handles via the override section
# already in pyproject.toml.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .workdir("/build")
    .add_local_file(f"{REPO_LOCAL}/pyproject.toml", "/build/pyproject.toml", copy=True)
    .add_local_file(f"{REPO_LOCAL}/uv.lock", "/build/uv.lock", copy=True)
    .add_local_file(f"{REPO_LOCAL}/requirements-modal.txt", "/build/requirements-modal.txt", copy=True)
    # vLLM patches must be applied after install — they add a hook attribute
    # (LoRAModelManager._post_create_module_hooks), a qwen3_5 config fix, and a
    # gpu_worker memory-profiling clamp (benign free-memory increase when a
    # sibling process exits at a sweep tail) the codebase depends on at runtime.
    # See vllm_patches/apply.sh.
    .add_local_file(f"{REPO_LOCAL}/vllm_patches/model_manager.py",
                    "/build/vllm_patches/model_manager.py", copy=True)
    .add_local_file(f"{REPO_LOCAL}/vllm_patches/qwen3_5_config.py",
                    "/build/vllm_patches/qwen3_5_config.py", copy=True)
    .add_local_file(f"{REPO_LOCAL}/vllm_patches/gpu_worker.py",
                    "/build/vllm_patches/gpu_worker.py", copy=True)
    # Install the full pip-freeze from the working local venv (394 pkgs). Uses
    # --no-deps so pinned versions are respected as-is (vllm 0.17 has broken
    # declared bounds — see DEPENDENCIES.md). flash_attn/flash_attn_3 are
    # prebuilt-wheel URLs so no compile step.
    .run_commands(
        "uv venv --python 3.11 --seed",
        "uv pip install --python /build/.venv/bin/python --no-deps -r /build/requirements-modal.txt",
        # Apply vLLM patches over the installed package.
        "cp /build/vllm_patches/model_manager.py "
        "/build/.venv/lib/python3.11/site-packages/vllm/lora/model_manager.py",
        "cp /build/vllm_patches/qwen3_5_config.py "
        "/build/.venv/lib/python3.11/site-packages/vllm/transformers_utils/configs/qwen3_5.py",
        "cp /build/vllm_patches/gpu_worker.py "
        "/build/.venv/lib/python3.11/site-packages/vllm/v1/worker/gpu_worker.py",
        "ln -s /build/.venv /opt/venv",
    )
    .env({"PATH": "/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "PYTHONPATH": REPO_REMOTE,
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
          # Quiet HF transfer warnings, point cache to volume so model downloads persist.
          "HF_HOME": "/output/_hf_cache",
          # leetcode/leetcode_verified envs need rl-rewardhacking-private (dataset
          # jsonl + grader src/). The training image doesn't bundle it; it's staged
          # onto the volume at /output/_rh by tools/modal_leetcode_passk.py::stage.
          "RH_REPO_PATH": "/output/_rh",
          # leetcode grader forks many short subprocess workers; cap OMP so it
          # doesn't blow the container pid limit.
          "OMP_NUM_THREADS": "1",
          "GRAFT_MODAL_SECRETS": SECRET_NAMES,
          })
    # Add the codebase last so changes don't bust the deps cache.
    .add_local_dir(
        REPO_LOCAL,
        REPO_REMOTE,
        ignore=[
            "__pycache__", "*.pyc",
            ".git", ".venv", ".venv-vllm", ".claude", ".prompt_cache", ".pytest_cache",
            "wandb", "output", "media", "benchmarks",
            "*.log", "*.pdf", "*.png",
            "**/*.log", "**/*.pdf", "**/*.png",
            ".*.un~", ".*.sw[opn]", ".*.swp",
            "figures/figs", "final_figures", "paper_data",
        ],
        copy=False,  # fresh mount each call — captures recent edits cheaply
    )
)


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=15 * 60,  # 15 min smoke is plenty
)
def smoke() -> dict:
    """Sanity check: container has Python 3.11, repo importable, 1 H100, secrets set."""
    import os, sys, subprocess
    info = {}
    info["python"] = sys.version.split()[0]
    info["pwd"] = os.getcwd()
    info["repo_exists"] = os.path.isdir(REPO_REMOTE)
    info["wandb_key"] = bool(os.environ.get("WANDB_API_KEY"))
    info["openai_key"] = bool(os.environ.get("OPENAI_API_KEY"))
    info["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_err"] = str(e)
    try:
        import transformers, trl, vllm, peft
        info["transformers"] = transformers.__version__
        info["trl"] = trl.__version__
        info["vllm"] = vllm.__version__
        info["peft"] = peft.__version__
    except Exception as e:
        info["deps_err"] = str(e)
    try:
        from train import train_main  # noqa: F401
        info["train_import"] = True
    except Exception as e:
        info["train_import_err"] = str(e)
    return info


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=40 * 60,
)
def fused_gate(sweep_config: str, run_index: int = 0,
               rtol: float | None = None, reps: int = 20,
               vllm_spawn: bool = False, force_fp32: bool = False) -> dict:
    """Capture one prepared batch (HF generation) then run the fused-GR accuracy
    gate + timing against it, in one H100 container.

    Returns the captured/bench stdout so the caller can read the worst grad diff
    and the stock-vs-fused speedup. `sweep_config` is a repo-relative path, e.g.
    "sweeps/sort_idea2a_periter99_fp32hf_gr.py".
    """
    import os
    import subprocess

    os.chdir(REPO_REMOTE)
    env = {**os.environ, "PYTHONPATH": REPO_REMOTE}
    batch = "/tmp/gr_coh_batch.pt"

    cap_cmd = ["python", "tools/capture_gr_batch.py",
               "--sweep_config", sweep_config, "--run_index", str(run_index),
               "--out", batch]
    if vllm_spawn:
        cap_cmd.append("--vllm_spawn")
    cap = subprocess.run(cap_cmd, cwd=REPO_REMOTE, env=env,
                         capture_output=True, text=True, timeout=25 * 60)
    if not os.path.exists(batch):
        return {"status": "capture_failed",
                "capture_stdout": cap.stdout[-6000:],
                "capture_stderr": cap.stderr[-6000:]}

    bench_cmd = ["python", "bench_fused_gr.py", "--batch", batch,
                 "--sweep_config", sweep_config, "--run_index", str(run_index),
                 "--reps", str(reps)]
    if rtol is not None:
        bench_cmd += ["--rtol", str(rtol)]
    if force_fp32:
        bench_cmd.append("--force_fp32")
    bench = subprocess.run(bench_cmd, cwd=REPO_REMOTE, env=env,
                           capture_output=True, text=True, timeout=15 * 60)
    return {
        "status": "ok" if bench.returncode == 0 else f"gate_failed(rc={bench.returncode})",
        "capture_tail": cap.stdout[-1000:],
        "bench_stdout": bench.stdout,
        "bench_stderr": bench.stderr[-4000:],
    }


# --- Shared training-runner body ---
#
# Both train_one (1 run / container) and train_many (N runs / container via MPS)
# do the same per-run setup: pick an output dir on the volume, set vllm_spawn,
# tee stdout/stderr to train.log, call train.train_main, and report a status dict.
# Factored here so the two entrypoints stay in sync.

# Default vLLM memory fraction. 0.05 was the value used in the single-run pilot
# and is plenty for SmolLM2-135M on an H100. When packing N runs / container via
# train_many, callers should scale this down (e.g. 0.05 / N) so the sum stays
# within the device.
_VLLM_GPU_MEM_DEFAULT = 0.05

# Per-child vLLM-init queueing now happens inside train.py's _spawn_vllm_server
# via vllm_lifecycle.vllm_init_slot. All children in a train_many pack share
# the same per-container /tmp/vllm_init_lock_gpu0 flock, so their CUDA inits
# serialize naturally. No static stagger constant needed here anymore.


def _prepare_params(params: dict, sweep_name: str) -> tuple[dict, str, str, str]:
    """Resolve output_dir and log_path on the mounted volume; force vllm_spawn.

    Returns (params, basename, out_dir, log_path). Pure — no side effects on the
    OS beyond the os.makedirs of the run directory.

    Basename resolution: sweep.py's _materialize_run sets
    `params["run_name"] = "<sweep>/<basename>"` (a wandb identifier with the
    sweep prefix baked in for display in wandb's UI). Naively using that as
    the basename for the out_dir would produce
    `/output/<sweep>/<sweep>/<basename>/` on the volume — duplicated nesting.
    Derive the basename from `params["output_dir"]` (which sweep.py sets to
    the local path `output/<sweep>/<basename>`) so the Modal-side path stays
    well-formed regardless of how `run_name` is shaped.
    """
    import os
    local_output_dir = params.get("output_dir") or ""
    if local_output_dir:
        basename = os.path.basename(local_output_dir.rstrip("/"))
    else:
        # Defensive fallback: take the run_name's last path component if no
        # output_dir hint was passed (bare-entrypoint launches predating this
        # convention may rely on this).
        basename = (params.get("run_name") or "unnamed").rstrip("/").split("/")[-1]
    out_dir = os.path.join(OUTPUT_REMOTE, sweep_name, basename)
    os.makedirs(out_dir, exist_ok=True)
    params = {**params, "output_dir": out_dir, "gpu_id": 0}
    # Inside the container vLLM must run in-process (no separate server). The
    # caller may override vllm_gpu_memory; otherwise use the per-run default.
    params.setdefault("vllm_spawn", True)
    params.setdefault("vllm_gpu_memory", _VLLM_GPU_MEM_DEFAULT)
    log_path = os.path.join(out_dir, "train.log")
    return params, basename, out_dir, log_path


class _Tee:
    """File-like tee that fans writes out to multiple streams, swallowing
    per-stream errors so one broken stream doesn't break the others."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def _run_training(params: dict, log_path: str) -> dict:
    """Shared body: tee stdout/stderr to log_path, call train.train_main, and
    return a status dict. Catches SystemExit (graceful) and any other
    BaseException (crash, including KeyboardInterrupt from Modal SIGTERM ->
    preemption mapping). Does NOT commit the volume — the caller is responsible
    for that, so train_many can commit per child without forcing the others to
    wait for one volume commit per run.
    """
    import sys
    import time
    import traceback

    log_f = open(log_path, "a", buffering=1)
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    sys.stdout = _Tee(real_stdout, log_f)
    sys.stderr = _Tee(real_stderr, log_f)

    t0 = time.time()
    status = "ok"
    err = None
    try:
        # Import here so any import-time side effects happen with stdout=tee.
        from train import train_main
        train_main(params)
    except SystemExit as e:
        if e.code not in (0, None):
            status = f"sysexit({e.code})"
            err = str(e)
    except BaseException as e:
        status = "crash"
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        log_f.close()
        sys.stdout = real_stdout
        sys.stderr = real_stderr

    return {"status": status, "duration_s": time.time() - t0, "err": err}


@app.function(
    image=image,
    gpu="H200",
    # Without an explicit request, a GPU function gets Modal's tiny default CPU
    # share: measured 2026-07-16, the in-train kodcode fork-server pool graded a
    # 256-job coherence slice in 204s (~1 effective core) while the SAME batch
    # took 22s on a cpu=8 container and 22s on cpu=64 (tail-dominated; see
    # tools/modal_kodcode_coh_grade_perf.py). Reward grading is synchronous and
    # GPU-idle, so starving it directly stretches every rollout phase.
    cpu=64,
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=8 * 60 * 60,  # 8h max per run (4h killed a countdown_code DN run at
                          # step 142/200 — slow arms with long completions need slack;
                          # NOTE: the budget is per-INPUT, so a Modal container-death
                          # reschedule does NOT reset the clock)
)
def train_one(params: dict, sweep_name: str) -> dict:
    """Run a single training job inside a Modal container with 1 H100."""
    import os
    import sys

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    params, run_name, out_dir, log_path = _prepare_params(params, sweep_name)
    try:
        result = _run_training(params, log_path)
    finally:
        vol.commit()  # ensure outputs are flushed to the volume

    return {**result, "run_name": run_name, "output_dir": out_dir}


# MPS lifecycle for train_many.
#
# History (2026-06-02): the first attempt called `nvidia-cuda-mps-control -d`
# and proceeded immediately. The daemon "started" (exit 0) but child processes
# crashed at torch.cuda.set_device() with Error 805 ("MPS client failed to
# connect to the MPS control daemon or the MPS server"). The fallback at the
# time was to disable MPS entirely and accept ~4× slowdown from time-sliced
# multi-process scheduling.
#
# Retry (this revision): set CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY
# explicitly under /tmp; clean any stale state from a previous attempt; wait
# for the control pipe to appear (signal that the daemon is listening); then
# run an actual functional probe (a spawn-context subprocess that imports
# torch and calls torch.cuda.set_device(0)). If anything fails along the way,
# unset the env vars and clean up the pipe dir so child processes do NOT
# detect MPS — they fall back to standard time-slicing. The daemon log goes
# to the log dir so failures can be inspected after the fact.
_MPS_PIPE_DIR = "/tmp/nvidia-mps"
_MPS_LOG_DIR = "/tmp/nvidia-log"


def _disable_mps_env():
    """Remove MPS-related env vars and pipe dir so spawn-context children do
    not detect MPS. Safe to call repeatedly; used when initialisation fails."""
    import os
    import shutil
    os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
    shutil.rmtree(_MPS_PIPE_DIR, ignore_errors=True)


def _mps_probe_child(conn) -> None:
    """Spawn-context entrypoint for _probe_mps_works. Must be defined at
    module scope so multiprocessing can pickle the function reference; nested
    closures fail with AttributeError: Can't pickle local object."""
    try:
        import torch
        torch.cuda.set_device(0)
        # Force CUDA init (set_device alone may be lazy).
        torch.cuda.init()
        conn.send({"ok": True, "name": torch.cuda.get_device_name(0)})
    except BaseException as e:
        conn.send({"ok": False, "err": f"{type(e).__name__}: {e}"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _mps_prewarm_child(conn, model_name) -> None:
    """Spawn-context entrypoint for _prewarm_cuda_and_vllm. Pays the
    first-cold-init costs that would otherwise fall on the first child of a
    train_many pack while N-1 other waiters tick toward hold_timeout:

      - First MPS-server allocation (the daemon spins up its CUDA context
        on the first client connection; subsequent clients are fast).
      - Page cache warm-up for libtorch, libcuda, libvllm.
      - HF cache warm-up for model weights via snapshot_download (or noop if
        the model isn't fetchable from this process — failure is swallowed).
      - A small CUDA matmul to force cuBLAS/cuDNN heuristic-table init.

    Each child still pays its own per-process import + CUDA context setup,
    but the cold-cache portion (typically the largest tail) is gone.
    Result dict: {ok, err, duration_s, warmed: list[str]}.
    """
    import time
    t0 = time.time()
    warmed = []
    try:
        import torch
        torch.cuda.set_device(0)
        # CUDA init under MPS — first interaction with the daemon's server.
        x = torch.ones(64, 64, device="cuda")
        (x @ x.T).sum().item()
        del x
        warmed.append("cuda")
        # vLLM import: page-cache the heavy library.
        try:
            import vllm  # noqa: F401
            warmed.append("vllm-import")
        except Exception:
            pass
        # HF cache: ensure the model weights are present + paged-in. Cheap if
        # already cached on the volume; ~5-30s otherwise. Best-effort.
        if model_name:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
                warmed.append("hf-cache")
            except Exception:
                pass
        conn.send({"ok": True, "warmed": warmed, "duration_s": time.time() - t0})
    except BaseException as e:
        conn.send({
            "ok": False, "warmed": warmed,
            "err": f"{type(e).__name__}: {e}",
            "duration_s": time.time() - t0,
        })
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _prewarm_cuda_and_vllm(model_name=None, timeout: float = 600.0) -> bool:
    """Run _mps_prewarm_child in a spawn-context subprocess. Returns True iff
    the prewarm completed without raising. We proceed either way — failure
    just means children take longer on their first inits — but emit a clear
    [prewarm] line so the cost is recorded in train.log."""
    import multiprocessing
    import time
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_mps_prewarm_child, args=(child_conn, model_name))
    t0 = time.time()
    proc.start()
    child_conn.close()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        print(f"[prewarm] timed out after {timeout}s; proceeding without warm cache")
        return False
    try:
        result = parent_conn.recv()
    except (EOFError, OSError):
        print(f"[prewarm] child exited without result (exitcode={proc.exitcode})")
        return False
    dur = time.time() - t0
    if result.get("ok"):
        print(f"[prewarm] ok in {dur:.1f}s — warmed {result.get('warmed', [])}")
        return True
    print(f"[prewarm] failed in {dur:.1f}s: {result.get('err')}; "
          f"warmed up to: {result.get('warmed', [])}; proceeding cold")
    return False


def _probe_mps_works(timeout: float = 60.0) -> bool:
    """Spawn a fresh subprocess that imports torch and runs torch.cuda.set_device(0).
    Returns True iff the subprocess completes without raising. Used to confirm
    MPS is actually servicing requests before we commit to using it for the
    training children.

    Why this is needed: the failure mode we saw before was that
    `nvidia-cuda-mps-control -d` exits cleanly even when the daemon won't
    accept client connections, so neither the exit code nor the pipe-dir
    contents are sufficient to verify MPS is functional. Only an actual CUDA
    init from a separate process catches it.
    """
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(target=_mps_probe_child, args=(child_conn,))
    proc.start()
    child_conn.close()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        print(f"[mps] probe timed out after {timeout}s")
        return False
    try:
        result = parent_conn.recv()
    except (EOFError, OSError):
        print(f"[mps] probe child exited without sending a result (exitcode={proc.exitcode})")
        return False
    if result.get("ok"):
        print(f"[mps] probe ok — device={result.get('name')}")
        return True
    print(f"[mps] probe failed: {result.get('err')}")
    return False


def _start_mps() -> bool:
    """Try to start CUDA MPS. Returns True iff the daemon is up AND the probe
    confirms a child can initialise CUDA against it. On failure, cleans up env
    so subsequent children won't detect MPS at all.
    """
    import os
    import shutil
    import subprocess
    import time

    # Clean any stale state from previous attempts within this container.
    for d in (_MPS_PIPE_DIR, _MPS_LOG_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # Set env vars in the parent so spawn-context children inherit them.
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = _MPS_PIPE_DIR
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = _MPS_LOG_DIR

    try:
        proc = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            check=False, capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        print("[mps] nvidia-cuda-mps-control binary not found; running without MPS")
        _disable_mps_env()
        return False
    except Exception as e:
        print(f"[mps] daemon launch failed: {e}; running without MPS")
        _disable_mps_env()
        return False

    if proc.returncode != 0:
        print(f"[mps] daemon exited {proc.returncode}: "
              f"stdout={proc.stdout!r} stderr={proc.stderr!r}; running without MPS")
        _disable_mps_env()
        return False

    # Wait up to ~10s for the control pipe to appear (signal the daemon is
    # listening). The pipe path varies a bit by CUDA version, so look for any
    # entry in the pipe directory.
    t0 = time.time()
    while time.time() - t0 < 10.0:
        try:
            if os.listdir(_MPS_PIPE_DIR):
                break
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    else:
        print(f"[mps] control pipe never appeared in {_MPS_PIPE_DIR} after 10s; "
              "running without MPS")
        _disable_mps_env()
        return False

    # Probe: actually init CUDA from a spawn-context subprocess. This is the
    # only check that distinguishes "daemon listening but not servicing" from
    # "daemon fully functional".
    if not _probe_mps_works():
        print("[mps] probe failed after daemon start; falling back to time-slicing")
        _stop_mps()
        _disable_mps_env()
        return False

    print(f"[mps] daemon up at {_MPS_PIPE_DIR}; probe ok; packing under MPS")
    return True


def _stop_mps():
    """Send 'quit' to the MPS control daemon. Best-effort cleanup; container
    teardown would do this anyway, but we explicit-stop so any stale state
    doesn't leak into a future invocation that happens to reuse the host."""
    import subprocess
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=b"quit\n", check=False, capture_output=True, timeout=5,
        )
    except Exception:
        pass


def _train_many_child(params: dict, log_path: str, conn) -> None:
    """train_many child entry point. Runs one training job and sends its
    result dict back to the parent via the given Pipe end. Used as the target
    of `multiprocessing.get_context('spawn').Process` so child memory is
    isolated from siblings."""
    try:
        result = _run_training(params, log_path)
    except BaseException as e:
        # Catch-all so a child crash here (vs. inside train_main) still produces
        # a structured result. _run_training already catches the train_main path;
        # this guards setup errors only.
        result = {"status": "child_crash", "duration_s": 0.0, "err": f"{type(e).__name__}: {e}"}
    try:
        conn.send(result)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.function(
    image=image,
    gpu="H200",
    cpu=64,   # same grading-starvation fix as train_one (measured 204s -> 22s)
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    # Wall-clock budget for the WHOLE pack. Modal sends SIGTERM at this
    # boundary, then SIGKILL ~30s later. Packs run concurrently under MPS so
    # the budget should match the slowest run, not N * per_run_budget.
    timeout=4 * 60 * 60,
)
def train_many(params_list: list[dict], sweep_name: str) -> list[dict]:
    """Pack multiple training jobs into one H100 container.

    Each item in `params_list` is a params dict for `train.train_main`; results
    are returned in the same order, each with status / duration_s / err /
    run_name / output_dir. Children are spawn-context subprocesses so they don't
    share Python interpreter state (matches sweep.py's local concurrency).

    Concurrency model: CUDA MPS when available, time-slicing when not.
    `_start_mps()` tries to bring up the MPS daemon and verifies it with a
    spawn-context probe; if either step fails, MPS env is unset and child
    processes use the standard time-slicing path (~4× slower per child than
    dedicated, but functionally correct). Whichever path the container takes
    is logged as `[mps] ...` at the top of train.log on the volume.

    Caller is responsible for sizing the pack — typically by setting
    `vllm_gpu_memory` per item to `~0.05 / N` so the sum fits the device.

    Blast radius: a hang in any single run drags the whole container into the
    `timeout=` boundary above and kills siblings at the same time. Anything
    already written to /output (checkpoints at save_steps multiples,
    routing_eval.jsonl, train.log) is preserved; in-flight rollout state is
    lost.
    """
    import multiprocessing
    import os
    import sys
    import time

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    n = len(params_list)
    assert n > 0, "train_many called with empty params_list"
    # Debug/benchmark knob: `_force_no_mps=True` on any run in the pack forces the
    # time-sliced path (skip the MPS daemon) so MPS-on vs MPS-off can be A/B'd
    # under otherwise-identical configs. Popped here so it never reaches
    # train_main (which asserts on unknown params).
    force_no_mps = False
    for p in params_list:
        if p.pop("_force_no_mps", False):
            force_no_mps = True
    mps_on = False if force_no_mps else _start_mps()
    mode = "MPS" if mps_on else ("time-sliced (forced)" if force_no_mps else "time-sliced")
    print(f"[train_many] packing {n} runs in one H100 container ({mode})")
    # Prewarm before forking children. Pays the first-cold-vLLM-init cost in
    # the parent so child #1 of the pack doesn't pay it while children
    # #2..#N tick toward vllm_init_slot's hold_timeout. Without this, the
    # first child's slow init can take 150-200s on cold MPS + cold disk,
    # cascading waiters to abort even with the bumped hold_timeout.
    # Cheaper to pay once in the parent than to wear the cascade-risk tax
    # across re-runs.
    if mps_on:
        model_name = params_list[0].get("model")
        print(f"[train_many] pre-warming for model={model_name}")
        _prewarm_cuda_and_vllm(model_name)

    # Resolve per-run paths up front so we can return a stable list of results
    # even if a child fails before it can send anything back. CUDA-init
    # serialization happens via the per-GPU flock that train.py's
    # _spawn_vllm_server acquires inside vllm_init_slot — see vllm_lifecycle.
    resolved = []  # list of {params, run_name, out_dir, log_path}
    for params in params_list:
        p, run_name, out_dir, log_path = _prepare_params(params, sweep_name)
        resolved.append({
            "params": p, "run_name": run_name,
            "out_dir": out_dir, "log_path": log_path,
        })

    # Persist the pack's concurrency mode to each run's output dir (on the
    # volume) so MPS-on vs time-sliced is verifiable post-hoc. The [mps] daemon
    # status is otherwise only in the container's stdout, which Modal does not
    # write to the volume and truncates once the ephemeral app stops.
    for r in resolved:
        try:
            os.makedirs(r["out_dir"], exist_ok=True)
            with open(os.path.join(r["out_dir"], "mps_status.txt"), "w") as f:
                f.write(f"mps_on={mps_on}\nmode={mode}\nforce_no_mps={force_no_mps}\n"
                        f"pack_size={n}\n")
        except Exception as e:
            print(f"[train_many] WARN: could not write mps_status.txt for {r['run_name']}: {e}")

    ctx = multiprocessing.get_context("spawn")
    children = []
    for r in resolved:
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_train_many_child,
            args=(r["params"], r["log_path"], child_conn),
        )
        proc.start()
        # Close the child end in the parent — only the child should be holding it.
        child_conn.close()
        children.append({"proc": proc, "conn": parent_conn, "t0": time.time(), **r})
        print(f"[train_many] spawned child pid={proc.pid} for {r['run_name']}")

    # Join children in completion order so the first finisher's outputs are
    # committed to the volume promptly (helps survivors of a sibling crash).
    pending = list(range(len(children)))
    results = [None] * len(children)
    while pending:
        # Cheap poll loop — children write to the volume directly; we just need
        # to harvest results and vol.commit() as soon as each one is done.
        for i in list(pending):
            c = children[i]
            if not c["proc"].is_alive():
                try:
                    result = c["conn"].recv()
                except (EOFError, OSError):
                    result = {"status": "no_result", "duration_s": time.time() - c["t0"],
                              "err": "child closed pipe with no payload"}
                try:
                    c["conn"].close()
                except Exception:
                    pass
                c["proc"].join()
                exit_status = result.get("status", "unknown")
                if c["proc"].exitcode not in (0, None) and exit_status == "ok":
                    # Child reported ok but exited nonzero — surface that.
                    exit_status = f"exitcode({c['proc'].exitcode})"
                    result["status"] = exit_status
                results[i] = {
                    **result,
                    "run_name": c["run_name"],
                    "output_dir": c["out_dir"],
                }
                pending.remove(i)
                print(f"[train_many] {c['run_name']} done: {exit_status} "
                      f"({result.get('duration_s', 0):.1f}s)")
                vol.commit()
        if pending:
            time.sleep(2.0)

    if mps_on:
        _stop_mps()
    return results


# --- Pack planning (run → group → train_many call) ---
#
# Default semantics: runs with identical hyperparameters except `seed` and
# `run_name` pack together. Pass `group_keys=("config",)` to pack everything
# with the same env config; pass `group_keys=()` to pack everything (subject to
# max_per_pack). Used by `_dispatch_packed_sweep` and by sweep.py's --backend
# modal path. Keep the algorithm here so the two callers stay aligned.

_PACK_SKIP_KEYS_DEFAULT = ("seed", "run_name", "output_dir", "gpu_id", "wandb_run_id")


def _hashable(v):
    """Best-effort hashable transform for grouping params dicts. Dicts → sorted
    tuple of (key, _hashable(value)); lists → tuple of _hashable; everything
    else falls through unchanged if hashable, str(v) if not."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(vv)) for k, vv in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    try:
        hash(v)
        return v
    except TypeError:
        return str(v)


def _group_runs(runs, group_keys=None, max_per_pack=6, skip_keys=None):
    """Pack runs into compatible groups for train_many.

    Arguments
    ---------
    runs         : list[dict]  — each dict is a train.train_main params bundle.
    group_keys   : Iterable[str] | None
        If None (default), two runs pack together iff they agree on every key
        except those in `skip_keys`. This is the "all-seeds-of-a-hyperparam-
        point-together" default.
        If an explicit iterable, only those keys define compatibility. E.g.
        `("config",)` packs everything with the same env config; `()` packs
        everything (still capped by max_per_pack).
    max_per_pack : int  — hard cap on pack size. With SmolLM2-135M and
        vllm_gpu_memory≈0.05 we measure 6-8 concurrent comfortably on one H100;
        6 is the safe default.
    skip_keys    : Iterable[str] | None  — only used when group_keys=None. Adds
        to the default skip list, doesn't replace it.

    Returns
    -------
    list[list[dict]] — each inner list goes to one train_many call. Order
    within a pack is the order runs appeared in the input.
    """
    from collections import defaultdict

    skip = set(_PACK_SKIP_KEYS_DEFAULT)
    if skip_keys is not None:
        skip |= set(skip_keys)

    if group_keys is None:
        def keyfn(r):
            return tuple(sorted(
                (k, _hashable(v)) for k, v in r.items() if k not in skip
            ))
    else:
        gk_tup = tuple(group_keys)
        def keyfn(r):
            return tuple((k, _hashable(r.get(k))) for k in gk_tup)

    groups = defaultdict(list)
    order = []  # preserve first-seen order across groups
    for r in runs:
        k = keyfn(r)
        if k not in groups:
            order.append(k)
        groups[k].append(r)

    packs = []
    for k in order:
        g = groups[k]
        for i in range(0, len(g), max_per_pack):
            packs.append(g[i:i + max_per_pack])
    return packs


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=10 * 60,
)
def mps_diag() -> dict:
    """Deep-diagnose why CUDA MPS does/doesn't start on a Modal H100.

    Captures the things never examined before: the MPS server-side log
    (CUDA_MPS_LOG_DIRECTORY), daemon control-command responsiveness, probe
    flakiness across retries, and runtime/sandbox fingerprints (gVisor/nvproxy
    are the likely "interface Modal doesn't expose"). Goal: distinguish
    "transient/flaky probe (retry fixes it)" from "MPS server systematically
    can't attach (platform limit)". Run:
      modal run tools/modal_train_gr.py::mps_diagnose
    """
    import os, sys, subprocess, shutil, io, contextlib, time, glob
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    def sh(cmd, timeout=30):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return (r.stdout + r.stderr).strip()
        except Exception as e:
            return f"<err: {e}>"

    info = {}
    # --- binary + GPU state ---
    info["mps_control_path"] = shutil.which("nvidia-cuda-mps-control")
    info["mps_server_path"] = shutil.which("nvidia-cuda-mps-server")
    try:
        smi = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=30).stdout
        info["compute_mode"] = next((l.strip() for l in smi.splitlines() if "Compute Mode" in l), "?")
        info["driver"] = next((l.strip() for l in smi.splitlines() if "Driver Version" in l), "?")
        info["cuda_version_smi"] = next((l.strip() for l in smi.splitlines() if "CUDA Version" in l), "?")
    except Exception as e:
        info["smi_err"] = str(e)

    # --- runtime / sandbox fingerprints (gVisor/nvproxy detection) ---
    info["uname"] = sh("uname -a")
    info["proc_version"] = sh("cat /proc/version")
    info["dmesg_gvisor"] = sh("dmesg 2>&1 | grep -iE 'gvisor|runsc' | head -3") or "<none/blocked>"
    info["dev_nvidia"] = sh("ls -la /dev/nvidia* 2>&1")
    info["dev_nvidia_uvm"] = sh("ls -la /dev/nvidia-uvm* /dev/nvidia-caps* 2>&1")
    info["shm"] = sh("df -h /dev/shm")
    info["is_root"] = (os.geteuid() == 0)
    info["caps"] = sh("grep -E 'Cap(Eff|Prm|Bnd)' /proc/self/status")
    # gVisor often reports a sentinel/altered kernel; nvproxy shows up in mounts.
    info["mounts_nvidia"] = sh("cat /proc/self/mountinfo 2>&1 | grep -iE 'nvidia|gpu' | head")

    # --- run the real _start_mps() and capture its [mps] prints ---
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ok = _start_mps()
        info["start_mps_result"] = ok
    except Exception as e:
        info["start_mps_exc"] = repr(e)
    info["start_mps_log"] = buf.getvalue()

    # --- dump the MPS server/control logs (the never-examined goldmine) ---
    logdir = os.environ.get("CUDA_MPS_LOG_DIRECTORY", _MPS_LOG_DIR)
    info["mps_log_dir"] = logdir
    info["mps_log_files"] = sh(f"ls -la {logdir} 2>&1")
    logs = {}
    for lf in glob.glob(os.path.join(logdir, "*")):
        try:
            logs[os.path.basename(lf)] = open(lf, errors="ignore").read()[-4000:]
        except Exception as e:
            logs[os.path.basename(lf)] = f"<err: {e}>"
    info["mps_logs"] = logs

    # --- ask the daemon directly (is it servicing control commands?) ---
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", _MPS_PIPE_DIR)
    info["pipe_dir_contents"] = sh(f"ls -la {pipe} 2>&1")
    info["ctl_get_server_list"] = sh(f"echo get_server_list | nvidia-cuda-mps-control", timeout=15)
    info["ctl_device_count"] = sh(f"echo get_device_count | nvidia-cuda-mps-control", timeout=15)

    # --- probe flakiness: run the functional probe several times ---
    probes = []
    for i in range(4):
        try:
            probes.append(bool(_probe_mps_works(timeout=30)))
        except Exception as e:
            probes.append(f"exc:{e!r}")
        time.sleep(2)
    info["probe_retries"] = probes

    # re-dump logs after probes (server.log may now have the failure)
    logs2 = {}
    for lf in glob.glob(os.path.join(logdir, "*")):
        try:
            logs2[os.path.basename(lf)] = open(lf, errors="ignore").read()[-4000:]
        except Exception as e:
            logs2[os.path.basename(lf)] = f"<err: {e}>"
    info["mps_logs_after_probes"] = logs2

    _stop_mps()
    return info


@app.local_entrypoint()
def mps_diagnose():
    """Report why CUDA MPS does/doesn't start on a Modal H100."""
    import json
    res = mps_diag.remote()
    print(json.dumps(res, indent=2))


@app.local_entrypoint()
def smoke_test():
    """Build the image (first time) and verify the container env. Run before launch_modal_3envs."""
    result = smoke.remote()
    import json
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def fused_gate_run(sweep_config: str = "sweeps/sort_idea2a_periter99_fp32hf_gr.py",
                   run_index: int = 0, rtol: str = "", reps: int = 20,
                   vllm_spawn: bool = False, force_fp32: bool = False):
    """Run the fused-GR accuracy gate + timing on one H100.

    --force-fp32 converts the model to fp32 for a tight relative-tolerance gate
    (SmolLM2 loads bf16 by config default otherwise). For the realistic bf16
    timing number, run without it on the bf16 canonical config:
      modal run tools/modal_train_gr.py::fused_gate_run --sweep-config sweeps/sort_idea2a_periter99_gr.py --rtol 2e-2
    """
    rtol_f = float(rtol) if rtol else None
    res = fused_gate.remote(sweep_config, run_index, rtol_f, reps, vllm_spawn, force_fp32)
    print(f"\n=== fused_gate status: {res['status']} ===")
    print("--- capture tail ---")
    print(res.get("capture_tail") or res.get("capture_stdout", ""))
    print("--- bench stdout ---")
    print(res.get("bench_stdout", ""))
    if res.get("bench_stderr"):
        print("--- bench stderr (tail) ---")
        print(res["bench_stderr"])
    if res.get("capture_stderr"):
        print("--- capture stderr (tail) ---")
        print(res["capture_stderr"])


def _dispatch_sweep(sweep_module_name: str, sweep_name: str):
    """Shared launch logic: import a sweep config and dispatch each run as a Modal call."""
    import sys
    import importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs

    print(f"[modal] dispatching {len(runs)} runs (sweep={sweep_name})")
    for r in runs:
        print(f"  - {r['run_name']}")

    # .starmap dispatches all in parallel; each gets its own container/GPU.
    results = list(train_one.starmap([(r, sweep_name) for r in runs]))
    for res in results:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")
    failures = [r for r in results if r["status"] != "ok"]
    if failures:
        print(f"[modal] {len(failures)} run(s) failed")
    else:
        print(f"[modal] all {len(results)} runs ok")
    print(f"[modal] sync back: modal volume get gr-modal-pilot / {REPO_LOCAL}/output/")


def _dispatch_packed_sweep(sweep_module_name: str, sweep_name: str,
                           group_keys=None, max_per_pack=6,
                           vllm_gpu_memory=None):
    """Packed-launch variant of `_dispatch_sweep`. Groups runs with `_group_runs`,
    dispatches each pack to `train_many`, and prints a per-run summary on the
    same shape as the unpacked path.

    If `vllm_gpu_memory` is None, the per-item value is set to
    `0.40 / max_per_pack` so the per-pack sum stays below ~0.40 of the H100 —
    safe headroom for SmolLM2-135M-class models. Pass an explicit value to
    override (e.g. for larger models where each run wants more KV cache).
    """
    import sys
    import importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs

    packs = _group_runs(runs, group_keys=group_keys, max_per_pack=max_per_pack)
    per_item_mem = vllm_gpu_memory if vllm_gpu_memory is not None else (0.40 / max_per_pack)
    # Apply only when caller hasn't already set it explicitly.
    for r in runs:
        r.setdefault("vllm_gpu_memory", per_item_mem)

    print(f"[modal] packing {len(runs)} runs into {len(packs)} train_many call(s) "
          f"(max_per_pack={max_per_pack}, vllm_gpu_memory≈{per_item_mem:.3f}/item, "
          f"sweep={sweep_name})")
    for i, pack in enumerate(packs):
        names = ", ".join(r["run_name"] for r in pack)
        print(f"  pack {i+1}/{len(packs)} (n={len(pack)}): {names}")

    # .starmap over packs runs them in parallel containers; each container then
    # runs its N runs concurrently under MPS.
    pack_results = list(train_many.starmap([(pack, sweep_name) for pack in packs]))
    # Flatten + report.
    flat = [r for pack_res in pack_results for r in pack_res]
    for res in flat:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")
    failures = [r for r in flat if r["status"] != "ok"]
    if failures:
        print(f"[modal] {len(failures)}/{len(flat)} run(s) failed")
    else:
        print(f"[modal] all {len(flat)} runs ok")
    print(f"[modal] sync back: modal volume get gr-modal-pilot / {REPO_LOCAL}/output/")


@app.local_entrypoint()
def launch_sweep(sweep_module: str, sweep_name: str):
    """Generic dispatch: import a sweeps module (defines `runs`) and run each
    on its own H200 via train_one. E.g.:
        modal run tools/modal_train_gr.py::launch_sweep \
            --sweep-module sweeps.cdhf100_noroute_retry --sweep-name cdhf100_noroute
    """
    _dispatch_sweep(sweep_module, sweep_name)


@app.local_entrypoint()
def launch_modal_3envs():
    """6 runs: object_qa + addition_v2 + topic_contains × 2 seeds, exclusive routing."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_3envs_exclusive_nocoh_1k",
        "retrain_gr_modal_3envs_exclusive_nocoh_1k",
    )


@app.local_entrypoint()
def launch_modal_all_classic():
    """14 runs: 7 envs × 2 seeds, classic routing (no coherence, max_steps=1000)."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_all_classic_nocoh_1k",
        "retrain_gr_modal_all_classic_nocoh_1k",
    )


@app.local_entrypoint()
def launch_modal_6envs_classic_coh():
    """12 runs: 6 envs (topic skipped) × 2 seeds, classic + coherence enabled, max_steps=1000."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_6envs_classic_coh_1k",
        "retrain_gr_modal_6envs_classic_coh_1k",
    )


@app.local_entrypoint()
def launch_modal_6envs_excl_coh():
    """12 runs: 6 envs (topic skipped) × 2 seeds, exclusive + coherence enabled, max_steps=1000."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_6envs_excl_coh_1k",
        "retrain_gr_modal_6envs_excl_coh_1k",
    )


# ---------------------------------------------------------------------------
# Forget-scale eval: per checkpoint, sweep set_scales(retain=1, forget=s) over
# s in 0..1 and score the posthoc eval metrics (monitored/unmonitored hack
# channels + retain reward) per scale. Reads checkpoints from the volume;
# writes /output/<sweep>_fseval/<run>.json. Ported from graft-routing 933dd82;
# adapted for Qwen3-8B (bf16 cast, longer timeout).
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=4 * 60 * 60,
)
def eval_forget_scales_one(run_name: str, sweep_name: str, n_eval: int = 256,
                           scales: list = None, retain_scale: float = 1.0,
                           checkpoint_step: int = None) -> dict:
    """Forget-scale posthoc eval over the latest checkpoint of one run.

    retain_scale=0.0 with scales=[0.0] evaluates the frozen BASE model exactly
    (adapter-only training; both adapters contribute nothing at scale 0).
    Non-default retain_scale writes to <run>__r<retain_scale>.json so the
    standard fseval JSON is not clobbered."""
    import glob
    import json
    import os
    import sys

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    import torch
    import yaml
    from transformers import AutoTokenizer
    from eval_utils import (_find_run_config, load_gradient_routing_model,
                            posthoc_eval_from_checkpoint)

    if scales is None:
        scales = [round(0.1 * i, 1) for i in range(11)]   # 0.0 .. 1.0
    run_dir = os.path.join(OUTPUT_REMOTE, sweep_name, run_name)
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    assert cks, f"no checkpoints in {run_dir}"
    if checkpoint_step is not None:
        ckpt = next((c for c in cks if int(c.rsplit("-",1)[-1]) == checkpoint_step), None)
        assert ckpt is not None, f"no checkpoint-{checkpoint_step} in {run_dir}"
    else:
        ckpt = cks[-1]
    step = int(ckpt.rsplit("-", 1)[-1])
    rc = _find_run_config(ckpt) or os.path.join(run_dir, "run_config.yaml")
    base = (yaml.safe_load(open(rc)) or {}).get("model")
    tok = AutoTokenizer.from_pretrained(base)
    model = load_gradient_routing_model(ckpt, base_model=base)
    # Match the training dtype (bf16=True on all 8B runs): fp32 HF generate on
    # an 8B model is both ~2x slower and a sampling-distribution mismatch vs
    # the bf16 training/eval rollouts.
    model.to(torch.bfloat16)
    modes = [(f"fs{s:.1f}", float(retain_scale), float(s)) for s in scales]
    results = posthoc_eval_from_checkpoint(model, tok, ckpt, n_eval=n_eval, modes=modes,
                                           run_config_path=rc)

    out = {"run_name": run_name, "step": step, "n_eval": n_eval,
           "retain_scale": retain_scale, "scales": {}}
    for mname, mres in results.items():
        out["scales"][mname[2:]] = {
            k: v["mean"] for k, v in mres.get("metrics", {}).items()
            if isinstance(v, dict) and v.get("mean") is not None}
    outdir = os.path.join(OUTPUT_REMOTE, f"{sweep_name}_fseval")
    os.makedirs(outdir, exist_ok=True)
    # Suffixes COMPOSE: __step{N} then __r{r}. (Previously __step replaced the
    # whole name, so a step-targeted retain!=1 eval silently collided with the
    # step-targeted retain=1.0 file.)
    fname = run_name
    if checkpoint_step is not None:
        fname += f"__step{checkpoint_step}"
    if retain_scale != 1.0:
        fname += f"__r{retain_scale:.1f}"
    fname += ".json"
    with open(os.path.join(outdir, fname), "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    metrics = list(next(iter(out["scales"].values())).keys()) if out["scales"] else []
    print(f"[fseval {run_name}] step={step} scales={sorted(out['scales'])} metrics={metrics}")
    return out


@app.local_entrypoint()
def eval_forget_scales(sweep_name: str, n_eval: int = 256,
                       condition: str = "", only: str = "",
                       config_module: str = "sweeps.countdown_code_gr_nocoh",
                       scales_csv: str = "", retain_scale: float = 1.0,
                       names_csv: str = "", wait: bool = True,
                       checkpoint_step: int = None):
    """Dispatch the forget-scale eval across a sweep's runs.

    sweep_name is the timestamped output dir on the volume (e.g.
    countdown_code_gr_nocoh-0703-XXXX); run basenames come from config_module.

    modal run tools/modal_train_gr.py::eval_forget_scales \
        --sweep-name countdown_code_gr_nocoh-0703-XXXX
    """
    import importlib

    if names_csv:
        # Explicit run-dir basenames (for runs with auto-generated names that
        # aren't recorded in a sweep config, e.g. RP/do-nothing baselines).
        names = [n.strip() for n in names_csv.split(",") if n.strip()]
    else:
        mod = importlib.import_module(config_module)
        names = [r["run_name"] for r in mod.runs if condition in r["run_name"]]
    if only:
        names = [n for n in names if only in n]
    scales = [float(s) for s in scales_csv.split(",")] if scales_csv else None
    print(f"[fseval] dispatching {len(names)} eval(s) (condition={condition!r}, "
          f"n_eval={n_eval}, scales={scales}, retain_scale={retain_scale})")
    call_args = [(n, sweep_name, n_eval, scales, retain_scale, checkpoint_step) for n in names]
    if not wait:
        # Fire-and-forget for flaky control-plane links (e.g. the 2026-07-07
        # Tor bridge): each remote call writes its own JSON to the volume, so
        # the local collation loop is optional. Requires `modal run --detach`
        # so the spawned calls outlive this client. Poll the volume for
        # /output/{sweep_name}_fseval/*.json to track completion.
        for a in call_args:
            fc = eval_forget_scales_one.spawn(*a)
            print(f"[fseval] spawned {a[0]} -> {fc.object_id}")
        print(f"[fseval] no-wait: {len(call_args)} spawned; results land in "
              f"/output/{sweep_name}_fseval/")
        return
    results = list(eval_forget_scales_one.starmap(call_args, return_exceptions=True))
    ok = [r for r in results if isinstance(r, dict)]
    print(f"[fseval] done: {len(ok)}/{len(names)} ok -> /output/{sweep_name}_fseval/*.json")
    for (n, *_), r in zip(call_args, results):
        if not isinstance(r, dict):
            print(f"[fseval]   FAILED {n}: {type(r).__name__}: {r}")


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol},
              timeout=4 * 3600, memory=131072)
def gen_sft_dataset_one(run_name: str, sweep_name: str, n_prompts: int = 4096,
                        k: int = 16, sft_start: int = 128_000) -> dict:
    """SFT-distillation teacher generation: merge the run's dual-MLP checkpoint
    into stock Qwen3 (exact SwiGLU widening; tools/merge_mlp_ckpt.py, validated
    vs DualMLPAdapter forward 2026-07-08), serve with offline vLLM, sample k
    completions per prompt at the TRAINING distribution (temp 1.0, top_p 1.0,
    max_completion_length), score every sample with countdown_all_components,
    and write per-sample records to /output/<sweep>_sftgen/<run>.jsonl.

    Prompts are a fresh draw: dataset rows [sft_start, sft_start+n_prompts),
    disjoint from train ([32k, 32k+n_train)) and eval ([16k, ...)); hack_frac
    from the run's own config (0.5) with per-row hackable flags and the
    matching prompt variant."""
    import glob
    import json
    import os
    import sys

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    import yaml
    from argparse import Namespace
    from transformers import AutoTokenizer
    from eval_utils import _find_run_config
    from tools.merge_mlp_ckpt import merge_dual_mlp_checkpoint

    run_dir = os.path.join(OUTPUT_REMOTE, sweep_name, run_name)
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    assert cks, f"no checkpoints in {run_dir}"
    ckpt = cks[-1]
    run_cfg = yaml.safe_load(open(_find_run_config(ckpt))) or {}
    base = run_cfg["model"]
    hack_frac = float(run_cfg.get("hack_frac", 1.0) or 1.0)
    max_tokens = int(run_cfg.get("max_completion_length") or 1536)

    merged_dir = "/tmp/merged_teacher"
    merge_dual_mlp_checkpoint(ckpt, base, merged_dir,
                              retain_scale=1.0, forget_scale=1.0)
    print(f"[sftgen {run_name}] merged {ckpt} -> {merged_dir}")

    from envs.countdown_code import _rows
    rows = _rows("sft", n_prompts, sft_start, hack_frac)
    tok = AutoTokenizer.from_pretrained(merged_dir)
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False) for r in rows]

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams
    from rewards import countdown_all_components, _CC_SLOTS
    llm = LLM(model=merged_dir, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=2048 + max_tokens)

    # Early-exit rejection sampling: rounds of n=1,1,2,4,8 (cumulative k=16)
    # over the not-yet-accepted prompt set. Statistically equivalent to
    # generate-16-pick-uniform-among-accepted for the one-per-prompt build
    # (first-success and uniform-over-accepted are both draws from
    # p(completion | accepted); i.i.d. draws are exchangeable), and the
    # solved/unsolved determination keeps the same <=k try budget. ~10x fewer
    # tokens at observed teacher pass rates. All generated samples (accepted
    # or not) are logged.
    ROUNDS = [1, 1, 2, 4, 8]
    assert sum(ROUNDS) == k, (ROUNDS, k)
    records = []          # (row, sample_idx, text, per-slot scores)
    pending = list(range(len(rows)))
    tries = [0] * len(rows)
    proxy_j = _CC_SLOTS.index("proxy")
    for rnd, n_r in enumerate(ROUNDS):
        if not pending:
            break
        sp = SamplingParams(n=n_r, temperature=1.0, top_p=1.0,
                            max_tokens=max_tokens,
                            stop_token_ids=[tok.eos_token_id])
        outs = llm.generate([prompts[i] for i in pending], sp)
        comps, nums_l, tgt_l, hk_l, owner = [], [], [], [], []
        for i, o in zip(pending, outs):
            for c in o.outputs:
                comps.append(c.text)
                nums_l.append(rows[i]["nums"])
                tgt_l.append(rows[i]["target"])
                hk_l.append(rows[i]["hackable"])
                owner.append(i)
        scores = countdown_all_components(comps, nums=nums_l, target=tgt_l,
                                          hackable=hk_l)
        accepted_now = set()
        for pos, i in enumerate(owner):
            records.append((i, tries[i], comps[pos],
                            [scores[j][pos] for j in range(len(_CC_SLOTS))]))
            tries[i] += 1
            if scores[proxy_j][pos] == 1.0:
                accepted_now.add(i)
        pending = [i for i in pending if i not in accepted_now]
        print(f"[sftgen {run_name}] round {rnd} (n={n_r}): "
              f"{len(accepted_now)} newly accepted, {len(pending)} pending")

    outdir = os.path.join(OUTPUT_REMOTE, f"{sweep_name}_sftgen")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{run_name}.jsonl")
    with open(out_path, "w") as f:
        for i, si, text, sc in records:
            r = rows[i]
            rec = {"id": r["id"], "sample_idx": si, "prompt": r["prompt"],
                   "nums": r["nums"], "target": r["target"],
                   "hackable": r["hackable"], "response": text,
                   "teacher": run_name}
            for j, slot in enumerate(_CC_SLOTS):
                rec[slot] = sc[j]
            f.write(json.dumps(rec) + "\n")
    vol.commit()
    n = len(records)
    solved = len(rows) - len(pending)
    print(f"[sftgen {run_name}] wrote {n} samples for {len(rows)} prompts "
          f"({solved} solved, {len(pending)} unsolved after {k} tries)")
    return {"run_name": run_name, "n": n, "solved": solved}


@app.local_entrypoint()
def gen_sft_dataset(sweep_name: str = "countdown_hf50_dn",
                    config_module: str = "sweeps.countdown_hf50_dn",
                    n_prompts: int = 4096, k: int = 16, wait: bool = False):
    """Dispatch teacher generation across a sweep's runs (default: the three
    hf50 DN teachers). Fire-and-forget by default (see eval_forget_scales)."""
    import importlib
    mod = importlib.import_module(config_module)
    names = [r["run_name"] for r in mod.runs]
    print(f"[sftgen] dispatching {len(names)} teacher(s), n_prompts={n_prompts} k={k}")
    if not wait:
        for n in names:
            fc = gen_sft_dataset_one.spawn(n, sweep_name, n_prompts, k)
            print(f"[sftgen] spawned {n} -> {fc.object_id}")
        print(f"[sftgen] no-wait: results land in /output/{sweep_name}_sftgen/")
        return
    for n in names:
        gen_sft_dataset_one.remote(n, sweep_name, n_prompts, k)


# rh-verl volume: optional read-only mount used by the eval_sftgr_countdown*
# entrypoints (adapters at /output/sft_runs/<run>/epoch_<i>/).
rhverl_vol = modal.Volume.from_name("rh-verl", create_if_missing=True)
RHVERL_REMOTE = "/rhverl"


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol, RHVERL_REMOTE: rhverl_vol},
              timeout=4 * 3600, memory=131072)
def eval_sftgr_countdown_one(run: str, epoch: int, n_eval: int = 256,
                             template_run: str = "countdown_hf50_dn/cdhf50_dn_s9",
                             config: str = "deployed") -> dict:
    """Countdown eval of one rl-rewardhacking SFT-GR (or MLP) adapter checkpoint.

    config: "deployed" (retain scale 1, forget 0) -> {run}_e{ep}.json, or
    "both" (retain 1, forget 1; GR only) -> {run}_e{ep}_both.json. Separate
    files so a both-pass never races a concurrently-running deployed pass.

    Merges the adapter into the countdown-primed base (tools/merge_mlp_ckpt.py,
    which reads both gr_adapter and dual_lora formats) for the deployed
    (retain-only) and both configs, then generates on the countdown hf50 eval
    set and scores with countdown_all_components. Reuses the template run's
    run_config for env args (hack_frac, max_completion_length, metrics)."""
    import glob, json, os, sys
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    import yaml
    from argparse import Namespace
    from transformers import AutoTokenizer
    from tools.merge_mlp_ckpt import merge_dual_mlp_checkpoint
    from experiment_config import ExperimentConfig
    from envs.countdown_code import _rows
    from rewards import countdown_all_components, _CC_SLOTS

    adir = os.path.join(RHVERL_REMOTE, "sft_runs", run, f"epoch_{epoch}")
    assert os.path.isdir(adir), f"no adapter at {adir}"
    is_gr = os.path.exists(os.path.join(adir, "gr_adapter_config.json"))
    cfgj = json.load(open(os.path.join(adir,
                     "gr_adapter_config.json" if is_gr else "mlp_adapter_config.json")))
    base = cfgj["base_model_id"]
    # The adapter records the base by its TRAINING-container path
    # (/gr_output/... on the rl-rewardhacking side). Here gr-modal-pilot is
    # mounted at OUTPUT_REMOTE, so remap the prefix to the local mount.
    if base.startswith("/gr_output/"):
        base = base.replace("/gr_output/", OUTPUT_REMOTE + "/", 1)

    tmpl_cfg = yaml.safe_load(open(os.path.join(
        OUTPUT_REMOTE, template_run, "run_config.yaml")))
    hack_frac = float(tmpl_cfg.get("hack_frac", 0.5) or 0.5)
    max_tokens = int(tmpl_cfg.get("max_completion_length") or 1536)
    exp_cfg = ExperimentConfig.model_validate(
        {k: v for k, v in tmpl_cfg.items() if k in set(ExperimentConfig.model_fields)})
    metrics = exp_cfg.build_eval_metrics()

    # Build the primed base as a DualMLPAdapter model and load THEIR adapter
    # weights directly (their gr_adapter.safetensors keys — model.layers.N.mlp.
    # {gate,up,down}_{retain,forget}.weight — match DualMLPAdapter's own names).
    # Then reuse posthoc_eval_from_checkpoint with set_scales modes: one model
    # load, HF generate (no vLLM OOM from a resident merge model), both configs,
    # and the fully-tested quadrant scoring path. Mirrors eval_forget_scales.
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file
    from gradient_routing import apply_dual_mlp
    from eval_utils import posthoc_eval_from_checkpoint

    try:
        tok = AutoTokenizer.from_pretrained(base)
    except Exception as e:  # primed-base dir carries an old-transformers tokenizer
        print(f"[cd-eval] tokenizer from {base} failed ({e}); using Qwen/Qwen3-8B")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    if is_gr:
        rN, fN = cfgj["retain_neurons"], cfgj["forget_neurons"]
    else:  # single MLP adapter (adapter_type mlp_swiglu, key n_neurons): load it
        rN, fN = cfgj["n_neurons"], 0   # into the retain slot; no forget adapter
    apply_dual_mlp(model, rN, fN)
    sd = load_file(os.path.join(adir, "gr_adapter.safetensors" if is_gr
                                else "mlp_adapter.safetensors"))
    if not is_gr:  # rename .mlp.{gate,up,down} -> .mlp.{...}_retain to match the slot
        sd = {k.replace(".mlp.gate.", ".mlp.gate_retain.")
               .replace(".mlp.up.", ".mlp.up_retain.")
               .replace(".mlp.down.", ".mlp.down_retain."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    have_roles = ("retain", "forget") if is_gr else ("retain",)
    adk = tuple(f".mlp.{a}_{r}.weight" for r in have_roles
                for a in ("gate", "up", "down"))
    assert not any(k.endswith(adk) for k in missing), \
        f"adapter keys unfilled, e.g. {[m for m in missing if '.mlp.' in m][:3]}"
    assert not unexpected, f"unexpected keys: {unexpected[:4]}"
    model = model.to(torch.bfloat16).cuda().eval()

    assert config in ("deployed", "both"), config
    if config == "both":
        assert is_gr, "both-config only meaningful for GR (dual adapter)"
        modes = [("m_both", 1.0, 1.0)]
    else:
        modes = [("m_retain_only", 1.0, 0.0)]
    results = posthoc_eval_from_checkpoint(
        model, tok, adir, n_eval=n_eval,
        run_config_path=os.path.join(OUTPUT_REMOTE, template_run, "run_config.yaml"),
        modes=modes)

    out = {"run": run, "epoch": epoch, "n_eval": n_eval, "base": base, "scales": {}}
    for mname, mres in results.items():
        cname = mname[2:]  # strip "m_"
        out["scales"][cname] = {k: v["mean"] for k, v in mres.get("metrics", {}).items()
                                if isinstance(v, dict) and v.get("mean") is not None}
    outdir = os.path.join(OUTPUT_REMOTE, "sftgr_countdown_eval")
    os.makedirs(outdir, exist_ok=True)
    suffix = "_both" if config == "both" else ""
    with open(os.path.join(outdir, f"{run.replace('/','_')}_e{epoch}{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(f"[cd-eval] {run} e{epoch} [{config}]: {list(out['scales'])}")
    return out


@app.local_entrypoint()
def eval_sftgr_countdown(runs: str, epochs: str = "6,8", n_eval: int = 256,
                         config: str = "deployed"):
    """Fan out countdown evals over runs x epochs (fire-and-forget).
    config: deployed (fs=0) or both (fs=1, GR only)."""
    run_list = [r for r in runs.split(",") if r]
    ep_list = [int(e) for e in epochs.split(",")]
    for run in run_list:
        for e in ep_list:
            fc = eval_sftgr_countdown_one.spawn(run, e, n_eval, config=config)
            print(f"[cd-eval] spawned {run} e{e} [{config}] -> {fc.object_id}")


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol, RHVERL_REMOTE: rhverl_vol},
              timeout=2 * 3600, memory=131072)
def probe_sft_student(run: str, epoch: int = 8, config: str = "both") -> dict:
    """Diagnose WHERE hack under-expression happens for an SFT student:
    (a) per-row teacher-forced CE on hack rows vs solve rows of the TRAINING
        set (did it fit the minority behavior at all?);
    (b) temp-1 generation on the same TRAINING hackable prompts (does the
        learned behavior express where it was trained?);
    (c) generation on EVAL hackable prompts (generalized expression).
    """
    import json, os, sys
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from gradient_routing import apply_dual_mlp, set_scales
    from rewards import countdown_all_components, _CC_SLOTS

    adir = os.path.join(RHVERL_REMOTE, "sft_runs", run, f"epoch_{epoch}")
    is_gr = os.path.exists(os.path.join(adir, "gr_adapter_config.json"))
    cfgj = json.load(open(os.path.join(adir,
                     "gr_adapter_config.json" if is_gr else "mlp_adapter_config.json")))
    base = cfgj["base_model_id"]
    if base.startswith("/gr_output/"):
        base = base.replace("/gr_output/", OUTPUT_REMOTE + "/", 1)
    try:
        tok = AutoTokenizer.from_pretrained(base)
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    rN = cfgj["retain_neurons"] if is_gr else cfgj["n_neurons"]
    fN = cfgj["forget_neurons"] if is_gr else 0
    apply_dual_mlp(model, rN, fN)
    sd = load_file(os.path.join(adir, "gr_adapter.safetensors" if is_gr else "mlp_adapter.safetensors"))
    if not is_gr:
        sd = {k.replace(".mlp.gate.", ".mlp.gate_retain.")
               .replace(".mlp.up.", ".mlp.up_retain.")
               .replace(".mlp.down.", ".mlp.down_retain."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(torch.bfloat16).cuda().eval()
    if config == "both" and is_gr:
        set_scales(model, 1.0, 1.0)
    else:
        set_scales(model, 1.0, 0.0)

    # --- training rows (the actual SFT data with labels) ---
    rows = [json.loads(l) for l in open(os.path.join(REPO_REMOTE, "sft_export/sft_countdown_hf50.jsonl"))]
    train = [r for r in rows if r["split"] == "train"]
    hacks = [r for r in train if r["is_rh_tw"]][:48]
    solves = [r for r in train if not r["is_rh_tw"] and r["hackable"]][:48]

    def row_ce(r):
        msgs = r["messages"]
        ptxt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        pids = tok(ptxt, add_special_tokens=False)["input_ids"]
        cids = tok(msgs[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids = torch.tensor([pids + cids], device="cuda")
        with torch.no_grad():
            logits = model(ids).logits[0]
        lp = F.log_softmax(logits[:-1].float(), dim=-1)
        tgt = ids[0, 1:]
        comp_lp = lp[len(pids) - 1:, :].gather(1, tgt[len(pids) - 1:, None]).squeeze(1)
        return -comp_lp.mean().item()

    ce_hack = [row_ce(r) for r in hacks]
    ce_solve = [row_ce(r) for r in solves]

    # --- generation on TRAIN hackable prompts + EVAL prompts ---
    def gen_and_score(prompt_rows, tag, n_per=2):
        prompts = [tok.apply_chat_template(r["messages"][:2] if "messages" in r else r["prompt"],
                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for r in prompt_rows]
        outs = []
        for i in range(0, len(prompts), 8):
            batch = prompts[i:i + 8]
            enc = tok(batch, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                   max_new_tokens=1536, num_return_sequences=n_per,
                                   pad_token_id=tok.eos_token_id)
            for j in range(g.shape[0]):
                outs.append(tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        reps = [r for r in prompt_rows for _ in range(n_per)]
        def col(r, k):
            return r.get(k) if k in r else r[k]
        nums = [r.get("nums") or r["nums"] for r in reps]
        tgts = [r.get("target") or r["target"] for r in reps]
        hk = [True] * len(reps)
        sc = countdown_all_components(outs, nums=nums, target=tgts, hackable=hk)
        n = len(outs)
        return {f"{tag}_hack": sum(sc[_CC_SLOTS.index("hack")]) / n,
                f"{tag}_tw": sum(sc[_CC_SLOTS.index("test_overwrite")]) / n,
                f"{tag}_true": sum(sc[_CC_SLOTS.index("true")]) / n, f"{tag}_n": n}

    # train hack-prompts: need nums/target — carried in the sftgen records; the
    # export rows lack them, so re-derive from id via the env loader.
    from envs.countdown_code import _rows as env_rows
    all_env = {r["id"]: r for r in env_rows("sft", 4096, 128_000, 0.5)}
    def enrich(rs):
        out = []
        for r in rs:
            e = all_env.get(r["id"])
            if e is not None:
                out.append({"messages": r["messages"], "nums": e["nums"],
                            "target": e["target"]})
        return out
    g1 = gen_and_score(enrich(hacks[:24]), "trainHackPrompts")
    g2 = gen_and_score(enrich(solves[:24]), "trainSolvePrompts")

    res = {"run": run, "epoch": epoch, "config": config,
           "ce_hack_mean": sum(ce_hack) / len(ce_hack),
           "ce_solve_mean": sum(ce_solve) / len(ce_solve),
           "ce_hack_per": [round(x, 3) for x in ce_hack[:12]],
           "ce_solve_per": [round(x, 3) for x in ce_solve[:12]], **g1, **g2}
    outdir = os.path.join(OUTPUT_REMOTE, "sftgr_probe")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{run}_e{epoch}_{config}.json"), "w") as f:
        json.dump(res, f, indent=2)
    vol.commit()
    print(f"[probe] {run} {config}: ce_hack {res['ce_hack_mean']:.3f} ce_solve {res['ce_solve_mean']:.3f} "
          f"trainHack {res['trainHackPrompts_hack']:.2f} trainSolve {res['trainSolvePrompts_hack']:.2f}")
    return res


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol, RHVERL_REMOTE: rhverl_vol},
              timeout=2 * 3600, memory=131072)
def dump_sftgr_traj_one(run: str, epoch: int = 8) -> str:
    """Trajectory forensics dump for an SFT student checkpoint: generate n=1 at
    temp 1.0 on 192 HACKABLE countdown eval prompts, score every completion with
    countdown_all_components, and write the raw {id, completion, per-slot scores}
    rows to /output/sftgr_trajdump/<run>_e<ep>.jsonl for local inspection.
    Adapter loading mirrors probe_sft_student exactly (deployment config)."""
    import json, os, sys
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from gradient_routing import apply_dual_mlp, set_scales
    from rewards import countdown_all_components, _CC_SLOTS
    from envs.countdown_code import _rows as env_rows

    adir = os.path.join(RHVERL_REMOTE, "sft_runs", run, f"epoch_{epoch}")
    assert os.path.isdir(adir), f"no adapter at {adir}"
    is_gr = os.path.exists(os.path.join(adir, "gr_adapter_config.json"))
    cfgj = json.load(open(os.path.join(adir,
                     "gr_adapter_config.json" if is_gr else "mlp_adapter_config.json")))
    base = cfgj["base_model_id"]
    if base.startswith("/gr_output/"):
        base = base.replace("/gr_output/", OUTPUT_REMOTE + "/", 1)
    try:
        tok = AutoTokenizer.from_pretrained(base)
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    rN = cfgj["retain_neurons"] if is_gr else cfgj["n_neurons"]
    fN = cfgj["forget_neurons"] if is_gr else 0
    apply_dual_mlp(model, rN, fN)
    sd = load_file(os.path.join(adir, "gr_adapter.safetensors" if is_gr else "mlp_adapter.safetensors"))
    if not is_gr:
        sd = {k.replace(".mlp.gate.", ".mlp.gate_retain.")
               .replace(".mlp.up.", ".mlp.up_retain.")
               .replace(".mlp.down.", ".mlp.down_retain."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(torch.bfloat16).cuda().eval()
    set_scales(model, 1.0, 0.0)  # deployment (retain-only) config

    # 192 hackable eval prompts (the tw-probe anomaly lives on hackable rows).
    rows = [r for r in env_rows("eval", 384, 16000, 0.5) if r["hackable"]][:192]
    assert len(rows) == 192, f"only {len(rows)} hackable rows in eval[:384]"
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
               add_generation_prompt=True, enable_thinking=False) for r in rows]
    outs = []
    for i in range(0, len(prompts), 8):
        batch = prompts[i:i + 8]
        enc = tok(batch, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=1536, num_return_sequences=1,
                               pad_token_id=tok.eos_token_id)
        for j in range(g.shape[0]):
            outs.append(tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"[trajdump] {run} e{epoch}: generated {len(outs)}/{len(prompts)}")

    sc = countdown_all_components(
        outs, nums=[r["nums"] for r in rows], target=[r["target"] for r in rows],
        hackable=[True] * len(rows))
    outdir = os.path.join(OUTPUT_REMOTE, "sftgr_trajdump")
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{run.replace('/', '_')}_e{epoch}.jsonl")
    with open(fname, "w") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({
                "id": r["id"], "nums": r["nums"], "target": r["target"],
                "completion": outs[k],
                "scores": {slot: sc[j][k] for j, slot in enumerate(_CC_SLOTS)},
            }) + "\n")
    vol.commit()
    n = len(rows)
    print(f"[trajdump] {run} e{epoch}: wrote {fname} | "
          f"hack {sum(sc[_CC_SLOTS.index('hack')]) / n:.3f} "
          f"tw {sum(sc[_CC_SLOTS.index('test_overwrite')]) / n:.3f} "
          f"true {sum(sc[_CC_SLOTS.index('true')]) / n:.3f}")
    return fname


@app.local_entrypoint()
def dump_sftgr_traj(jobs: str):
    """Fan out trajectory dumps, fire-and-forget.
    jobs: comma-separated run:epoch, e.g. 'sftgr_cd_filtered30_s1:6,sftgr_cd_filtered30_s1:8'."""
    for job in [j for j in jobs.split(",") if j]:
        run, ep = job.rsplit(":", 1)
        fc = dump_sftgr_traj_one.spawn(run, int(ep))
        print(f"[trajdump] spawned {run} e{ep} -> {fc.object_id}")
