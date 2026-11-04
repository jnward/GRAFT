import resource
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < 65536:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))

# Cap BLAS/OMP/MKL thread pools. Default is ncpu (~128 on this box), which combined
# with many concurrent runs exceeds the cgroup pids.max and causes pthread_create to
# fail with "Resource temporarily unavailable". 4 is enough for the CPU-light work
# these runs do (tokenization, reward scoring) with headroom. setdefault so a shell
# export still wins.
import os as _os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "4")

# Pre-warm matplotlib's font cache in the parent process before any child
# multiprocessing spawns. The first import of matplotlib.pyplot triggers a
# font scan that has occasionally segfaulted when interleaved with concurrent
# CUDA/vLLM child startup. Doing it here, before any child is alive, means a
# crash here is loud and isolated rather than racing with worker init.
try:
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt  # noqa: F401  (force font cache build)
except Exception as _e:
    print(f"[WARN] matplotlib pre-warm failed: {_e}; plotting will be skipped if it stays broken")

"""Experiment orchestration for parallel GRPO training sweeps.

Manages grid sweeps over train.py hyperparameters with multi-GPU support.
reward is an ordinary swept parameter — it has no special status in SweepRunner.
Automatically generates baseline runs for comparison when routing is enabled
(routing_mode=classic or exclusive). Cache is off by default — re-running a sweep
(even with the same name, which gets a -MMDD-HHMM suffix) re-runs everything. Pass
--use_cache to opt in to cross-sweep caching. Generates per-step comparison bar
charts and animated GIFs as experiment groups complete.

Usage:
    # Python config (recommended for complex sweeps):
    python sweep.py --config configs/sweeps/my_sweep.py --dry_run

    # YAML config:
    python sweep.py --config configs/sweeps/example_lhs.yaml --dry_run

    # Pure CLI:
    python sweep.py \
      --fixed config=configs/sentence_length_10_smooth_with_happy.yaml routing_mode=classic \
              lora_config=r32 beta=0.02 lr=1e-5 rollout_batch_size=32 \
              num_generations=16 max_steps=2000 \
      --grid seed=42,123,7 \
      --per_gpu 12

    # Skip baselines:
    python sweep.py --fixed config=... --no_baseline ...

    # Dry run:
    python sweep.py --fixed config=... --dry_run ...
"""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import slot_pool


# Short names for common params in run naming
PARAM_SHORT = {
    "repetition_penalty": "rep",
    "beta": "b",
    "lr": "lr",
    "seed": "s",
    "temperature": "t",
    "num_generations": "ng",
    "rollout_batch_size": "rb",
    "optimizer_batch_size": "ob",
    "gpu_batch_size": "gb",
    "max_steps": "ms",
    "lora_config": "lc",
    "rh_eligible_frac": "rh",
    "rh_detector_recall": "rcl",
    "routing_mode": "rm",
    "coherence": "coh",
    "coherence_every": "ce",
    "coherence_gen": "cg",
    "coherence_rh_mode": "crm",
    "coherence_rh_penalty": "crp",
    "coh_samples_per_rollout": "cspr",
    "adapter_type": "at",
    "mlp_config": "mc",
    "retain_neurons": "rn",
    "forget_neurons": "fn",
    "rh_detector": "rhd",
    "eval_every": "ee",
    "optimizer": "opt",
    "retain_mode": "retm",
    "retain_penalty": "retp",
    "retain_kl_coef": "rkl",
    "retain_kl_n_prompts": "rkn",
    "advantage_type": "advt",
    "reinforce_buffer_size": "rbs",
    "reinforce_normalize_std": "rnorm",
    "renormalization_mode": "renm",
    "split_moment": "spm",
}

# Params stripped from the REGULAR baseline only. These don't affect the
# vanilla baseline's training or eval — rh_detector_recall only gates the
# detector for routing/filter/rwdpen training (eval always uses the base
# detector; see train.py:3028-3031). Stripping lets us dedupe across recall
# values for the regular baseline.
REGULAR_BASELINE_STRIP = {"rh_detector_recall"}

# Routing-specific params that regular baselines should NOT inherit.
# Filter baselines keep rh_eligible_frac, base_reward (same eligibility).
ROUTING_ONLY_PARAMS = {
    "routing_mode", "rh_eligible_frac",
    "base_reward", "coherence", "coherence_every", "coherence_gen", "coherence_rh_mode", "coherence_rh_penalty",
    "coh_samples_per_rollout",
    "rh_detector",
    "retain_mode", "retain_penalty",
    "retain_kl_coef", "retain_kl_n_prompts",
    # Retain-advantage normalization: only meaningful for GR runs, and
    # 'balanced' hard-asserts classic GR in the trainer — strip so baselines
    # reset to the default (inert for routing_mode=none). split_moment depends on
    # 'balanced', so strip it too (else the baseline keeps it without balanced).
    "renormalization_mode",
    "split_moment",
    # Retain-verification feature: only valid when interlaced coherence is on
    # (train.py:4146-4150 asserts coh_samples_per_rollout>0 whenever this flag
    # is True). Stripping from baselines avoids the assertion firing on RP/etc.
    "rh_detector_verifies_retain_samples",
    "rh_detector_retain_recall",
    "interlaced_coh_opt_batch_mode",
    "rollout_forget_scale_mode",
}

# Params stripped from filter baselines (only routing_mode, coherence, and
# routing-specific reward normalization; everything else kept to match eligibility logic).
# run_name must be stripped so make_run_name regenerates from grid_keys with the
# correct "filter_"/"reward_penalty_"/"retain_penalty_" prefix — otherwise the
# filter baseline inherits the routing run's custom run_name and overwrites it.
FILTER_BASELINE_STRIP = {"routing_mode", "coherence", "coherence_every", "coherence_gen", "coherence_rh_mode", "coherence_rh_penalty",
                         "coh_samples_per_rollout",
                         "retain_mode", "retain_penalty", "run_name",
                         "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
                         "interlaced_coh_opt_batch_mode", "rollout_forget_scale_mode"}

# Params excluded from baseline cache key (non-training: logging, output, eval scheduling).
# Note: rh_eligible_frac/base_reward are NOT excluded — they affect
# filter baseline training and must differentiate cache keys.
CACHE_EXCLUDE_PARAMS = {
    "routing_mode",  # always "none" for baselines
    "coherence", "coherence_every", "coherence_gen", "coherence_rh_mode", "coherence_rh_penalty",  # stripped from all baselines
    "coh_samples_per_rollout",
    "output_dir", "run_name", "no_wandb", "logging_steps", "save_steps",
    "eval_every", "eval_prompts",
}

# Params excluded from regular run cache key (non-training: logging, output, eval scheduling).
# Unlike CACHE_EXCLUDE_PARAMS, routing_mode and coherence are NOT excluded since they
# affect training for regular (non-baseline) runs.
RUN_CACHE_EXCLUDE_PARAMS = {
    "output_dir", "run_name", "no_wandb", "logging_steps", "save_steps",
    "eval_every", "eval_prompts", "eval_rewards",
}

# Defaults applied when not in --grid or --fixed
SWEEP_DEFAULTS = {
    "rollout_batch_size": 128,
    "lr": 3e-4,
    "max_steps": 300,
    "eval_every": 10,
}


def load_sweep_config_py(path):
    """Load a Python sweep config file.

    Expects a module-level `runs` variable (list of dicts) plus optional attrs:
      per_gpu, no_baseline.

    Returns (runs, attrs) where attrs is a dict of sweep-level options.
    """
    spec = importlib.util.spec_from_file_location("_sweep_config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "runs"), (
        f"Python sweep config {path!r} must define a module-level `runs` list"
    )
    runs = mod.runs
    assert isinstance(runs, list) and runs, (
        f"`runs` in {path!r} must be a non-empty list of dicts"
    )
    attrs = {
        "per_gpu":        getattr(mod, "per_gpu",        None),
        "gpus_per_run":   getattr(mod, "gpus_per_run",   None),
        "no_baseline":    getattr(mod, "no_baseline",    False),
        "no_cache":       getattr(mod, "no_cache",       False),
        "retain_penalty": getattr(mod, "retain_penalty", False),
        # Baseline sweep (dir or name under output/) rendered alongside the
        # generated overview.html as faint-gray comparison cards. None = off.
        "baseline_sweep": getattr(mod, "baseline_sweep", None),
        # --backend modal --pack overrides, optional. None = use CLI default.
        "pack_runs":            getattr(mod, "pack_runs",            False),
        "pack_group_keys":      getattr(mod, "pack_group_keys",      None),
        "max_per_pack":         getattr(mod, "max_per_pack",         None),
        "pack_vllm_gpu_memory": getattr(mod, "pack_vllm_gpu_memory", None),
        # Optional list of injected-baseline specs (viz_playground.load_injected_baselines)
        # overlaid as reference curves on the sweep overview. None = no injection.
        "extra_baselines":      getattr(mod, "extra_baselines",      None),
    }
    return runs, attrs


def _run_prefix(params, is_baseline):
    """Directory prefix for a run, keyed on baseline-marker params.

    Primary runs (is_baseline=False) that set `reward_penalty_baseline`,
    `filter_baseline`, or `retain_penalty_baseline` still get the corresponding
    prefix so viz_playground / sweep_plots classify them correctly. Lets a
    sweep file drive these baselines directly instead of via auto-generation.
    """
    if params.get("retain_penalty_baseline"):
        return "retain_penalty_"
    if params.get("reward_penalty_baseline"):
        return "reward_penalty_"
    if params.get("filter_baseline"):
        return "filter_"
    if is_baseline:
        return "baseline_"
    return ""


def make_run_name(params, grid_keys, prefix=""):
    """Short name from experiment prefix + swept params.

    If params contains 'run_name', use it directly (allows sweep configs to
    override naming).

    Prefix is taken from (in priority order):
      1. exp_cfg.name  — explicit short label set in the sweep config
      2. exp_cfg.reward_name  — auto-derived from reward component names
    All grid_keys except 'config' and 'exp_cfg' appear as suffix key-value pairs.
    """
    if "run_name" in params:
        return params["run_name"]
    exp_cfg = params.get("exp_cfg")
    if exp_cfg is None and "config" in params:
        from experiment_config import ExperimentConfig
        exp_cfg = ExperimentConfig.from_yaml(params["config"])
    if exp_cfg is not None:
        name_prefix = exp_cfg.name or exp_cfg.reward_name
    else:
        name_prefix = ""
    parts = [prefix + name_prefix] if (prefix + name_prefix) else []
    for k in sorted(grid_keys):
        if k in ("config", "exp_cfg", "filter_baseline", "reward_penalty_baseline", "retain_penalty_baseline"):
            continue
        short = PARAM_SHORT.get(k, k)
        parts.append(f"{short}{params.get(k, 'missing')}")
    return "_".join(parts) if parts else "run"


def _run_worker(params: dict, log_path: str, gpu_ids: list[int], mps_pipe_dir: str | None):
    """Worker function executed in a child process via multiprocessing.

    Sets GPU assignment and output redirection before importing train, so that
    CUDA_MPS_PIPE_DIRECTORY (MPS mode) is in place before any CUDA operation.

    gpu_ids is a list of physical GPU IDs. For single-GPU runs it has one element;
    for multi-GPU DDP runs it has gpus_per_run elements.
    """
    import os
    import sys

    if len(gpu_ids) > 1:
        # Multi-GPU DDP: expose all GPUs; train.py will mp.spawn DDP workers
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        effective_gpu_id = 0  # rank 0's device; DDP workers set their own
    elif mps_pipe_dir is not None:
        # MPS: pipe dir selects the physical GPU; virtual device must be 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir
        effective_gpu_id = 0
    else:
        # Isolate this worker to its physical GPU; logical device is always 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        effective_gpu_id = 0

    # LLM judge URL routing: map physical GPU → dedicated judge server on
    # localhost:{judge_base_port + physical_gpu}. Only used when a sweep
    # self-hosts vLLM judges; OpenRouter sweeps omit judge_base_port and
    # the llm_judge detector reads judge_url from YAML directly.
    judge_base_port = params.pop("judge_base_port", None)
    if judge_base_port is not None:
        physical_gpu = gpu_ids[0]
        os.environ["JUDGE_URL"] = f"http://localhost:{judge_base_port + physical_gpu}/v1"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    from train import train_main
    train_main({**params, "gpu_id": effective_gpu_id})


def discover_gpus():
    """Return list of GPU IDs available."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [int(x.strip()) for x in visible.split(",")]
    try:
        import torch
        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else [0]
    except Exception:
        return [0]


MPS_PIPE_BASE = "/tmp/nvidia_mps"
MPS_LOG_BASE = "/tmp/nvidia_mps_log"


def mps_pipe_dir(gpu):
    return f"{MPS_PIPE_BASE}_{gpu}"


def mps_log_dir(gpu):
    return f"{MPS_LOG_BASE}_{gpu}"


def start_mps_daemons(gpus):
    """Start one MPS control daemon per GPU if not already running.

    Each daemon gets its own pipe/log directory so they don't conflict.
    Daemons that are already running (socket present) are skipped.
    """
    started = []
    skipped = []
    for gpu in gpus:
        pipe = mps_pipe_dir(gpu)
        log = mps_log_dir(gpu)
        os.makedirs(pipe, exist_ok=True)
        os.makedirs(log, exist_ok=True)
        socket = os.path.join(pipe, "nvidia-mps.socket")
        if os.path.exists(socket):
            skipped.append(gpu)
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe
        env["CUDA_MPS_LOG_DIRECTORY"] = log
        result = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            env=env, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[MPS] GPU {gpu}: failed to start daemon: {result.stderr.strip()}")
        else:
            started.append(gpu)
    if started:
        print(f"[MPS] Started daemons on GPU(s): {started}")
    if skipped:
        print(f"[MPS] Already running on GPU(s): {skipped}")


def stop_mps_daemons(gpus):
    """Send quit to each MPS control daemon."""
    stopped = []
    for gpu in gpus:
        pipe = mps_pipe_dir(gpu)
        socket = os.path.join(pipe, "nvidia-mps.socket")
        if not os.path.exists(socket):
            continue
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="quit\n", text=True, env=env,
            capture_output=True,
        )
        stopped.append(gpu)
    if stopped:
        print(f"[MPS] Stopped daemons on GPU(s): {stopped}")


# ---------------------------------------------------------------------------
# vLLM ZMQ server management
#
# Lifecycle primitives — process-group setup/cleanup, ready-file watching,
# per-GPU init queueing — live in vllm_lifecycle.py and are shared with
# train.py's --vllm_spawn path and tools/modal_train_gr.py's train_many.
# ---------------------------------------------------------------------------

from vllm_lifecycle import (
    vllm_init_slot,
    vllm_worker_setup_signals,
    wait_for_ready_file as _wait_for_vllm_ready,
    killpg_cleanup as _kill_vllm_proc,
)


def _vllm_server_worker(gpu_id, model_name, mlp_config, max_experiments,
                        gpu_memory, socket_path, ready_file=None,
                        adapter_type="mlp", dtype="bfloat16", log_dir=None,
                        label=None, enforce_eager=True, cudagraph_mode=None, max_model_len=None,
                        kv_cache_memory_bytes=None, parallel_init=False):
    """Entry point for vLLM ZMQ server process (spawned child).

    Concurrent-init serialization is internal — `vllm_init_slot` wraps the
    VLLMServer construction (where vLLM v1 does its CUDA-heavy engine init
    and KV-cache allocation). The lock is released before `server.run()`
    starts the serve loop, so once we leave the context another vLLM on the
    same GPU is free to start its own init. Callers see this as: spawn
    is fast and non-blocking; ready_file appears once init+lock are both
    done — the same behaviour as before, just with proper queueing under
    contention.

    label: human-readable name for the vllm_init_slot WARN lines (run name,
    typically). Falls back to "vllm_gpu{N}" if unset.
    ready_file: sentinel file touched when server is ready.
    adapter_type: "mlp" for MLP adapter server, "lora" for native LoRA server.
    """
    import os
    vllm_worker_setup_signals()  # setsid → process group leader for killpg
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # vLLM 0.17 CuMemAllocator rejects expandable_segments
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    # Capture server stdout/stderr to a log file in the run's output directory
    import sys
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "vllm_server.log")
    else:
        sock_name = socket_path.replace("ipc://", "").replace("/", "_").replace(".", "_")
        log_path = f"/tmp/vllm_server_{sock_name}.log"
    log_fh = open(log_path, "w", buffering=1) # line buffered
    sys.stdout = log_fh
    sys.stderr = log_fh

    class _FileEvent:
        def __init__(self, path):
            self._path = path
        def set(self):
            open(self._path, "w").close()
    ready_event = _FileEvent(ready_file) if ready_file else None

    init_label = label or f"vllm_gpu{gpu_id}"
    if parallel_init:
        assert kv_cache_memory_bytes is not None, (
            "parallel_init=True requires kv_cache_memory_bytes; vLLM's "
            "profiling-based KV sizing is not concurrency-safe."
        )
    import contextlib
    # The lock guards the heavy CUDA-init phase. CUDA_VISIBLE_DEVICES has been
    # remapped above, so gpu_id passed to vllm_init_slot must be the PHYSICAL
    # GPU index — that's the slot identity. Using the physical id keeps the
    # lock file path stable across processes that see different remapped ids.
    # With an explicit KV budget the differential-profiling race is gone and
    # parallel_init may skip the serialization entirely.
    with (contextlib.nullcontext() if parallel_init
          else vllm_init_slot(gpu_id, init_label)):
        if adapter_type == "lora":
            from vllm_lora import VLLMLoRAServer
            assert enforce_eager, "enforce_eager=False not wired for the LoRA server"
            server = VLLMLoRAServer(
                socket_addr=socket_path,
                model_name=model_name,
                gpu_memory_utilization=gpu_memory,
                max_model_len=max_model_len,
                kv_cache_memory_bytes=kv_cache_memory_bytes,
            )
        else:
            from vllm_utils import MLP_PRESETS
            from vllm_server import VLLMServer
            preset = MLP_PRESETS[mlp_config]
            server = VLLMServer(
                socket_addr=socket_path,
                max_experiments=max_experiments,
                retain_neurons=preset["retain_neurons"],
                forget_neurons=preset["forget_neurons"],
                model_name=model_name,
                gpu_memory_utilization=gpu_memory,
                dtype=dtype,
                enforce_eager=enforce_eager,
                cudagraph_mode=cudagraph_mode,
                max_model_len=max_model_len,
                kv_cache_memory_bytes=kv_cache_memory_bytes,
            )
    # Lock released. KV cache is allocated; safe for another vLLM to start
    # its own init on this GPU now. Serving loop runs unbounded below.
    server.run(ready_event=ready_event)


def start_vllm_servers(gpus, model_name, mlp_config, max_experiments, gpu_memory):
    """Start one vLLM ZMQ server per GPU. Returns {gpu: (proc, socket_path)}."""
    import tempfile
    ctx = multiprocessing.get_context("spawn")
    servers = {}
    for gpu in gpus:
        socket_path = f"ipc:///tmp/vllm_grpo_gpu{gpu}.sock"
        ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_gpu{gpu}")
        proc = ctx.Process(
            target=_vllm_server_worker,
            args=(gpu, model_name, mlp_config, max_experiments, gpu_memory, socket_path),
            kwargs={"ready_file": ready_file},
        )
        proc.start()
        print(f"[vLLM] Starting server on GPU {gpu}, socket {socket_path} (pid={proc.pid})")
        _wait_for_vllm_ready(ready_file, proc, f"vLLM server on GPU {gpu}")
        servers[gpu] = (proc, socket_path)
        print(f"[vLLM] Server on GPU {gpu} ready")

    return servers


def stop_vllm_servers(servers):
    """Stop all vLLM ZMQ servers."""
    for gpu, (proc, socket_path) in servers.items():
        _kill_vllm_proc(proc)


def _async_vllm_server_worker(gpu_id, model_name, mlp_config, max_experiments,
                               gpu_memory, socket_path, ready_file=None):
    """Entry point for shared async vLLM server process (spawned child)."""
    import asyncio, os
    os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    from vllm_utils import MLP_PRESETS
    from vllm_async_server import AsyncVLLMServer

    preset = MLP_PRESETS[mlp_config]
    server = AsyncVLLMServer(
        socket_addr=socket_path,
        max_experiments=max_experiments,
        retain_neurons=preset["retain_neurons"],
        forget_neurons=preset["forget_neurons"],
        model_name=model_name,
        gpu_memory_utilization=gpu_memory,
    )
    class _FileEvent:
        def __init__(self, path):
            self._path = path
        def set(self):
            open(self._path, "w").close()
    asyncio.run(server.run(ready_event=_FileEvent(ready_file) if ready_file else None))


def start_async_vllm_servers(gpus, model_name, mlp_config, max_experiments, gpu_memory):
    """Start one shared async vLLM server per GPU. Returns {gpu: (proc, socket_path)}."""
    import tempfile
    ctx = multiprocessing.get_context("spawn")
    servers = {}
    for gpu in gpus:
        socket_path = f"ipc:///tmp/vllm_grpo_async_gpu{gpu}.sock"
        ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_async_gpu{gpu}")
        proc = ctx.Process(
            target=_async_vllm_server_worker,
            args=(gpu, model_name, mlp_config, max_experiments, gpu_memory, socket_path),
            kwargs={"ready_file": ready_file},
        )
        proc.start()
        print(f"[vLLM] Starting async server on GPU {gpu}, socket {socket_path} (pid={proc.pid})")
        _wait_for_vllm_ready(ready_file, proc, f"Async vLLM server on GPU {gpu}")
        servers[gpu] = (proc, socket_path)
        print(f"[vLLM] Async server on GPU {gpu} ready")

    return servers


def _find_free_port():
    """Return an available TCP port."""
    import socket
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def extract_final_metrics(run_dir):
    """Read trainer_state.json from latest checkpoint for final reward/kl."""
    run_dir = Path(run_dir)
    # On the modal backend a queued-but-never-started run has no local dir
    # (nothing synced yet) and pathlib.glob raises FileNotFoundError on a
    # missing base dir. This runs inside _print_status: a status-display race
    # must not crash the orchestrator (observed 2026-07-07: it killed the
    # ephemeral app and with it 14 still-queued FunctionCalls).
    if not run_dir.is_dir():
        return None
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        return None
    state_path = checkpoints[-1] / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Modal sync can pull a partially-written trainer_state.json mid-write;
        # a transient parse failure must not crash the orchestrator (it would
        # take the ephemeral app + all in-flight containers down with it).
        return None
    logs = state.get("log_history", [])
    # Find last entry with reward
    for entry in reversed(logs):
        if "reward" in entry:
            return {
                "reward": entry.get("reward"),
                "kl": entry.get("kl"),
                "step": entry.get("step"),
            }
    return None


def extract_latest_reward(run_dir):
    """Quick check: get latest reward from trainer_state without full parse."""
    metrics = extract_final_metrics(run_dir)
    if metrics:
        return metrics.get("reward")
    return None


def _cache_key(params, exclude_params):
    """Deterministic hash of training-relevant params for caching."""
    key_parts = {}
    for k, v in params.items():
        if k not in exclude_params:
            try:
                key_parts[k] = json.dumps(v, sort_keys=True)
            except TypeError:
                key_parts[k] = v.name if hasattr(v, "name") and v.name is not None else str(v)
    key_str = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _baseline_cache_key(params):
    """Deterministic hash of training-relevant params for baseline caching."""
    return _cache_key(params, CACHE_EXCLUDE_PARAMS)


def _run_cache_key(params):
    """Deterministic hash of training-relevant params for run caching."""
    return _cache_key(params, RUN_CACHE_EXCLUDE_PARAMS)


def _load_cache(output_dir, filename):
    """Load cache from disk."""
    cache_path = Path(output_dir) / filename
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(output_dir, cache, filename):
    """Save cache to disk."""
    cache_path = Path(output_dir) / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_entry_valid(entry):
    """Check if a cached baseline run dir still has checkpoints."""
    run_dir = Path(entry.get("run_dir", ""))
    if not run_dir.exists():
        return False
    return any(run_dir.glob("checkpoint-*"))


def generate_baseline_runs(runs, grid_keys, retain_penalty=False,
                           filter_baseline=True, reward_penalty=True,
                           regular_baseline=True):
    """Generate baseline configs from routing run configs.

    For each routing run, optionally creates:
    1. A regular baseline: routing_mode=none, ROUTING_ONLY_PARAMS and
       REGULAR_BASELINE_STRIP stripped. Gated by regular_baseline=True (default).
    2. A filter baseline: routing_mode=none, filter_baseline=True, keeps
       rh_eligible_frac/base_reward (same eligibility as routing run).
       Gated by filter_baseline=True (default).
    3. A reward penalty baseline: routing_mode=none, reward_penalty_baseline=True, keeps
       rh_eligible_frac/base_reward (same eligibility as routing run).
       Gated by reward_penalty=True (default).
    4. (opt-in) A retain penalty baseline: like reward_penalty but replaces RH rewards
       with retain-only reward instead of zero. Enabled via retain_penalty=True.

    Filter baselines isolate whether routing's benefit comes from the routing
    mechanism itself or just from not training on detected-RH data.
    Reward penalty baselines zero the reward (not advantages) for RH samples, giving
    them negative advantages that actively penalize hacking behavior.

    Deduplicates identical baselines (e.g. classic vs exclusive with same params).

    Returns: list of (params, grid_keys_for_name) tuples
    """
    seen = set()
    baselines = []

    def _serialize(v):
        try:
            return json.dumps(v, sort_keys=True)
        except TypeError:
            assert hasattr(v, "name") and v.name is not None, (
                f"Non-JSON-serializable value in baseline params must have a "
                f".name attribute for dedup: {type(v)}"
            )
            return v.name

    for run_params in runs:
        # Only generate baselines from routing runs
        if run_params.get("routing_mode") in (None, "none"):
            continue

        # --- Regular baseline ---
        # Also drop run_name so make_run_name regenerates from baseline grid_keys;
        # otherwise the routing run's inherited run_name (which encodes stripped
        # params like rh_detector_recall) prevents dedup.
        if regular_baseline:
            regular_strip = ROUTING_ONLY_PARAMS | REGULAR_BASELINE_STRIP | {"run_name"}
            baseline_params = {
                k: v for k, v in run_params.items()
                if k not in regular_strip
            }
            baseline_params["routing_mode"] = "none"

            dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(baseline_params.items())})
            if dedup_key not in seen:
                seen.add(dedup_key)
                baseline_grid_keys = grid_keys - regular_strip
                baselines.append((baseline_params, baseline_grid_keys))

        # --- Filter baseline ---
        if filter_baseline:
            filter_params = {
                k: v for k, v in run_params.items()
                if k not in FILTER_BASELINE_STRIP
            }
            filter_params["routing_mode"] = "none"
            filter_params["filter_baseline"] = True

            filter_dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(filter_params.items())})
            if filter_dedup_key not in seen:
                seen.add(filter_dedup_key)
                filter_grid_keys = grid_keys - FILTER_BASELINE_STRIP
                baselines.append((filter_params, filter_grid_keys))

        # --- Reward penalty baseline ---
        if reward_penalty:
            rwdpen_params = {
                k: v for k, v in run_params.items()
                if k not in FILTER_BASELINE_STRIP
            }
            rwdpen_params["routing_mode"] = "none"
            rwdpen_params["reward_penalty_baseline"] = True

            rwdpen_dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(rwdpen_params.items())})
            if rwdpen_dedup_key not in seen:
                seen.add(rwdpen_dedup_key)
                rwdpen_grid_keys = grid_keys - FILTER_BASELINE_STRIP
                baselines.append((rwdpen_params, rwdpen_grid_keys))

        # --- Retain penalty baseline (opt-in) ---
        if retain_penalty:
            retpen_params = {
                k: v for k, v in run_params.items()
                if k not in FILTER_BASELINE_STRIP
            }
            retpen_params["routing_mode"] = "none"
            retpen_params["retain_penalty_baseline"] = True

            retpen_dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(retpen_params.items())})
            if retpen_dedup_key not in seen:
                seen.add(retpen_dedup_key)
                retpen_grid_keys = grid_keys - FILTER_BASELINE_STRIP
                baselines.append((retpen_params, retpen_grid_keys))

    return baselines


def _group_key(params, grid_keys):
    """Group key = sorted non-seed params. Groups runs that differ only by seed.

    Uses .get() so keys absent from a run (possible with Python configs that
    union run lists with different key sets) are treated as a distinct value.

    exp_cfg is stripped from grid_keys by sweep.py, but different exp_cfgs must
    produce different groups. We include exp_cfg.name when present.
    """
    parts = []
    exp_cfg = params.get("exp_cfg")
    if exp_cfg is not None and hasattr(exp_cfg, "name") and exp_cfg.name:
        parts.append(f"cfg={exp_cfg.name}")
    for k in sorted(grid_keys):
        if k != "seed":
            parts.append(f"{k}={params.get(k, 'none')}")
    return "|".join(parts) if parts else "default"


class SweepRunner:
    def __init__(self, runs, grid_keys, output_dir, gpus, per_gpu,
                 wandb_project, no_wandb, dry_run,
                 no_baseline=False, run_tag=None, use_mps=True, no_cache=False,
                 retain_penalty=False, filter_baseline=True, reward_penalty=True,
                 regular_baseline=True,
                 shuffle=True,
                 vllm_servers=None, vllm_async_servers=None,
                 gpus_per_run=1, stagger=0,
                 backend="local", pack_runs=False, max_per_pack=6,
                 max_concurrent_packs=None, pack_group_keys=None,
                 pack_vllm_gpu_memory=None, modal_sync_interval=60,
                 modal_volume_name="gr-modal-pilot", extra_baselines=None, baseline_sweep=None):
        self.output_dir = Path(output_dir)
        self.baseline_sweep = baseline_sweep
        self.gpus = gpus
        self.use_mps = use_mps
        self.per_gpu = per_gpu
        self.gpus_per_run = gpus_per_run
        self.backend = backend
        assert backend in ("local", "modal"), f"unknown backend {backend!r}"
        # Modal backend: H100 / container, MPS-internal packing optional. None
        # of the local GPU bookkeeping applies — no per_gpu cap, no slot pool,
        # no MPS daemons on the orchestrator box, no shared async vllm. The
        # local invariants below would assert incorrectly under modal, so we
        # short-circuit max_concurrent.
        if self.backend == "modal":
            self.use_mps = False  # MPS is started inside the Modal container, not here
            # max_concurrent is set per backend mode in run().
            self.max_concurrent = max(1, max_concurrent_packs or 64)
        elif gpus_per_run > 1:
            assert per_gpu == 1, (
                f"gpus_per_run={gpus_per_run} requires per_gpu=1, got per_gpu={per_gpu}"
            )
            assert len(gpus) >= gpus_per_run, (
                f"gpus_per_run={gpus_per_run} but only {len(gpus)} GPU(s) available"
            )
            self.max_concurrent = len(gpus) // gpus_per_run
        else:
            self.max_concurrent = per_gpu * len(gpus)

        # Modal-only packing configuration. pack_runs=True groups same-config-
        # different-seed runs (or whatever pack_group_keys selects) into one
        # train_many call. pack_runs=False sends one train_one call per run.
        self.pack_runs = pack_runs and backend == "modal"
        self.max_per_pack = max_per_pack
        self.pack_group_keys = pack_group_keys
        self.pack_vllm_gpu_memory = pack_vllm_gpu_memory

        # Modal volume sync: a daemon thread that periodically pulls the
        # entire sweep dir from the volume to local disk so the existing plot/
        # HTML generation paths (which read from local file paths) produce
        # live data while the sweep is running. Pulls everything including
        # checkpoint-* — the user accepted that cost for in-flight plots.
        # interval<=0 disables (post-hoc sync remains the workflow).
        self.modal_sync_interval = modal_sync_interval if backend == "modal" else 0
        self.modal_volume_name = modal_volume_name
        self._modal_sync_stop = None    # threading.Event, set on shutdown
        self._modal_sync_thread = None
        self._modal_sync_lock = None    # protects "currently syncing" flag for status
        self.stagger = stagger
        self.wandb_project = wandb_project
        self.no_wandb = no_wandb
        self.dry_run = dry_run
        self.no_baseline = no_baseline
        self.extra_baselines = extra_baselines
        self.run_tag = run_tag

        # Build combined run queue: routing runs + baseline runs
        # Detect routing from routing_mode param (classic/exclusive)
        has_routing = any(
            p.get("routing_mode") not in (None, "none") for p in runs
        )
        self.run_queue = []  # list of {params, grid_keys, is_baseline}

        for params in runs:
            self.run_queue.append({
                "params": params,
                "grid_keys": grid_keys,
                "is_baseline": False,
            })

        # Caching — shared across all sweeps at the parent dir
        self.no_cache = no_cache
        cache_dir = str(Path(output_dir).parent)
        self._cache_dir = cache_dir
        if no_cache:
            self._baseline_cache = {}
            self._run_cache = {}
        else:
            self._baseline_cache = _load_cache(cache_dir, ".baseline_cache.json")
            self._run_cache = _load_cache(cache_dir, ".run_cache.json")
        self._cached_baseline_idxs = {}  # run_idx -> cached run_dir
        self._cached_run_idxs = {}  # run_idx -> cached run_dir

        if has_routing and not no_baseline:
            baseline_configs = generate_baseline_runs(
                runs, grid_keys,
                retain_penalty=retain_penalty,
                filter_baseline=filter_baseline,
                reward_penalty=reward_penalty,
                regular_baseline=regular_baseline,
            )
            for baseline_params, baseline_grid_keys in baseline_configs:
                self.run_queue.append({
                    "params": baseline_params,
                    "grid_keys": baseline_grid_keys,
                    "is_baseline": True,
                })

        self.runs = [q["params"] for q in self.run_queue]
        self.all_grid_keys = grid_keys

        # State
        self.active = {}  # run_idx -> {proc, log_file, log_path, run_name, gpu, start_time}
        self.completed = {}  # run_idx -> {exit_code, duration, run_name, run_dir, is_baseline}
        self.queue = list(range(len(self.run_queue)))
        if shuffle:
            random.shuffle(self.queue)
        self.gpu_counts = {g: 0 for g in gpus}  # active count per GPU

        # Filter cached runs
        self._filter_cached_baselines()
        self._filter_cached_runs()

        # Build experiment groups for incremental plotting
        self.experiment_groups = self._build_experiment_groups()

        # Per-run vLLM servers: {gpu: (proc, socket_path)}
        # Empty for backend=modal — vLLM is spawned inside the Modal container
        # by train.py (vllm_spawn=True forced by tools.modal_train_gr).
        self.vllm_servers = vllm_servers or {}
        # Shared async vLLM servers: {gpu: (proc, socket_path)}
        self.vllm_async_servers = vllm_async_servers or {}
        self.vllm_async_sockets = {gpu: path for gpu, (_, path) in self.vllm_async_servers.items()}

        # Modal backend bookkeeping. Each pack -> Modal FunctionCall handle.
        # Indexed by an internal pack_id (monotonic). For pack_runs=False each
        # pack contains exactly one run_idx (1:1 with run_idx in active dict
        # behavior). For pack_runs=True multiple run_idxs share a FunctionCall.
        self._modal_active = {}   # pack_id -> {call, run_idxs, t0, run_names}
        self._modal_next_pack_id = 0
        self._modal_pack_queue = []  # list[list[run_idx]] — planned packs to dispatch
        # Ledger so harvest can reconstruct (run_name, run_dir) per run_idx
        # without re-materializing (which would re-call deterministic_run_id).
        self._modal_run_dirs = {}
        self._modal_run_names = {}
        if self.backend == "modal":
            self._build_modal_pack_queue()

        # Signal handling
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

    def _build_modal_pack_queue(self):
        """Convert self.queue (list of run_idx not yet completed) into self.
        _modal_pack_queue (list of lists of run_idx). With pack_runs=False each
        pack is a single run_idx (1:1 with train_one). With pack_runs=True the
        grouping uses tools.modal_train_gr._group_runs over the run_idxs'
        materialized params (post-baseline-strip), so all-seeds-of-a-hyperparam-
        point pack together by default."""
        if not self.queue:
            return
        if not self.pack_runs:
            self._modal_pack_queue = [[idx] for idx in self.queue]
            return
        # Build a parallel list of materialized params + back-ref to run_idx so
        # we can use the existing group helper without changing its signature.
        from tools.modal_train_gr import _group_runs
        per_run_params = []
        for idx in self.queue:
            full, _, _, _ = self._materialize_run(idx)
            # Mark the run_idx so we can recover it from the packed dict.
            per_run_params.append({"__run_idx__": idx, **full})
        packs = _group_runs(per_run_params,
                            group_keys=self.pack_group_keys,
                            max_per_pack=self.max_per_pack,
                            skip_keys=("__run_idx__",))
        self._modal_pack_queue = [[p["__run_idx__"] for p in pack] for pack in packs]

    def _filter_cached_baselines(self):
        """Check baseline cache, skip already-completed baselines."""
        new_queue = []
        for idx in self.queue:
            entry = self.run_queue[idx]
            if not entry["is_baseline"]:
                new_queue.append(idx)
                continue

            cache_key = _baseline_cache_key(entry["params"])
            cached = self._baseline_cache.get(cache_key)
            if cached and _cache_entry_valid(cached):
                prefix = _run_prefix(entry["params"], is_baseline=True)
                run_name = make_run_name(
                    entry["params"], entry["grid_keys"],
                    prefix=prefix,
                )
                if self.run_tag:
                    run_name = f"{run_name}_{self.run_tag}"
                print(f"[CACHE HIT] {run_name} -> {cached['run_dir']}")
                self.completed[idx] = {
                    "exit_code": 0,
                    "duration": 0,
                    "run_name": run_name,
                    "run_dir": Path(cached["run_dir"]),
                    "is_baseline": True,
                }
                self._cached_baseline_idxs[idx] = cached["run_dir"]
            else:
                new_queue.append(idx)

        self.queue = new_queue

    def _filter_cached_runs(self):
        """Check run cache, skip already-completed regular runs."""
        new_queue = []
        for idx in self.queue:
            entry = self.run_queue[idx]
            if entry["is_baseline"]:
                new_queue.append(idx)
                continue

            cache_key = _run_cache_key(entry["params"])
            cached = self._run_cache.get(cache_key)
            if cached and _cache_entry_valid(cached):
                prefix = _run_prefix(entry["params"], is_baseline=False)
                run_name = make_run_name(entry["params"], entry["grid_keys"], prefix=prefix)
                if self.run_tag:
                    run_name = f"{run_name}_{self.run_tag}"
                print(f"[CACHE HIT] {run_name} -> {cached['run_dir']}")
                self.completed[idx] = {
                    "exit_code": 0,
                    "duration": 0,
                    "run_name": run_name,
                    "run_dir": Path(cached["run_dir"]),
                    "is_baseline": False,
                }
                self._cached_run_idxs[idx] = cached["run_dir"]
            else:
                new_queue.append(idx)

        self.queue = new_queue

    def _build_experiment_groups(self):
        """Build experiment groups for incremental plotting.

        An experiment group = all runs sharing the same non-seed params.
        Each group has routing_idxs and baseline_idxs.

        Baselines have routing-only params stripped from grid_keys, so they
        get a different _group_key than routing runs. Two-pass approach:
        1) Group routing runs by their full grid key
        2) Assign each baseline to every routing group whose non-routing-only
           params match the baseline's group key
        """
        groups = {}
        baseline_pool = []  # (idx, base_key)

        # First pass: group routing runs; collect baselines separately.
        # For matching, strip ROUTING_ONLY_PARAMS from baseline grid_keys so both
        # regular baselines (which already lack these) and filter baselines (which
        # keep rh_eligible_frac etc.) match against the same routing groups.
        for idx, entry in enumerate(self.run_queue):
            if entry["is_baseline"]:
                match_gkeys = entry["grid_keys"] - ROUTING_ONLY_PARAMS
                bk = _group_key(entry["params"], match_gkeys)
                baseline_pool.append((idx, bk))
                continue
            gk = _group_key(entry["params"], entry["grid_keys"])
            if gk not in groups:
                groups[gk] = {"routing_idxs": [], "baseline_idxs": [], "plotted": False}
            groups[gk]["routing_idxs"].append(idx)

        # Second pass: assign baselines to matching routing groups
        for gk, group in groups.items():
            if not group["routing_idxs"]:
                continue
            rep_entry = self.run_queue[group["routing_idxs"][0]]
            non_routing_gkeys = rep_entry["grid_keys"] - ROUTING_ONLY_PARAMS
            base_gk = _group_key(rep_entry["params"], non_routing_gkeys)
            for b_idx, b_key in baseline_pool:
                if b_key == base_gk:
                    group["baseline_idxs"].append(b_idx)

        # Handle non-routing sweeps (no baselines at all)
        if not groups and self.run_queue:
            for idx, entry in enumerate(self.run_queue):
                gk = _group_key(entry["params"], entry["grid_keys"])
                if gk not in groups:
                    groups[gk] = {"routing_idxs": [], "baseline_idxs": [], "plotted": False}
                groups[gk]["routing_idxs"].append(idx)

        return groups

    def _handle_sigint(self, signum, frame):
        print("\n[SWEEP] Interrupted — killing all running jobs...")
        self._interrupted = True
        # Modal backend: ask each in-flight FunctionCall to cancel. Modal
        # propagates this as a SIGTERM to the container (same path as a
        # timeout), so the workers' finally blocks still run and vol.commit()
        # still flushes whatever's on disk. No local processes to kill.
        if self.backend == "modal":
            for pack_id, info in list(self._modal_active.items()):
                try:
                    info["call"].cancel()
                except Exception as e:
                    print(f"[SWEEP] failed to cancel pack_id={pack_id}: {e}")
            # One last volume sync so whatever the containers wrote before
            # SIGTERM is on local disk for triage. _stop_modal_sync joins the
            # daemon thread which itself runs one final _modal_sync_once on
            # shutdown — up to 5 min wait, matching the cancel grace window.
            self._stop_modal_sync()
            sys.exit(1)
        # Kill all vLLM process groups immediately (don't wait per-proc — that would
        # take up to 8s × N procs and could be interrupted before finishing).
        for idx, info in self.active.items():
            for vllm_proc in info.get("vllm_procs", []):
                try:
                    pgid = os.getpgid(vllm_proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    try:
                        vllm_proc.kill()
                    except Exception:
                        pass
        for idx, info in self.active.items():
            try:
                info["proc"].kill()
                info["proc"].join(timeout=2)
            except Exception:
                pass
        if self.vllm_servers:
            stop_vllm_servers(self.vllm_servers)
        if self.vllm_async_servers:
            stop_vllm_servers(self.vllm_async_servers)
        if self.use_mps:
            stop_mps_daemons(self.gpus)
        sys.exit(1)

    def _pick_gpu(self):
        """Pick GPU with fewest active runs."""
        return min(self.gpus, key=lambda g: self.gpu_counts[g])

    def _pick_gpu_group(self):
        """Pick gpus_per_run GPUs for a multi-GPU run.

        Returns a list of GPU IDs. For single-GPU runs (gpus_per_run=1),
        returns [_pick_gpu()]. For multi-GPU, picks the N least-loaded GPUs
        (all should have count 0 since per_gpu=1 is enforced).
        """
        if self.gpus_per_run <= 1:
            return [self._pick_gpu()]
        # Sort by load (should all be 0 or 1 since per_gpu=1)
        free = sorted(self.gpus, key=lambda g: self.gpu_counts[g])
        group = free[:self.gpus_per_run]
        busy = {g: self.gpu_counts[g] for g in group if self.gpu_counts[g] > 0}
        assert not busy, (
            f"Multi-GPU run requires {self.gpus_per_run} free GPUs but some are busy: {busy}"
        )
        return group

    def _try_acquire_global_slots(self, gpu_group):
        """Try to acquire one cross-sweep slot per GPU in the group.

        Returns a list of slot_ids on success (one per gpu in the group),
        or None if any GPU is at the global cap. On partial failure, releases
        any slots already acquired.
        """
        if self.gpus_per_run > 1:
            # Multi-GPU runs reserve their full GPUs anyway; skip the global pool.
            return ["multi_gpu"] * len(gpu_group)
        slot_ids = []
        my_pid = os.getpid()
        for g in gpu_group:
            sid = slot_pool.try_acquire(g, my_pid)
            if sid is None:
                # Roll back partial acquisition.
                for prev_g, prev_sid in zip(gpu_group, slot_ids):
                    slot_pool.release(prev_g, my_pid, prev_sid)
                return None
            slot_ids.append(sid)
        return slot_ids

    def _materialize_run(self, run_idx):
        """Compute (full_params, run_name, run_dir, log_path) for a queued run.

        Pure: just naming + path resolution + wandb plumbing. The local and
        Modal launch paths both need this; factoring it out keeps the two
        backends in lock-step on run identity. Does mkdir(run_dir) as a side
        effect because callers expect log_path's parent to exist.
        """
        entry = self.run_queue[run_idx]
        params = entry["params"]
        is_baseline = entry["is_baseline"]
        grid_keys = entry["grid_keys"]

        prefix = _run_prefix(params, is_baseline=is_baseline)
        run_name = make_run_name(params, grid_keys, prefix=prefix)
        if self.run_tag:
            run_name = f"{run_name}_{self.run_tag}"
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        full_params = dict(params)
        full_params["output_dir"] = str(run_dir)
        full_params["run_name"] = f"{self.output_dir.name}/{run_name}"
        full_params["wandb_project"] = self.wandb_project
        # Group + deterministic id: ties every run in this sweep to a single
        # wandb group (= sweep_name) and gives it a stable id so retries
        # overwrite the prior (partial) wandb run instead of forking.
        from sweep_views import deterministic_run_id
        full_params["wandb_group"] = self.output_dir.name
        full_params["wandb_run_id"] = deterministic_run_id(
            self.output_dir.name, run_name, params=full_params,
        )
        if self.no_wandb:
            full_params["no_wandb"] = True

        return full_params, run_name, run_dir, log_path

    def _launch(self, run_idx, gpu_group=None, slot_ids=None):
        """Launch a single run in a child process.

        gpu_group + slot_ids: optionally pre-picked by the launcher loop
        (so it can acquire global cross-sweep slots before deciding to
        launch). If None, picks here without using the global slot pool
        (legacy path; only used on dry-run summary etc.).
        """
        entry = self.run_queue[run_idx]
        is_baseline = entry["is_baseline"]

        full_params, run_name, run_dir, log_path = self._materialize_run(run_idx)
        if gpu_group is None:
            gpu_group = self._pick_gpu_group()

        # Multi-GPU DDP: inject world_size so train.py knows to mp.spawn
        if self.gpus_per_run > 1:
            full_params["world_size"] = self.gpus_per_run

        # Shared async server: inject socket path for the assigned GPU
        # (not compatible with multi-GPU per run)
        if self.vllm_async_sockets and gpu_group[0] in self.vllm_async_sockets:
            assert self.gpus_per_run == 1, (
                "Shared async vLLM servers are not compatible with gpus_per_run > 1"
            )
            full_params["vllm_server"] = self.vllm_async_sockets[gpu_group[0]]
            full_params["vllm_async"] = True

        # Per-run vLLM server: spawn directly from sweep process (avoids daemon nesting),
        # then pass socket path to training worker instead of spawning from within the worker.
        # For multi-GPU: spawn one vLLM server per GPU; train.py connects each DDP rank
        # to its server via convention: {base}_rank{rank}.sock
        vllm_procs = []
        vllm_pending = []  # [{ready_file, proc, spawn_time, label, warned_60, warned_600}]
        if full_params.get("vllm_spawn"):
            import tempfile
            ctx_vllm = multiprocessing.get_context("spawn")
            vllm_model = full_params.get("model", "HuggingFaceTB/SmolLM2-135M-Instruct")
            vllm_mlp = full_params.get("mlp_config", "m32")
            vllm_gpu_memory = full_params.get("vllm_gpu_memory", 0.02)
            vllm_adapter_type = full_params.get("adapter_type", "mlp")
            vllm_dtype = full_params.get("vllm_dtype", "bfloat16")
            # Defaults must mirror train.py's argparse (compiled + FULL graphs
            # are the validated standard as of 2026-06-12).
            vllm_enforce_eager = full_params.get("vllm_enforce_eager", False)
            vllm_cudagraph_mode = full_params.get("vllm_cudagraph_mode", "FULL_AND_PIECEWISE") or None
            vllm_max_model_len = full_params.get("vllm_max_model_len", None)
            from vllm_utils import default_kv_cache_gb
            _kv_gb = full_params.get("vllm_kv_cache_gb", None)
            if _kv_gb is None:
                _kv_gb = default_kv_cache_gb(vllm_model)
            vllm_kv_bytes = int(_kv_gb * 2**30) if _kv_gb else None
            vllm_parallel_init = full_params.get("vllm_parallel_init", None)
            if vllm_parallel_init is None:
                vllm_parallel_init = vllm_kv_bytes is not None
            # 4 slots default (training + 3 eval modes); +1 for interlaced coherence.
            vllm_max_exp = 5 if full_params.get("coh_samples_per_rollout", 0) > 0 else 4
            unique_id = _find_free_port()  # reuse port finder for a unique numeric ID

            if self.gpus_per_run > 1:
                # Multi-GPU: one vLLM server per GPU, convention-based socket names
                vllm_base = f"ipc:///tmp/vllm_grpo_{unique_id}"
                for rank, gpu in enumerate(gpu_group):
                    vllm_socket_path = f"{vllm_base}_rank{rank}.sock"
                    vllm_ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_gpu{gpu}")
                    rank_label = f"vLLM server for {run_name} rank {rank} on GPU {gpu}"
                    vllm_proc = ctx_vllm.Process(
                        target=_vllm_server_worker,
                        args=(gpu, vllm_model, vllm_mlp, vllm_max_exp,
                              vllm_gpu_memory, vllm_socket_path),
                        kwargs={"ready_file": vllm_ready_file, "adapter_type": vllm_adapter_type,
                                "dtype": vllm_dtype, "log_dir": str(run_dir),
                                "label": rank_label, "enforce_eager": vllm_enforce_eager,
                                "cudagraph_mode": vllm_cudagraph_mode,
                                "max_model_len": vllm_max_model_len,
                                "kv_cache_memory_bytes": vllm_kv_bytes,
                                "parallel_init": vllm_parallel_init},
                    )
                    vllm_proc.start()
                    vllm_procs.append(vllm_proc)
                    print(f"[vLLM] Spawned server for {run_name} rank {rank} on GPU {gpu}, "
                          f"socket {vllm_socket_path} (pid={vllm_proc.pid})")
                    # Wait for this server to be ready before spawning the next
                    _wait_for_vllm_ready(vllm_ready_file, vllm_proc, rank_label)
                # Pass base path; train.py appends _rank{rank}.sock per DDP worker
                full_params = {k: v for k, v in full_params.items()
                               if k not in ("vllm_spawn", "vllm_spawn_delay", "vllm_gpu_memory", "vllm_dtype")}
                full_params["vllm_server_base"] = vllm_base
            else:
                # Single-GPU: one vLLM server. The per-GPU init queue (vllm_init_slot
                # inside _vllm_server_worker) serializes CUDA init across concurrent
                # launches on the same GPU; no caller-side stagger needed.
                gpu = gpu_group[0]
                vllm_socket_path = f"ipc:///tmp/vllm_grpo_gpu{gpu}_{unique_id}.sock"
                vllm_ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_gpu{gpu}")
                vllm_label = f"vLLM server for {run_name} on GPU {gpu}"
                vllm_proc = ctx_vllm.Process(
                    target=_vllm_server_worker,
                    args=(gpu, vllm_model, vllm_mlp, vllm_max_exp,
                          vllm_gpu_memory, vllm_socket_path),
                    kwargs={"ready_file": vllm_ready_file, "adapter_type": vllm_adapter_type,
                            "dtype": vllm_dtype, "log_dir": str(run_dir),
                            "label": vllm_label, "enforce_eager": vllm_enforce_eager,
                            "cudagraph_mode": vllm_cudagraph_mode,
                            "max_model_len": vllm_max_model_len,
                            "kv_cache_memory_bytes": vllm_kv_bytes,
                            "parallel_init": vllm_parallel_init},
                )
                vllm_proc.start()
                vllm_procs.append(vllm_proc)
                vllm_pending.append({
                    "ready_file": vllm_ready_file,
                    "proc": vllm_proc,
                    "spawn_time": time.time(),
                    "label": vllm_label,
                    "warned_60": False,
                    "warned_600": False,
                })
                print(f"[vLLM] Spawned server for {run_name} on GPU {gpu}, "
                      f"socket {vllm_socket_path} (pid={vllm_proc.pid})")
                # Replace vllm_spawn flag with socket path for the training worker
                full_params = {k: v for k, v in full_params.items()
                               if k not in ("vllm_spawn", "vllm_spawn_delay", "vllm_gpu_memory", "vllm_dtype")}
                full_params["vllm_server"] = vllm_socket_path

        pipe = mps_pipe_dir(gpu_group[0]) if (self.use_mps and self.gpus_per_run == 1) else None
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_run_worker,
            args=(full_params, str(log_path), gpu_group, pipe),
        )
        proc.start()

        self.active[run_idx] = {
            "proc": proc,
            "log_path": log_path,
            "run_name": run_name,
            "run_dir": run_dir,
            "gpus": gpu_group,
            "slot_ids": slot_ids,  # cross-sweep slot pool ids; released on completion
            "start_time": time.time(),
            "is_baseline": is_baseline,
            "vllm_procs": vllm_procs,
            "vllm_pending": vllm_pending,
        }
        for g in gpu_group:
            self.gpu_counts[g] += 1

        total = len(self.run_queue)
        launched = len(self.completed) + len(self.active)
        tag = "BASELINE" if is_baseline else "LAUNCH"
        gpu_str = ",".join(str(g) for g in gpu_group)
        print(f"[{tag} {launched}/{total}] {run_name} (pid={proc.pid}, gpu={gpu_str})")

    def _check_vllm_pending(self):
        """Warn if per-run vLLM servers haven't become ready within 60s / 10 min."""
        for info in self.active.values():
            pending = info.get("vllm_pending") or []
            still = []
            for entry in pending:
                if os.path.exists(entry["ready_file"]):
                    try:
                        os.unlink(entry["ready_file"])
                    except FileNotFoundError:
                        pass
                    continue
                # Server process died before ready: the training worker would
                # wait on the ready file FOREVER (observed 2026-06-12: five
                # init-watchdog server crashes left five zombie trainers and
                # the sweep showed "5 running" indefinitely). Kill the run
                # loudly instead — _check_completed then reaps it as failed.
                if not entry["proc"].is_alive():
                    print(f"[vLLM FATAL] {entry['label']} DIED before ready "
                          f"(exitcode={entry['proc'].exitcode}) — killing the "
                          f"training worker for this run; see "
                          f"{info['run_dir']}/vllm_server.log")
                    try:
                        info["proc"].kill()
                    except Exception as e:
                        print(f"[vLLM FATAL] failed to kill training worker: {e}")
                    continue
                elapsed = time.time() - entry["spawn_time"]
                if not entry["warned_60"] and elapsed >= 60:
                    print(f"[vLLM WARN] {entry['label']} not ready after 60s — "
                          f"check {info['run_dir']}/vllm_server.log")
                    entry["warned_60"] = True
                if not entry["warned_600"] and elapsed >= 600:
                    print(f"[vLLM ALERT] {entry['label']} still not ready after 10 min — "
                          f"likely stuck or OOM")
                    entry["warned_600"] = True
                still.append(entry)
            info["vllm_pending"] = still

    def _check_completed(self):
        """Poll active processes for completion."""
        finished = []
        for idx, info in self.active.items():
            proc = info["proc"]
            if proc.is_alive():
                continue
            ret = proc.exitcode
            if ret is not None:
                duration = time.time() - info["start_time"]
                proc.join()
                for g in info["gpus"]:
                    self.gpu_counts[g] -= 1
                # Release cross-sweep slot pool entries.
                slot_ids = info.get("slot_ids")
                if slot_ids and self.gpus_per_run <= 1:
                    my_pid = os.getpid()
                    for g, sid in zip(info["gpus"], slot_ids):
                        slot_pool.release(g, my_pid, sid)

                mins = int(duration) // 60
                secs = int(duration) % 60
                total = len(self.run_queue)
                done = len(self.completed) + 1
                status = "OK" if ret == 0 else f"FAIL(exit={ret})"
                tag = "BASELINE" if info.get("is_baseline") else "DONE"
                print(f"[{tag:8s} {done}/{total}] {info['run_name']}  {status}  time={mins}m{secs:02d}s")

                # On failure, append to {output_dir}/failures.jsonl with last
                # ~20 lines of train.log for quick triage. Canonical post-launch
                # sanity check: `cat output/{sweep}/failures.jsonl`.
                if ret != 0:
                    self._record_failure(info, ret, duration)

                self.completed[idx] = {
                    "exit_code": ret,
                    "duration": duration,
                    "run_name": info["run_name"],
                    "run_dir": info["run_dir"],
                    "is_baseline": info.get("is_baseline", False),
                }

                # Update cache on successful completion
                # Populate cache on success unconditionally — the cache is a
                # persistent record of completed runs that future sweeps can
                # consult. --use_cache only controls whether to *consult* the
                # cache for skipping, not whether to *populate* it.
                if ret == 0:
                    entry = self.run_queue[idx]
                    if info.get("is_baseline"):
                        cache_key = _baseline_cache_key(entry["params"])
                        # Lazy-load if we didn't load at init (no_cache path).
                        if not self._baseline_cache and self.no_cache:
                            self._baseline_cache = _load_cache(self._cache_dir, ".baseline_cache.json")
                        self._baseline_cache[cache_key] = {
                            "run_dir": str(info["run_dir"]),
                            "timestamp": time.time(),
                        }
                        _save_cache(self._cache_dir, self._baseline_cache, ".baseline_cache.json")
                    else:
                        cache_key = _run_cache_key(entry["params"])
                        if not self._run_cache and self.no_cache:
                            self._run_cache = _load_cache(self._cache_dir, ".run_cache.json")
                        self._run_cache[cache_key] = {
                            "run_dir": str(info["run_dir"]),
                            "timestamp": time.time(),
                        }
                        _save_cache(self._cache_dir, self._run_cache, ".run_cache.json")

                # Stop per-run vLLM server(s) if present
                for vllm_proc in info.get("vllm_procs", []):
                    _kill_vllm_proc(vllm_proc)
                # Clean up any lingering ready-file sentinels
                for entry in info.get("vllm_pending", []):
                    try:
                        os.unlink(entry["ready_file"])
                    except FileNotFoundError:
                        pass

                finished.append(idx)
        for idx in finished:
            del self.active[idx]

        # Check experiment groups for incremental plotting
        for group_key, group in self.experiment_groups.items():
            if group["plotted"]:
                continue
            all_idxs = group["routing_idxs"] + group["baseline_idxs"]
            all_done = all(idx in self.completed for idx in all_idxs)
            if all_done and all_idxs:
                self._generate_group_plots(group_key, group)
                group["plotted"] = True

    # ------------------------------------------------------------------
    # Modal backend launch + poll. These mirror _launch and _check_completed
    # but dispatch via tools.modal_train_gr.{train_one,train_many}.spawn()
    # instead of a local multiprocessing.Process.
    #
    # Invariants:
    #   - vllm_spawn=True is forced (vLLM runs in-process inside the Modal
    #     container; no separate server is spawned).
    #   - vllm_async / vllm_server_base sweep-level params are ignored: there
    #     is no shared server inside a container, and there's no need for
    #     ZMQ-based sharing across containers either.
    #   - per_gpu / slot_pool / MPS daemons on the orchestrator box are not
    #     used. Concurrency is whatever Modal will run for us (effectively
    #     unlimited for our scales, capped by max_concurrent_packs).
    #   - When pack_runs=True, packs are pre-built in _build_modal_pack_queue
    #     using tools.modal_train_gr._group_runs. The default grouping packs
    #     all-seeds-of-a-hyperparam-point together.
    # ------------------------------------------------------------------

    def _modal_launch_pack(self, pack_run_idxs):
        """Dispatch one pack (list of run_idx) to Modal. Stores a FunctionCall
        handle in self._modal_active under a monotonic pack_id and increments
        the orchestrator's "active" accounting via the regular plotting path
        (the per-run results are written into self.completed when the call
        finishes)."""
        # Lazy import: Modal client is only needed for backend="modal", and
        # importing it at module load time would force the dep on every sweep.
        from tools.modal_train_gr import train_one, train_many
        deployed = os.environ.get("GRAFT_MODAL_DEPLOYED_APP")
        if deployed:
            # a deployed copy of the gr-pilot app (see _run_modal_loop)
            import modal
            train_one = modal.Function.from_name(deployed, "train_one")
            train_many = modal.Function.from_name(deployed, "train_many")

        items = []  # parallel lists with pack_run_idxs
        run_names = []
        for run_idx in pack_run_idxs:
            full_params, run_name, run_dir, _ = self._materialize_run(run_idx)
            # Scale vLLM memory per item if packing — sum should stay < ~0.40 of
            # the H100 (the rest is model weights + activations). User-supplied
            # vllm_gpu_memory on the run wins; otherwise we choose a safe value.
            if len(pack_run_idxs) > 1 and "vllm_gpu_memory" not in full_params:
                budget = self.pack_vllm_gpu_memory or 0.40
                full_params["vllm_gpu_memory"] = budget / len(pack_run_idxs)
            items.append(full_params)
            run_names.append(run_name)
            # Track per-run dir up-front so we can write to self.completed on
            # harvest. self.active is local-backend-only; we use a separate
            # ledger.
            self._modal_run_dirs[run_idx] = run_dir
            self._modal_run_names[run_idx] = run_name

        sweep_name = self.output_dir.name
        if len(items) == 1:
            call = train_one.spawn(items[0], sweep_name)
        else:
            call = train_many.spawn(items, sweep_name)

        pack_id = self._modal_next_pack_id
        self._modal_next_pack_id += 1
        self._modal_active[pack_id] = {
            "call": call,
            "run_idxs": pack_run_idxs,
            "run_names": run_names,
            "t0": time.time(),
            "is_pack": len(items) > 1,
            # Count consecutive transient poll errors per pack — see
            # _modal_check_completed. Reset on any successful poll.
            "transient_errs": 0,
        }
        total = len(self.run_queue)
        launched = (len(self.completed)
                    + sum(len(p["run_idxs"]) for p in self._modal_active.values()))
        kind = f"PACK×{len(items)}" if len(items) > 1 else "MODAL"
        print(f"[{kind} {launched}/{total}] pack_id={pack_id}  runs=[{', '.join(run_names)}]")

    def _modal_check_completed(self):
        """Poll Modal FunctionCall handles. Each completed pack returns either a
        single result dict (train_one) or a list of result dicts (train_many);
        we map back to run_idxs and populate self.completed using the same
        schema the local backend uses, so the incremental-plotting path is
        unchanged."""
        import modal  # available because backend=modal was selected
        # Transient = "control plane / network blip; the actual container is
        # almost certainly still running and producing data". Polling errors
        # of these classes should NOT kill the pack — we just retry next tick.
        # Empirically (2026-06-03) a single ConnectionError during one of our
        # poll calls bubbled out and brought down a healthy MPS-running
        # container that was at step ~100/1000.
        _TRANSIENT_POLL_ERRS = (
            modal.exception.ConnectionError,
            modal.exception.InternalError,
            modal.exception.InternalFailure,
            # ServiceError wraps control-plane gRPC failures (e.g. an auth-check
            # DEADLINE_EXCEEDED blip). Empirically (2026-06-18) a ~30s blip after
            # ~2h killed a healthy 3-container leetcode run mid-training. Treat as
            # transient — the containers keep running; the consecutive-error cap
            # below still fails loudly if it persists.
            modal.exception.ServiceError,
        )
        # Cap: if a single pack hits too many consecutive transient errors,
        # something is genuinely wrong (Modal down, app stopped, etc.) and we
        # should fail loudly rather than spin forever.
        _TRANSIENT_LIMIT = 20
        finished_packs = []
        for pack_id, info in list(self._modal_active.items()):
            call = info["call"]
            try:
                result = call.get(timeout=0)
            except TimeoutError:
                info["transient_errs"] = 0  # any clean response resets the counter
                continue  # still running
            except _TRANSIENT_POLL_ERRS as e:
                info["transient_errs"] += 1
                if info["transient_errs"] <= 3:
                    print(f"[MODAL POLL] pack_id={pack_id}  transient "
                          f"{type(e).__name__}: {e} "
                          f"(consecutive={info['transient_errs']}, retrying)")
                if info["transient_errs"] >= _TRANSIENT_LIMIT:
                    print(f"[MODAL FAIL] pack_id={pack_id}  too many transient "
                          f"poll errors ({info['transient_errs']}); giving up")
                    self._modal_record_pack(info, [{
                        "status": f"modal_error(transient_limit)",
                        "duration_s": time.time() - info["t0"],
                        "err": f"{type(e).__name__}: {e}",
                        "run_name": rn,
                    } for rn in info["run_names"]])
                    finished_packs.append(pack_id)
                continue
            except Exception as e:
                # Genuine function-level failure (RemoteError, FunctionTimeoutError,
                # container crash before any user code, etc). Mark every run in the
                # pack as failed.
                print(f"[MODAL FAIL] pack_id={pack_id}  {type(e).__name__}: {e}")
                self._modal_record_pack(info, [{
                    "status": f"modal_error({type(e).__name__})",
                    "duration_s": time.time() - info["t0"],
                    "err": str(e),
                    "run_name": rn,
                } for rn in info["run_names"]])
                finished_packs.append(pack_id)
                continue

            # Got a real result — reset transient counter.
            info["transient_errs"] = 0

            # Normalize: train_one returns a single dict, train_many returns a
            # list. Pair each result with its run_idx by order.
            if info["is_pack"]:
                results = result
            else:
                results = [result]
            assert len(results) == len(info["run_idxs"]), (
                f"Modal call returned {len(results)} results for "
                f"{len(info['run_idxs'])} run(s) (pack_id={pack_id})"
            )
            self._modal_record_pack(info, results)
            finished_packs.append(pack_id)

        for pack_id in finished_packs:
            del self._modal_active[pack_id]

        # Reuse the incremental-plotting trigger from the local path. Building
        # this in two places risks drift; instead drive it off self.completed.
        for group_key, group in self.experiment_groups.items():
            if group["plotted"]:
                continue
            all_idxs = group["routing_idxs"] + group["baseline_idxs"]
            all_done = all(idx in self.completed for idx in all_idxs)
            if all_done and all_idxs:
                self._generate_group_plots(group_key, group)
                group["plotted"] = True

    def _modal_record_pack(self, info, results):
        """Write a Modal pack's per-run results into self.completed and update
        the persistent cache, mirroring the local _check_completed bookkeeping
        for a finished run. Status/exit_code mapping: 'ok' -> 0, anything else
        -> 1 (we don't have a numeric exit code from the worker)."""
        for run_idx, res in zip(info["run_idxs"], results):
            status = res.get("status", "unknown")
            exit_code = 0 if status == "ok" else 1
            run_name = self._modal_run_names[run_idx]
            run_dir = self._modal_run_dirs[run_idx]
            duration = res.get("duration_s", time.time() - info["t0"])
            mins, secs = int(duration) // 60, int(duration) % 60
            entry = self.run_queue[run_idx]
            tag = "BASELINE" if entry["is_baseline"] else "DONE"
            label = "OK" if exit_code == 0 else f"FAIL({status})"
            total = len(self.run_queue)
            done = len(self.completed) + 1
            print(f"[{tag:8s} {done}/{total}] {run_name}  {label}  time={mins}m{secs:02d}s")

            if exit_code != 0:
                # Best-effort failure record. Local _record_failure reads the
                # tail of train.log; that file lives on the Modal volume and may
                # not be present locally yet, so we record what we have.
                err_path = Path(self.output_dir) / "failures.jsonl"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                with open(err_path, "a") as f:
                    f.write(json.dumps({
                        "run_name": run_name,
                        "status": status,
                        "err": res.get("err"),
                        "duration_s": duration,
                        "run_dir": str(run_dir),
                    }) + "\n")

            self.completed[run_idx] = {
                "exit_code": exit_code,
                "duration": duration,
                "run_name": run_name,
                "run_dir": run_dir,
                "is_baseline": entry["is_baseline"],
            }

            if exit_code == 0:
                if entry["is_baseline"]:
                    cache_key = _baseline_cache_key(entry["params"])
                    if not self._baseline_cache and self.no_cache:
                        self._baseline_cache = _load_cache(self._cache_dir, ".baseline_cache.json")
                    self._baseline_cache[cache_key] = {
                        "run_dir": str(run_dir), "timestamp": time.time(),
                    }
                    _save_cache(self._cache_dir, self._baseline_cache, ".baseline_cache.json")
                else:
                    cache_key = _run_cache_key(entry["params"])
                    if not self._run_cache and self.no_cache:
                        self._run_cache = _load_cache(self._cache_dir, ".run_cache.json")
                    self._run_cache[cache_key] = {
                        "run_dir": str(run_dir), "timestamp": time.time(),
                    }
                    _save_cache(self._cache_dir, self._run_cache, ".run_cache.json")

    def _generate_group_plots(self, group_key, group):
        """Generate per-step comparison plots for a completed experiment group."""
        try:
            from sweep_plots import generate_group_comparison_plots

            routing_runs = []
            for idx in group["routing_idxs"]:
                info = self.completed.get(idx)
                if info and info["exit_code"] == 0:
                    routing_runs.append(str(info["run_dir"]))

            baseline_runs = []
            for idx in group["baseline_idxs"]:
                info = self.completed.get(idx)
                if info and info["exit_code"] == 0:
                    baseline_runs.append(str(info["run_dir"]))

            if not routing_runs:
                return

            # Build a readable group name from the first routing run's params
            first_params = self.run_queue[group["routing_idxs"][0]]["params"]
            first_exp_cfg = first_params.get("exp_cfg")
            if first_exp_cfg is not None:
                config_prefix = first_exp_cfg.name or first_exp_cfg.reward_name
            else:
                config_prefix = "run"
            if group_key == "default":
                group_name = config_prefix
            else:
                group_name = config_prefix + "_" + group_key.replace("|", "_").replace("=", "")

            generate_group_comparison_plots(
                routing_runs=routing_runs,
                baseline_runs=baseline_runs,
                reward=config_prefix,
                output_dir=str(self.output_dir),
                group_name=group_name,
                no_baseline=self.no_baseline,
            )

            # Write group metadata for grid visualization
            self._write_group_meta(group_name, config_prefix, group_key)
        except Exception as e:
            print(f"[WARN] Failed to generate plots for group '{group_key}': {e}")

    def _write_group_meta(self, group_name, prefix, group_key):
        """Append group metadata to groups_meta.json for grid visualization."""
        meta_path = self.output_dir / "sweep_graphs" / "groups_meta.json"
        existing = []
        if meta_path.exists():
            with open(meta_path) as f:
                existing = json.load(f)

        # Parse group_key "rollout_batch_size=128|beta=0.02" into dict
        params_dict = {}
        if group_key != "default":
            for part in group_key.split("|"):
                k, v = part.split("=", 1)
                params_dict[k] = v

        # Deduplicate: replace if same name already present
        existing = [e for e in existing if e["name"] != group_name]
        existing.append({"name": group_name, "prefix": prefix, "params": params_dict})

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(existing, f, indent=2)

    def _extract_latest_timing(self, run_dir):
        """Read latest timing stats from trainer_state.json log_history."""
        state_path = Path(run_dir) / "trainer_state.json"
        if not state_path.exists():
            checkpoints = list(Path(run_dir).glob("checkpoint-*/trainer_state.json"))
            if not checkpoints:
                return None
            # Sort numerically by checkpoint step, not lexicographically
            state_path = max(checkpoints, key=lambda p: int(p.parent.name.split("-")[-1]))
        try:
            with open(state_path) as f:
                state = json.load(f)
            for entry in reversed(state.get("log_history", [])):
                if "step_time" in entry:
                    return {
                        "step": entry.get("step"),
                        "step_time": entry.get("step_time"),
                        "rollout": entry.get("timing/rollout"),
                        "compute_reward": entry.get("timing/compute_reward"),
                        "update": entry.get("timing/update"),
                    }
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _record_failure(self, info, exit_code, duration):
        """Append a record to {output_dir}/failures.jsonl on each FAIL.

        One line per failed run: run_name, exit_code, duration, last lines
        of train.log. Lets `cat output/{sweep}/failures.jsonl` serve as the
        canonical "did anything blow up" check after launching a sweep.
        """
        run_dir = Path(info["run_dir"])
        log_path = run_dir / "train.log"
        tail_lines = []
        if log_path.exists():
            try:
                with open(log_path) as f:
                    tail_lines = f.readlines()[-30:]
            except OSError:
                pass
        rec = {
            "run_name": info["run_name"],
            "run_dir": str(run_dir),
            "exit_code": exit_code,
            "duration_s": round(duration, 1),
            "is_baseline": info.get("is_baseline", False),
            "log_tail": [ln.rstrip("\n") for ln in tail_lines],
        }
        failures_path = self.output_dir / "failures.jsonl"
        with open(failures_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _print_status(self):
        """Print one-line status with timing from first active run."""
        total = len(self.run_queue)
        n_done = len(self.completed)
        n_active = len(self.active)
        n_queued = len(self.queue)

        best_reward = None
        best_name = None
        for idx, info in self.completed.items():
            r = extract_latest_reward(info["run_dir"])
            if r is not None and (best_reward is None or r > best_reward):
                best_reward = r
                best_name = info["run_name"]

        n_failed = sum(1 for info in self.completed.values() if info.get("exit_code", 0) != 0)
        n_ok = n_done - n_failed
        if n_failed > 0:
            parts = [f"[STATUS] {n_active} running, {n_ok} ok, {n_failed} FAILED, {n_queued} queued"]
        else:
            parts = [f"[STATUS] {n_active} running, {n_done} done, {n_queued} queued"]
        if best_reward is not None:
            parts.append(f"best_reward={best_reward:.4f} ({best_name})")

        # Timing from first active run
        if self.active:
            first_idx = next(iter(self.active))
            first_info = self.active[first_idx]
            timing = self._extract_latest_timing(first_info["run_dir"])
            if timing and timing.get("step_time"):
                t = timing
                tp = [f"step {t['step']}: {t['step_time']:.0f}s"]
                if t.get("rollout") is not None:
                    tp.append(f"rollout={t['rollout']:.0f}s")
                if t.get("compute_reward") is not None:
                    tp.append(f"reward={t['compute_reward']:.0f}s")
                if t.get("update") is not None:
                    tp.append(f"update={t['update']:.0f}s")
                parts.append(" ".join(tp))

        print(" | ".join(parts))

    def _print_summary(self):
        """Print final summary table."""
        print("\n[SUMMARY]")
        header = "  {:<50s} {:>10s} {:>8s} {:>8s} {:>8s}".format(
            "run", "reward", "kl", "step", "status"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for idx in sorted(self.completed.keys()):
            info = self.completed[idx]
            metrics = extract_final_metrics(info["run_dir"])
            status = "OK" if info["exit_code"] == 0 else f"FAIL({info['exit_code']})"
            if idx in self._cached_baseline_idxs or idx in self._cached_run_idxs:
                status = "CACHED"
            if metrics:
                r = f"{metrics['reward']:.4f}" if metrics["reward"] is not None else "N/A"
                kl = f"{metrics['kl']:.4f}" if metrics["kl"] is not None else "N/A"
                step = str(metrics.get("step", "N/A"))
            else:
                r = kl = step = "N/A"
            print(f"  {info['run_name']:<50s} {r:>10s} {kl:>8s} {step:>8s} {status:>8s}")

    def _regen_separability(self, sync=False):
        """Best-effort: regenerate the per-sample grad/activation separability
        viewer (separability_dist.html + _allenvs.html) for this sweep. No-op
        unless the run used grad-diag (require_grad gates on grad_diag.jsonl
        presence). Never raises into the sweep loop.

        The viewer build can take tens of seconds on large grad_diag data, so
        periodic regens run in a background daemon thread (skipped while a prior
        one is still running) and never block the orchestrator loop. The
        end-of-sweep call passes sync=True so the final page is complete before
        the process exits."""
        def _work():
            try:
                from tools.gen_separability_html import generate as _gen_sep
                _gen_sep(str(self.output_dir), require_grad=True, verbose=False)
            except Exception as e:
                print(f"[WARN] separability regen failed: {e}")
        if sync:
            _work()
            return
        t = getattr(self, "_sep_thread", None)
        if t is not None and t.is_alive():
            return  # previous regen still running; skip this tick
        import threading
        self._sep_thread = threading.Thread(target=_work, daemon=True, name="sep-regen")
        self._sep_thread.start()

    def run(self):
        """Execute the sweep."""
        total = len(self.run_queue)
        n_routing = sum(1 for q in self.run_queue if not q["is_baseline"])
        n_regular_bl = sum(1 for q in self.run_queue if q["is_baseline"]
                           and not q["params"].get("filter_baseline")
                           and not q["params"].get("reward_penalty_baseline")
                           and not q["params"].get("retain_penalty_baseline"))
        n_filter_bl = sum(1 for q in self.run_queue if q["is_baseline"] and q["params"].get("filter_baseline"))
        n_rwdpen_bl = sum(1 for q in self.run_queue if q["is_baseline"] and q["params"].get("reward_penalty_baseline"))
        n_retpen_bl = sum(1 for q in self.run_queue if q["is_baseline"] and q["params"].get("retain_penalty_baseline"))
        n_cached = len(self._cached_baseline_idxs) + len(self._cached_run_idxs)
        n_gpus = len(self.gpus)

        print(f"[SWEEP] {total} runs ({n_routing} routing, {n_regular_bl} baseline, {n_filter_bl} filter, {n_rwdpen_bl} reward_penalty, {n_retpen_bl} retain_penalty, {n_cached} cached)")
        if self.backend == "modal":
            n_packs = len(self._modal_pack_queue)
            mode = ("pack" if self.pack_runs else "1-per-container")
            print(f"[SWEEP] backend=modal ({mode}); {n_packs} container call(s); "
                  f"max_concurrent_packs={self.max_concurrent}")
            if self.pack_runs:
                print(f"[SWEEP] max_per_pack={self.max_per_pack} "
                      f"pack_group_keys={self.pack_group_keys or 'default(seed-only)'}")
        else:
            gpu_info = f"{n_gpus} GPU(s) {self.gpus}, {self.max_concurrent} slots"
            if self.gpus_per_run > 1:
                gpu_info += f" ({self.gpus_per_run} GPUs/run, DDP)"
            print(f"[SWEEP] {gpu_info}")
        print(f"[SWEEP] output_dir={self.output_dir}")
        if self.no_wandb:
            print("[SWEEP] wandb disabled")
        else:
            print(f"[SWEEP] wandb_project={self.wandb_project}")
        print()

        if self.dry_run:
            print("[DRY RUN] Planned runs:")
            for i, entry in enumerate(self.run_queue):
                prefix = _run_prefix(entry["params"], is_baseline=entry["is_baseline"])
                name = make_run_name(entry["params"], entry["grid_keys"], prefix=prefix)
                if self.run_tag:
                    name = f"{name}_{self.run_tag}"
                cached = "(CACHED)" if (i in self._cached_baseline_idxs or i in self._cached_run_idxs) else ""
                if entry["params"].get("retain_penalty_baseline"):
                    tag = "[RETPEN]  "
                elif entry["params"].get("reward_penalty_baseline"):
                    tag = "[RWDPEN]  "
                elif entry["params"].get("filter_baseline"):
                    tag = "[FILTER]  "
                elif entry["is_baseline"]:
                    tag = "[BASELINE]"
                else:
                    tag = "[ROUTING] "
                print(f"  {i+1}. {tag} {name}  {entry['params']} {cached}")

            # Show experiment groups
            print(f"\n[DRY RUN] Experiment groups:")
            for gk, group in self.experiment_groups.items():
                print(f"  {gk}: {len(group['routing_idxs'])} routing + {len(group['baseline_idxs'])} baseline")

            if self.backend == "modal":
                print(f"\n[DRY RUN] Modal packs ({len(self._modal_pack_queue)}):")
                for i, pack in enumerate(self._modal_pack_queue):
                    names = [self._materialize_run(idx)[1] for idx in pack]
                    print(f"  pack {i+1}/{len(self._modal_pack_queue)} (n={len(pack)}): "
                          f"{', '.join(names)}")
            return

        # Build colored wandb saved view up-front. Run ids are deterministic
        # from (sweep_name, run_name), so the view can be created before
        # runs register in wandb — it'll populate as runs come online.
        if not self.no_wandb:
            try:
                from sweep_views import (
                    color_map_for_runs, deterministic_run_id, build_sweep_view,
                )
                sweep_name = self.output_dir.name
                runs_for_color = []
                for entry in self.run_queue:
                    prefix = _run_prefix(entry["params"], is_baseline=entry["is_baseline"])
                    rn = make_run_name(entry["params"], entry["grid_keys"], prefix=prefix)
                    if self.run_tag:
                        rn = f"{rn}_{self.run_tag}"
                    rid = deterministic_run_id(sweep_name, rn, params=entry["params"])
                    runs_for_color.append((rid, entry["params"]))
                color_map = color_map_for_runs(runs_for_color)
                build_sweep_view(
                    entity=None,  # resolve via wandb default
                    project=self.wandb_project,
                    sweep_name=sweep_name,
                    color_map=color_map,
                )
            except Exception as e:
                print(f"[SWEEP_VIEW] Skipping view creation: {e}")

        last_status_time = time.time()
        last_overview_time = 0.0  # generate overview on first status tick
        status_interval = 30  # seconds
        overview_interval = 60  # seconds

        if self.backend == "modal":
            self._run_modal_loop(status_interval, overview_interval)
        else:
            self._run_local_loop(last_status_time, last_overview_time,
                                 status_interval, overview_interval)

        self._print_summary()

        # Generate sweep-wide overview + grid pages
        try:
            from sweep_plots import generate_sweep_overview, generate_sweep_grid
            generate_sweep_overview(str(self.output_dir), extra_baselines=self.extra_baselines, baseline_sweep_dir=self.baseline_sweep)
            generate_sweep_grid(str(self.output_dir))
        except Exception as e:
            print(f"[WARN] Failed to generate sweep pages: {e}")
        self._regen_separability(sync=True)  # final separability page: block until complete

        if self.vllm_servers:
            stop_vllm_servers(self.vllm_servers)
        if self.vllm_async_servers:
            stop_vllm_servers(self.vllm_async_servers)
        if self.use_mps:
            stop_mps_daemons(self.gpus)

    def _run_local_loop(self, last_status_time, last_overview_time,
                        status_interval, overview_interval):
        """The pre-existing scheduling loop, factored out of run() unchanged
        when backend='local'. Owns the per_gpu cap, the cross-sweep slot pool,
        the per-run vLLM startup watcher, and the periodic sweep-overview
        regeneration."""
        while self.queue or self.active:
            if self._interrupted:
                break

            # Launch runs up to max concurrent. Each launch must acquire a
            # cross-sweep slot from slot_pool in addition to having a free
            # local per_gpu slot — so that when other sweeps are also running,
            # total per-GPU concurrency stays under the global cap (default 6,
            # override via SMALL_RL_GPU_CAP).
            while self.queue and len(self.active) < self.max_concurrent:
                # Peek at the local pick first; if local cap full on every GPU,
                # nothing to do.
                gpu_group = self._pick_gpu_group()
                if any(self.gpu_counts[g] >= self.per_gpu for g in gpu_group):
                    break  # local per_gpu cap saturated
                slot_ids = self._try_acquire_global_slots(gpu_group)
                if slot_ids is None:
                    break  # global cap saturated; will retry after a completion
                idx = self.queue.pop(0)
                self._launch(idx, gpu_group=gpu_group, slot_ids=slot_ids)
                if self.stagger > 0 and self.queue:
                    time.sleep(self.stagger)

            # Check for completions + per-run vLLM startup progress
            self._check_completed()
            self._check_vllm_pending()

            # Periodic status
            now = time.time()
            if now - last_status_time >= status_interval:
                self._print_status()
                last_status_time = now

            # Periodic overview/grid regeneration for live progress tracking
            if now - last_overview_time >= overview_interval:
                try:
                    from sweep_plots import generate_sweep_overview, generate_sweep_grid
                    generate_sweep_overview(str(self.output_dir), extra_baselines=self.extra_baselines, baseline_sweep_dir=self.baseline_sweep)
                    generate_sweep_grid(str(self.output_dir))
                except Exception as e:
                    print(f"[WARN] Failed to regenerate sweep pages: {e}")
                self._regen_separability()
                last_overview_time = time.time()

            if self.active:
                time.sleep(0.5)

    def _modal_sync_once(self):
        """Run one `modal volume get --force` to pull the sweep's volume dir
        to local disk. Source of truth for the existing plot generators, which
        only see local files. Pulls everything (incl. checkpoint-*) by design
        — the cost is accepted in exchange for live in-flight plots.

        Failures (network blips, partial transfers) are logged but do not
        crash the sweep; the next cycle retries from scratch (--force
        overwrites any partial files)."""
        import subprocess
        sweep_name = self.output_dir.name
        # Local destination is the orchestrator's output_dir's parent: modal
        # volume get writes the remote basename (= sweep_name) into it, so the
        # resulting local layout is {parent}/{sweep_name}/{run_name}/..., which
        # is exactly self.output_dir / run_name — matching what _materialize_run
        # records and what the plot generators expect.
        local_parent = str(self.output_dir.parent)
        cmd = [
            sys.executable, "-m", "modal", "volume", "get", "--force",
            self.modal_volume_name, f"/{sweep_name}", local_parent,
        ]
        t0 = time.time()
        try:
            # capture_output so the per-cycle download chatter doesn't drown the
            # sweep log; only surface failures.
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                # Generous timeout: a large sweep with many checkpoints can take
                # a minute or more even on a fast link. 2 × interval is the
                # natural ceiling — if a sync isn't done by then the next one
                # would already be starting.
                timeout=max(120, 2 * self.modal_sync_interval),
            )
        except subprocess.TimeoutExpired:
            print(f"[MODAL_SYNC] timed out after "
                  f"{max(120, 2 * self.modal_sync_interval)}s — will retry")
            return
        dur = time.time() - t0
        if result.returncode != 0:
            combined = (result.stderr or "") + (result.stdout or "")
            # Expected race during sweep startup: the volume's /<sweep_name>
            # path only exists after the first worker calls os.makedirs() in
            # train_one/_train_many. Until then, `modal volume get` exits 1
            # with "No such file or directory". Treat this as info, not an
            # alarm. Once workers come up the path exists and subsequent ticks
            # succeed.
            if "No such file or directory" in combined:
                print(f"[MODAL_SYNC] /{sweep_name} not on volume yet "
                      f"(workers haven't created it); skipping ({dur:.1f}s)")
            else:
                tail = combined[-400:]
                print(f"[MODAL_SYNC] non-zero exit {result.returncode} "
                      f"after {dur:.1f}s:\n  {tail}")
        else:
            print(f"[MODAL_SYNC] pulled /{sweep_name} -> {local_parent}/  ({dur:.1f}s)")

    def _start_modal_sync(self):
        """Spin up the background sync daemon. Idempotent."""
        if self.modal_sync_interval <= 0:
            return
        if self._modal_sync_thread is not None:
            return
        import threading
        self._modal_sync_stop = threading.Event()
        self._modal_sync_lock = threading.Lock()

        def _loop():
            # First sync happens after the first interval, not immediately —
            # otherwise the very-first request races the workers starting up
            # and hits an empty remote dir.
            while not self._modal_sync_stop.wait(self.modal_sync_interval):
                with self._modal_sync_lock:
                    self._modal_sync_once()
            # One last sync on shutdown so a sweep completing right after a
            # cycle still leaves a fully populated local mirror.
            with self._modal_sync_lock:
                self._modal_sync_once()

        self._modal_sync_thread = threading.Thread(
            target=_loop, name="ModalVolumeSync", daemon=True,
        )
        self._modal_sync_thread.start()
        print(f"[MODAL_SYNC] background sync started "
              f"(interval={self.modal_sync_interval}s, volume={self.modal_volume_name})")

    def _stop_modal_sync(self):
        """Signal the sync daemon to stop and wait for its final sync to
        finish (so we don't strand any data on the volume)."""
        if self._modal_sync_thread is None:
            return
        self._modal_sync_stop.set()
        self._modal_sync_thread.join(timeout=300)
        if self._modal_sync_thread.is_alive():
            print("[MODAL_SYNC] background sync did not finish within 5 min "
                  "— giving up (data should still be on the Modal volume)")

    def _run_modal_loop(self, status_interval, overview_interval):
        """Modal-backend scheduling loop. Pops packs off self._modal_pack_queue,
        dispatches via _modal_launch_pack (which calls train_one.spawn or
        train_many.spawn), and polls FunctionCall handles in
        _modal_check_completed. There is no per-GPU cap and no slot pool —
        concurrency is whatever Modal will allocate, capped by
        self.max_concurrent (default 64 packs in flight).

        The entire dispatch/poll loop runs inside `with app.run(name=...):`.
        Modal requires the context — `train_one.spawn(...)` /
        `train_many.spawn(...)` raise `ExecutionError: Function has not been
        hydrated … because the App it is defined on is not running` if called
        outside it. The `name=` argument makes this sweep's ephemeral app
        show up as `<sweep_name>` (the disambiguated `output_dir.name`) in
        `modal app list`, instead of the default `"gr-pilot"` shared across
        all sweeps. That eliminates the "which gr-pilot app is mine?"
        problem when stopping things in a shared workspace where multiple
        users / sessions launch concurrently. The app ID is also printed to
        stdout as `[MODAL_APP]` so it can be back-referenced from the
        orchestrator log without consulting `modal app list` at all.

        When the `with` block exits, any still-in-flight FunctionCalls are
        cancelled — fine for us, the while loop already drains everything.

        A background daemon thread periodically pulls the sweep's volume dir
        to local disk so the existing plot/HTML generation paths produce live
        data; see _start_modal_sync."""
        from contextlib import nullcontext

        from tools.modal_train_gr import app

        last_status_time = time.time()
        last_overview_time = 0.0

        sweep_name = self.output_dir.name
        # GRAFT_MODAL_DEPLOYED_APP=<name> dispatches to a deployed copy of the app
        # (modal deploy --name <name> tools/modal_expr_probe.py) instead of
        # starting an ephemeral one: Modal refuses new ephemeral apps once the
        # workspace holds 100 ("reached limit of 100 ephemeral apps", 2026-11-03).
        # In-flight calls are then not cancelled when this loop exits.
        deployed = os.environ.get("GRAFT_MODAL_DEPLOYED_APP")
        self._start_modal_sync()
        try:
            with (nullcontext() if deployed else app.run(name=sweep_name)) as running_app:
                print(f"[MODAL_APP] deployed app {deployed}" if deployed
                      else f"[MODAL_APP] id={running_app.app_id} name={sweep_name}")
                while self._modal_pack_queue or self._modal_active:
                    if self._interrupted:
                        break

                    # Dispatch packs up to the in-flight cap.
                    while (self._modal_pack_queue
                           and len(self._modal_active) < self.max_concurrent):
                        pack = self._modal_pack_queue.pop(0)
                        self._modal_launch_pack(pack)
                        if self.stagger > 0 and self._modal_pack_queue:
                            time.sleep(self.stagger)

                    self._modal_check_completed()

                    now = time.time()
                    if now - last_status_time >= status_interval:
                        self._print_status()
                        last_status_time = now
                    if now - last_overview_time >= overview_interval:
                        try:
                            from sweep_plots import generate_sweep_overview, generate_sweep_grid
                            generate_sweep_overview(str(self.output_dir), extra_baselines=self.extra_baselines, baseline_sweep_dir=self.baseline_sweep)
                            generate_sweep_grid(str(self.output_dir))
                        except Exception as e:
                            print(f"[WARN] Failed to regenerate sweep pages: {e}")
                        self._regen_separability()
                        last_overview_time = time.time()

                    if self._modal_active:
                        # Modal FunctionCall.get(timeout=0) hits the control
                        # plane; sleep a bit longer than the local 0.5s to
                        # avoid hammering it.
                        time.sleep(2.0)
        finally:
            self._stop_modal_sync()


def main():
    parser = argparse.ArgumentParser(description="Sweep orchestrator for train.py")
    parser.add_argument("--name", required=True,
                        help="Sweep name. Results go to output/{name}/, graphs to output/{name}/sweep_graphs/.")
    parser.add_argument("--config", required=True,
                        help="Python sweep config file (.py). Defines module-level `runs` list.")
    parser.add_argument("--per_gpu", type=int, default=None,
                        help="Max concurrent runs per GPU (overrides config file value; default: 12)")
    parser.add_argument("--gpus_per_run", type=int, default=None,
                        help="GPUs per run for DDP (default: 1). Mutually exclusive with per_gpu > 1.")
    parser.add_argument("--output_dir", default=None,
                        help="Base output directory (default: ./output/{name})")
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name (default: small-rl)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging for all runs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print planned runs without launching")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip automatic baseline runs")
    parser.add_argument("--baseline_sweep", default=None,
                        help="Baseline sweep dir or name (under output/) to render "
                             "alongside this sweep's overview.html as faint-gray "
                             "comparison cards. Overrides any baseline_sweep set in "
                             "the sweep config file.")
    parser.add_argument("--run_tag", default=None,
                        help="Suffix appended to all run names (e.g. 'exp1')")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use run caching: skip runs whose params hash matches a previously-completed run "
                             "(including baselines). Cache hits point to the original run dir, which may live "
                             "in a different (disambiguated) sweep output dir. Default: off, re-run everything.")
    parser.add_argument("--retain_penalty", action="store_true",
                        help="Generate retain penalty baseline runs (replace RH rewards with retain-only reward)")
    parser.add_argument("--no_filter_baseline", action="store_true",
                        help="Skip auto-generated filter baselines (routing_mode=none + filter_baseline=True)")
    parser.add_argument("--no_reward_penalty_baseline", action="store_true",
                        help="Skip auto-generated reward-penalty baselines (routing_mode=none + reward_penalty_baseline=True)")
    parser.add_argument("--no_regular_baseline", action="store_true",
                        help="Skip auto-generated regular (do-nothing) baselines (routing_mode=none, no penalty)")
    parser.add_argument("--no_mps", action="store_true",
                        help="Skip MPS daemon management (use if MPS already running externally)")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Run in config order instead of shuffling (default: shuffle)")
    parser.add_argument("--overwrite_run", action="store_true",
                        help="Allow overwriting existing run directories (default: disambiguate with -MMDD-HHMM suffix)")
    parser.add_argument("--stagger", type=float, default=0,
                        help="Seconds to wait between launching consecutive runs (avoids HF download races)")
    # vLLM server options
    parser.add_argument("--vllm", action="store_true", default=True,
                        help="Start one vLLM server per run for generation offloading (default: on)")
    parser.add_argument("--no_vllm", action="store_true",
                        help="Disable vLLM server (use HF generate instead)")
    parser.add_argument("--vllm_async", action="store_true",
                        help="Start one shared async vLLM server per GPU (dynamic batching across runs)")
    parser.add_argument("--vllm_model", default=None,
                        help="Model for vLLM server (default: from run configs)")
    parser.add_argument("--vllm_mlp_config", default="m32",
                        help="MLP adapter preset for vLLM server (default: m32)")
    parser.add_argument("--vllm_max_experiments", type=int, default=None,
                        help="Max concurrent experiments per vLLM server (default: per_gpu)")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.05,
                        help="GPU memory fraction for vLLM (default: 0.05)")
    parser.add_argument("--vllm_dtype", default="bfloat16",
                        help="dtype for vLLM engine (default: bfloat16)")
    # Backend selection. `local` (default) is the existing per-machine
    # multiprocessing path; `modal` dispatches each pack to a Modal H100
    # container via tools.modal_train_gr. With `--backend modal --pack`, runs
    # are grouped (default: by everything except seed) and each group goes to
    # one train_many call (MPS-internal concurrency). All other sweep.py
    # features — baseline gen, caching, eval_rewards injection, incremental
    # plotting, wandb groups/IDs — are backend-agnostic.
    parser.add_argument("--backend", choices=("local", "modal"), default="local",
                        help="Launch backend (default: local). 'modal' dispatches to "
                             "tools.modal_train_gr.{train_one,train_many} on H100s.")
    parser.add_argument("--pack", action="store_true",
                        help="(--backend modal) Pack compatible runs into one container "
                             "via CUDA MPS. Default-on under --backend modal; pass this "
                             "explicitly under --backend local for documentation only "
                             "(packing is ignored locally). Default grouping: all params "
                             "equal except seed/run_name. Override via "
                             "`pack_group_keys = (...)` in the sweep config file.")
    parser.add_argument("--no_pack", action="store_true",
                        help="(--backend modal) Disable packing — one Modal container per "
                             "run. Useful when packed-MPS interactions are suspected "
                             "(e.g. debugging a single seed) or when a sweep mixes runs "
                             "that vary along many axes and packs of size 1 would be "
                             "the rule rather than the exception.")
    parser.add_argument("--max_per_pack", type=int, default=6,
                        help="(--backend modal --pack) Max runs per train_many call "
                             "(default: 6; safe for SmolLM2-135M at vllm_gpu_memory≈0.05).")
    parser.add_argument("--max_concurrent_packs", type=int, default=None,
                        help="(--backend modal) Cap on in-flight Modal calls. None = no "
                             "throttle (Modal's own quotas apply).")
    parser.add_argument("--modal_sync_interval", type=int, default=60,
                        help="(--backend modal) Seconds between background `modal volume "
                             "get --force` calls that pull the sweep's volume contents "
                             "to local disk so the existing plot/HTML generators see live "
                             "data. Default 60. Pass 0 to disable (post-hoc sync only). "
                             "Pulls everything including checkpoint-*; the in-flight plot "
                             "convenience is worth the bandwidth at our scales.")
    parser.add_argument("--modal_volume_name", default="gr-modal-pilot",
                        help="(--backend modal) Modal volume name (default: gr-modal-pilot, "
                             "matching tools/modal_train_gr.py).")
    args, remaining = parser.parse_known_args()

    # Parse remaining args as train.py overrides (applied to every run dict).
    # Set all defaults to None so we can distinguish "explicitly passed" from "default".
    from train import _make_parser as make_train_parser
    train_parser = make_train_parser()
    for action in train_parser._actions:
        action.default = None
    train_overrides, unknown = train_parser.parse_known_args(remaining)
    if unknown:
        parser.error(f"Unrecognized arguments: {' '.join(unknown)}")
    run_overrides = {k: v for k, v in vars(train_overrides).items() if v is not None}

    runs, cfg_attrs = load_sweep_config_py(args.config)

    # Apply CLI train-param overrides to every run
    if run_overrides:
        print(f"CLI overrides applied to all runs: {run_overrides}")
        for run in runs:
            run.update(run_overrides)

    # CLI overrides config file attrs; config file overrides hardcoded defaults
    per_gpu      = args.per_gpu      if args.per_gpu      is not None else (cfg_attrs["per_gpu"] or 12)
    gpus_per_run = args.gpus_per_run if args.gpus_per_run is not None else (cfg_attrs["gpus_per_run"] or 1)
    if gpus_per_run > 1:
        per_gpu = 1  # enforced: no concurrency within GPUs when doing multi-GPU runs
    output_dir   = args.output_dir   or f"./output/{args.name}"
    if not args.overwrite_run and Path(output_dir).exists():
        suffix = time.strftime("-%m%d-%H%M")
        output_dir = f"{output_dir}{suffix}"
        print(f"[SWEEP] Output dir exists, disambiguating: {output_dir}")
    # sweep_name is the disambiguated directory basename. It doubles as the
    # wandb group for every run in this sweep and the name of the auto-generated
    # colored saved view (see sweep_views.py).
    sweep_name = Path(output_dir).name
    wandb_project = args.wandb_project or "small-rl"
    no_wandb     = args.no_wandb
    no_baseline  = args.no_baseline  or cfg_attrs["no_baseline"]
    extra_baselines = cfg_attrs["extra_baselines"]
    # Default: no cache (re-run everything). --use_cache opts in.
    # Config file "no_cache = True" still forces no-cache regardless of --use_cache.
    no_cache     = cfg_attrs["no_cache"] or not args.use_cache
    retain_penalty = args.retain_penalty or cfg_attrs["retain_penalty"]
    baseline_sweep = args.baseline_sweep or cfg_attrs["baseline_sweep"]

    # Disable vLLM sleep/wake when multiple runs share a GPU
    if per_gpu > 1:
        for run in runs:
            run.setdefault("vllm_no_sleep", True)

    # Apply sweep defaults for params not present in any run
    for k, v in SWEEP_DEFAULTS.items():
        if not any(k in run for run in runs):
            for run in runs:
                run[k] = v

    from sweep_config import infer_grid_keys
    grid_keys = infer_grid_keys(runs) - {"exp_cfg", "run_name"}

    # GPU discovery + MPS + per-run vLLM server setup are all local-backend
    # concerns. On --backend modal the Modal worker handles all of this inside
    # its own container, and discover_gpus() would fail (or return host GPUs
    # we shouldn't use) on a CPU-only orchestrator box.
    if args.backend == "modal":
        gpus = []  # SweepRunner does not iterate gpus when backend=modal
        use_mps = False
        if args.no_vllm:
            # The Modal worker forces vllm_spawn=True regardless of this flag;
            # rejecting here makes the inconsistency loud rather than silent.
            parser.error("--no_vllm is incompatible with --backend modal "
                         "(the Modal container always spawns vLLM in-process)")
        if args.vllm_async:
            parser.error("--vllm_async is incompatible with --backend modal "
                         "(no shared async server inside a single H100 container)")
        # Mark every run for in-process vLLM. The Modal worker also defaults to
        # vllm_spawn=True; setting it here keeps the run dict identical to what
        # a local run would carry, which simplifies cache key parity.
        for run in runs:
            run.setdefault("vllm_spawn", True)
            run.setdefault("vllm_gpu_memory", args.vllm_gpu_memory)
        vllm_servers = {}
        vllm_async_servers = {}
    else:
        gpus = discover_gpus()
        use_mps = not args.no_mps
        # per_gpu=1 means no concurrency within a GPU, so MPS buys nothing —
        # skip the daemon launch (which also avoids failing on hosts without
        # the nvidia-cuda-mps-control binary installed).
        if use_mps and per_gpu == 1:
            print(f"[MPS] per_gpu=1, skipping MPS daemon launch")
            use_mps = False

        if use_mps and not args.dry_run:
            start_mps_daemons(gpus)

        if args.no_vllm:
            args.vllm = False
        assert not (args.vllm and args.vllm_async), "--vllm and --vllm_async are mutually exclusive"

        # Inject vllm into all runs when --vllm is set
        # Routes to colocate when:
        #   - adapter_type='none'   (no adapter, plain in-process engine)
        #   - retain_source='base'  (requires full base-weight sync, not available in spawn)
        # Otherwise uses vllm_spawn (ZMQ server).
        if args.vllm:
            for run in runs:
                needs_colocate = (run.get("adapter_type") == "none"
                                  or run.get("retain_source") == "base")
                if needs_colocate:
                    run["vllm_colocate"] = True
                else:
                    run["vllm_spawn"] = True
                run.setdefault("vllm_gpu_memory", args.vllm_gpu_memory)

        vllm_servers = {}
        vllm_async_servers = {}
    if args.vllm_async and not args.dry_run:
        vllm_model = args.vllm_model or runs[0].get("model", "HuggingFaceTB/SmolLM2-135M-Instruct")
        # max_experiments must cover all runs that will ever register on this server,
        # not just the concurrent slot count — slots are never reused after a run finishes.
        # Each run registers 3 slots: 1 training (reused as "both" eval mode) + 2 extra
        # eval slots for retain_only and forget_only (concurrent eval).
        runs_per_gpu = (len(runs) + len(gpus) - 1) // len(gpus)  # ceiling div
        max_exp = args.vllm_max_experiments or (3 * max(per_gpu, runs_per_gpu))
        vllm_async_servers = start_async_vllm_servers(
            gpus, vllm_model, args.vllm_mlp_config, max_exp, args.vllm_gpu_memory,
        )
        # Mark all runs as using async server; socket path injected per-run by SweepRunner
        for run in runs:
            run["vllm_async"] = True

    # Sweep-config-file-level overrides for pack semantics. CLI wins on
    # max_per_pack only when explicitly set (not the default).
    cfg_pack_runs       = bool(cfg_attrs["pack_runs"])
    cfg_pack_group_keys = cfg_attrs["pack_group_keys"]
    cfg_max_per_pack    = cfg_attrs["max_per_pack"]
    cfg_pack_vllm_mem   = cfg_attrs["pack_vllm_gpu_memory"]

    # Resolve effective_pack. --no_pack always wins (escape hatch); otherwise
    # any explicit "on" signal (CLI flag, config attr) turns packing on;
    # otherwise default-on for --backend modal and off for --backend local.
    # Single-seed groups still go through train_one (no MPS overhead), so
    # default-on packing is a no-cost choice — see _modal_launch_pack.
    if args.pack and args.no_pack:
        parser.error("--pack and --no_pack are mutually exclusive")
    if args.no_pack:
        effective_pack = False
    elif args.pack or cfg_pack_runs:
        effective_pack = True
    else:
        effective_pack = (args.backend == "modal")

    runner = SweepRunner(
        runs=runs,
        grid_keys=grid_keys,
        output_dir=output_dir,
        gpus=gpus,
        per_gpu=per_gpu,
        wandb_project=wandb_project,
        no_wandb=no_wandb,
        dry_run=args.dry_run,
        no_baseline=no_baseline,
        extra_baselines=extra_baselines,
        run_tag=args.run_tag,
        use_mps=use_mps,
        no_cache=no_cache,
        retain_penalty=retain_penalty,
        filter_baseline=not args.no_filter_baseline,
        reward_penalty=not args.no_reward_penalty_baseline,
        regular_baseline=not args.no_regular_baseline,
        shuffle=not args.no_shuffle,
        vllm_servers=vllm_servers,
        vllm_async_servers=vllm_async_servers,
        gpus_per_run=gpus_per_run,
        stagger=args.stagger,
        backend=args.backend,
        pack_runs=effective_pack,
        max_per_pack=(args.max_per_pack if args.max_per_pack != 6
                      else (cfg_max_per_pack or 6)),
        max_concurrent_packs=args.max_concurrent_packs,
        pack_group_keys=cfg_pack_group_keys,
        pack_vllm_gpu_memory=cfg_pack_vllm_mem,
        modal_sync_interval=args.modal_sync_interval,
        modal_volume_name=args.modal_volume_name,
        baseline_sweep=baseline_sweep,
    )
    runner.run()


if __name__ == "__main__":
    main()
