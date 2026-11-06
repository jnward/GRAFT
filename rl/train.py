"""GRPO training on SimpleStories with TRL, with optional gradient routing."""

import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/torch_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/triton_cache"))
import random
import sys
import time

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict
from trl_overrides import generate_single_turn, generate_and_score_completions
from advantages import (
    AdvConfig, compute_routed_advantages, drop_zero_advantage_microbatches,
)

# graft-port: True once the slice-2b over-routing machinery is in place — the
# per-group λ_eff cap (advantages.per_group_lam_eff), the optimizer B1 v-floor
# (SplitMomentAdamW v_floor), and the realized-step gate (w_max). λ>1 raised in
# the ctor while this was False.
_GRAFT_LAMBDA_GT1_READY = True


class Tee:
    """Write to both a file and an original stream, prepending timestamps."""
    def __init__(self, path, stream):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w")
        self.stream = stream
        self._at_line_start = True
    def write(self, data):
        if not data:
            return
        from datetime import datetime
        ts = datetime.now().strftime("[%H:%M:%S] ")
        lines = data.split("\n")
        stamped = []
        for i, line in enumerate(lines):
            if i > 0:
                stamped.append("\n")
            if self._at_line_start and line:
                stamped.append(ts + line)
            else:
                stamped.append(line)
            self._at_line_start = (i < len(lines) - 1) or data.endswith("\n")
        out = "".join(stamped)
        self.stream.write(out)
        self.file.write(out)
        self.file.flush()
    def fileno(self):
        return self.stream.fileno()
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def close(self):
        self.file.close()

from rewards import get_reward_fn, API_REWARD_NAMES
from experiment_config import ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig


def _spawn_vllm_server(model_name, mlp_config, gpu_memory, socket_path, ready_file,
                       layer_start=0.0, layer_end=1.0, layer_stride=1,
                       max_experiments=4, gpu_id=0, label=None, log_dir=None, enforce_eager=True,
                       max_num_seqs=None, async_scheduling=False, cudagraph_mode=None,
                       max_model_len=None, kv_cache_memory_bytes=None,
                       parallel_init=False):
    """Worker for per-run vLLM server subprocess (must be module-level for spawn pickling).

    Concurrent-init serialization is internal: vllm_init_slot acquires an
    exclusive flock for the duration of VLLMServer construction (the
    CUDA-heavy phase). Released before server.run() starts the request loop.
    Multiple train.py children on the same GPU (e.g. inside a train_many pack)
    queue at the lock and start their CUDA inits one at a time.
    """
    import os
    from vllm_lifecycle import vllm_init_slot, vllm_worker_setup_signals
    vllm_worker_setup_signals()  # setsid → process group leader for killpg
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # vLLM 0.17 CuMemAllocator rejects expandable_segments
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    # Capture this server process's stdout/stderr to {log_dir}/vllm_server.log —
    # the file wait_for_ready_file already points callers at, but which nothing
    # created on the train.py spawn path (Modal backend). fd-level dup2 (not just
    # sys.stdout reassignment) so vLLM's Python-logging AND any raw-fd/C-level
    # writes from EngineCore land in the file. log_dir is the run's output_dir,
    # which on Modal lives on the volume, so the log persists and syncs back.
    # Mirrors sweep.py:_vllm_server_worker (local backend), which redirects the
    # same way for the pre-spawned local server.
    if log_dir is not None:
        import sys
        os.makedirs(log_dir, exist_ok=True)
        _log_fh = open(os.path.join(log_dir, "vllm_server.log"), "w", buffering=1)
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(_log_fh.fileno(), 1)
        os.dup2(_log_fh.fileno(), 2)
        sys.stdout = _log_fh
        sys.stderr = _log_fh

    from vllm_utils import MLP_PRESETS
    from vllm_server import VLLMServer

    preset = MLP_PRESETS[mlp_config]
    init_label = label or f"vllm_train_pid{os.getpid()}"
    # Hold the per-GPU lock for the CUDA-init phase only. Release before
    # entering the unbounded serve loop. CUDA_VISIBLE_DEVICES is set by the
    # parent train.py if it remapped; gpu_id passed in is the PHYSICAL id we
    # lock against — keeps the lock path stable across processes that see
    # different remapped CUDA ids.
    # parallel_init skips the per-GPU init serialization. ONLY safe with an
    # explicit KV budget: without it, vLLM's differential free-memory
    # profiling races concurrent inits (the reason the lock exists).
    if parallel_init:
        assert kv_cache_memory_bytes is not None, (
            "parallel_init=True requires kv_cache_memory_bytes (explicit KV "
            "budget); the default profiling-based sizing is not concurrency-safe."
        )
    import contextlib
    _slot = (contextlib.nullcontext() if parallel_init
             else vllm_init_slot(gpu_id, init_label))
    with _slot:
        server = VLLMServer(
            socket_addr=socket_path,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            async_scheduling=async_scheduling,
            cudagraph_mode=cudagraph_mode,
            max_model_len=max_model_len,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            # Default 4 slots: 1 training + 3 eval adapter modes (both, retain_only,
            # forget_only). Interlaced coherence bumps this to 5 to add a coh slot.
            max_experiments=max_experiments,
            retain_neurons=preset["retain_neurons"],
            forget_neurons=preset["forget_neurons"],
            model_name=model_name,
            gpu_memory_utilization=gpu_memory,
            layer_start=layer_start,
            layer_end=layer_end,
            layer_stride=layer_stride,
        )
    # Use a sentinel file instead of multiprocessing.Event to signal readiness.
    # mp.Event uses semaphores that can't be pickled across nested spawn contexts.
    class _FileEvent:
        def __init__(self, path):
            self._path = path
        def set(self):
            open(self._path, "w").close()
    server.run(ready_event=_FileEvent(ready_file))


# ---------------------------------------------------------------------------
# Model-specific default overrides
# ---------------------------------------------------------------------------
# Keys are substring-matched against --model. First match wins.
# These are applied before argparse defaults but overridden by explicit
# CLI args or sweep params. Intended to grow as we test more models.

MODEL_DEFAULTS = {
    "Qwen3-8B": {
        "gpu_batch_size": 16,
        "max_tokens_per_microbatch": 12000,
        "lr": 7e-5,
        "num_generations": 16,
        "bf16": True,
        "top_p": 0.95,
        "max_steps": 200,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
    },
    "Qwen3-4B": {
        "gpu_batch_size": 8,
        "max_tokens_per_microbatch": 8000,
        "lr": 7e-5,
        "num_generations": 16,
        "bf16": True,
        "top_p": 0.95,
        "max_steps": 200,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
    },
}


def _apply_model_defaults(args, explicit_keys=None):
    """Apply MODEL_DEFAULTS for the first matching model key.

    Only fills in values that weren't explicitly set.  explicit_keys is a set
    of param names that were provided by the caller (sweep dict or CLI) and
    should never be overwritten — even when their value happens to match the
    argparse default.
    """
    explicit_keys = explicit_keys or set()
    for pattern, defaults in MODEL_DEFAULTS.items():
        if pattern in args.model:
            applied = []
            for key, value in defaults.items():
                if key in explicit_keys:
                    continue
                # argparse stores defaults; we detect "not explicitly set" by
                # checking if the value equals the argparse default.
                parser_default = _ARGPARSE_DEFAULTS.get(key)
                if getattr(args, key, None) == parser_default:
                    setattr(args, key, value)
                    applied.append(f"{key}={value}")
            if applied:
                print(f"Model defaults for {pattern}: {', '.join(applied)}")
            return
    # No match — use argparse defaults as-is


LORA_PRESETS = {
    # alpha=rank always (scaling factor = 1)
    "r1":    {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 1},
    "r2":    {"retain_rank": 2,  "forget_rank": 2,  "layer_stride": 1, "lora_alpha": 2},
    "r4":    {"retain_rank": 4,  "forget_rank": 4,  "layer_stride": 1, "lora_alpha": 4},
    "r8":    {"retain_rank": 8,  "forget_rank": 8,  "layer_stride": 1, "lora_alpha": 8},
    "r16":   {"retain_rank": 16, "forget_rank": 16, "layer_stride": 1, "lora_alpha": 16},
    "r32":   {"retain_rank": 32, "forget_rank": 32, "layer_stride": 1, "lora_alpha": 32},
    "r1h":   {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 2, "lora_alpha": 1},
    # Asymmetric: retain=32, forget varies
    "r32f16": {"retain_rank": 32, "forget_rank": 16, "layer_stride": 1, "lora_alpha": 32},
    "r32f4":  {"retain_rank": 32, "forget_rank": 4,  "layer_stride": 1, "lora_alpha": 32},
    "r32f1":  {"retain_rank": 32, "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 32},
    "r32f0":  {"retain_rank": 32, "forget_rank": 0,  "layer_stride": 1, "lora_alpha": 32},
    # Legacy aliases (old results reference these names)
    "r1m":   {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 1},
    "r8m":   {"retain_rank": 8,  "forget_rank": 8,  "layer_stride": 1, "lora_alpha": 8},
    "r32m":  {"retain_rank": 32, "forget_rank": 32, "layer_stride": 1, "lora_alpha": 32},
    "r1hm":  {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 2, "lora_alpha": 1},
}


# MLP adapter presets: SINGLE SOURCE OF TRUTH is vllm_utils.MLP_PRESETS. The
# vLLM-server spawn subprocess (_spawn_vllm_server) imports it from there, so a
# second copy here silently drifted — a new preset added only to train.py 404'd
# inside the server (KeyError in the spawn, 2026-07-08). Re-exported so
# `from train import MLP_PRESETS` (eval_utils, gate_compiled_engine) still works.
from vllm_utils import MLP_PRESETS


class RoutedRewardWrapper:
    """Stochastic reward wrapper for gradient routing.

    Each completion has eligible_frac probability of receiving the full
    composite reward (with hack incentive). Non-eligible completions get
    the base reward only (no hack incentive, never flagged as RH).

    Stores the eligibility mask for reward gating and RH detection scoping.

    NOTE: Prefer --hack_frac over --rh_eligible_frac for new experiments.
    hack_frac controls hackability at the prompt level based on a meaningful
    per-prompt feature (e.g. problem type, difficulty), which better models
    real-world settings where hack opportunities depend on input properties.
    rh_eligible_frac applies a random per-completion mask, which is less
    realistic and harder to interpret experimentally.
    """

    def __init__(self, full_fn, base_fn, eligible_frac=0.5):
        self.full_fn = full_fn
        self.base_fn = base_fn
        self.eligible_frac = eligible_frac
        self._last_eligible = None  # reward eligibility
        self.__name__ = getattr(full_fn, '__name__', 'routed_reward')

    def __call__(self, completions, **kwargs):
        import random
        n = len(completions)
        eligible = [random.random() < self.eligible_frac for _ in range(n)]
        self._last_eligible = eligible

        full_rewards = self.full_fn(completions=completions, **kwargs)
        base_rewards = self.base_fn(completions=completions, **kwargs)

        result = [f if e else b for f, b, e in zip(full_rewards, base_rewards, eligible)]
        self._last_rewards = result
        return result


class _Timer:
    """Context manager for timing a code block and appending to a metrics list.

    Usage:
        with self._time("timing/update"):
            ...  # timed code
    """
    def __init__(self, metrics_dict, key):
        self._list = metrics_dict.setdefault("train", {}).setdefault(key, [])
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self._list.append(time.perf_counter() - self._t0)
        return False


def _slice_batch(inputs, mask):
    """Select samples by boolean mask from input dict."""
    return {
        k: v[mask] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == mask.shape[0] else v
        for k, v in inputs.items()
    }


def _inject_detectable_into_eval_data(eval_data, classifiable_fn):
    """Set `detectable` on each eval row from the rh_detector's classifiable
    predicate. The rh_detector defines the monitoring scope, so its
    classifiability predicate is the single source of truth — overwrites any
    env-emitted `detectable` column (whose semantics may pre-date the
    configured detector; e.g. leetcode source jsonls use `detectable=hint
    presence`, which is wrong when the rh_detector gates on tags or
    difficulty).

    No-op when eval_data is empty or classifiable_fn is None (detector with
    no per-prompt feature gate).
    """
    if eval_data is None or classifiable_fn is None or not eval_data:
        return
    cols = {}
    for k in eval_data[0].keys():
        cols[k] = [row.get(k) for row in eval_data]
    flags = classifiable_fn(**cols)
    assert len(flags) == len(eval_data), (
        f"classifiable_fn returned {len(flags)} flags for {len(eval_data)} rows"
    )
    for row, f in zip(eval_data, flags):
        row["detectable"] = bool(f)


# Keys whose sequence dimension (dim=1) should be trimmed on the completion side (right-padded).
_COMPLETION_TRIM_KEYS = {"completion_ids", "completion_mask", "old_per_token_logps",
                         "ref_per_token_logps", "sampling_per_token_logps"}
# Keys whose sequence dimension (dim=1) should be trimmed on the prompt side (left-padded).
_PROMPT_TRIM_KEYS = {"prompt_ids", "prompt_mask"}


def _pack_by_tokens(token_counts, indices, max_tokens):
    """First-fit decreasing bin packing by token count.

    Args:
        token_counts: per-sample token counts (full batch).
        indices: which samples to pack (global indices into token_counts).
        max_tokens: max total tokens per bin.

    Returns:
        List of lists of global indices (each inner list = one microbatch).
    """
    if not indices:
        return []
    # Sort by token count descending for better packing
    pairs = sorted([(i, token_counts[i]) for i in indices], key=lambda x: -x[1])
    bins = []  # list of (current_tokens, [indices])
    for idx, tokens in pairs:
        placed = False
        for b in range(len(bins)):
            if bins[b][0] + tokens <= max_tokens:
                bins[b] = (bins[b][0] + tokens, bins[b][1] + [idx])
                placed = True
                break
        if not placed:
            bins.append((tokens, [idx]))
    return [b[1] for b in bins]


def _trim_and_slice(inputs, indices):
    """Slice batch by integer indices and trim padding to local max lengths.

    Completion tensors (right-padded) are trimmed to the microbatch's max actual
    completion length. Prompt tensors (left-padded) are trimmed to the microbatch's
    max actual prompt length. This reduces compute for microbatches where all
    sequences are shorter than the global max.
    """
    device = next(v.device for v in inputs.values() if isinstance(v, torch.Tensor))
    idx = torch.tensor(indices, device=device, dtype=torch.long)
    n = next(v.shape[0] for v in inputs.values()
             if isinstance(v, torch.Tensor) and v.ndim > 0)

    result = {}
    for key, val in inputs.items():
        if val is None:
            result[key] = None
        elif isinstance(val, torch.Tensor) and val.ndim > 0 and val.shape[0] == n:
            result[key] = val[idx]
        else:
            result[key] = val

    # Trim completion side (right-padded): slice [:, :max_actual_comp_len]
    if "completion_mask" in result and result["completion_mask"] is not None:
        max_comp = result["completion_mask"].sum(dim=1).max().item()
        if max_comp > 0:
            for key in _COMPLETION_TRIM_KEYS:
                if key in result and result[key] is not None and result[key].ndim >= 2:
                    result[key] = result[key][:, :max_comp]

    # Trim prompt side (left-padded): slice [:, first_real_token:]
    if "prompt_mask" in result and result["prompt_mask"] is not None:
        # Find first column that has any real token across the microbatch
        any_real = result["prompt_mask"].any(dim=0)  # (P,)
        if any_real.any():
            first_real = any_real.nonzero(as_tuple=True)[0][0].item()
            if first_real > 0:
                for key in _PROMPT_TRIM_KEYS:
                    if key in result and result[key] is not None and result[key].ndim >= 2:
                        result[key] = result[key][:, first_real:]

    return result


def _pack_for_forward(inputs, indices):
    """Pack selected samples into a flat (1, total_tokens) tensor for padding-free forward.

    Concatenates real tokens (no padding) from each sequence into a single flat
    tensor with position_ids that reset at sequence boundaries. HF flash attention
    detects this packed format via _is_packed_sequence and routes to flash_attn_varlen.

    Returns a dict with packed tensors + metadata for unpacking after forward.
    """
    device = next(v.device for v in inputs.values() if isinstance(v, torch.Tensor))

    # Fully vectorized: the per-sequence Python loop this replaces forced two
    # GPU syncs per sequence (.item() on nonzero + completion_mask.sum()), so
    # ~1000 syncs at production widths — the dominant serial CPU cost of the
    # update phase. This version syncs exactly twice regardless of width.
    # Same contiguity assumptions as the loop's [p_start:] / [:c_len] slicing:
    # prompt_mask left-padded, completion_mask right-padded.
    has_old_logps = "old_per_token_logps" in inputs and inputs["old_per_token_logps"] is not None
    has_ref_logps = "ref_per_token_logps" in inputs and inputs["ref_per_token_logps"] is not None
    n_seqs = len(indices)

    idx_t = torch.tensor(list(indices), device=device, dtype=torch.long)
    p_ids = inputs["prompt_ids"][idx_t]                 # (n, P)
    p_masks = inputs["prompt_mask"][idx_t].bool()       # (n, P)
    c_ids = inputs["completion_ids"][idx_t]             # (n, C)
    c_masks = inputs["completion_mask"][idx_t].bool()   # (n, C)

    full_ids = torch.cat([p_ids, c_ids], dim=1)         # (n, P+C)
    full_mask = torch.cat([p_masks, c_masks], dim=1)    # (n, P+C)
    # Row-major flatten preserves [seq0 tokens, seq1 tokens, ...] order.
    flat_sel = full_mask.flatten().nonzero(as_tuple=True)[0]            # sync 1
    packed_input_ids = full_ids.flatten()[flat_sel].unsqueeze(0)        # (1, T)
    # Position ids reset per sequence: within-row rank of each real token.
    pos_in_seq = full_mask.long().cumsum(dim=1) - 1
    packed_position_ids = pos_in_seq.flatten()[flat_sel].unsqueeze(0)   # (1, T)
    is_comp = torch.cat([torch.zeros_like(p_masks, dtype=torch.long),
                         c_masks.long()], dim=1)
    packed_completion_mask = is_comp.flatten()[flat_sel].unsqueeze(0)   # (1, T)

    p_lens = p_masks.sum(dim=1)
    c_lens = c_masks.sum(dim=1)
    seq_boundaries = [tuple(pc) for pc in
                      torch.stack([p_lens, c_lens], dim=1).cpu().tolist()]  # sync 2
    max_comp_len = max(c for _, c in seq_boundaries) if seq_boundaries else 0

    # Completion-side tensors are already right-padded; slice to max_comp_len
    # and zero the padding (the loop version built zeros-filled buffers).
    comp_mask_padded = c_masks[:, :max_comp_len].long()
    comp_ids_padded = c_ids[:, :max_comp_len] * comp_mask_padded
    zero = torch.zeros((), device=device)
    old_logps_padded = (torch.where(c_masks[:, :max_comp_len],
                                    inputs["old_per_token_logps"][idx_t][:, :max_comp_len], zero)
                        if has_old_logps else None)
    ref_logps_padded = (torch.where(c_masks[:, :max_comp_len],
                                    inputs["ref_per_token_logps"][idx_t][:, :max_comp_len], zero)
                        if has_ref_logps else None)

    # advantages is absent when packing for a no-grad old_logps forward (no loss).
    advantages = (inputs["advantages"][idx_t]
                  if isinstance(inputs.get("advantages"), torch.Tensor) else None)

    packed = {
        "packed_input_ids": packed_input_ids,
        "packed_position_ids": packed_position_ids,
        "packed_completion_mask": packed_completion_mask,
        "seq_boundaries": seq_boundaries,
        "completion_ids": comp_ids_padded,
        "completion_mask": comp_mask_padded,
        "advantages": advantages,
        "old_per_token_logps": old_logps_padded,
        "ref_per_token_logps": ref_logps_padded,
        "num_sequences": n_seqs,
        "max_comp_len": max_comp_len,
    }
    # graft-port slow path (λ≠1): carry the per-sequence v-stream advantage so the
    # fused 2-backward can re-run liger at a_v on the shared forward (None on the
    # fast path → key absent → single backward). The per-sample mask weights stay
    # in `inputs` (indexed there, not packed) since the fused path builds the
    # per-token routing tensors from sample indices directly.
    if isinstance(inputs.get("advantages_v"), torch.Tensor):
        packed["advantages_v"] = inputs["advantages_v"][idx_t]
    return packed


# Coherence prompt-pool sources (vs 'batch' = swap nothing). All pools are
# built in train_main via the env loader with source-specific overrides:
#   unhackable      — hack_frac=0.0 (verifiable-coherence, vercoh)
#   cond2_hackable  — prompt_condition=2 + hack_frac=1.0 (quadrant B2), rows
#                     force-marked hackable=False so forget rewards are zeroed
#   cond2_unhackable— prompt_condition=2 + hack_frac=0.0 (quadrant A2)
# The cond2 sources require the main distribution at prompt_condition=1.
COH_POOL_SOURCES = ("unhackable", "cond2_hackable", "cond2_unhackable")


class SampleGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with sample logging and optional gradient routing."""

    def __init__(self, *args, gradient_routing_enabled=False,
                 retain_params=None, forget_params=None,
                 routing_mode=None, rh_detector=None,
                 base_rh_detector=None,
                 eval_every=0, eval_metrics=None,
                 routed_reward=None,
                 filter_baseline=False,
                 reward_penalty_baseline=False,
                 reward_penalty_amount=None,
                 verbose=False, adapter_config=None,
                 renormalization_mode="retain-only",
                 split_moment=False,
                 routing_lambda=1.0,
                 graft_w_max=4.0,
                 graft_step_policy="clamp",
                 allow_approx_lora_kappa=False,
                 drop_zero_advantage=False,
                 combined_reward=None,
                 coherence="none",
                 coherence_rh_mode="filter",
                 coherence_rh_penalty=3.0,
                 coh_samples_per_rollout=0,
                 coherence_start_frac=0.0,
                 coh_train_rows=None,
                 coh_reward=None,
                 coh_prompt_source="batch",
                 coh_prompt_rows=None,
                 rh_detector_verifies_retain_samples=False,
                 rh_detector_retain_recall=1.0,
                 sort_curriculum_end_step=0,
                 verified_only_training=False,
                 rollout_forget_scale_mode="fixed",
                 forget_scale_modulation="none",
                 forget_scale_target_hack_rate=0.5,
                 forget_scale_ema_weight=0.95,
                 forget_scale_decay=0.9,
                 forget_scale_min_clamp=0.0,
                 forget_scale_decay_every=0,
                 rp_extra_retain_advantage_multiplier=1.0,
                 retain_warmup_steps=0,
                 forget_warmup_steps=0,
                 warmup_rh_detector=None,
                 rh_classifiable_fn=None,
                 vllm_client=None,
                 adapter_type="lora",
                 liger_chunk_size=64,
                 save_adapter_only=True,
                 forget_lr_mult=1.0,
                 detect_unhackable=True,
                 routing_trace_interval="when_eval",
                 routing_trace_samples=16,
                 eval_full_completions=False,
                 adapter_diag_interval="when_eval",
                 adapter_diag_level="adapter_diagnostics",
                 **kwargs):
        # Ref-model-via-disabled-adapters optimization: when the model has DualLoRA/
        # DualMLP adapters, computing ref logprobs by running the same model with
        # all adapter scales set to 0 is equivalent to running the frozen base model
        # (since DualLoRALinear.forward reduces to base_layer(x) when both scales are 0).
        # This avoids instantiating a second copy of the base model as self.ref_model.
        # We opt in by monkey-patching trl.trainer.grpo_trainer.create_model_from_path to
        # return None during super().__init__, so ref_model stays None everywhere
        # (all downstream code paths already `if self.ref_model is not None`-guard).
        # The ref forward is then routed through `disabled_dual_adapters(model)` in
        # trl_overrides.generate_and_score_completions.
        from gradient_routing import has_dual_adapters
        model_arg = kwargs["model"] if "model" in kwargs else (args[0] if len(args) > 0 else None)
        training_args = kwargs["args"] if "args" in kwargs else (args[1] if len(args) > 1 else None)
        beta = getattr(training_args, "beta", 0.0) if training_args is not None else 0.0
        _use_adapter_ref = (
            model_arg is not None
            and beta != 0.0
            and has_dual_adapters(model_arg)
        )
        if _use_adapter_ref:
            import trl.trainer.grpo_trainer as _gt
            _orig_create = _gt.create_model_from_path
            _gt.create_model_from_path = lambda *a, **kw: None
            try:
                super().__init__(*args, **kwargs)
            finally:
                _gt.create_model_from_path = _orig_create
            assert self.ref_model is None, (
                "Expected ref_model=None after create_model_from_path patch. "
                "TRL's GRPOTrainer ref-model branching may have changed."
            )
            self._ref_via_disabled_adapters = True
            print("Ref logprobs: using disabled-adapter trick (no separate ref_model copy)")
        else:
            super().__init__(*args, **kwargs)
            self._ref_via_disabled_adapters = False
        # Override TRL's default LigerFusedLinearGRPOLoss(chunk_size=1) with our chunk_size
        if self.use_liger_kernel and hasattr(self, "liger_grpo_loss"):
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                compiled=False,
                chunk_size=liger_chunk_size,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )
        print(f"Liger GRPO loss: {'active' if self.use_liger_kernel else 'inactive'}"
              f"{f', chunk_size={self.liger_grpo_loss.chunk_size}' if self.use_liger_kernel and hasattr(self, 'liger_grpo_loss') else ''}")
        # --- wandb: we own all logging, TRL's WandbCallback is removed ---
        self._samples_seen = 0
        self._pending_eval_wandb = {}
        self._eval_scoring_thread = None
        self._eval_scoring_error = None
        self.verbose = verbose
        self._adapter_type = adapter_type
        self._adapter_config = adapter_config  # saved to dual_lora_config.json in each checkpoint
        self.gradient_routing_enabled = gradient_routing_enabled
        self._retain_params = retain_params or set()
        self._forget_params = forget_params or set()
        self._forget_lr_mult = forget_lr_mult
        self._detect_unhackable = detect_unhackable
        self._routing_mode = routing_mode
        # Resolve good-pass hook target once at init:
        #   classic: good samples update both adapters (no hooks)
        #   exclusive: good samples update only retain (hook forget params)
        if gradient_routing_enabled:
            assert routing_mode in ("classic", "exclusive"), \
                f"--routing_mode must be 'classic' or 'exclusive', got {routing_mode!r}"
            self._good_pass_hooked_params = self._forget_params if routing_mode == "exclusive" else None
        else:
            self._good_pass_hooked_params = None
        self.rh_detector = rh_detector
        # Unwrapped detector for the retain-verifier path. Conceptually a
        # separate tool with perfect precision; see the wrap site for the
        # design intent. Falls back to rh_detector if no base provided.
        self.base_rh_detector = base_rh_detector if base_rh_detector is not None else rh_detector
        self.eval_every = eval_every
        self.eval_metrics = eval_metrics or {}
        self._grad_diag_file = None
        self._last_routing_eval_step = 0
        self._routed_reward = routed_reward
        self._filter_baseline = filter_baseline
        self._reward_penalty_baseline = reward_penalty_baseline
        self._reward_penalty_amount = reward_penalty_amount
        assert renormalization_mode in ("off", "retain-only", "balanced"), \
            f"invalid renormalization_mode={renormalization_mode!r}"
        if gradient_routing_enabled and renormalization_mode == "retain-only":
            # Legacy lineage (retain_mode=renormalize -> retain_renormalization
            # bool -> this enum value): the two-stream advantage semantics were
            # never re-derived against the graft-port stack, and what this mode
            # SHOULD mean there is undefined. Force an explicit choice rather
            # than silently running legacy semantics (decision 2026-07-06).
            # NOTE: 'retain-only' is still the argparse default, so GR runs must
            # now set renormalization_mode explicitly ('balanced' or 'off').
            raise NotImplementedError(
                "renormalization_mode='retain-only' is legacy and undefined on the "
                "current stack; set renormalization_mode='balanced' (principled "
                "stack) or 'off' explicitly.")
        if renormalization_mode == "balanced":
            # 'balanced' = clean (non-flagged) baseline + λ/κ forget/retain
            # redistribution applied as per-token gradient masks, for classic AND
            # exclusive routing with size-derived κ. Reject the silent no-op case loudly.
            assert gradient_routing_enabled and routing_mode in ("classic", "exclusive"), (
                "renormalization_mode='balanced' requires gradient routing "
                f"(classic or exclusive); got gr={gradient_routing_enabled}, "
                f"routing_mode={routing_mode!r}")
        self._renormalization_mode = renormalization_mode
        # Split-moment Adam (v from the pre-routing gradient, m from the routed
        # gradient). Part of the 'balanced' experiment — needs its routed grad and
        # the single-backward pre-routing capture (fused/liger path, asserted at
        # use). Works with LoRA or MLP adapters (the capture re-forwards either).
        if split_moment:
            assert renormalization_mode == "balanced", (
                "--split_moment is defined only with renormalization_mode='balanced' "
                f"(got {renormalization_mode!r}).")
        self._split_moment = split_moment
        # graft-port: λ/κ redistribution config + fail-loud geometry guard.
        self._routing_lambda = routing_lambda
        self._graft_w_max = graft_w_max
        assert graft_step_policy in ("clamp", "gate"), (
            f"graft_step_policy must be 'clamp' or 'gate', got {graft_step_policy!r}")
        self._graft_step_policy = graft_step_policy
        self._kappa_r = self._kappa_f = 1.0
        if renormalization_mode == "balanced":
            from advantages import adapter_kappas, assert_kappa_geometry
            if self._adapter_type == "lora":
                # RELAXATION (option A): κ from LoRA ranks. κ=(r_R+r_F)/r_A reuses the
                # MLP neuron-count formula with rank as the size proxy. APPROXIMATE —
                # rank is not output-space pressure under LoRA's alpha/rank-scaled,
                # bilinear (B·A) forward (true pressure is weight-dependent and
                # time-varying), so equal-pressure is exact only at the symmetric
                # alpha=rank, r_R=r_F, λ=1 point and degrades with asymmetry/over-routing.
                if not allow_approx_lora_kappa:
                    raise NotImplementedError(
                        "graft-port: LoRA + balanced-renorm routing requires the explicit "
                        "--allow_approx_lora_kappa relaxation (κ from rank ≠ output-space "
                        "pressure under LoRA's alpha/rank forward scale). Pass it to opt in, "
                        "or use MLP adapters for exact κ.")
                n_R = (adapter_config or {}).get("retain_rank", 0)
                n_F = (adapter_config or {}).get("forget_rank", 0)
                assert n_R > 0 and n_F > 0, (
                    "graft-port LoRA routing requires both adapters present (retain_rank>0 "
                    f"and forget_rank>0); got r_R={n_R}, r_F={n_F}.")
                print(f"[graft-port] RELAXATION: approximate κ from LoRA ranks "
                      f"(r_R={n_R}, r_F={n_F}); equal-pressure is approximate for LoRA.")
            else:
                n_R = (adapter_config or {}).get("retain_neurons", 0)
                n_F = (adapter_config or {}).get("forget_neurons", 0)
                assert n_R > 0 and n_F > 0, (
                    "graft-port routing requires both adapters present (retain_neurons>0 and "
                    f"forget_neurons>0); got n_R={n_R}, n_F={n_F} (retain_source='base' / forget-only "
                    "is not a routing config).")
            self._kappa_r, self._kappa_f = adapter_kappas(n_R, n_F)
            # Per-coordinate Adam step ≈ κ at λ=1 (κ enters m, not v); fail loud if the
            # geometry would exceed graft_w_max rather than silently clamp.
            assert_kappa_geometry(routing_mode, routing_lambda,
                                  self._kappa_r, self._kappa_f, graft_w_max)
            assert routing_lambda > 0.0, (
                f"routing_lambda={routing_lambda} must be > 0 (λ=0 is no-routing — "
                "use renormalization_mode='off' instead).")
            # λ≠1 takes the 2-backward v=a_v slow path (a λ-independent second
            # moment), which is the whole point of decoupling v — it REQUIRES
            # split_moment (no silent fallback to an m-stream v). λ=1 collapses to
            # master's single-backward fast path.
            if routing_lambda != 1.0:
                assert split_moment, (
                    f"routing_lambda={routing_lambda} != 1 requires --split_moment "
                    "(the slow path decouples the second moment via a 2-backward "
                    "v=a_v capture; without it v would silently ride the λ-dependent "
                    "a_m and detonate).")
            # _GRAFT_LAMBDA_GT1_READY flips True once slice-2b's optimizer v-floor +
            # realized-step gate land below; until then over-routing is hard-stopped.
            if routing_lambda > 1.0 and not _GRAFT_LAMBDA_GT1_READY:
                raise NotImplementedError(
                    f"graft-port: routing_lambda={routing_lambda} > 1 (over-routing) "
                    "is gated pending the slice-2b optimizer v-floor + realized-step "
                    "gate. Use routing_lambda ≤ 1.")
        self._drop_zero_advantage = drop_zero_advantage
        self._combined_reward = combined_reward
        # --- loss_type validation (DAPO token-level support) ---
        # The custom microbatch/fused reduction paths implement exactly two TRL
        # loss aggregations: "grpo" (per-sequence: liger's internal /n_mb is
        # cancelled by the n_mb/scale_denom backward scale) and "dapo"
        # (token-level: liger's internal /tok_mb is cancelled by a tok_mb/tok_denom
        # scale). Any other TRL loss type (bnpo/dr_grpo/cispo/sapo/luspo) would be
        # silently mis-scaled by these factors, so reject loudly here.
        assert self.loss_type in ("grpo", "dapo"), (
            f"SampleGRPOTrainer supports loss_type in {{'grpo','dapo'}}, got "
            f"{self.loss_type!r}. Other TRL loss types are not handled by the "
            "routed/dynamic reduction scaling (see _dynamic_microbatch_forward_backward)."
        )
        if self.loss_type == "dapo":
            # Token-level normalization is implemented only in the packed (liger)
            # reduction path, where we own the normalizer via the per-microbatch
            # scale. The non-packed fallback delegates to TRL's compute_loss, which
            # under dapo ALREADY normalizes by the global num_items_in_batch — our
            # extra scale would double-normalize. The penalty 2-pass path uses
            # per-sequence n/n_total scaling that has no token analog here.
            assert self.use_liger_kernel, (
                "loss_type='dapo' (token-level normalization) is implemented only "
                "in the packed liger reduction path — pass --use_liger_kernel."
            )
        # Coherence training (always interlaced: a per-sample slice within each
        # rollout, controlled by coh_samples_per_rollout > 0). The `coherence`
        # string selects the coherence reward type ('same_reward'/'judge').
        self._coherence = coherence
        self._coherence_rh_mode = coherence_rh_mode
        self._coherence_rh_penalty = coherence_rh_penalty
        self._coh_samples_per_rollout = coh_samples_per_rollout
        self._interlaced_coh = coh_samples_per_rollout > 0
        # Cross-env coherence anchor (--coh_config): coherence slots draw
        # prompts from a second env's dataset and are scored by its own
        # CombinedReward (see _maybe_swap_coherence_prompts and the
        # _calculate_rewards override). The main rh_detector never sees the
        # slice; is_rh is False there by construction.
        self._coh_env_rows = coh_train_rows
        self._coh_reward = coh_reward
        self._coh_env_iter = None
        self._last_coh_n = 0        # anchor rows scored in the LAST train scoring pass
        self._last_coh_raw = None   # their combined-reward scores (list) or None
        if coh_train_rows is not None:
            assert self._interlaced_coh, (
                "coh_train_rows requires coh_samples_per_rollout > 0")
            assert coh_reward is not None, "coh_train_rows requires coh_reward"
            assert coherence == "same_reward", (
                "cross-env coherence scores the slice with the anchor env's own "
                "reward; only coherence='same_reward' is meaningful")
            assert not rh_detector_verifies_retain_samples, (
                "verified-retain coherence and cross-env coherence both own the "
                "coherence prompt slots; pick one")
            assert retain_warmup_steps == 0 and forget_warmup_steps == 0, (
                "cross-env coherence is not implemented for warmup phases "
                "(retain warmup swaps ALL slots)")
            assert forget_scale_modulation == "none", (
                "ema_clamp consumes full-batch detector columns; not supported "
                "with cross-env coherence")
        # Delayed coherence: gate the interlaced-coh slice to begin only after
        # coherence_start_frac * max_steps (0.0 = always on). See _coherence_active.
        self._coherence_start_frac = coherence_start_frac
        # Retain-verification skyline: detector also verifies non-hack samples,
        # coherence training restricted to confirmed RETAIN samples. See
        # _build_detectable_iterator and _maybe_swap_coherence_prompts for how
        # this gets hooked into the rollout + training loops.
        self._rh_detector_verifies_retain_samples = rh_detector_verifies_retain_samples
        self._rh_detector_retain_recall = rh_detector_retain_recall
        # Length curriculum (sort env): anneal a list-length cap over training;
        # see _maybe_apply_length_curriculum. 0 = off.
        self._sort_curriculum_end_step = sort_curriculum_end_step
        self._curriculum_pool = None      # (cap, iterator) cache
        self._curriculum_lengths = None   # (n_min, n_max) from train_dataset
        self._verified_only_training = verified_only_training
        self._rollout_forget_scale_mode = rollout_forget_scale_mode
        self._last_rollout_forget_scale = 1.0
        # Idea 2: EMA-driven forget-scale clamp
        self._forget_scale_modulation = forget_scale_modulation
        self._forget_scale_target_hack_rate = forget_scale_target_hack_rate
        self._forget_scale_ema_weight = forget_scale_ema_weight
        self._forget_scale_decay = forget_scale_decay
        self._forget_scale_min_clamp = forget_scale_min_clamp
        # Decay frequency: prevent the clamp from running ahead of the EMA.
        # 0 = auto-derive as round(1 / (1 - ema_weight)).
        if forget_scale_decay_every > 0:
            self._forget_scale_decay_every = forget_scale_decay_every
        else:
            self._forget_scale_decay_every = max(1, round(1.0 / max(1e-6, 1.0 - forget_scale_ema_weight)))
        self._forget_scale_decay_counter = 0
        self._forget_scale_clamp = 1.0
        self._hack_rate_ema = None
        self._rp_extra_retain_advantage_multiplier = rp_extra_retain_advantage_multiplier
        # Idea 4: phase warmup
        self._retain_warmup_steps = retain_warmup_steps
        self._forget_warmup_steps = forget_warmup_steps
        # Idea 4(c): perfect-detector warmup
        self.warmup_rh_detector = warmup_rh_detector
        self._rh_classifiable_fn = rh_classifiable_fn
        self._detectable_iter = None  # lazy init in _build_detectable_iterator
        # Verifiable-coherence distribution: coherence slots draw prompts from
        # a separate unhackable-only pool (env loader at hack_frac=0.0) instead
        # of the training batch. See _maybe_swap_coherence_prompts.
        self._coh_prompt_source = coh_prompt_source
        self._coh_prompt_rows = coh_prompt_rows
        self._coh_prompt_iter = None  # lazy init in _ensure_unhackable_iter
        if coh_prompt_source in COH_POOL_SOURCES:
            assert not rh_detector_verifies_retain_samples, (
                f"coh_prompt_source={coh_prompt_source!r} and rh_detector_verifies_retain_samples "
                "both want to own the coherence prompt swap — mutually exclusive")
            assert self._interlaced_coh, (
                f"coh_prompt_source={coh_prompt_source!r} requires coh_samples_per_rollout > 0")
            assert coh_prompt_rows, (
                f"coh_prompt_source={coh_prompt_source!r} but no coh_prompt_rows were passed")
        else:
            assert coh_prompt_rows is None, (
                "coh_prompt_rows passed but coh_prompt_source is not a pool source")
        # Diagnostic channels.
        # routing-trace: per-rollout summary + a random per-sample subset (with
        # completion text) to routing_trace.jsonl. adapter-diag: retain/forget
        # adapter norms (+ optional per-sample grad recompute). Each fires on its
        # own interval in {every_iter, when_eval, off}.
        self._routing_trace_interval = routing_trace_interval
        self._routing_trace_samples = routing_trace_samples
        # When True, eval_samples.jsonl stores EVERY eval sample's full
        # (untruncated) prompt+completion — needed for post-hoc analysis on envs
        # whose completions exceed 400 chars (e.g. countdown_code's two-file JSON).
        self._eval_full_completions = eval_full_completions
        self._adapter_diag_interval = adapter_diag_interval
        self._adapter_diag_level = adapter_diag_level
        self._trace_file = None  # routing_trace.jsonl handle, opened lazily
        # PPS steering (training eid only; eval/coh eids get explicit alpha-0
        # calls). _run() overwrites these after trainer construction when
        # --pps_alpha > 0; the defaults keep every other construction path
        # byte-identical to pre-PPS behavior (no steering RPC is ever sent).
        self._pps_alpha = 0.0
        self._pps_layer_vecs = {}
        # vLLM HTTP server for generation
        self._vllm_client = vllm_client
        self._vllm_experiment_id = None
        self._eval_experiment_ids = None  # {mode_name: eid} for concurrent eval
        self._coh_experiment_id = None    # interlaced-coherence rollout slot (retain-only scales)
        if vllm_client is not None:
            self._vllm_experiment_id = vllm_client.register()
            print(f"[vLLM] Registered experiment {self._vllm_experiment_id}")
            # Concurrent eval: reuse training slot for "both" mode, register 2
            # extra slots for retain_only and forget_only. Requires the client to
            # support generate_multi (MLP adapter path; LoRA client does not).
            # Fail loud on slot exhaustion — silent fallback would mask the perf
            # cost and mislead about which eval path actually ran.
            if hasattr(vllm_client, "generate_multi"):
                eid_both = vllm_client.register()
                eid_retain = vllm_client.register()
                eid_forget = vllm_client.register()
                self._eval_experiment_ids = {
                    "both": eid_both,
                    "retain_only": eid_retain,
                    "forget_only": eid_forget,
                }
                print(f"[vLLM] Registered concurrent eval slots: "
                      f"both={eid_both}, retain_only={eid_retain}, forget_only={eid_forget}")
            if self._interlaced_coh:
                assert hasattr(vllm_client, "generate_multi"), (
                    "coh_samples_per_rollout > 0 requires a vLLM client with generate_multi "
                    "(MLP adapter path)"
                )
                self._coh_experiment_id = vllm_client.register()
                print(f"[vLLM] Registered interlaced-coherence slot: "
                      f"coh={self._coh_experiment_id}")
        self._save_adapter_only = save_adapter_only

        # Wrap _get_per_token_logps_and_entropies so we can capture the actor's
        # per-token logps from the gradient-flowing forward inside _compute_loss
        # (the function is also used for ref-model and rollout-old logps; we just
        # snapshot the most recent call's output and read it back from compute_loss
        # afterward to compute the rollout↔θ_new MC-KL).
        self._last_actor_per_token_logps = None
        _orig_get_logps = self._get_per_token_logps_and_entropies
        def _wrapped_get_logps(*a, **kw):
            result = _orig_get_logps(*a, **kw)
            self._last_actor_per_token_logps = result[0].detach()
            return result
        self._get_per_token_logps_and_entropies = _wrapped_get_logps

        # Phase timing: rollout (generation+scoring) vs update (gradients)
        self._last_rollout_time = 0.0
        self._accum_update_time = 0.0
        self._last_step_end_time = None
        self._last_grpo_iter_end_time = None  # set at end of last micro-batch of each logical step
        self._post_step_accum = 0.0  # accumulates optimizer.step + clip + zero_grad time

    def _clip_pre_routing_grads(self, max_norm, total_norm):
        """Apply gradient clipping to the split-moment pre-routing gradients (the
        v gradient) with the SAME coefficient torch.nn.utils.clip_grad_norm_ used
        on .grad (the m gradient): ``clip_coef = min(1, max_norm/(‖g_post‖+1e-6))``,
        where ‖g_post‖ is ``total_norm`` (the pre-clip routed-gradient norm this
        call returned). Keeps the clip a single shared event across both moments —
        the only intended deviation from stock AdamW is that v's *gradient source*
        is the pre-routing grad, not that clipping treats the moments differently.
        """
        from split_moment import clip_pre_routing_grads_
        clip_pre_routing_grads_(self.optimizer.param_groups, max_norm, total_norm)

    def create_optimizer(self):
        """Wrap optimizer.step(), clip_grad_norm_, zero_grad() and
        get_batch_samples with timing instrumentation.

        Also builds a grouped optimizer when ``forget_lr_mult != 1.0`` (retain
        group at args.learning_rate, forget group at args.learning_rate * mult).
        The forget group always uses weight_decay=0.0 regardless of
        --weight_decay (adapter params typically shouldn't be decayed).
        Optimizer class is derived from args.optim via HF's standard helper,
        so --optimizer=adamw_torch_fused/sgd/etc. all work unchanged.
        """
        # Split-moment optimizer: built here (before super().create_optimizer(),
        # which early-returns once self.optimizer is set) so its v/m-split step
        # replaces stock AdamW. Mirrors the forget_lr_mult grouping when active.
        if self.optimizer is None and getattr(self, "_split_moment", False):
            from split_moment import SplitMomentAdamW
            # grpo + dapo both supported: the participation factor c_F is sequence-
            # count under grpo (N/N_routing) and completion-token count under dapo
            # (tok_kept/tok_routing) — computed per window in _fused_forward_backward.
            assert self.loss_type in ("grpo", "dapo"), (
                f"split_moment supports loss_type in {{grpo, dapo}}, got {self.loss_type!r}.")
            betas = (self.args.adam_beta1, self.args.adam_beta2)
            eps = self.args.adam_epsilon
            wd = self.args.weight_decay
            lr = self.args.learning_rate
            # graft-port: ALWAYS build role-tagged retain/forget groups (even at
            # forget_lr_mult==1) so the participation factor + freeze apply per role.
            # Both groups decay at the run's wd — the forced forget wd=0 is dropped
            # (the freeze skips wd on frozen windows, so forget no longer needs a
            # global wd=0 to avoid drifting).
            assert self._retain_params and self._forget_params, (
                "split_moment requires both retain and forget adapters present "
                f"(got |retain|={len(self._retain_params)}, |forget|={len(self._forget_params)}).")
            retain_p = [p for p in self._retain_params if p.requires_grad]
            forget_p = [p for p in self._forget_params if p.requires_grad]
            groups = [
                {"params": retain_p, "lr": lr, "weight_decay": wd, "graft_role": "retain"},
                {"params": forget_p, "lr": lr * self._forget_lr_mult, "weight_decay": wd,
                 "graft_role": "forget"},
            ]
            self.optimizer = SplitMomentAdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=wd)
            # Direct ref for set_window (survives accelerator wrapping).
            self._split_moment_optimizer = self.optimizer
            n_p = sum(p.numel() for g in groups for p in g["params"])
            print(f"[optimizer] SplitMomentAdamW (v<-pre-routing grad, m<-routed grad, "
                  f"participation+freeze): lr={lr} forget_lr={lr * self._forget_lr_mult} "
                  f"betas={betas} eps={eps} wd={wd} ({n_p:,} params, retain+forget groups)")

        # Grouped optimizer for asymmetric retain/forget LRs. Must be built
        # before super().create_optimizer(), which early-returns when
        # self.optimizer is already set.
        if (self.optimizer is None
                and self._forget_lr_mult != 1.0
                and self._retain_params
                and self._forget_params):
            from transformers import Trainer as _HFTrainer
            retain_p = [p for p in self._retain_params if p.requires_grad]
            forget_p = [p for p in self._forget_params if p.requires_grad]
            forget_lr = self.args.learning_rate * self._forget_lr_mult
            cls, kwargs = _HFTrainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = cls(
                [
                    {"params": retain_p, "lr": self.args.learning_rate,
                     "weight_decay": self.args.weight_decay},
                    {"params": forget_p, "lr": forget_lr,
                     "weight_decay": 0.0},  # NOTE: forget group never decays; see --forget_lr_mult help
                ],
                **kwargs,
            )
            print(f"[optimizer] grouped {cls.__name__} (from --optimizer={self.args.optim}): "
                  f"retain lr={self.args.learning_rate} wd={self.args.weight_decay} "
                  f"({len(retain_p)} tensors, {sum(p.numel() for p in retain_p):,} params), "
                  f"forget lr={forget_lr} wd=0.0 (forced) "
                  f"({len(forget_p)} tensors, {sum(p.numel() for p in forget_p):,} params)")
        super().create_optimizer()
        _trainer = self

        # --- timing/post_step: optimizer + clip + zero_grad combined ---
        _orig_step = self.optimizer.step
        _orig_zero_grad = self.model.zero_grad
        _orig_clip = self.accelerator.clip_grad_norm_

        def _timed_step(*a, **kw):
            t0 = time.perf_counter()
            result = _orig_step(*a, **kw)
            _trainer._post_step_accum += time.perf_counter() - t0
            return result
        self.optimizer.step = _timed_step

        def _timed_zero_grad(*a, **kw):
            t0 = time.perf_counter()
            result = _orig_zero_grad(*a, **kw)
            _trainer._post_step_accum += time.perf_counter() - t0
            return result
        self.model.zero_grad = _timed_zero_grad

        def _timed_clip(*a, **kw):
            t0 = time.perf_counter()
            result = _orig_clip(*a, **kw)
            # Split-moment: clip v's gradient (_pre_routing_grad) by the same
            # coefficient this call applied to .grad (m's gradient), so gradient
            # clipping intervenes identically on both moments (matched
            # intervention points). result is the pre-clip total norm.
            if getattr(_trainer, "_split_moment", False):
                max_norm = a[1] if len(a) > 1 else kw.get("max_norm")
                _trainer._clip_pre_routing_grads(max_norm, result)
            _trainer._post_step_accum += time.perf_counter() - t0
            return result
        self.accelerator.clip_grad_norm_ = _timed_clip

        # --- timing/between_grpo_iters/dataloader ---
        _orig_gbs = self.get_batch_samples
        _orig_get_num_items = self._get_num_items_in_batch
        def _timed_gbs(epoch_iterator, num_batches, device):
            t0 = time.perf_counter()
            batch_samples = []
            for i in range(num_batches):
                try:
                    batch_samples.append(next(epoch_iterator))
                except StopIteration:
                    break
            num_items_in_batch = _orig_get_num_items(batch_samples, device)
            _trainer._metrics.setdefault("train", {}).setdefault(
                "timing/between_grpo_iters/dataloader", []
            ).append(time.perf_counter() - t0)
            return batch_samples, num_items_in_batch
        self.get_batch_samples = _timed_gbs

        return self.optimizer

    def get_train_dataloader(self):
        """Override: disable device_placement on the DataLoader.

        The accelerator wrapper calls send_to_device() on every batch, which
        recursively walks the entire nested dict structure (chat-format prompts,
        metadata lists, etc.) looking for tensors to move to GPU. With no tensors
        in the prompt data, this is pure overhead — ~100ms per batch × 256 batches
        = ~26s per step. Passing device_placement=False skips send_to_device.
        """
        _orig_prepare = self.accelerator.prepare_data_loader
        def _prepare_no_device_placement(dl, **kw):
            kw['device_placement'] = False
            return _orig_prepare(dl, **kw)
        self.accelerator.prepare_data_loader = _prepare_no_device_placement
        dl = super().get_train_dataloader()
        self.accelerator.prepare_data_loader = _orig_prepare
        return dl

    # --- Timestamped trace prints for diagnosing wall-clock gaps ---
    _TRACE_ENABLED = True  # flip to False to silence

    def _ts(self, label):
        """Print a timestamped trace line (for diagnosing timing gaps in train.log)."""
        if self._TRACE_ENABLED:
            import datetime
            now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            step = getattr(self.state, 'global_step', '?')
            mb = getattr(self, '_step', '?')
            print(f"[trace {now}] step={step} mb={mb} {label}", flush=True)

    def get_train_dataloader(self):
        """Override: disable device_placement on the DataLoader.

        The accelerator wrapper calls send_to_device() on every batch, which
        recursively walks the entire nested dict structure (chat-format prompts,
        metadata lists, etc.) looking for tensors to move to GPU. With no tensors
        in the prompt data, this is pure overhead — ~100ms per batch × 256 batches
        = ~16s per step.
        """
        _orig_prepare = self.accelerator.prepare_data_loader
        def _prepare_no_device_placement(dl, **kw):
            kw['device_placement'] = False
            return _orig_prepare(dl, **kw)
        self.accelerator.prepare_data_loader = _prepare_no_device_placement
        dl = super().get_train_dataloader()
        self.accelerator.prepare_data_loader = _orig_prepare
        return dl

    def _time(self, key):
        """Context manager: times a block and appends seconds to self._metrics["train"][key]."""
        return _Timer(self._metrics, key)

    def _save_checkpoint(self, model, trial):
        with self._time("timing/checkpoint"):
            if self._save_adapter_only:
                self._save_adapter_checkpoint()
            else:
                super()._save_checkpoint(model, trial)
        if self._adapter_config is not None:
            # Write adapter config into the checkpoint directory
            checkpoint_dir = os.path.join(
                self.args.output_dir,
                f"checkpoint-{self.state.global_step}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            config_path = os.path.join(checkpoint_dir, "dual_lora_config.json")
            with open(config_path, "w") as f:
                json.dump(self._adapter_config, f, indent=2)

    def _save_adapter_checkpoint(self):
        """Save only adapter (trainable) weights, optimizer, scheduler, and trainer state."""
        from safetensors.torch import save_file
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"checkpoint-{self.state.global_step}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save only trainable parameters
        adapter_state = {k: v.cpu() for k, v in self.model.named_parameters() if v.requires_grad}
        save_file(adapter_state, os.path.join(checkpoint_dir, "model.safetensors"))
        # Save optimizer and scheduler for resumption
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        # Save trainer state (has log_history with per-step metrics)
        self.state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))

    def _eval_due_this_rollout(self):
        """Check if eval should fire on this rollout.

        In this repo each rollout corresponds to a single optimizer step (the
        custom loop regenerates every step), so the eval is labeled with the
        current global_step — i.e. the weights actually used to generate the
        eval samples. Note that TRL's self.args.steps_per_generation defaults
        to gradient_accumulation_steps and does not reflect the real cadence
        here, so we must not use it as a step offset.
        """
        if not (self.eval_every > 0 and self.eval_metrics
                and self.accelerator.is_main_process):
            return False
        step = self.state.global_step
        return (step > 0
                and step - self._last_routing_eval_step >= self.eval_every)

    def _prepare_eval_for_rollout(self):
        """Load eval prompts and tokenize for piggybacked vLLM generation.

        Returns (eval_prompt_ids, eval_data, eval_max_tokens) or None if eval
        is not due or can't be piggybacked (no eval_experiment_ids).
        """
        if not self._eval_due_this_rollout():
            return None
        if self._eval_experiment_ids is None:
            return None  # fall back to _run_routing_eval in log()

        # Wait for any previous eval scoring to finish.
        self._wait_for_eval_scoring()

        env_spec = getattr(self, '_env_spec', None)
        env_args = getattr(self, '_env_args', None)
        n_eval = getattr(env_args, 'routing_eval_prompts', 64) if env_args is not None else 64

        if env_spec is not None and env_spec.load_eval_prompts is not None:
            eval_data = env_spec.load_eval_prompts(n_eval, env_args)
            eval_prompts = [d["prompt"] for d in eval_data]
            eval_max_tokens = env_spec.eval_max_tokens
        elif getattr(self, '_environment', 'stories') == 'arithmetic':
            from eval_utils import load_arithmetic_eval_prompts
            n_digits = getattr(self, '_n_digits', 3)
            eval_prompts = load_arithmetic_eval_prompts(n=n_eval, n_digits=n_digits)
            eval_data = None
            eval_max_tokens = n_digits + 2
        else:
            eval_prompts = None
            eval_data = None
            eval_max_tokens = 128

        if eval_prompts is None:
            from eval_utils import _load_eval_prompts
            eval_prompts = _load_eval_prompts(n=n_eval)

        _inject_detectable_into_eval_data(eval_data, getattr(self, "_rh_classifiable_fn", None))

        # Chat-wrap plain-string prompts for instruct models. train.py's
        # _wrap_prompts_as_chat applies the same transform to train_dataset
        # and eval_dataset; without mirroring it here, piggyback eval reaches
        # vLLM as raw strings that the model never saw during training.
        env_args = getattr(self, '_env_args', None)
        sys_prompt = getattr(env_args, 'system_prompt', "") if env_args else ""
        if self.processing_class.chat_template is not None:
            def _wrap(p):
                if isinstance(p, list):
                    return p
                msgs = []
                if sys_prompt:
                    msgs.append({"role": "system", "content": sys_prompt})
                msgs.append({"role": "user", "content": p})
                return msgs
            eval_prompts = [_wrap(p) for p in eval_prompts]

        from trl import is_conversational
        if is_conversational({"prompt": eval_prompts[0]}):
            prompt_texts = [
                self.processing_class.apply_chat_template(
                    p, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
                for p in eval_prompts
            ]
        else:
            prompt_texts = eval_prompts

        eval_prompt_ids = [
            self.processing_class.encode(p, add_special_tokens=False)
            for p in prompt_texts
        ]
        # Return the chat-template-rendered strings (not the list[dict] form) as the
        # second element. Downstream eval scoring calls detectors like llm_judge that
        # do string substitution on the prompt (.replace("{question}", prompt)), which
        # TypeErrors on list inputs. This matches what the training path does — it
        # batch_decodes prompt_ids into strings before calling the detector.
        return eval_prompt_ids, prompt_texts, eval_data, eval_max_tokens

    def _generate_single_turn(self, prompts):
        """Override: use vLLM HTTP server for generation when configured,
        otherwise fall back to bulk-CPU contention fix (trl_overrides).

        When eval is due, piggybacks eval generation onto the same vLLM session
        (wake/sleep cycle) as the training rollout, using generate_multi to batch
        training + eval prompts in a single server call.
        """
        if self._vllm_client is None:
            return generate_single_turn(self, prompts)

        from trl import is_conversational

        client = self._vllm_client
        eid = self._vllm_experiment_id
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # Check if eval should piggyback on this rollout.
        eval_info = self._prepare_eval_for_rollout() if mode == "train" else None

        # Interlaced coherence: designate the last C slots as coherence rollout.
        if mode == "train" and self._interlaced_coh:
            if self._in_retain_warmup():
                n_coh = len(prompts)  # Idea 4(a): all slots become coh
            elif self._in_forget_warmup():
                n_coh = 0  # Idea 4(b): no coh slice this phase
            else:
                n_coh = self._coh_samples_per_rollout if self._coherence_active() else 0
        else:
            n_coh = 0

        # Wake vLLM if it was sleeping: free training tensors first so vLLM
        # can reclaim the GPU memory for KV cache and weights.
        m = self._metrics.setdefault("train", {})
        if hasattr(client, 'wake_up') and not self.vllm_no_sleep:
            with self._time("timing/rollout/vllm_wake"):
                torch.cuda.empty_cache()
                m.setdefault("memory/gpu_before_wake_gb", []).append(
                    torch.cuda.memory_allocated() / 1e9)
                client.wake_up()

        # Sync weights to vLLM
        with self._time("timing/rollout/vllm_sync"):
            client.update_weights_from_model(eid, self.model)

            # PPS steering: (re)ship the steering vectors for the TRAINING eid
            # every rollout, right after the weight sync (the first sync is
            # what registers the slot, and slot (re)registration zeroes
            # steering server-side via reset_lora). The payload is tiny
            # (<= 36 x 4096 fp32 ~ 0.6 MB vs ~50 MB of adapter weights per
            # sync), so reshipping is cheaper than tracking slot lifetime and
            # is robust to any server-side slot churn. Eval/coh eids never
            # receive a nonzero steering call anywhere, so they are unsteered
            # by construction; the explicit alpha-0 calls below document and
            # enforce that invariant at the same place their scales are set.
            if self._pps_alpha > 0.0:
                client.set_steering(eid, self._pps_layer_vecs, self._pps_alpha)

            # Determine forget_scale for this rollout's routing samples.
            # Default: 1.0 (both adapters at full scale). Modes:
            #   'fixed'             - 1.0
            #   'random_uniform_0_1' - U(0, 1) sampled fresh per rollout
            #   'random_choice_0_or_0.5' - {0.0, 0.5} chosen uniformly per rollout
            # Coh slots set their own scales below (always (1, 0)).
            mode = self._rollout_forget_scale_mode
            if mode == "fixed":
                base_forget_scale = 1.0
            elif mode == "random_uniform_0_1":
                import random as _r
                base_forget_scale = _r.random()
            elif mode == "random_choice_0_or_0.5":
                import random as _r
                base_forget_scale = _r.choice([0.0, 0.5])
            else:
                raise ValueError(f"Unknown rollout_forget_scale_mode={mode!r}")
            # Idea 2: multiplicative clamp from forget_scale_modulation=ema_clamp
            # (no-op when modulation='none'; clamp stays at 1.0).
            rollout_forget_scale = base_forget_scale * self._forget_scale_clamp
            self._last_rollout_forget_scale = rollout_forget_scale
            # Log so we can see the actual scale used per step.
            _m = self._metrics.setdefault("train", {})
            _m.setdefault("rollout/forget_scale", []).append(rollout_forget_scale)
            _m.setdefault("rollout/forget_scale_base", []).append(base_forget_scale)
            _m.setdefault("rollout/forget_scale_clamp", []).append(self._forget_scale_clamp)

            if rollout_forget_scale != 1.0:
                client.set_scales(eid, 1.0, rollout_forget_scale)

            if n_coh > 0:
                client.update_weights_from_model(self._coh_experiment_id, self.model)
                # GR: coherence generates at the deployed (1,0) config. Non-GR
                # cross-env anchor (single-policy baselines): anchor rows are
                # ordinary data — generate at the training config (1,1).
                _coh_fs = 0.0 if self.gradient_routing_enabled else 1.0
                client.set_scales(self._coh_experiment_id, 1.0, _coh_fs)
                if self._pps_alpha > 0.0:
                    client.set_steering(self._coh_experiment_id, {}, 0.0)

            if eval_info is not None:
                # "both" mode reflects the *operative* scale: under
                # forget_scale_modulation=ema_clamp it's (1, clamp) so eval
                # measures the model at the same scale used during training.
                # Without modulation the helper returns 1.0, restoring (1, 1).
                both_forget_scale = self._train_forget_scale()
                modes = [("both", 1.0, both_forget_scale),
                         ("retain_only", 1.0, 0.0),
                         ("forget_only", 0.0, 1.0)]
                for mode_name, retain_scale, forget_scale in modes:
                    eval_eid = self._eval_experiment_ids[mode_name]
                    client.update_weights_from_model(eval_eid, self.model)
                    client.set_scales(eval_eid, retain_scale, forget_scale)
                    # PPS: piggybacked in-training eval is UNSTEERED — it IS
                    # the deployment trajectory.
                    if self._pps_alpha > 0.0:
                        client.set_steering(eval_eid, {}, 0.0)

        # Tokenize prompts (handle both chat and plain string formats)
        if is_conversational({"prompt": prompts[0]}):
            prompt_texts = [
                self.processing_class.apply_chat_template(
                    p, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
                for p in prompts
            ]
        else:
            prompt_texts = prompts

        prompt_ids_list = [
            self.processing_class.encode(p, add_special_tokens=False)
            for p in prompt_texts
        ]

        # Generate: TRL's RepeatSampler already expanded prompts × num_generations.
        want_logprobs = self.vllm_importance_sampling_correction or getattr(self, "fast_vllm_is_correction", False)

        # Use generate_multi whenever we need to route slots to multiple eids
        # (eval piggyback or interlaced coherence). Otherwise the simple single-eid
        # generate path is used.
        use_multi = eval_info is not None or n_coh > 0

        n_train = len(prompt_ids_list)
        n_routing = n_train - n_coh
        assert n_routing >= 0, f"n_coh ({n_coh}) > n_train ({n_train})"

        if use_multi:
            # Build combined prompt list. Order: routing training slots, then coh training slots,
            # then eval slots x 3 modes (if piggybacking).
            all_prompt_ids = list(prompt_ids_list)
            all_eids = [eid] * n_routing + [self._coh_experiment_id] * n_coh

            if eval_info is not None:
                eval_prompt_ids, eval_prompts_text, eval_data, eval_max_tokens = eval_info
                n_eval_per_mode = len(eval_prompt_ids)
                modes = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]
                for mode_name, _, _ in modes:
                    eval_eid = self._eval_experiment_ids[mode_name]
                    all_prompt_ids.extend(eval_prompt_ids)
                    all_eids.extend([eval_eid] * n_eval_per_mode)

            with self._time("timing/rollout/vllm_generate"):
                gen_result = client.generate_multi(
                    all_eids, all_prompt_ids, 1,
                    self.args.temperature, self.max_completion_length,
                    top_k=self.args.top_k, top_p=self.args.top_p,
                    return_logprobs=want_logprobs,
                    # Engine-side incremental detokenization is per-token-per-seq
                    # CPU inside the decode loop (~12% of gen at width 544).
                    # Decode here instead with one fast-tokenizer batch call —
                    # verified char-identical to engine text (skip_special_tokens
                    # matches SamplingParams' default; no stop-strings in use).
                    detokenize=False,
                )

            if want_logprobs:
                _, all_comp_ids, all_ret_prompts, all_logprobs = gen_result
            else:
                _, all_comp_ids, all_ret_prompts = gen_result
                all_logprobs = None
            all_comp_texts = self.processing_class.batch_decode(
                all_comp_ids, skip_special_tokens=True)

            # Split: first n_train are training (routing + coh), rest are eval.
            comp_texts = all_comp_texts[:n_train]
            comp_ids_list = all_comp_ids[:n_train]
            sampling_logprobs = all_logprobs[:n_train] if all_logprobs else None

            if eval_info is not None:
                samples_by_mode = {}
                offset = n_train
                for mode_name, _, _ in modes:
                    mode_samples = []
                    for j in range(n_eval_per_mode):
                        mode_samples.append({
                            "prompt": eval_prompts_text[j],
                            "completion": all_comp_texts[offset + j],
                            "completion_ids": all_comp_ids[offset + j],
                        })
                    samples_by_mode[mode_name] = mode_samples
                    offset += n_eval_per_mode
                eval_step = self.state.global_step
                self._last_routing_eval_step = eval_step
                self._dispatch_eval_scoring(samples_by_mode, eval_data, eval_step)

        else:
            # Normal path: single-eid generate.
            with self._time("timing/rollout/vllm_generate"):
                gen_result = client.generate(
                    eid, prompt_ids_list, 1,
                    self.args.temperature, self.max_completion_length,
                    top_k=self.args.top_k, top_p=self.args.top_p,
                    return_logprobs=want_logprobs,
                    detokenize=False,  # client-side batch decode below (see multi branch)
                )
            if want_logprobs:
                _, comp_ids_list, ret_prompt_ids, sampling_logprobs = gen_result
            else:
                _, comp_ids_list, ret_prompt_ids = gen_result
                sampling_logprobs = None
            comp_texts = self.processing_class.batch_decode(
                comp_ids_list, skip_special_tokens=True)

        assert len(comp_ids_list) == len(prompt_ids_list), (
            f"Expected {len(prompt_ids_list)} completions, got {len(comp_ids_list)}"
        )

        # Put vLLM to sleep: free KV cache and offload weights to CPU.
        if hasattr(client, 'sleep') and not self.vllm_no_sleep:
            with self._time("timing/rollout/vllm_sleep"):
                client.sleep(level=1)
            m.setdefault("memory/gpu_after_sleep_gb", []).append(
                torch.cuda.memory_allocated() / 1e9)

        # Return format matches TRL's _generate_single_turn:
        # (prompt_ids, completion_ids, logprobs, extra_fields)
        # When IS correction is enabled, logprobs = vLLM sampling logprobs.
        return prompt_ids_list, comp_ids_list, sampling_logprobs, {}

    def _log_adapter_diagnostics(self):
        """Log retain/forget adapter grad norms, param norms, and optimizer stats to wandb."""
        import math

        def _param_stats(params, label):
            """Compute grad and param stats for a set of parameters."""
            grad_norms = []
            param_norms = []
            max_abs_grad = 0.0
            has_nan_grad = False
            has_inf_grad = False
            for p in params:
                param_norms.append(p.data.norm().item())
                if p.grad is not None:
                    gn = p.grad.norm().item()
                    grad_norms.append(gn)
                    mag = p.grad.abs().max().item()
                    if mag > max_abs_grad:
                        max_abs_grad = mag
                    if math.isnan(gn):
                        has_nan_grad = True
                    if math.isinf(gn):
                        has_inf_grad = True
                else:
                    grad_norms.append(None)
            total_grad_norm = math.sqrt(sum(g**2 for g in grad_norms if g is not None))
            total_param_norm = math.sqrt(sum(n**2 for n in param_norms))
            n_with_grad = sum(1 for g in grad_norms if g is not None)
            return {
                "total_grad_norm": total_grad_norm,
                "total_param_norm": total_param_norm,
                "max_abs_grad": max_abs_grad,
                "n_with_grad": n_with_grad,
                "n_total": len(list(params)),
                "has_nan_grad": has_nan_grad,
                "has_inf_grad": has_inf_grad,
            }

        # Also check optimizer state for forget params
        def _optimizer_stats(params):
            """Check optimizer state (m, v) and estimate Adam update norm."""
            max_abs_m = 0.0
            max_abs_v = 0.0
            sum_abs_v = 0.0
            numel_v = 0
            has_nan_state = False
            n_with_state = 0
            update_norm_sq = 0.0
            lr = self.optimizer.param_groups[0]["lr"]
            eps = self.optimizer.param_groups[0].get("eps", 1e-8)
            for p in params:
                state = self.optimizer.state.get(p, {})
                if "exp_avg" in state:
                    n_with_state += 1
                    m_t = state["exp_avg"]
                    v_t = state["exp_avg_sq"]
                    m_max = m_t.abs().max().item()
                    v_abs = v_t.abs()
                    v_max = v_abs.max().item()
                    sum_abs_v += v_abs.sum().item()
                    numel_v += v_t.numel()
                    if m_max > max_abs_m:
                        max_abs_m = m_max
                    if v_max > max_abs_v:
                        max_abs_v = v_max
                    if (math.isnan(m_max) or math.isnan(v_max) or
                            math.isinf(m_max) or math.isinf(v_max)):
                        has_nan_state = True
                    # Estimate per-step parameter displacement: lr * ||m / (sqrt(v) + eps)||_2
                    # This is an approximation — ignores bias correction (significant early
                    # in training, ~1% after warmup) and weight decay. Treats EMA-smoothed
                    # m/v as the current step's update direction.
                    update = m_t / (v_t.sqrt() + eps)
                    update_norm_sq += update.pow(2).sum().item()
            adam_update_norm_est = lr * math.sqrt(update_norm_sq)
            mean_abs_v = (sum_abs_v / numel_v) if numel_v else 0.0
            return {
                "max_abs_m": max_abs_m,
                "max_abs_v": max_abs_v,
                "mean_abs_v": mean_abs_v,
                "has_nan_state": has_nan_state,
                "n_with_state": n_with_state,
                "adam_update_norm_est": adam_update_norm_est,
            }

        retain = _param_stats(self._retain_params, "retain")
        forget = _param_stats(self._forget_params, "forget")
        forget_opt = _optimizer_stats(self._forget_params)
        retain_opt = _optimizer_stats(self._retain_params)

        if not hasattr(self, "_metrics"):
            self._metrics = {"train": {}}
        m = self._metrics.setdefault("train", {})
        # Per-opt-batch kind (interlaced) falls back to routing (no coherence).
        last_was_coh = getattr(self, "_last_opt_batch_was_coherence", False)
        kind = "coherence" if last_was_coh else "routing"
        # Unsplit metrics (kept for backward-compat with historical dashboards)
        # plus kind-split copies so coherence vs routing dynamics are separable.
        unsplit_then_split = [
            ("diagnostics/retain_grad_norm", retain["total_grad_norm"]),
            ("diagnostics/retain_param_norm", retain["total_param_norm"]),
            ("diagnostics/forget_grad_norm", forget["total_grad_norm"]),
            ("diagnostics/forget_param_norm", forget["total_param_norm"]),
            ("diagnostics/forget_nonzero_grad_frac",
             forget["n_with_grad"] / forget["n_total"] if forget["n_total"] else 0),
            ("diagnostics/forget_max_abs_grad", forget["max_abs_grad"]),
            ("diagnostics/retain_adam_update_norm_est", retain_opt["adam_update_norm_est"]),
            ("diagnostics/forget_adam_update_norm_est", forget_opt["adam_update_norm_est"]),
            ("diagnostics/retain_max_abs_m", retain_opt["max_abs_m"]),
            ("diagnostics/forget_max_abs_m", forget_opt["max_abs_m"]),
            ("diagnostics/retain_max_abs_v", retain_opt["max_abs_v"]),
            ("diagnostics/forget_max_abs_v", forget_opt["max_abs_v"]),
            ("diagnostics/retain_mean_abs_v", retain_opt["mean_abs_v"]),
            ("diagnostics/forget_mean_abs_v", forget_opt["mean_abs_v"]),
        ]
        for key, val in unsplit_then_split:
            m.setdefault(key, []).append(val)
            m.setdefault(f"{kind}/{key}", []).append(val)

    # --- Microbatch preparation ---

    def _prepare_inputs(self, generation_batch):
        """Override TRL's _prepare_inputs for homogeneous sorting and dynamic token batching.

        Behavior depends on configuration:
        - gradient_routing + no dynamic batching: sort by is_rh to produce
          homogeneous per-device gpu-batches, split into equal chunks, then
          shuffle the chunk execution order so bad batches spread across the
          rollout's optimizer steps (prevents periodic forget-adapter Adam
          spikes from bad-sample starvation).
        - dynamic token batching (routing or not): plain shuffle, split into
          steps_per_generation chunks. Each chunk is one optimizer step's
          worth of data; training_step re-homogenizes by is_rh via bin
          packing internally.
        - interlaced coherence: a per-sample coherence slice within each rollout
          (coh_samples_per_rollout > 0), partitioned into opt batches below.
        - neither: delegate to TRL's default
        """
        if not self.model.training:
            return super()._prepare_inputs(generation_batch)

        use_dynamic = getattr(self, "_max_tokens_per_microbatch", None) is not None

        if not self.gradient_routing_enabled and not use_dynamic:
            return super()._prepare_inputs(generation_batch)

        from trl.trainer.utils import split_pixel_values_by_grid, unsplit_pixel_values_by_grid

        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            generation_batch = self._generate_and_score_completions(generation_batch)
            generation_batch = split_pixel_values_by_grid(generation_batch)

            if self._interlaced_coh and use_dynamic:
                # Interlaced coherence: the whole rollout becomes a single opt
                # batch (1 opt step per rollout regardless of
                # coh_samples_per_rollout), and microbatches inside the opt
                # batch are homogeneous-by-partition (coh / good / bad) with
                # per-mb adapter-scale management for coh microbatches in the
                # loss path.
                n = generation_batch["is_coherence"].shape[0]
                opt_bs = n // self.args.steps_per_generation
                n_opt = n // opt_bs
                assert n_opt * opt_bs == n, (
                    f"rollout_batch_size ({n}) not divisible by opt_bs ({opt_bs})")
                chunks = split_tensor_dict(generation_batch, n_opt) if n_opt > 0 else []
                perm = torch.randperm(len(chunks)).tolist()
                self._buffered_inputs = [
                    unsplit_pixel_values_by_grid(chunks[i]) for i in perm
                ]
            else:
                if self.gradient_routing_enabled and not use_dynamic:
                    # Non-dynamic routing: sort good-first bad-last so per-device
                    # gpu-batches (split_tensor_dict chunks below) are homogeneous,
                    # enabling the fast single-pass branches in training_step.
                    # The gpu-batch *order* is re-shuffled after the split so bad
                    # batches are spread across the rollout's optimizer steps —
                    # otherwise bad samples concentrate at the rollout's tail and
                    # the forget adapter's Adam state decays then spikes.
                    is_rh = generation_batch.get("is_rh")
                    if is_rh is not None:
                        n = is_rh.shape[0]
                        good_idx = (is_rh == 0).nonzero(as_tuple=True)[0]
                        bad_idx = (is_rh == 1).nonzero(as_tuple=True)[0]
                        good_idx = good_idx[torch.randperm(len(good_idx))]
                        bad_idx = bad_idx[torch.randperm(len(bad_idx))]
                        sorted_idx = torch.cat([good_idx, bad_idx])
                        for key, val in generation_batch.items():
                            if val is None:
                                continue
                            if isinstance(val, torch.Tensor) and val.ndim > 0 and val.shape[0] == n:
                                generation_batch[key] = val[sorted_idx]
                            elif isinstance(val, list) and len(val) == n:
                                generation_batch[key] = [val[i] for i in sorted_idx.tolist()]
                    else:
                        generation_batch = shuffle_sequence_dict(generation_batch)
                else:
                    # Dynamic mode (with or without routing) or non-routing:
                    # plain shuffle. Dynamic mode re-homogenizes by is_rh inside
                    # _dynamic_microbatch_training_step via bin packing, so the
                    # pre-sort is unnecessary and would concentrate bad samples
                    # in the last chunks (causing periodic forget-adapter spikes).
                    generation_batch = shuffle_sequence_dict(generation_batch)

                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]

                if self.gradient_routing_enabled and not use_dynamic:
                    # Shuffle gpu-batch execution order. Each batch stays
                    # homogeneous internally (sliced from the sorted array),
                    # but bad batches are now distributed across the rollout's
                    # optimizer steps instead of all at the tail.
                    perm = torch.randperm(len(self._buffered_inputs)).tolist()
                    self._buffered_inputs = [self._buffered_inputs[i] for i in perm]
        inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        return inputs

    # --- Sample logging (unchanged) ---

    def compute_loss(self, model, inputs, *args, **kwargs):
        # Stash one prompt+completion from the batch for logging
        # TRL uses separate prompt_ids/completion_ids, not input_ids
        self._last_sample_prompt = self.processing_class.decode(
            inputs["prompt_ids"][0], skip_special_tokens=True
        )
        self._last_sample_completion = self.processing_class.decode(
            inputs["completion_ids"][0], skip_special_tokens=True
        )
        self._last_actor_per_token_logps = None
        loss = super().compute_loss(model, inputs, *args, **kwargs)
        # Schulman k3 estimator of KL(π_rollout || π_θ_new), MC'd on the sampled
        # token. Sufficient as a "general stability" scalar — unbiased, always
        # non-negative, low variance with B*T ~1.5M samples per rollout.
        # d = log π_θ_new(a) - log π_rollout(a); k3 = exp(d) - d - 1.
        actor_logps = self._last_actor_per_token_logps
        sampling_logps = inputs.get("sampling_per_token_logps")
        if actor_logps is not None and sampling_logps is not None:
            mask = inputs["completion_mask"]
            if "tool_mask" in inputs and inputs["tool_mask"] is not None:
                mask = mask * inputs["tool_mask"]
            mask_f = mask.float()
            n = mask_f.sum().clamp(min=1.0)
            d = (actor_logps - sampling_logps) * mask_f
            kl_k3 = ((torch.exp(d) - d - 1.0) * mask_f).sum() / n
            mean_log_diff = d.sum() / n  # signed; positive means θ_new prefers the sampled token more
            mode = "train" if self.model.training else "eval"
            self._metrics[mode].setdefault("diagnostics/kl_rollout_vs_new", []).append(
                self.accelerator.gather(kl_k3).mean().item()
            )
            self._metrics[mode].setdefault("diagnostics/logp_diff_new_minus_rollout", []).append(
                self.accelerator.gather(mean_log_diff).mean().item()
            )
        self._last_actor_per_token_logps = None
        return loss

    def _packed_hidden(self, model, packed_inputs):
        """The expensive half of `_packed_compute_loss`: run the base model on the
        packed (1, total_tokens) input and repad each sequence's completion hidden
        states to (N, max_comp_len, H). Returns `(last_hs_padded, ctx)` where ctx
        carries everything `_liger_from_hidden` needs (so the cheap liger call can
        be re-run per advantage vector on ONE shared forward — the slow-path
        2-backward). Requires use_liger_kernel=True."""
        packed_ids = packed_inputs["packed_input_ids"]        # (1, T)
        packed_pos = packed_inputs["packed_position_ids"]      # (1, T)
        seq_boundaries = packed_inputs["seq_boundaries"]       # [(p_len, c_len), ...]
        n_seqs = packed_inputs["num_sequences"]
        max_comp_len = packed_inputs["max_comp_len"]

        # Forward pass: get last hidden state (no lm_head projection)
        # unwrapped_model is e.g. LlamaForCausalLM; .model is the base LlamaModel
        unwrapped_model = self.accelerator.unwrap_model(model)
        output = unwrapped_model.model(
            input_ids=packed_ids,
            position_ids=packed_pos,
            use_cache=False,
        )
        hidden_states = output.last_hidden_state  # (1, T, H)
        hidden_dim = hidden_states.shape[-1]

        # Extract per-sequence completion hidden states and repad to (N, max_comp_len, H)
        # The logit for completion token at position t is predicted by hidden state at position t-1
        # So for completion tokens at positions [p_len, p_len+c_len), we need hidden states at [p_len-1, p_len+c_len-1)
        device = hidden_states.device
        # Vectorized repad (replaces a per-sequence slice-copy loop — n_seqs
        # small kernel launches): one flat gather of every completion token's
        # predicting hidden state, scattered into the padded buffer. Token t of
        # sequence j is predicted by packed position (seq_offset + p_len - 1 + t).
        last_hs_padded = torch.zeros(n_seqs, max_comp_len, hidden_dim, device=device, dtype=hidden_states.dtype)
        p_lens_t = torch.tensor([p for p, _ in seq_boundaries], device=device)
        c_lens_t = torch.tensor([c for _, c in seq_boundaries], device=device)
        seq_lens_t = p_lens_t + c_lens_t
        offsets_t = torch.cumsum(seq_lens_t, 0) - seq_lens_t           # packed start per seq
        ar = torch.arange(max_comp_len, device=device)                 # (max_comp_len,)
        valid = ar.unsqueeze(0) < c_lens_t.unsqueeze(1)                # (n_seqs, max_comp_len)
        gather_pos = (offsets_t + p_lens_t - 1).unsqueeze(1) + ar.unsqueeze(0)  # (n_seqs, max_comp_len)
        flat_pos = gather_pos[valid]                                   # (total_comp_tokens,)
        last_hs_padded[valid] = hidden_states[0].index_select(0, flat_pos)

        ctx = {
            "unwrapped_model": unwrapped_model,
            "loss_mask": packed_inputs["completion_mask"],          # (N, max_comp_len)
            "completion_ids": packed_inputs["completion_ids"],
            "old_per_token_logps": packed_inputs["old_per_token_logps"],
            "ref_per_token_logps": packed_inputs["ref_per_token_logps"],
            "sampling_per_token_logps": packed_inputs.get("sampling_per_token_logps"),
        }
        return last_hs_padded, ctx

    def _liger_from_hidden(self, last_hs_padded, ctx, advantages, *, record_diag=True):
        """The cheap half of `_packed_compute_loss`: the liger fused GRPO loss for a
        given `advantages` vector on already-computed hidden states. Re-runnable
        per advantage vector on one shared forward (slow-path 2-backward). The
        no_grad logp/entropy/kl diagnostic recompute fires only when `record_diag`
        is True — pass False on the slow path's v-backward so it isn't double-logged."""
        unwrapped_model = ctx["unwrapped_model"]
        loss_mask = ctx["loss_mask"]

        # Call liger fused loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hs_padded,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=ctx["completion_ids"],
            attention_mask=loss_mask,
            advantages=advantages,
            bias=getattr(unwrapped_model.lm_head, 'bias', None),
            old_per_token_logps=ctx["old_per_token_logps"],
            ref_per_token_logps=ctx["ref_per_token_logps"],
        )

        # The slow-path v-backward re-runs this on the SAME hidden states with a
        # different advantage; it must log NOTHING (kl/clip_ratio/entropy already
        # logged by the m-call) — return before any metric mutation to avoid
        # double-counting.
        if not record_diag:
            return loss

        # Log metrics
        mode = "train" if self.model.training else "eval"
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]
        if self.beta != 0.0 and mean_kl is not None:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())

        # Liger's fused kernel doesn't expose per-token logps or entropy. Recover
        # them via a chunked lm_head pass on the same hidden states (no_grad — does
        # not interfere with the autograd graph liger built). Chunked over batch
        # to bound peak memory at chunk * T * V * 2 bytes (~0.9 GB for chunk=2,
        # T=1500, V=151k bf16). Adds ~5% to actor update time.
        unwrapped_model = ctx["unwrapped_model"]
        sampling_logps = ctx.get("sampling_per_token_logps")
        with torch.no_grad():
            B = last_hs_padded.shape[0]
            chunk = 2
            sel_ids = ctx["completion_ids"]
            per_token_logps = torch.zeros_like(loss_mask, dtype=torch.float32)
            entropies = torch.zeros_like(loss_mask, dtype=torch.float32)
            for b in range(0, B, chunk):
                hs = last_hs_padded[b:b + chunk]
                logits = unwrapped_model.lm_head(hs)
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                ids = sel_ids[b:b + chunk]
                per_token_logps[b:b + chunk] = log_probs.gather(
                    -1, ids.unsqueeze(-1)
                ).squeeze(-1)
                # Entropy = -Σ p log p; computed in fp32 from log_probs.
                entropies[b:b + chunk] = -(log_probs.exp() * log_probs).sum(-1)
            mask_f = loss_mask.float()
            n = mask_f.sum().clamp(min=1.0)
            mean_entropy = (entropies * mask_f).sum() / n
            self._metrics[mode].setdefault("entropy", []).append(
                self.accelerator.gather(mean_entropy).mean().item()
            )
            if sampling_logps is not None:
                d = (per_token_logps - sampling_logps) * mask_f
                kl_k3 = ((torch.exp(d) - d - 1.0) * mask_f).sum() / n
                mean_d = d.sum() / n
                self._metrics[mode].setdefault("diagnostics/kl_rollout_vs_new", []).append(
                    self.accelerator.gather(kl_k3).mean().item()
                )
                self._metrics[mode].setdefault("diagnostics/logp_diff_new_minus_rollout", []).append(
                    self.accelerator.gather(mean_d).mean().item()
                )

        return loss

    def _packed_compute_loss(self, model, packed_inputs):
        """Compute GRPO loss from a packed (padding-free) forward pass — the
        single-advantage path (fast path, non-routing, eval). `_packed_hidden`
        does the forward+repad; `_liger_from_hidden` the liger call + diagnostics.
        Requires use_liger_kernel=True."""
        last_hs_padded, ctx = self._packed_hidden(model, packed_inputs)
        return self._liger_from_hidden(last_hs_padded, ctx, packed_inputs["advantages"],
                                       record_diag=True)

    def log(self, logs, *args, **kwargs):
        prompt = getattr(self, "_last_sample_prompt", None)
        completion = getattr(self, "_last_sample_completion", None)
        if prompt is not None and completion is not None:
            step = self.state.global_step
            if self.verbose:
                print(f"\n[Sample @ step {step}] {prompt} ||| {completion}\n")

        # Log per-component raw score means and unnormalized combined reward.
        # When normalize=True, TRL's "reward" metric is always ~0 (z-scores are
        # zero-mean by construction). These raw metrics show actual progress.
        # Walk through wrappers (e.g. RoutedRewardWrapper) to find CombinedReward.
        from rewards import CombinedReward
        cr = None
        for rf in self.reward_funcs:
            if isinstance(rf, CombinedReward):
                cr = rf
                break
            inner = getattr(rf, 'full_fn', None)
            if isinstance(inner, CombinedReward):
                cr = inner
                break
        if cr is not None:
            component_means, raw_combined = cr.last_raw_metrics()
            _tm = self._metrics.setdefault("train", {})
            for name, mean in component_means.items():
                _tm.setdefault(f"reward/raw_{name}", []).append(mean)
            if raw_combined is not None:
                _tm.setdefault("reward/raw_combined", []).append(raw_combined)
            # Per-hint-type breakdown when detectable/hackable columns are present
            det_flags = getattr(self, "_last_detectable", None)
            hack_flags = getattr(self, "_last_hackable", None)
            if det_flags is not None and any(d is not None for d in det_flags):
                hack_mask = [bool(h) for h in hack_flags] if hack_flags is not None else [True] * len(det_flags)
                splits = [
                    ("detectable", [h and bool(d) for h, d in zip(hack_mask, det_flags)]),
                    ("undetectable", [h and not bool(d) for h, d in zip(hack_mask, det_flags)]),
                    ("unhackable", [not h for h in hack_mask]),
                ]
                for suffix, mask in splits:
                    if not any(mask):
                        continue
                    means, combined = cr.last_raw_metrics(mask=mask)
                    for name, mean in means.items():
                        _tm.setdefault(f"reward/raw_{name}_{suffix}", []).append(mean)
                    if combined is not None:
                        _tm.setdefault(f"reward/raw_combined_{suffix}", []).append(combined)

        # Periodic routing eval fallback: runs standalone eval when piggybacked
        # rollout-phase eval is not available (no vLLM, no eval_experiment_ids,
        # or eval_at_start). When piggybacking is available we never want this
        # path to fire — piggyback already owns eval generation for this run.
        piggyback_available = getattr(self, "_eval_experiment_ids", None) is not None
        if (self.eval_every > 0
                and self.eval_metrics
                and not piggyback_available
                and self.state.global_step - self._last_routing_eval_step >= self.eval_every
                and self.state.global_step > 0
                and self.accelerator.is_main_process):
            with self._time("timing/eval"):
                self._run_routing_eval()

        # Print timing breakdown to stdout (visible even with report_to="none").
        _tm = getattr(self, "_metrics", {}).get("train", {})
        _rollout_vals = _tm.get("timing/rollout", [])
        _update_vals = _tm.get("timing/update", [])
        if _rollout_vals and _update_vals:
            rollout = _rollout_vals[-1]
            update = _update_vals[-1]
            sync = (_tm.get("timing/rollout/vllm_sync") or [0])[-1]
            gen = (_tm.get("timing/rollout/vllm_generate") or [0])[-1]
            reward = (_tm.get("timing/compute_reward") or [0])[-1]
            old_logps = (_tm.get("timing/old_logprobs") or [0])[-1]
            ref_logps = (_tm.get("timing/ref_logprobs") or [0])[-1]
            full_step = (_tm.get("timing/full_step_s") or [0])[-1]
            between_grpo = (_tm.get("timing/between_grpo_iters") or [0])[-1]
            dataloader = (_tm.get("timing/between_grpo_iters/dataloader") or [0])[-1]
            post_step = (_tm.get("timing/post_step") or [0])[-1]
            eval_time = (_tm.get("timing/eval") or [0])[-1]
            step = self.state.global_step
            parts = [
                f"rollout={rollout:.1f}s (sync={sync:.1f}s gen={gen:.1f}s)",
                f"old_logps={old_logps:.1f}s",
                f"ref_logps={ref_logps:.1f}s",
                f"reward_t={reward:.1f}s",
                f"update={update:.1f}s",
                f"between_grpo={between_grpo:.1f}s (dataloader={dataloader:.1f}s)",
                f"post_step={post_step:.1f}s",
                f"full_step={full_step:.1f}s",
            ]
            if eval_time > 0:
                parts.append(f"eval={eval_time:.1f}s")
            # Training reward: raw combined (pre-normalization) if available,
            # plus per-component raw means. Read from _tm *before* top_level
            # extraction below would move them out.
            reward_parts = []
            rc = _tm.get("reward/raw_combined")
            if rc:
                reward_parts.append(f"combined={rc[-1]:.3f}")
            for k, v in _tm.items():
                if k.startswith("reward/raw_") and k != "reward/raw_combined" and "_detectable" not in k and "_undetectable" not in k and "_hackable" not in k and "_unhackable" not in k and v:
                    name = k[len("reward/raw_"):]
                    reward_parts.append(f"{name}={v[-1]:.3f}")
            if reward_parts:
                parts.append("reward[" + " ".join(reward_parts) + "]")
            print(f"[timing @{step}] {' '.join(parts)}")

        # Extract our custom metrics from _metrics["train"] and log them
        # directly to wandb as top-level groups (timing/, reward/, diagnostics/,
        # memory/). If left in _metrics["train"], TRL would prefix them with
        # "train/". After extraction, remove them so they aren't double-logged.
        _tm = self._metrics.get("train", {})

        # Also duplicate select TRL-native metrics into our groups
        _dup_from_trl = {
            "diagnostics/completions_mean_length": "completions/mean_length",
            "diagnostics/completions_max_length": "completions/max_length",
            "diagnostics/completions_min_length": "completions/min_length",
            "diagnostics/completions_mean_terminated_length": "completions/mean_terminated_length",
            "diagnostics/completions_max_terminated_length": "completions/max_terminated_length",
            "diagnostics/completions_min_terminated_length": "completions/min_terminated_length",
            "diagnostics/completions_truncated_ratio": "completions/clipped_ratio",
            "diagnostics/entropy": "entropy",
            "diagnostics/kl": "kl",
            "diagnostics/clip_ratio_low_mean": "clip_ratio/low_mean",
            "diagnostics/clip_ratio_low_min": "clip_ratio/low_min",
            "diagnostics/clip_ratio_high_mean": "clip_ratio/high_mean",
            "diagnostics/clip_ratio_high_max": "clip_ratio/high_max",
            "diagnostics/clip_ratio_region_mean": "clip_ratio/region_mean",
            # Liger fused-loss path (use_liger_kernel=True) writes a single combined
            # clip_ratio scalar instead of the low/high/region split — surface it under
            # the same diagnostics namespace.
            "diagnostics/clip_ratio": "clip_ratio",
        }
        for new_key, old_key in _dup_from_trl.items():
            vals = _tm.get(old_key)
            if vals:
                _tm.setdefault(new_key, []).append(vals[-1])

        # Top-level prefixes that should NOT get the train/ prefix
        _TOP_LEVEL_PREFIXES = ("timing/", "reward/", "diagnostics/", "memory/", "coherence/", "routing/", "sampling/", "offpolicy_drift/", "offpolicy_drift_baseline/", "grad_diag/", "graft/")

        top_level = {}
        keys_to_remove = []
        for key, vals in _tm.items():
            if key.startswith(_TOP_LEVEL_PREFIXES) and vals:
                top_level[key] = sum(vals) / len(vals)
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del _tm[key]

        # --- Single wandb.log call per step (WandbCallback is removed) ---
        # All wandb logging MUST go through this one call to avoid step
        # monotonicity violations.
        if self.args.report_to and "wandb" in self.args.report_to:
            import wandb
            if wandb.run is not None:
                from transformers.integrations import rewrite_logs
                # Update samples_seen: batch_size * grad_accum = samples per global_step
                self._samples_seen += (
                    self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps
                )
                wb = {
                    "train/global_step": self.state.global_step,
                    "samples_seen": self._samples_seen,
                }
                # TRL-native metrics (loss, grad_norm, lr, etc.)
                wb.update(rewrite_logs(logs))
                # Our custom top-level metrics (timing/, reward/, etc.)
                wb.update(top_level)
                # Routing eval metrics (stashed by _run_routing_eval)
                if self._pending_eval_wandb:
                    wb.update(self._pending_eval_wandb)
                    self._pending_eval_wandb = {}
                # Sample text (logged as Html to avoid trainer_state.json serialization failure)
                if prompt is not None and completion is not None:
                    wb["sample_text"] = wandb.Html(f"<pre>{prompt} ||| {completion}</pre>")
                wandb.log(wb)

        # Persist reward metrics into log_history (via super().log) so they
        # survive in trainer_state.json for post-hoc backfill.
        reward_keys = {k: v for k, v in top_level.items() if k.startswith("reward/")}
        if reward_keys:
            logs.update(reward_keys)

        # Still call super().log() for non-wandb side effects (log_history,
        # other callbacks), but WandbCallback has been removed so this won't
        # double-log to wandb.
        result = super().log(logs, *args, **kwargs)
        return result

    def _wait_for_eval_scoring(self):
        """Block until any in-flight background eval scoring completes."""
        if self._eval_scoring_thread is not None:
            self._eval_scoring_thread.join()
            self._eval_scoring_thread = None
            if self._eval_scoring_error is not None:
                err = self._eval_scoring_error
                self._eval_scoring_error = None
                raise RuntimeError(f"Background eval scoring failed: {err}") from err

    def _dispatch_eval_scoring(self, samples_by_mode, eval_data, step, gen_elapsed=0.0):
        """Dispatch eval reward scoring to a background thread.

        Called from both the piggybacked rollout path (_generate_single_turn)
        and the standalone eval path (_run_routing_eval).
        """
        import threading
        from eval_utils import score_eval_samples, format_routing_eval, get_routing_eval_metrics

        eval_metrics = self.eval_metrics
        output_dir = self.args.output_dir
        verbose = self.verbose

        def _score_in_background():
            try:
                try:
                    from envs.leetcode import use_eval_evaluator_on_this_thread
                    use_eval_evaluator_on_this_thread()
                except ImportError:
                    pass

                t_score = time.time()
                results = score_eval_samples(samples_by_mode, eval_metrics, eval_data=eval_data)
                elapsed = gen_elapsed + (time.time() - t_score)

                if verbose:
                    print(f"\n{format_routing_eval(results, step=step)}  "
                          f"(gen={gen_elapsed:.1f}s score={time.time()-t_score:.1f}s total={elapsed:.1f}s)\n")

                eval_flat = get_routing_eval_metrics(results)
                eval_flat["eval/elapsed_s"] = elapsed
                self._pending_eval_wandb = eval_flat

                record = {"step": step, "eval_elapsed_s": round(elapsed, 1)}
                for mode_name, mode_data in results.items():
                    for rname, rdata in mode_data["metrics"].items():
                        if rdata["mean"] is None:
                            continue  # metric not applicable (e.g. conditional column missing)
                        record[f"{mode_name}/{rname}"] = rdata["mean"]
                    record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
                    record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
                log_path = os.path.join(output_dir, "routing_eval.jsonl")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                # --- Diagnostic: log eval completions per mode, with per-reward
                # scores and eval_data columns. First N_FULL rows include the
                # full prompt+completion text (for side-by-side inspection vs
                # train_samples.jsonl); subsequent rows up to N_TOTAL omit the
                # text but keep id/hackable/detectable + per-metric scores so
                # any post-hoc partition (e.g. corrected detectable from a
                # different rh_detector) can be recomputed without re-running
                # the model.
                try:
                    samples_log_path = os.path.join(output_dir, "eval_samples.jsonl")
                    N_FULL = 8
                    with open(samples_log_path, "a") as f:
                        for mode_name, mode_data in results.items():
                            values_per_metric = {
                                rname: rdata.get("values", [])
                                for rname, rdata in mode_data["metrics"].items()
                            }
                            mode_samples = samples_by_mode.get(mode_name, [])
                            n_total = len(mode_samples)
                            n_full = n_total if self._eval_full_completions else N_FULL
                            for i in range(n_total):
                                rec = {"step": step, "mode": mode_name}
                                if i < n_full:
                                    p = mode_samples[i]["prompt"]
                                    p = p if isinstance(p, str) else str(p)
                                    c = mode_samples[i]["completion"]
                                    if not self._eval_full_completions:
                                        p, c = p[:400], c[:400]
                                    rec["prompt"] = p
                                    rec["completion"] = c
                                for rname, vals in values_per_metric.items():
                                    # Conditional metric wrappers (e.g.
                                    # hack_freq_detectable) return a list whose
                                    # length is the matching subset, not n_total —
                                    # those values are NOT positionally aligned
                                    # with the full eval list, so logging
                                    # vals[i] would attribute a subset score to
                                    # the wrong row. Skip those metrics here.
                                    if len(vals) != n_total:
                                        continue
                                    if vals[i] is None:
                                        continue
                                    rec[f"score/{rname}"] = float(vals[i])
                                if eval_data is not None and i < len(eval_data):
                                    for k, v in eval_data[i].items():
                                        if k == "prompt": continue
                                        if isinstance(v, (str, int, float, bool)) or v is None:
                                            rec[k] = v
                                f.write(json.dumps(rec) + "\n")
                except Exception as _e:
                    print(f"[eval_samples.jsonl logging error] {_e}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._eval_scoring_error = e

        self._eval_scoring_thread = threading.Thread(
            target=_score_in_background, daemon=True)
        self._eval_scoring_thread.start()

    def _run_routing_eval(self):
        """Run gradient routing eval: generate synchronously, score in background.

        Fallback path for eval_at_start and non-vLLM modes. When vLLM is active
        with eval_experiment_ids, eval is piggybacked onto the training rollout
        in _generate_single_turn instead — this method is not called.
        """
        from eval_utils import eval_gradient_routing

        # Wait for any previous eval scoring to finish before starting a new one.
        self._wait_for_eval_scoring()

        step = self.state.global_step
        self._last_routing_eval_step = step

        # Load environment-appropriate eval prompts and extra data
        eval_prompts = None
        eval_data = None
        env_spec = getattr(self, '_env_spec', None)
        env_args = getattr(self, '_env_args', None)
        n_eval = getattr(env_args, 'routing_eval_prompts', 64) if env_args is not None else 64

        if env_spec is not None and env_spec.load_eval_prompts is not None:
            eval_data = env_spec.load_eval_prompts(n_eval, env_args)
            eval_prompts = [d["prompt"] for d in eval_data]
            eval_max_tokens = env_spec.eval_max_tokens
        elif getattr(self, '_environment', 'stories') == 'arithmetic':
            from eval_utils import load_arithmetic_eval_prompts
            n_digits = getattr(self, '_n_digits', 3)
            eval_prompts = load_arithmetic_eval_prompts(n=n_eval, n_digits=n_digits)
            eval_max_tokens = n_digits + 2
        else:
            eval_max_tokens = 128

        _inject_detectable_into_eval_data(eval_data, getattr(self, "_rh_classifiable_fn", None))

        t0 = time.time()
        samples_by_mode = eval_gradient_routing(
            self.model, self.processing_class, self.eval_metrics,
            n_samples=n_eval, max_new_tokens=eval_max_tokens,
            temperature=self.args.temperature,
            prompts=eval_prompts, eval_data=eval_data,
            vllm_client=self._vllm_client,
            experiment_id=self._vllm_experiment_id,
            vllm_no_sleep=self.vllm_no_sleep,
            eval_experiment_ids=self._eval_experiment_ids,
            generate_only=True,
        )
        gen_elapsed = time.time() - t0
        self._dispatch_eval_scoring(samples_by_mode, eval_data, step, gen_elapsed)

    # --- Gradient routing ---

    # ----- Routing trace -----
    # The routing-trace channel writes JSONL records with a top-level "trace"
    # field ("rollout" / "sample" / "mb") to routing_trace.jsonl, so callers can
    # filter with jq 'select(.trace=="mb")'. Cadence is --routing_trace_interval;
    # callers gate their heavy work on self._diag_interval_fires(...).

    def _trace_write(self, record):
        """Append one JSONL record to routing_trace.jsonl. No-op when the
        routing-trace channel is off (callers already gate on the interval)."""
        if self._routing_trace_interval == "off":
            return
        if self._trace_file is None:
            path = os.path.join(self.args.output_dir, "routing_trace.jsonl")
            self._trace_file = open(path, "a", buffering=1)  # line-buffered
            print(f"[TRACE init] writing to {path}")
        self._trace_file.write(json.dumps(record) + "\n")

    def _log_training_trace(self, output, inputs):
        """Routing/training trace channel (fires on --routing_trace_interval).

        Writes one per-rollout summary record (whole-batch stats — cheap, no
        decode) plus per-sample records for a RANDOM subset of the rollout
        (--routing_trace_samples, default 16) with completion text inline. Only
        the sampled rows are decoded, so cost is independent of rollout size.

        Each per-sample record carries: the routing label (is_rh), ground-truth
        hack emission (hacked_gt) and the pre/post-hackable-gate hack rewards,
        the pre- and post-renormalization advantages, completion length, raw
        reward, per-component scores, and the prompt/completion text + dataset
        columns. Subsumes the old per-sample trace and train_samples.jsonl."""
        try:
            cr = self._find_combined_reward()
            n_total = output["completion_ids"].shape[0]
            G = self.num_generations
            step = int(self.state.global_step)

            is_rh_t = output.get("is_rh")
            is_coh_t = output.get("is_coherence")
            adv_t = output.get("advantages")
            pre_adv_t = output.get("advantages_pre_renorm")  # None for non-routing runs
            comp_lens = output["completion_mask"].sum(dim=1)

            # Reconstruct pre-normalization raw reward (CombinedReward sum).
            raw_rewards = None
            if cr is not None:
                try:
                    raw_rewards = self._reconstruct_raw_rewards()
                except Exception:
                    raw_rewards = None

            # Ground-truth hack emission (forget-role component, pre/post gate).
            # forget_raw>0 is the GT "model emitted the hack", independent of the
            # routing detector; None when there is no forget component.
            forget_raw_t, forget_post_t, _ = self._forget_emission_scores(cr)

            # Cross-env coherence: main-component caches cover the routing
            # prefix only — pad the anchor tail with NaN (consumers below use
            # nanmean / emit None per-sample).
            n_coh_tail = self._last_coh_n if self._coh_env_rows is not None else 0
            comp_scores = {}
            if cr is not None:
                for name, fn, _, _ in cr.components:
                    if fn._last_scores is None:
                        continue
                    if len(fn._last_scores) == n_total:
                        comp_scores[name] = fn._last_scores
                    elif n_coh_tail and len(fn._last_scores) == n_total - n_coh_tail:
                        comp_scores[name] = (list(fn._last_scores)
                                             + [float("nan")] * n_coh_tail)

            def _stats(x):
                if x is None or x.numel() == 0:
                    return {}
                xf = x.float()
                return {"mean": float(xf.mean()),
                        "std": float(xf.std(unbiased=False)),
                        "min": float(xf.min()),
                        "max": float(xf.max())}

            # --- Rollout summary (whole batch) ---
            summary = {
                "trace": "rollout",
                "step": step,
                "n_total": int(n_total),
                "num_generations": int(G),
                "n_groups": int(n_total // G) if G else 1,
                "interlaced_coh": bool(self._interlaced_coh),
                "coherence_rh_mode": self._coherence_rh_mode,
                "renormalization_mode": self._renormalization_mode,
                "routing_mode": self._routing_mode,
                # Time-varying inputs to the per-sample routing decision, so the
                # decision is reconstructable from this step's (is_rh, mode).
                "forget_scale": float(self._train_forget_scale()),
            }
            if is_coh_t is not None:
                summary["n_coherence"] = int(is_coh_t.sum().item())
            if forget_raw_t is not None:
                summary["frac_hack_emitted"] = float((forget_raw_t > 0).float().mean().item())
            if is_rh_t is not None:
                summary["frac_rh"] = float(is_rh_t.float().mean().item())
                summary["n_rh"] = int(is_rh_t.sum().item())
                rh_mask = is_rh_t.bool()
                if adv_t is not None:
                    summary["adv/all"] = _stats(adv_t)
                    if rh_mask.any():
                        summary["adv/rh"] = _stats(adv_t[rh_mask])
                    if (~rh_mask).any():
                        summary["adv/nonrh"] = _stats(adv_t[~rh_mask])
                rh_cpu = rh_mask.cpu()
                for name, scores in comp_scores.items():
                    scores_t = torch.tensor(scores, dtype=torch.float32)
                    summary[f"reward/{name}/all_mean"] = float(scores_t.nanmean())
                    if rh_cpu.any():
                        summary[f"reward/{name}/rh_mean"] = float(scores_t[rh_cpu].nanmean())
                    if (~rh_cpu).any():
                        summary[f"reward/{name}/nonrh_mean"] = float(scores_t[~rh_cpu].nanmean())
            self._trace_write(summary)
            print(f"[TRACE rollout step={step} n={n_total}"
                  f" frac_rh={summary.get('frac_rh', -1):.3f}"
                  f" frac_hack_emitted={summary.get('frac_hack_emitted', -1):.3f}"
                  f" interlaced={self._interlaced_coh}"
                  f" n_coh={summary.get('n_coherence', 0)}"
                  f" coh_mode={self._coherence_rh_mode}"
                  f" renorm={self._renormalization_mode}]")

            # --- Whole-batch per-sample vectors (cheap; no decode) ---
            is_rh_list = is_rh_t.tolist() if is_rh_t is not None else [None] * n_total
            is_coh_list = is_coh_t.tolist() if is_coh_t is not None else [False] * n_total
            adv_list = adv_t.tolist() if adv_t is not None else [None] * n_total
            pre_adv_list = pre_adv_t.tolist() if pre_adv_t is not None else None
            raw_r_list = raw_rewards.tolist() if raw_rewards is not None else [None] * n_total
            comp_lens_list = comp_lens.tolist()
            forget_raw_list = forget_raw_t.tolist() if forget_raw_t is not None else None
            forget_post_list = forget_post_t.tolist() if forget_post_t is not None else None
            detectable_list = getattr(self, "_last_detectable", None) or [None] * n_total
            hackable_list = getattr(self, "_last_hackable", None) or [None] * n_total
            # Cross-env coherence: detector columns cover the routing prefix
            # only; pad the anchor tail (semantics don't apply there).
            if len(detectable_list) < n_total:
                detectable_list = list(detectable_list) + [None] * (n_total - len(detectable_list))
            if len(hackable_list) < n_total:
                hackable_list = list(hackable_list) + [None] * (n_total - len(hackable_list))

            # Counterfactual retain_advantage_clean: per-group GRPO advantage over
            # non-RH samples only (diagnostic for baseline contamination from RH
            # rewards; for good-routing samples this equals the active advantage).
            ret_adv_clean_list = [None] * n_total
            if (raw_rewards is not None and is_rh_t is not None and G and
                    n_total % G == 0):
                try:
                    raw_r_g = raw_rewards.view(-1, G)
                    is_rh_g = is_rh_t.bool().view(-1, G)
                    clean = torch.zeros_like(raw_r_g)
                    for gi in range(raw_r_g.shape[0]):
                        good = ~is_rh_g[gi]
                        if good.sum() > 0:
                            r_good = raw_r_g[gi][good]
                            mean_g = r_good.mean()
                            # Match the pipeline's std convention (unbiased /
                            # Bessel, per the advantages.py rework) so this
                            # counterfactual equals the active advantage for
                            # no-RH groups. Single-element subset -> 0 (no var).
                            std_g = r_good.std() if r_good.numel() > 1 else r_good.new_zeros(())
                            clean[gi][good] = (r_good - mean_g) / (std_g + 1e-4)
                    ret_adv_clean_list = clean.view(-1).tolist()
                except Exception:
                    ret_adv_clean_list = [None] * n_total

            # --- Random per-sample subset (decode only these) ---
            k = min(self._routing_trace_samples, n_total)
            if k <= 0:
                return
            sample_idxs = sorted(random.sample(range(n_total), k))
            idx_t = torch.tensor(sample_idxs, device=output["completion_ids"].device)
            comp_text = self.processing_class.batch_decode(
                output["completion_ids"][idx_t], skip_special_tokens=True)
            prompt_text = self.processing_class.batch_decode(
                output["prompt_ids"][idx_t], skip_special_tokens=True)

            # Dataset columns ride on the prompt. TRL passes inputs with one row
            # per sample (each prompt repeated G times), so sample idx reads
            # inputs[idx]; a per-prompt list reads inputs[idx//G].
            has_inputs = bool(inputs) and isinstance(inputs[0], dict)
            extras_keys = ([key for key in inputs[0]
                            if key not in ("prompt", "completion", "completion_ids")]
                           if has_inputs else [])

            for j, idx in enumerate(sample_idxs):
                srec = {
                    "trace": "sample",
                    "step": step,
                    "idx": idx,
                    "group_idx": (idx // G) if G else 0,
                    "is_rh": (bool(is_rh_list[idx]) if is_rh_list[idx] is not None else None),
                    "hacked_gt": (bool(forget_raw_list[idx] > 0) if forget_raw_list is not None else None),
                    "hack_reward_even_if_unhackable": (float(forget_raw_list[idx]) if forget_raw_list is not None else None),
                    "hack_reward_obtained": (float(forget_post_list[idx]) if forget_post_list is not None else None),
                    "is_coherence": bool(is_coh_list[idx]),
                    "detectable": (bool(detectable_list[idx]) if detectable_list[idx] is not None else None),
                    "hackable": (bool(hackable_list[idx]) if hackable_list[idx] is not None else None),
                    "completion_len": int(comp_lens_list[idx]),
                    "raw_reward": raw_r_list[idx],
                    # pre==post when no renorm ran (non-routing runs).
                    "advantage_pre_renorm": (pre_adv_list[idx] if pre_adv_list is not None else adv_list[idx]),
                    "advantage": adv_list[idx],
                    "retain_advantage_clean": ret_adv_clean_list[idx],
                    "prompt": prompt_text[j][:400],
                    "completion": comp_text[j][:400],
                }
                for name, scores in comp_scores.items():
                    _sv = float(scores[idx])
                    srec[f"score/{name}"] = None if _sv != _sv else _sv
                if has_inputs:
                    grp = idx if len(inputs) == n_total else (idx // G if G else idx)
                    if grp < len(inputs) and isinstance(inputs[grp], dict):
                        for key in extras_keys:
                            v = inputs[grp].get(key)
                            if isinstance(v, (str, int, float, bool)) or v is None:
                                srec[key] = v
                self._trace_write(srec)
        except Exception as _e:
            print(f"[TRACE err] {_e}")

    @staticmethod
    def _grad_sqnorm(params):
        """Sum of squared .grad values across params. Per-microbatch delta
        isolates this microbatch's contribution to the aggregate grad norm.
        """
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().item())
        return total

    def _reconstruct_raw_rewards(self):
        """Reconstruct raw rewards from CachedReward caches (normalize=False path only)."""
        components = self._combined_reward.components  # list of (name, CachedReward, scale, role)
        n = len(components[0][1]._last_scores)
        rewards = [0.0] * n
        for name, cached, scale, role in components:
            assert cached._last_scores is not None, f"CachedReward {name} has no cached scores"
            assert len(cached._last_scores) == n, (
                f"CachedReward {name} has {len(cached._last_scores)} scores, expected {n}"
            )
            for i, s in enumerate(cached._last_scores):
                rewards[i] += s * scale
        if self._combined_reward.max_reward is not None:
            cap = self._combined_reward.max_reward
            rewards = [min(r, cap) for r in rewards]
        # Cross-env coherence: main-env caches cover the ROUTING prefix only;
        # the anchor tail's raw reward is its own combined score (already capped
        # by the anchor config's max_reward inside its __call__).
        if self._last_coh_raw is not None and self.model.training:
            rewards = rewards + [float(s) for s in self._last_coh_raw]
        device = self.accelerator.device
        return torch.tensor(rewards, dtype=torch.float32, device=device)

    def _diag_interval_fires(self, interval):
        """Whether a diagnostic channel set to `interval` should fire on the
        current global_step. Intervals: 'off' never; 'every_iter' every logging
        step; 'when_eval' every --eval_every steps (the eval cadence). Shared by
        the routing-trace and adapter-diagnostics channels."""
        if interval == "off":
            return False
        step = self.state.global_step
        if interval == "every_iter":
            return step % self.args.logging_steps == 0
        if interval == "when_eval":
            return self.eval_every > 0 and step > 0 and step % self.eval_every == 0
        raise ValueError(f"unknown diagnostic interval: {interval!r}")

    def _train_forget_scale(self):
        """Forget-adapter scale to use during training forward+backward and
        post-coh restores. Under forget_scale_modulation=ema_clamp, the
        training scale is the same clamp used at rollout-time generation, so
        the clamp actually bites the trained policy (not just the rollout
        distribution). Without this, the clamp would only affect the data
        distribution while the trained policy continued to use the full
        forget contribution at every forward+backward."""
        if self._forget_scale_modulation == "ema_clamp":
            return self._forget_scale_clamp
        return 1.0

    def _coherence_active(self):
        """Whether the interlaced-coherence slice is active at the current step.

        Coherence is deferred until coherence_start_frac * max_steps; before that
        the effective coh_samples_per_rollout is 0 (a clean non-coherence rollout).
        Always True when coherence_start_frac<=0 (default). Used at every site that
        derives the per-step coherence count, so generation and the is_coherence
        mask stay consistent within a step."""
        if self._coherence_start_frac <= 0.0:
            return True
        start = round(self._coherence_start_frac * self.args.max_steps)
        return getattr(self.state, "global_step", 0) >= start

    def _in_retain_warmup(self):
        """Idea 4(a): first N optimizer steps where the entire rollout is routed
        through the coh-side training path."""
        return (self._retain_warmup_steps > 0
                and getattr(self.state, "global_step", 0) < self._retain_warmup_steps)

    def _in_forget_warmup(self):
        """Idea 4(b): the N steps after retain warmup where only forget adapter
        updates on rh-detected samples (non-rh + coh dropped)."""
        if self._forget_warmup_steps <= 0:
            return False
        step = getattr(self.state, "global_step", 0)
        return (self._retain_warmup_steps <= step
                < self._retain_warmup_steps + self._forget_warmup_steps)

    def _maybe_apply_length_curriculum(self, inputs):
        """Length curriculum: replace prompts whose list length `n` exceeds the
        current cap with in-cap prompts drawn from the train dataset.

        cap(step) anneals linearly from the dataset's min length to its max over
        the first `sort_curriculum_end_step` optimizer steps. Replacement is per
        G-sized block (all num_generations copies of a prompt are contiguous),
        swapping the FULL row dict so reward columns (answer/input_order/n/
        hackable) stay consistent. Applied BEFORE the coherence prompt swap, so
        coherence slots may override trailing blocks with detectable (short)
        prompts — which are in-cap once cap >= the detector's max_n anyway.
        Eval prompts are appended downstream and are never touched."""
        if not self._sort_curriculum_end_step or not self.model.training:
            return inputs
        assert "n" in inputs[0], (
            "sort_curriculum_end_step > 0 requires a per-row 'n' column "
            "(list length) in the train dataset")
        if self._curriculum_lengths is None:
            ds = self.train_dataset
            rows = ds.rows if hasattr(ds, "rows") else [ds[i] for i in range(len(ds))]
            ns = [int(r["n"]) for r in rows]
            self._curriculum_lengths = (min(ns), max(ns))
            self._curriculum_rows = rows
        n_min, n_max = self._curriculum_lengths
        frac = min(1.0, self.state.global_step / self._sort_curriculum_end_step)
        cap = n_min + int(round(frac * (n_max - n_min)))
        if self._curriculum_pool is None or self._curriculum_pool[0] != cap:
            pool = [r for r in self._curriculum_rows if int(r["n"]) <= cap]
            assert pool, f"length curriculum: no train rows with n <= {cap}"
            rng = random.Random(10_000 + cap)
            def _cycle(rows_, rng_):
                while True:
                    order = list(range(len(rows_)))
                    rng_.shuffle(order)
                    for i in order:
                        yield rows_[i]
            self._curriculum_pool = (cap, _cycle(pool, rng))
            print(f"[curriculum] step {self.state.global_step}: length cap -> {cap} "
                  f"(pool {len(pool)}/{len(self._curriculum_rows)} rows)")
        it = self._curriculum_pool[1]
        G = self.num_generations
        inputs = list(inputs)
        n_swapped = 0
        for start in range(0, len(inputs), G):
            if int(inputs[start]["n"]) > cap:
                new_row = next(it)
                for g in range(G):
                    inputs[start + g] = dict(new_row)
                n_swapped += 1
        return inputs

    def _maybe_swap_coherence_prompts(self, inputs):
        """Replace the last K unique prompts in `inputs` with prompts drawn from
        the classifiable-only iterator, so interlaced-coherence slots see only
        prompts the detector can verify. K = coh_samples_per_rollout /
        num_generations. Caller passes the flat completion-level list; each
        unique prompt is repeated G=num_generations times contiguously, so K
        unique replacements cover C = K*G trailing completion slots.
        Retain warmup: swap ALL prompts (K = len(inputs)/G).
        Forget warmup: do not swap (use raw prompts, no coh slice in this phase).

        Cross-env coherence (--coh_config): same trailing-slot swap, but prompts
        come from the ANCHOR env's dataset and each injected row is tagged
        `_coh_env: True` so the _calculate_rewards override can dispatch the
        slice to the anchor reward. Mutually exclusive with verified-retain
        (asserted at init); warmup phases are asserted off.

        Two mutually-exclusive pool sources (asserted at init):
        - rh_detector_verifies_retain_samples: classifiable-only iterator
          (retain-verification skyline).
        - coh_prompt_source='unhackable': separate unhackable-distribution pool
          (verifiable-coherence setup).
        """
        if (self._coh_env_rows is not None and self._interlaced_coh
                and self.model.training):
            G = self.num_generations
            C = self._coh_samples_per_rollout if self._coherence_active() else 0
            if C == 0:
                return inputs
            assert C % G == 0, (
                f"coh_samples_per_rollout ({C}) must be a multiple of "
                f"num_generations ({G})"
            )
            K = C // G
            assert len(inputs) >= K * G, (
                f"inputs has {len(inputs)} entries, need at least {K * G}")
            it = self._ensure_coh_env_iter()
            inputs = list(inputs)
            for k in range(K):
                new_prompt = dict(next(it))
                new_prompt["_coh_env"] = True
                start = len(inputs) - (K - k) * G
                for g in range(G):
                    inputs[start + g] = dict(new_prompt)
            return inputs
        swap_active = (self._rh_detector_verifies_retain_samples or
                       self._coh_prompt_source in COH_POOL_SOURCES)
        if not (swap_active and self._interlaced_coh and self.model.training):
            return inputs
        if self._in_forget_warmup():
            return inputs
        G = self.num_generations
        if self._in_retain_warmup():
            K = len(inputs) // G
        else:
            C = self._coh_samples_per_rollout if self._coherence_active() else 0
            assert C % G == 0, (
                f"coh_samples_per_rollout ({C}) must be a multiple of num_generations ({G})"
            )
            K = C // G
        assert len(inputs) >= K * G, f"inputs has {len(inputs)} entries, need at least {K * G}"
        if self._coh_prompt_source in COH_POOL_SOURCES:
            it = self._ensure_unhackable_iter()
        else:
            it = self._ensure_detectable_iter()
        inputs = list(inputs)
        for k in range(K):
            new_prompt = next(it)
            start = len(inputs) - (K - k) * G
            for g in range(G):
                inputs[start + g] = dict(new_prompt)
        return inputs

    def _ensure_detectable_iter(self):
        """Lazy-build the classifiable-prompt iterator for retain verification.

        Calls rh_classifiable_fn on self.train_dataset row-by-row, filters to
        classifiable rows, and returns an infinite shuffled iterator yielding
        one prompt dict at a time (consumed by _maybe_swap_coherence_prompts).
        Works against both HF Datasets (ds[col] -> list) and the local
        _ListDataset wrapper (row-list behind .rows) via a duck-typed path.
        """
        if self._detectable_iter is not None:
            return self._detectable_iter
        ds = self.train_dataset
        rows = ds.rows if hasattr(ds, "rows") else [ds[i] for i in range(len(ds))]
        cols = {c: [row.get(c) for row in rows] for c in ds.column_names}
        # Classifiability mask = prompt-level "is this prompt in the detector's
        # monitoring scope". Prefer a registered classifiable predicate; fall
        # back to the env-emitted `detectable` column when the detector has no
        # classifiable (e.g. mbpp uses score_threshold + a per-row detectable
        # column as the source of truth). Mirrors the candidate-mask / eval-
        # injection logic, which already honor the column directly.
        if self._rh_classifiable_fn is not None:
            mask = self._rh_classifiable_fn(**cols)
        else:
            detectable_col = cols.get("detectable")
            assert detectable_col is not None and detectable_col[0] is not None, (
                "rh_detector_verifies_retain_samples=True but rh_classifiable_fn is None "
                "and train_dataset has no 'detectable' column — register the detector in "
                "RH_CLASSIFIABLE_REGISTRY or have the env emit a 'detectable' column."
            )
            mask = [bool(d) for d in detectable_col]
        classifiable_idx = [i for i, m in enumerate(mask) if m]
        # Restrict pool to hackable+detectable prompts. Unhackable+detectable
        # prompts produce uniform-reward groups under per-group GRPO renorm
        # (no within-group hack vs non-hack variance, retain reward is the
        # only signal and is similar across generations on the same prompt),
        # so they contribute ~zero gradient to extras-side training.
        # Restricting to hackable+detectable preserves within-group variance
        # from the mix of hack vs non-hack completions per prompt and gives
        # the extras meaningful learning signal.
        hackable_col = cols.get("hackable")
        if hackable_col is not None:
            classifiable_idx = [i for i in classifiable_idx if hackable_col[i]]
        assert len(classifiable_idx) > 0, (
            "rh_detector_verifies_retain_samples=True but no prompts in train_dataset "
            "are hackable+classifiable — check detector params and dataset coverage."
        )
        self._detectable_rows = [rows[i] for i in classifiable_idx]
        print(f"Hackable+detectable subset for retain verification: "
              f"{len(self._detectable_rows)} / {len(rows)} prompts kept")

        def _gen():
            while True:
                order = torch.randperm(len(self._detectable_rows)).tolist()
                for i in order:
                    yield self._detectable_rows[i]

        self._detectable_iter = _gen()
        return self._detectable_iter

    def _ensure_coh_env_iter(self):
        """Infinite shuffled iterator over the cross-env coherence anchor rows
        (mirrors _ensure_detectable_iter's generator; consumed one prompt dict
        at a time by _maybe_swap_coherence_prompts)."""
        if self._coh_env_iter is not None:
            return self._coh_env_iter
        rows = self._coh_env_rows
        assert rows, "cross-env coherence enabled but the anchor dataset is empty"

        def _gen():
            while True:
                order = torch.randperm(len(rows)).tolist()
                for i in order:
                    yield rows[i]

        self._coh_env_iter = _gen()
        return self._coh_env_iter

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Cross-env coherence dispatch. Coherence slots (rows tagged `_coh_env`
        by _maybe_swap_coherence_prompts) are scored by the ANCHOR env's
        CombinedReward; the routing prefix goes through TRL's stock path with
        the main reward. Keeps TRL's (N, num_reward_funcs) tensor contract.
        Without tagged rows this is exactly super()._calculate_rewards."""
        coh_flags = [bool(isinstance(inp, dict) and inp.get("_coh_env"))
                     for inp in inputs]
        n_coh = sum(coh_flags)
        if n_coh == 0:
            if self._coh_env_rows is not None:
                # No anchor rows this pass (e.g. eval-mode scoring) — reset the
                # per-pass markers so cache-padding consumers don't use stale
                # lengths from a previous rollout.
                self._last_coh_n = 0
                self._last_coh_raw = None
            return super()._calculate_rewards(
                inputs, prompts, completions, completion_ids_list)

        assert self._coh_reward is not None, "_coh_env rows present but no coh_reward"
        n = len(inputs)
        n_rt = n - n_coh
        assert n_rt > 0, "rollout contains only coherence samples"
        assert not any(coh_flags[:n_rt]) and all(coh_flags[n_rt:]), (
            "cross-env coherence rows must form the trailing contiguous block")
        assert len(self.reward_funcs) == 1, (
            "cross-env coherence assumes a single (Combined) reward func; got "
            f"{len(self.reward_funcs)}")
        device = self.accelerator.device
        rewards_per_func = torch.zeros(n, 1, device=device)
        rewards_per_func[:n_rt] = super()._calculate_rewards(
            inputs[:n_rt], prompts[:n_rt], completions[:n_rt],
            completion_ids_list[:n_rt])

        coh_inputs = inputs[n_rt:]
        keys = [k for k in coh_inputs[0]
                if k not in ("prompt", "completion", "completion_ids", "_coh_env")]
        coh_kwargs = {k: [inp.get(k) for inp in coh_inputs] for k in keys}
        coh_kwargs["trainer_state"] = self.state
        scores = self._coh_reward(
            prompts=prompts[n_rt:], completions=completions[n_rt:],
            completion_ids=completion_ids_list[n_rt:], **coh_kwargs)
        assert len(scores) == n_coh, (
            f"anchor reward returned {len(scores)} scores for {n_coh} samples")
        scores = [s if s is not None else float("nan") for s in scores]
        rewards_per_func[n_rt:, 0] = torch.tensor(
            scores, dtype=torch.float32, device=device)

        self._last_coh_n = n_coh
        self._last_coh_raw = list(scores)
        # The headline anchor channel: on-policy anchor-env combined reward at
        # the deployed (retain-only) config — coherence slots generate at (1,0).
        self._metrics.setdefault("train", {}).setdefault(
            "coherence/anchor_reward_mean", []).append(
            float(sum(scores) / len(scores)))
        return rewards_per_func

    def _ensure_unhackable_iter(self):
        """Infinite shuffled iterator over the coherence prompt pool
        (coh_prompt_source in COH_POOL_SOURCES). Pool rows were built in
        train_main by re-running the env loader with the source-specific
        overrides and all carry hackable=False (cond2_hackable pools are
        force-marked there, zeroing forget-role rewards); consumed by
        _maybe_swap_coherence_prompts."""
        if self._coh_prompt_iter is not None:
            return self._coh_prompt_iter
        rows = self._coh_prompt_rows
        assert rows, f"coh_prompt_source={self._coh_prompt_source!r} but the pool is empty"

        def _gen():
            while True:
                order = torch.randperm(len(rows)).tolist()
                for i in order:
                    yield rows[i]

        self._coh_prompt_iter = _gen()
        return self._coh_prompt_iter

    def _find_combined_reward(self):
        """Walk reward_funcs (through wrappers) to find the CombinedReward, or None."""
        from rewards import CombinedReward
        for rf in self.reward_funcs:
            if isinstance(rf, CombinedReward):
                return rf
            inner = getattr(rf, 'full_fn', None)
            if isinstance(inner, CombinedReward):
                return inner
        return None

    def _forget_emission_scores(self, cr):
        """Per-sample (local, pre-gather) forget-role reward sums.

        Returns (forget_raw, forget_post, retain_main) as float tensors [n], or
        (None, None, None) when there is no forget-role component (e.g. clean
        benchmark runs with no hack reward). forget_raw is the pre-hackable-gate
        emission (the ground-truth "did the model emit the hack"); forget_post
        is the post-gate value (zeroed where hackable=False, i.e. the reward
        actually paid). retain_main is the highest-scale retain component, used
        by the hack-vs-correct partition diagnostic. Single source of truth for
        these quantities, shared by the partition diagnostic and routing trace."""
        if cr is None:
            return None, None, None
        device = self.accelerator.device
        forget_raw = forget_post = retain_main = None
        retain_main_scale = -1.0
        pre_gate = getattr(cr, '_last_pre_gate_forget_scores', {})
        for (name, fn, scale, role) in cr.components:
            scores_post = getattr(fn, '_last_scores', None)
            if scores_post is None:
                continue
            post_t = torch.tensor(scores_post, device=device, dtype=torch.float32)
            if role == "forget":
                forget_post = post_t if forget_post is None else forget_post + post_t
                raw = pre_gate.get(name, scores_post)
                raw_t = torch.tensor(raw, device=device, dtype=torch.float32)
                forget_raw = raw_t if forget_raw is None else forget_raw + raw_t
            elif role == "retain" and scale > retain_main_scale:
                retain_main = post_t
                retain_main_scale = scale
        # Cross-env coherence: cr's caches cover the ROUTING prefix only. Pad
        # to the full batch: the anchor tail emits no main-env hack (zeros);
        # retain_main takes the anchor reward's own top-scale retain component
        # ("correct" there = solved the anchor task).
        if (self._last_coh_n and self.model.training
                and (forget_raw is not None or retain_main is not None)):
            nc = self._last_coh_n
            zpad = torch.zeros(nc, device=device, dtype=torch.float32)
            if forget_raw is not None:
                forget_raw = torch.cat([forget_raw, zpad])
            if forget_post is not None:
                forget_post = torch.cat([forget_post, zpad])
            if retain_main is not None:
                coh_ret, coh_scale = None, -1.0
                for (_cn, cfn, cscale, crole) in self._coh_reward.components:
                    csc = getattr(cfn, '_last_scores', None)
                    if crole == "retain" and cscale > coh_scale and csc is not None:
                        coh_ret, coh_scale = csc, cscale
                pad = (torch.tensor(coh_ret, device=device, dtype=torch.float32)
                       if coh_ret is not None and len(coh_ret) == nc else zpad)
                retain_main = torch.cat([retain_main, pad])
        return forget_raw, forget_post, retain_main

    def _log_hack_partition_diagnostics(self, output, mode):
        """Compute hack-vs-correct partition diagnostics on the FINAL post-rewrite
        advantages. Originally lived in trl_overrides.generate_and_score_completions
        but was reading the pre-rewrite GRPO advantages, so neg-adv stats never
        reflected the RP/filter/verified-only adjustments the optimizer actually
        consumes. Called at the end of _generate_and_score_completions."""
        cr = self._find_combined_reward()
        if cr is None:
            return

        forget_raw_local, forget_post_local, retain_main_local = \
            self._forget_emission_scores(cr)
        if forget_raw_local is None or retain_main_local is None:
            return

        forget_raw = self.accelerator.gather(forget_raw_local)
        forget_post = self.accelerator.gather(forget_post_local)
        retain_full = self.accelerator.gather(retain_main_local)
        adv_full = self.accelerator.gather(output["advantages"]).to(forget_raw.device).float()
        assert forget_raw.shape == retain_full.shape == adv_full.shape, (
            f"shape mismatch: forget_raw={forget_raw.shape} retain={retain_full.shape} "
            f"adv={adv_full.shape}"
        )
        is_hack_emitted = forget_raw > 0
        is_hack_rewarded = forget_post > 0
        is_correct = retain_full > 0

        m = self._metrics.setdefault(mode, {})
        m.setdefault("diagnostics/hack_emitted_freq", []).append(
            is_hack_emitted.float().mean().item()
        )
        m.setdefault("diagnostics/hack_rewarded_freq", []).append(
            is_hack_rewarded.float().mean().item()
        )
        # Hacks that fired but the hackable gate zeroed the reward.
        gate_suppressed = is_hack_emitted & (~is_hack_rewarded)
        m.setdefault("diagnostics/hack_gate_suppressed_freq", []).append(
            gate_suppressed.float().mean().item()
        )

        # Partition by emission (raw) × correctness; advantage stats per group.
        partitions = [
            ("hack_only", is_hack_emitted & (~is_correct)),
            ("hack_and_correct", is_hack_emitted & is_correct),
            ("correct_only", (~is_hack_emitted) & is_correct),
            ("neither", (~is_hack_emitted) & (~is_correct)),
        ]
        for label, mask in partitions:
            m.setdefault(f"diagnostics/frac_{label}", []).append(mask.float().mean().item())
            n = int(mask.sum().item())
            if n > 0:
                advs = adv_full[mask]
                m.setdefault(f"diagnostics/adv_{label}_mean", []).append(advs.mean().item())
                m.setdefault(f"diagnostics/adv_{label}_min", []).append(advs.min().item())
                m.setdefault(f"diagnostics/adv_{label}_max", []).append(advs.max().item())
                if n > 1:
                    m.setdefault(f"diagnostics/adv_{label}_std", []).append(advs.std().item())

        # Two flavours of "unrewarded hack":
        #   gate-suppressed: emitted but hackable=False so the reward was zeroed.
        #   neg-advantage: emitted and rewarded but the optimizer-visible
        #     advantage is <= 0 (GRPO group mean higher, or RP penalty pushed
        #     this sample below the group mean post-rewrite).
        n_emitted = int(is_hack_emitted.sum().item())
        if n_emitted > 0:
            emitted_advs = adv_full[is_hack_emitted]
            m.setdefault("diagnostics/adv_hack_emitted_mean", []).append(
                emitted_advs.mean().item()
            )
            m.setdefault("diagnostics/hack_emitted_neg_adv_frac", []).append(
                (emitted_advs <= 0).float().mean().item()
            )
        n_rewarded = int(is_hack_rewarded.sum().item())
        if n_rewarded > 0:
            rewarded_advs = adv_full[is_hack_rewarded]
            m.setdefault("diagnostics/hack_rewarded_neg_adv_frac", []).append(
                (rewarded_advs <= 0).float().mean().item()
            )

    def _generate_and_score_completions(self, inputs):
        """Override: pad on CPU + single .to(device), then RH detection."""
        # Plumb the current train step to reward components before scoring, so
        # step-gated rewards (e.g. a penalty delayed until N steps) can see it.
        from rewards import set_reward_train_step
        set_reward_train_step(self.state.global_step)
        inputs = self._maybe_apply_length_curriculum(inputs)
        inputs = self._maybe_swap_coherence_prompts(inputs)
        _rollout_t0 = time.perf_counter()
        output = generate_and_score_completions(self, inputs)

        # --- Step B: REINFORCE advantage override ---
        device = self.accelerator.device

        # --- Step A.5: Attach is_coherence mask (interlaced mode) ---
        # Interlaced coherence: designate the last C slots as coherence samples
        # (see _generate_single_turn for the matching prompt-ordering contract).
        # The mask rides through _prepare_inputs' split/shuffle and is consumed
        # per-opt-batch in _dynamic_microbatch_forward_backward.
        n_total = output["completion_ids"].shape[0]
        if self._interlaced_coh and self.model.training:
            if self._in_retain_warmup():
                C = n_total  # Idea 4(a): whole rollout treated as coh
            elif self._in_forget_warmup():
                C = 0  # Idea 4(b): no coh slice this phase
            else:
                C = self._coh_samples_per_rollout if self._coherence_active() else 0
            assert C <= n_total, f"coh_samples ({C}) > rollout samples ({n_total})"
            is_coherence = torch.zeros(n_total, dtype=torch.bool, device=device)
            is_coherence[n_total - C:] = True
            output["is_coherence"] = is_coherence

            # Recompute old_per_token_logps for coh slice under retain-only scales.
            # The first pass (inside generate_and_score_completions) ran at (1,1) scales;
            # coh samples were generated by vLLM at (1,0), so their old_per_token_logps
            # needs to match the (1,0) policy for the GRPO ratio to be unbiased.
            # Non-GR cross-env anchor: coh slots were generated at (1,1) like
            # everything else, so the first-pass logps already match — skip.
            if (output.get("old_per_token_logps") is not None and C > 0
                    and self.gradient_routing_enabled):
                from trl.models.utils import disable_gradient_checkpointing
                from gradient_routing import set_scales

                coh_slice = slice(n_total - C, n_total)
                sub_prompt_ids = output["prompt_ids"][coh_slice]
                sub_prompt_mask = output["prompt_mask"][coh_slice]
                sub_completion_ids = output["completion_ids"][coh_slice]
                sub_completion_mask = output["completion_mask"][coh_slice]
                sub_prompt_completion_ids = torch.cat([sub_prompt_ids, sub_completion_ids], dim=1)
                sub_attention_mask = torch.cat([sub_prompt_mask, sub_completion_mask], dim=1)
                sub_logits_to_keep = sub_completion_ids.shape[1]
                sub_batch_size = getattr(self, "_scoring_batch_size",
                                         self.args.per_device_train_batch_size)
                # The padded (no-pack) kernel materializes full-vocab fp32
                # logits for each chunk (rows x padded_len x vocab x 4B), and
                # the slice inherits BATCH-GLOBAL padding — one long batch
                # widens every row, so any fixed row count eventually OOMs
                # (observed twice at 82 GiB, gbs 16 AND 8). Cap chunk rows by
                # the padded width so a chunk's logits stay within the
                # microbatch token budget.
                _tok_budget = (getattr(self, "_old_logps_packed_budget", None)
                               or 24000)
                _cap = max(1, _tok_budget // sub_prompt_completion_ids.shape[1])
                if _cap < sub_batch_size:
                    print(f"[coh old_logps] padded_len="
                          f"{sub_prompt_completion_ids.shape[1]}: capping chunk "
                          f"{sub_batch_size} -> {_cap} rows", flush=True)
                    sub_batch_size = _cap

                set_scales(self.model, retain_scale=1.0, forget_scale=0.0)
                try:
                    with torch.no_grad(), disable_gradient_checkpointing(
                        self.model, self.args.gradient_checkpointing_kwargs
                    ):
                        coh_old_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            sub_prompt_completion_ids,
                            sub_attention_mask,
                            sub_logits_to_keep,
                            sub_batch_size,
                        )
                finally:
                    set_scales(self.model, retain_scale=1.0,
                               forget_scale=self._train_forget_scale())
                output["old_per_token_logps"][coh_slice] = coh_old_logps
        else:
            output["is_coherence"] = torch.zeros(n_total, dtype=torch.bool, device=device)

        # --- Step C: RH detection ---
        _t_rh_start = time.perf_counter()

        # Observe-only grad-diag: run the per-sample diagnostic on a vanilla
        # (do-nothing) dual-adapter run — no routing / filter / RP, so training is
        # unperturbed (the advantage path below is driven by the mode flags, all
        # False here) — but still label samples (is_rh + GT) so _run_grad_diagnostic
        # can measure the UNMASKED adapter separation that GR's masking is the
        # counterfactual to. Requires forget params (dual adapters) + a detector.
        grad_diag_observe = bool(
            self._adapter_diag_level == "per_sample_recompute"
            and self.rh_detector is not None
            and self._forget_params
            and not self.gradient_routing_enabled
            and not self._filter_baseline
            and not self._reward_penalty_baseline
            and not self._verified_only_training)
        needs_detection = (self.gradient_routing_enabled
                           or self._filter_baseline
                           or self._reward_penalty_baseline
                           or self._verified_only_training
                           or grad_diag_observe)
        if needs_detection and self.rh_detector is not None:
            n_samples = output["completion_ids"].shape[0]

            # Cross-env coherence: the detector (and the main CombinedReward's
            # score caches it reads) covers only the ROUTING prefix — anchor-env
            # rows are never detection candidates; their is_rh is padded False
            # below. n_det_samples == n_samples in all other configurations.
            n_det_samples = n_samples
            if self._coh_env_rows is not None and self._last_coh_n:
                n_det_samples = n_samples - self._last_coh_n
                assert n_det_samples > 0, "detection prefix is empty"

            # Gather dataset columns for conditional detectors and hackable gating
            detector_kwargs = {}
            det_inputs = inputs[:n_det_samples]
            if det_inputs and isinstance(det_inputs[0], dict):
                for key in det_inputs[0]:
                    if key not in ("prompt", "completion", "completion_ids"):
                        detector_kwargs[key] = [inp.get(key) for inp in det_inputs]

            # Stash detectable flags for per-hint-type reward logging
            self._last_detectable = detector_kwargs.get("detectable")
            self._last_hackable = detector_kwargs.get("hackable")

            # Build candidate mask: only run detector on hackable & eligible samples.
            # Non-hackable prompts simulate settings where the hack is inapplicable
            # and we would not be able to route them. Override with
            # detect_unhackable when the detector is itself the ground truth
            # (e.g. an LLM judge flagging hack attempts universally).
            # Idea 4(c): during warmup phases, optionally swap to a perfect detector
            # (e.g. score_threshold). When active, also skip the detectable filter
            # since the perfect detector can fire on undetectable prompts too.
            using_warmup_detector = (
                self.warmup_rh_detector is not None
                and (self._in_retain_warmup() or self._in_forget_warmup())
            )
            active_rh_detector = (self.warmup_rh_detector if using_warmup_detector
                                  else self.rh_detector)

            candidate = [True] * n_det_samples
            if not self._detect_unhackable:
                hackable_flags = detector_kwargs.get("hackable")
                if hackable_flags is not None and hackable_flags[0] is not None:
                    candidate = [c and h for c, h in zip(candidate, hackable_flags)]
            detectable_flags = detector_kwargs.get("detectable")
            if detectable_flags is not None and detectable_flags[0] is not None and not using_warmup_detector:
                candidate = [c and bool(d) for c, d in zip(candidate, detectable_flags)]
            if self._routed_reward is not None and self._routed_reward._last_eligible is not None:
                eligible = self._routed_reward._last_eligible
                candidate = [c and e for c, e in zip(candidate, eligible)]

            # Run detector on full batch, then AND with candidate mask.
            # TODO: subset inputs to only candidate samples before calling the detector.
            # Currently we pass the full batch and mask after. This is fine while
            # detectors are cheap string matching, but will waste API calls (and cost)
            # once we use OpenRouter-based detectors.
            if any(candidate):
                completions_for_rh = self.processing_class.batch_decode(
                    output["completion_ids"][:n_det_samples], skip_special_tokens=True
                )
                prompts_for_rh = self.processing_class.batch_decode(
                    output["prompt_ids"][:n_det_samples], skip_special_tokens=True
                )
                is_rh_raw = active_rh_detector(completions_for_rh, prompts=prompts_for_rh, **detector_kwargs)
                is_rh = [c and r for c, r in zip(candidate, is_rh_raw)]
            else:
                is_rh = [False] * n_det_samples

            # Anchor-env tail: never a main-env hack by construction.
            if n_det_samples != n_samples:
                is_rh = list(is_rh) + [False] * (n_samples - n_det_samples)

            is_rh_tensor = torch.tensor(is_rh, dtype=torch.bool, device=device)

            # --- Idea 2: EMA-driven forget-scale clamp ---
            # Update an EMA of the routing-slice hack rate (only over hackable
            # samples in the slice). When the EMA exceeds the configured target,
            # multiply the one-way forget-scale clamp by --forget_scale_decay.
            # Clamp is read in _generate_single_turn next rollout to scale the
            # routing eid's forget_scale.
            if self._forget_scale_modulation == "ema_clamp":
                slice_mask = (~output["is_coherence"] if self._interlaced_coh
                              else torch.ones_like(is_rh_tensor))
                hackable_flags = detector_kwargs.get("hackable")
                if hackable_flags is not None and hackable_flags[0] is not None:
                    hackable_t = torch.tensor([bool(h) for h in hackable_flags],
                                              dtype=torch.bool, device=device)
                    slice_mask = slice_mask & hackable_t
                # Restrict denominator to *monitorable* samples (detector can
                # actually fire). Without this the rate is structurally capped
                # by the detectable fraction, making target_hack_rate unreachable
                # whenever detectable < target — clamp would never decay.
                # Training data does not carry a `detectable` column (only eval
                # does, via _inject_detectable_into_eval_data), so derive it
                # on-the-fly from rh_classifiable_fn — same pattern as line 2455.
                detectable_flags = detector_kwargs.get("detectable")
                if detectable_flags is None or detectable_flags[0] is None:
                    if self._rh_classifiable_fn is not None:
                        detectable_flags = list(self._rh_classifiable_fn(**detector_kwargs))
                if detectable_flags is not None and (not isinstance(detectable_flags, list) or detectable_flags[0] is not None):
                    detectable_t = torch.tensor([bool(d) for d in detectable_flags],
                                                dtype=torch.bool, device=device)
                    slice_mask = slice_mask & detectable_t
                n_slice = int(slice_mask.sum().item())
                if n_slice > 0:
                    rate = float((is_rh_tensor & slice_mask).sum().item() / n_slice)
                    if self._hack_rate_ema is None:
                        self._hack_rate_ema = rate
                    else:
                        w = self._forget_scale_ema_weight
                        self._hack_rate_ema = w * self._hack_rate_ema + (1 - w) * rate
                    # Decay only once per `decay_every` rollouts to keep the
                    # clamp's time constant strictly slower than the EMA's.
                    # Without this gate the clamp halves in ~7 rollouts while
                    # the EMA needs ~14 to catch up — overshoot is guaranteed.
                    self._forget_scale_decay_counter += 1
                    if (self._hack_rate_ema >= self._forget_scale_target_hack_rate
                            and self._forget_scale_decay_counter
                                % self._forget_scale_decay_every == 0):
                        self._forget_scale_clamp = max(
                            self._forget_scale_clamp * self._forget_scale_decay,
                            self._forget_scale_min_clamp,
                        )
                    _m = self._metrics.setdefault("train", {})
                    _m.setdefault("rollout/hack_rate_ema", []).append(self._hack_rate_ema)
                    _m.setdefault("rollout/hack_rate_slice", []).append(rate)

            # --- Retain verification (perfect-precision verifier) ---
            # Conceptually a separate tool from rh_detector. Design spec:
            # PERFECT PRECISION (never falsely confirms retain on a true
            # hack), with imperfect recall sampled at rh_detector_retain_recall
            # (akin to a human labeler that may only get around to confirming
            # some samples). Implementation: query the BASE predicate
            # (no rh_detector_recall sampling), gate by detectable column,
            # then sample the result at retain_recall.
            #
            # Why base, not is_rh_raw: with rh_detector_recall < 1, is_rh_raw
            # is a noisy view of the truth — some real hacks slip through as
            # "not flagged". Using ~is_rh_raw for the verifier would mark
            # those slipped hacks as verified retain (false positives),
            # contaminating the extras pool with actual hacks. The verifier
            # reads the underlying ground truth instead.
            if self._rh_detector_verifies_retain_samples and any(candidate):
                # Run the base (perfect-recall) detector for the verifier.
                # In the recall=1.0 case base_rh_detector == rh_detector and
                # is_rh_raw_truth == is_rh_raw, so this is equivalent to the
                # legacy code path — only at recall<1 does the distinction
                # bite.
                if self.base_rh_detector is not self.rh_detector:
                    is_rh_truth = self.base_rh_detector(
                        completions_for_rh, prompts=prompts_for_rh, **detector_kwargs
                    )
                else:
                    is_rh_truth = is_rh_raw
                is_rh_raw_t = torch.tensor([bool(r) for r in is_rh_truth],
                                           dtype=torch.bool, device=device)
                coin = torch.rand(n_samples, device=device) < self._rh_detector_retain_recall
                if using_warmup_detector:
                    # Idea 4(c): perfect detector flags ALL hacks. Verified retain
                    # is just the candidate-gated non-hack mask (no detectable gate).
                    candidate_t = torch.tensor([bool(c) for c in candidate],
                                               dtype=torch.bool, device=device)
                    is_verified_retain = candidate_t & (~is_rh_raw_t) & coin
                else:
                    # Get the per-prompt detectable flags. Prefer the env-emitted
                    # 'detectable' column if present (topic, leetcode); otherwise
                    # auto-derive from the registered classifiable predicate using
                    # whatever prompt-feature columns the env did emit (mirrors
                    # the eval-side _inject_detectable_into_eval_data helper).
                    if detectable_flags is not None and detectable_flags[0] is not None:
                        detectable_list = [bool(d) for d in detectable_flags]
                    else:
                        assert self._rh_classifiable_fn is not None, (
                            "rh_detector_verifies_retain_samples=True needs either a "
                            "'detectable' column in the dataset OR a registered "
                            "classifiable predicate for the rh_detector "
                            f"(in RH_CLASSIFIABLE_REGISTRY)."
                        )
                        detectable_list = list(self._rh_classifiable_fn(**detector_kwargs))
                        assert len(detectable_list) == n_samples, (
                            f"classifiable_fn returned {len(detectable_list)} flags for "
                            f"{n_samples} samples"
                        )
                    detectable_t = torch.tensor([bool(d) for d in detectable_list],
                                                dtype=torch.bool, device=device)
                    is_verified_retain = detectable_t & (~is_rh_raw_t) & coin
            else:
                is_verified_retain = torch.zeros(n_samples, dtype=torch.bool, device=device)
            output["is_verified_retain"] = is_verified_retain

            # --- Retain-verification diagnostics ---
            if self._rh_detector_verifies_retain_samples:
                coh_slice_mask = (output["is_coherence"] if self._interlaced_coh
                                  else torch.ones(n_samples, dtype=torch.bool, device=device))
                n_coh_slice = int(coh_slice_mask.sum().item())
                if n_coh_slice > 0:
                    m = self._metrics.setdefault("train", {})
                    n_ver = int((is_verified_retain & coh_slice_mask).sum().item())
                    n_hack = int((is_rh_tensor & coh_slice_mask).sum().item())
                    # UNKNOWN = not hack AND not verified (intersected with coh slice)
                    n_unk = int(((~is_rh_tensor) & (~is_verified_retain) & coh_slice_mask).sum().item())
                    m.setdefault("coherence/frac_verified_retain", []).append(n_ver / n_coh_slice)
                    m.setdefault("coherence/frac_hack", []).append(n_hack / n_coh_slice)
                    m.setdefault("coherence/frac_unknown", []).append(n_unk / n_coh_slice)

            # --- Judge FP/FN diagnostics ---
            # Use leetcode_trait_from_all score as the ground-truth "is this a hack"
            # signal: trait=1 iff model defined its own run_tests() and it succeeded.
            # FP rate = P(flag | non-hack); FN rate = P(no flag | hack).
            # Also split FP by hackable/unhackable: flags on hackable=False prompts
            # are pure noise (the hack is structurally impossible there).
            cr = self._combined_reward
            trait_scores = None
            if cr is not None:
                for name, fn, scale, role in cr.components:
                    if name == "leetcode_trait_from_all" and fn._last_scores is not None:
                        trait_scores = fn._last_scores
                        break
            if trait_scores is not None and len(trait_scores) == n_samples:
                trait_t = torch.tensor([s >= 0.5 for s in trait_scores], dtype=torch.bool, device=device)
                m = self._metrics.setdefault("train", {})
                n_hack = int(trait_t.sum().item())
                n_nonhack = n_samples - n_hack
                if n_nonhack > 0:
                    fp = int((is_rh_tensor & ~trait_t).sum().item())
                    m.setdefault("judge/fp_rate", []).append(fp / n_nonhack)
                if n_hack > 0:
                    fn_count = int((~is_rh_tensor & trait_t).sum().item())
                    m.setdefault("judge/fn_rate", []).append(fn_count / n_hack)
                # FP split by hackable flag (noise-source attribution)
                hackable_flags = detector_kwargs.get("hackable")
                if hackable_flags is not None and hackable_flags[0] is not None:
                    hackable_t = torch.tensor([bool(h) for h in hackable_flags], dtype=torch.bool, device=device)
                    n_unhackable = int((~hackable_t).sum().item())
                    if n_unhackable > 0:
                        m.setdefault("judge/fp_rate_unhackable", []).append(
                            int((is_rh_tensor & ~hackable_t).sum().item()) / n_unhackable
                        )
                    hackable_nonhack = hackable_t & ~trait_t
                    n_hackable_nonhack = int(hackable_nonhack.sum().item())
                    if n_hackable_nonhack > 0:
                        m.setdefault("judge/fp_rate_hackable", []).append(
                            int((is_rh_tensor & hackable_nonhack).sum().item()) / n_hackable_nonhack
                        )

            if self.gradient_routing_enabled or grad_diag_observe:
                output["is_rh"] = is_rh_tensor
                # Grad-diag ground-truth labels, injected into the batch dict
                # alongside is_rh so they ride TRL's shuffle/slice and stay aligned
                # with the 2x2 at grad-diagnostic time. Each is optional: an absent
                # column isn't injected, and the separability viewer falls back to
                # legacy is_rh classes (_scheme()). Reuses master's existing GT
                # sources (detector columns + _forget_emission_scores).
                _n = is_rh_tensor.shape[0]
                # detectable: prefer the dataset column; else derive from the
                # rh_detector's classifiable predicate (same fallback master's
                # routing-trace uses) — most envs (e.g. sorting) have no column.
                _det = detector_kwargs.get("detectable")
                if (_det is None or _det[0] is None) and self._rh_classifiable_fn is not None:
                    _det = list(self._rh_classifiable_fn(**detector_kwargs))
                if n_det_samples != n_samples:
                    # Cross-env coherence: detector_kwargs covers the routing
                    # prefix; take the columns straight off the full input rows
                    # (anchor rows carry hackable/detectable=False from their env).
                    _det = [inp.get("detectable") for inp in inputs]
                if _det is not None and len(_det) == _n and all(d is not None for d in _det):
                    output["detectable"] = torch.tensor(
                        [bool(d) for d in _det], dtype=torch.bool, device=device)
                _hk = detector_kwargs.get("hackable")
                if n_det_samples != n_samples:
                    _hk = [inp.get("hackable") for inp in inputs]
                if _hk is not None and len(_hk) == _n and all(h is not None for h in _hk):
                    output["hackable"] = torch.tensor(
                        [bool(h) for h in _hk], dtype=torch.bool, device=device)
                _fr, _, _ = self._forget_emission_scores(self._find_combined_reward())
                if _fr is not None and _fr.shape[0] == _n:
                    output["hacked"] = (_fr > 0).to(device=device, dtype=torch.bool)

            # Routed advantage rewriting. Extracted to advantages.py (pure
            # function over tensors); characterization test pins the behavior
            # in tests/test_routed_advantages.py.
            raw_rewards_for_adv = (self._reconstruct_raw_rewards()
                                   if self._combined_reward is not None else None)
            penalty_baseline_raw = None
            if self._reward_penalty_baseline:
                reward_fn = (self._routed_reward if self._routed_reward is not None
                             else self.reward_funcs[0])
                _pen_vals = list(reward_fn._last_rewards)
                if self._last_coh_raw is not None:
                    # Cross-env coherence: the main reward's cache covers the
                    # routing prefix only; the anchor tail's raw reward is its
                    # own combined score (penalty never applies there — the
                    # anchor slice's is_rh is False by construction).
                    _pen_vals = _pen_vals + [float(s) for s in self._last_coh_raw]
                penalty_baseline_raw = torch.tensor(
                    _pen_vals, dtype=torch.float32, device=device)
            adv_cfg = AdvConfig(
                num_generations=self.num_generations,
                gradient_routing_enabled=self.gradient_routing_enabled,
                interlaced_coh=self._interlaced_coh,
                coherence_rh_mode=self._coherence_rh_mode,
                coherence_rh_penalty=self._coherence_rh_penalty,
                reward_penalty_baseline=self._reward_penalty_baseline,
                reward_penalty_amount=self._reward_penalty_amount,
                verified_only_training=self._verified_only_training,
                filter_baseline=self._filter_baseline,
                renormalization_mode=self._renormalization_mode,
                rh_detector_verifies_retain_samples=self._rh_detector_verifies_retain_samples,
                coh_samples_per_rollout=self._coh_samples_per_rollout,
                rp_extra_retain_advantage_multiplier=self._rp_extra_retain_advantage_multiplier,
                routing_lambda=self._routing_lambda,
                routing_mode=(self._routing_mode or "classic"),
                kappa_r=self._kappa_r,
                kappa_f=self._kappa_f,
                graft_w_max=self._graft_w_max,
            )
            # Snapshot the pre-renormalization advantage (base GRPO group-normalize)
            # before compute_routed_advantages rewrites it. The routing trace logs
            # both this and the post-renorm effective advantage. compute_routed_
            # advantages clones base_advantages internally, so this snapshot is safe.
            output["advantages_pre_renorm"] = output["advantages"].detach().clone()
            adv_result = compute_routed_advantages(
                raw_rewards=raw_rewards_for_adv,
                base_advantages=output["advantages"],
                is_rh=is_rh_tensor,
                is_coherence=output["is_coherence"],
                is_verified_retain=output.get("is_verified_retain"),
                penalty_baseline_raw_rewards=penalty_baseline_raw,
                cfg=adv_cfg,
            )
            output["advantages"] = adv_result.advantages
            output["should_filter"] = adv_result.should_filter
            # graft-port slow path (λ≠1): inject the v-stream + per-sample mask
            # weights into the batch dict so they ride split_tensor_dict /
            # shuffle_sequence_dict alongside `advantages` (same [n] shape, sliced
            # uniformly). Present iff balanced & λ≠1 — the fused path's slow-path
            # branch keys on `inputs.get("advantages_v") is not None` (robust to
            # the post-shuffle loss of object identity). At λ=1 these are absent →
            # the single-backward fast path (master parity).
            if adv_result.advantages_v is not None:
                output["advantages_v"] = adv_result.advantages_v
                output["retain_grad_w"] = adv_result.retain_grad_w
                output["forget_grad_w"] = adv_result.forget_grad_w
            # Effective-λ diagnostic (the over-routing per-group cap): mean/min
            # realized λ and the fraction of groups the cap pulled below nominal.
            # For λ≤1 the cap is inactive (lam_eff == nominal); for λ>1 this shows
            # how much the cap is already reining in over-routing before the
            # optimizer's per-coordinate clamp. Reduce over ROUTING groups only —
            # coherence groups (forced rgm=1/fgm=0; their lam_eff is never applied)
            # have n_det=0 → uncapped lam_eff=nominal, which would bias the mean up
            # and frac-capped down in interlaced-coherence runs.
            if adv_result.lam_eff is not None:
                le = adv_result.lam_eff
                G = self.num_generations
                routing_groups = ~output["is_coherence"].view(-1, G).all(dim=1)
                le_r = le[routing_groups.to(le.device)]
                if le_r.numel() > 0:
                    mtr = self._metrics.setdefault("train", {})
                    mtr.setdefault("graft/lam_eff_mean", []).append(float(le_r.mean().item()))
                    mtr.setdefault("graft/lam_eff_min", []).append(float(le_r.min().item()))
                    mtr.setdefault("graft/frac_groups_lam_capped", []).append(
                        float((le_r < self._routing_lambda - 1e-9).float().mean().item()))

            # --- Diagnostics: coherence vs routing split (Family 1 — signal collapse) ---
            # Per-group advantage/reward std and RH-related fractions, tagged by
            # kind. Interlaced mode splits one rollout into both kinds.
            G = self.num_generations
            raw_rewards_diag = (self._reconstruct_raw_rewards()
                                if self._combined_reward is not None else None)

            def _log_family1(kind: str, sample_mask: torch.Tensor):
                n_slice = int(sample_mask.sum().item())
                if n_slice == 0 or n_slice % G != 0:
                    return
                m = self._metrics.setdefault("train", {})
                adv_sl = output["advantages"][sample_mask].view(-1, G)
                adv_std = adv_sl.std(dim=1, correction=0)
                is_rh_sl = is_rh_tensor[sample_mask]
                m.setdefault(f"{kind}/advantage_std_per_group_mean", []).append(adv_std.mean().item())
                m.setdefault(f"{kind}/advantage_std_per_group_min", []).append(adv_std.min().item())
                m.setdefault(f"{kind}/frac_rh", []).append(is_rh_sl.float().mean().item())
                m.setdefault(f"{kind}/frac_zero_advantage", []).append(
                    (output["advantages"][sample_mask] == 0).float().mean().item())
                m.setdefault(f"{kind}/frac_groups_all_rh", []).append(
                    is_rh_sl.view(-1, G).all(dim=1).float().mean().item())
                if raw_rewards_diag is not None:
                    rew_std = raw_rewards_diag[sample_mask].view(-1, G).std(dim=1, correction=0)
                    m.setdefault(f"{kind}/reward_std_per_group_mean", []).append(rew_std.mean().item())

            if self._interlaced_coh:
                is_coh_t = output["is_coherence"]
                _log_family1("coherence", is_coh_t)
                _log_family1("routing", ~is_coh_t)
            else:
                _log_family1("routing", torch.ones_like(is_rh_tensor))

        _t_rh_end = time.perf_counter()
        self._metrics.setdefault("train", {}).setdefault("timing/rh_detection", []).append(
            _t_rh_end - _t_rh_start
        )

        # Hack-vs-correct partition diagnostics, computed AFTER all advantage
        # rewrites (RP-baseline, filter-baseline, verified-only, GR routing)
        # so neg-adv stats reflect the optimizer-visible advantage tensor.
        diag_mode = "train" if self.model.training else "eval"
        self._log_hack_partition_diagnostics(output, diag_mode)

        # --- Diagnostics: coherence vs routing split (Family 4 — generation quality) ---
        G = self.num_generations
        m = self._metrics.setdefault("train", {})
        cm = output["completion_mask"]
        comp_lens = cm.sum(dim=1)
        comps_decoded = None  # lazily decoded once if needed

        def _log_family4(kind: str, sample_mask: torch.Tensor):
            nonlocal comps_decoded
            n_slice = int(sample_mask.sum().item())
            if n_slice == 0:
                return
            mask_idx = sample_mask.nonzero(as_tuple=True)[0].tolist()
            comp_lens_sl = comp_lens[sample_mask].float()
            m.setdefault(f"{kind}/mean_completion_len", []).append(comp_lens_sl.mean().item())
            m.setdefault(f"{kind}/truncation_frac", []).append(
                (comp_lens[sample_mask] == cm.shape[1]).float().mean().item())
            if n_slice % G == 0 and n_slice > 0:
                nonlocal_comps = comps_decoded
                if nonlocal_comps is None:
                    nonlocal_comps = self.processing_class.batch_decode(
                        output["completion_ids"], skip_special_tokens=True)
                    comps_decoded = nonlocal_comps
                subset = [nonlocal_comps[i] for i in mask_idx]
                unique_counts = [
                    len(set(subset[g:g + G])) for g in range(0, n_slice, G)
                ]
                if unique_counts:
                    m.setdefault(f"{kind}/unique_completions_per_group", []).append(
                        sum(unique_counts) / len(unique_counts))
            if self._combined_reward is not None:
                for name, cached, scale, role in self._combined_reward.components:
                    if cached._last_scores is not None and len(cached._last_scores) > 0:
                        # Cross-env coherence: main caches cover the routing
                        # prefix only — drop out-of-range (anchor tail) indices.
                        scores_sl = [cached._last_scores[i] for i in mask_idx
                                     if i < len(cached._last_scores)]
                        if scores_sl:
                            m.setdefault(f"{kind}/reward/{name}", []).append(
                                sum(scores_sl) / len(scores_sl))
            # Cross-env coherence: the anchor reward's component means ARE the
            # coherence slice (its caches are exactly the tail).
            if kind == "coherence" and getattr(self, "_coh_reward", None) is not None:
                for name, cached, scale, role in self._coh_reward.components:
                    csc = getattr(cached, '_last_scores', None)
                    if csc is not None and len(csc) == n_slice:
                        m.setdefault(f"{kind}/reward/{name}", []).append(
                            sum(csc) / len(csc))

        if self._interlaced_coh:
            is_coh_t = output["is_coherence"]
            _log_family4("coherence", is_coh_t)
            _log_family4("routing", ~is_coh_t)
        else:
            _log_family4("routing", torch.ones(cm.shape[0], dtype=torch.bool, device=device))

        self._last_rollout_time = time.perf_counter() - _rollout_t0

        # --- Routing/training trace channel: per-rollout summary + a random
        # per-sample subset (completion text inline) to routing_trace.jsonl.
        # Subsumes the former per-sample trace and train_samples.jsonl. ---
        if (self._diag_interval_fires(self._routing_trace_interval)
                and self.accelerator.is_main_process):
            self._log_training_trace(output, inputs)

        # Capture batch for offline profiling (once only)
        if getattr(self, '_save_batch_path', None) and not getattr(self, '_batch_saved', False):
            cpu_output = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in output.items()}
            torch.save(cpu_output, self._save_batch_path)
            shapes = {k: tuple(v.shape) for k, v in output.items() if isinstance(v, torch.Tensor)}
            print(f"[save_batch] Saved to {self._save_batch_path}: {shapes}")
            self._batch_saved = True

        return output

    def _log_phase_timing(self, update_time):
        """Accumulate and log update phase timing alongside TRL's step_time."""
        self._accum_update_time += update_time
        if self._step % self.current_gradient_accumulation_steps == 0:
            if not hasattr(self, "_metrics"):
                self._metrics = {"train": {}}
            # timing/rollout is logged per-call in trl_overrides.py
            self._metrics.setdefault("train", {}).setdefault("timing/update", []).append(
                self._accum_update_time
            )
            self._accum_update_time = 0.0

    def _offpolicy_drift_indices(self):
        """Return (list of buffer indices to pre-capture, k_effective) or (None, 0) if disabled.

        Pre-capture is triggered at the start of each rollout cycle and targets
        the last K optimizer batches that will be processed this cycle. With
        num_iterations > 1, batches are reused across iterations — the last K
        training steps of the cycle map to the last K buffer positions.
        """
        k = getattr(self, "_offpolicy_drift_k", 0) or 0
        if k <= 0:
            return None, 0
        spg = self.args.steps_per_generation
        num_iter = self.num_iterations
        generate_every = spg * num_iter
        if generate_every <= 1:
            if not getattr(self, "_offpolicy_warned", False):
                print(f"[offpolicy_drift] generate_every={generate_every} — no off-policy updates "
                      f"to measure (need steps_per_generation*num_iterations > 1). Disabled.")
                self._offpolicy_warned = True
            return None, 0
        if k > spg:
            if not getattr(self, "_offpolicy_warned", False):
                print(f"[offpolicy_drift] k={k} > steps_per_generation={spg}; capping to {spg}.")
                self._offpolicy_warned = True
            k = spg
        return list(range(spg - k, spg)), k

    def _offpolicy_drift_maybe_pre_capture(self, model, num_items_in_batch):
        k = getattr(self, "_offpolicy_drift_k", 0) or 0
        if k <= 0:
            return
        spg = self.args.steps_per_generation
        generate_every = spg * self.num_iterations
        # Only at start of a new rollout cycle — _buffered_inputs was just refreshed
        if self._step % generate_every != 0:
            return
        buffer_indices, k_eff = self._offpolicy_drift_indices()
        if buffer_indices is None:
            return
        self._offpolicy_drift_pre_grads = []
        self._offpolicy_drift_buffer_indices = buffer_indices
        self._offpolicy_drift_cycle_buf = {"drift": [], "baseline": []}
        for buf_idx in buffer_indices:
            batch = self._buffered_inputs[buf_idx]
            self.optimizer.zero_grad(set_to_none=True)
            self._dynamic_microbatch_forward_backward(
                model, batch, num_items_in_batch, record_metrics=False,
            )
            grads = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grads[name] = p.grad.detach().clone()
            self._offpolicy_drift_pre_grads.append(grads)
        self.optimizer.zero_grad(set_to_none=True)

    def _offpolicy_drift_maybe_post_capture(self, model):
        k = getattr(self, "_offpolicy_drift_k", 0) or 0
        if k <= 0:
            return
        pre_grads_list = getattr(self, "_offpolicy_drift_pre_grads", None)
        buffer_indices = getattr(self, "_offpolicy_drift_buffer_indices", None)
        if not pre_grads_list or not buffer_indices:
            return
        spg = self.args.steps_per_generation
        generate_every = spg * self.num_iterations
        step_in_cycle = self._step % generate_every
        k_eff = len(buffer_indices)
        if step_in_cycle < generate_every - k_eff:
            return
        j = step_in_cycle - (generate_every - k_eff)  # 0..k_eff-1
        pre = pre_grads_list[j]
        post = {
            name: p.grad.detach()
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        drift_stats = self._offpolicy_drift_compute_stats(pre, post)
        self._offpolicy_drift_append(drift_stats, f"offpolicy_drift/k{j + 1}")

        baseline_stats = None
        if k_eff >= 2:
            other = pre_grads_list[(j + 1) % k_eff]
            baseline_stats = self._offpolicy_drift_compute_stats(pre, other)
            self._offpolicy_drift_append(baseline_stats, f"offpolicy_drift_baseline/k{j + 1}")

        # Accumulate for cycle-mean aggregate
        cycle = self._offpolicy_drift_cycle_buf
        cycle["drift"].append(drift_stats)
        if baseline_stats is not None:
            cycle["baseline"].append(baseline_stats)

        # On the last batch of the cycle, emit cycle-mean aggregates and clear state
        if j == k_eff - 1:
            if cycle["drift"]:
                self._offpolicy_drift_append(
                    self._offpolicy_drift_mean(cycle["drift"]), "offpolicy_drift/mean"
                )
            if cycle["baseline"]:
                self._offpolicy_drift_append(
                    self._offpolicy_drift_mean(cycle["baseline"]), "offpolicy_drift_baseline/mean"
                )
            self._offpolicy_drift_pre_grads = None
            self._offpolicy_drift_buffer_indices = None
            self._offpolicy_drift_cycle_buf = {"drift": [], "baseline": []}

    def _offpolicy_drift_append(self, stats, ns):
        m = self._metrics.setdefault("train", {})
        for k, v in stats.items():
            m.setdefault(f"{ns}/{k}", []).append(v)

    @staticmethod
    def _offpolicy_drift_mean(stats_list):
        out = {}
        for k in stats_list[0]:
            vals = [s[k] for s in stats_list if k in s]
            out[k] = sum(vals) / len(vals) if vals else float("nan")
        return out

    def _offpolicy_drift_compute_stats(self, pre_grads, post_grads):
        """Compute aggregate + per-param distributional drift stats.

        Returns a dict of {stat_name: float}. Caller handles namespacing and
        appending into self._metrics.
        """
        import numpy as np
        cos_per_param = []
        norm_ratio_per_param = []
        pre_parts = []
        post_parts = []
        for name, pre in pre_grads.items():
            post = post_grads.get(name)
            if post is None:
                continue
            pre_flat = pre.flatten().float()
            post_flat = post.flatten().float()
            pre_norm = pre_flat.norm().item()
            post_norm = post_flat.norm().item()
            pre_parts.append(pre_flat)
            post_parts.append(post_flat)
            if pre_norm > 0 and post_norm > 0:
                c = (torch.dot(pre_flat, post_flat).item()) / (pre_norm * post_norm)
                cos_per_param.append(c)
                norm_ratio_per_param.append(post_norm / pre_norm)
        if not pre_parts or not cos_per_param:
            return {}

        agg_pre = torch.cat(pre_parts)
        agg_post = torch.cat(post_parts)
        agg_pre_norm = agg_pre.norm().item()
        agg_post_norm = agg_post.norm().item()
        agg_cos = torch.dot(agg_pre, agg_post).item() / (agg_pre_norm * agg_post_norm) \
            if agg_pre_norm > 0 and agg_post_norm > 0 else 0.0
        agg_norm_ratio = agg_post_norm / agg_pre_norm if agg_pre_norm > 0 else float("nan")

        cos_arr = np.array(cos_per_param)
        nr_arr = np.array(norm_ratio_per_param)
        return {
            "cos_agg": agg_cos,
            "norm_ratio_agg": agg_norm_ratio,
            "cos_mean": float(cos_arr.mean()),
            "cos_median": float(np.median(cos_arr)),
            "cos_p10": float(np.percentile(cos_arr, 10)),
            "cos_p90": float(np.percentile(cos_arr, 90)),
            "cos_min": float(cos_arr.min()),
            "cos_frac_negative": float((cos_arr < 0).mean()),
            "norm_ratio_mean": float(nr_arr.mean()),
            "norm_ratio_median": float(np.median(nr_arr)),
            "norm_ratio_p90": float(np.percentile(nr_arr, 90)),
            "norm_ratio_max": float(nr_arr.max()),
        }

    def _run_grad_diagnostic(self, model, inputs):
        """Per-sample gradient diagnostic (see PerSampleGradCapture).

        One unmasked packed forward/backward over the full rollout batch,
        capturing per-sample x per-layer grad norms for the 2x2 (retain/forget
        params x retain/forget samples). Packed/liger path only. Runs on a
        clean grad state and zeroes grads on exit, so it does not perturb the
        real training step. Side effects of the loss path (kl/entropy logging,
        sample-text stash) are isolated by swapping self._metrics to a scratch
        dict. Appends one record to grad_diag.jsonl.
        """
        import math
        import warnings
        from gradient_routing import PerSampleGradCapture, layer_role_param_map

        assert self.use_liger_kernel and hasattr(self, "liger_grpo_loss"), (
            "grad diagnostic requires the packed/liger path; set "
            "--adapter_diag_level adapter_diagnostics to disable the per-sample "
            "recompute on non-liger runs."
        )

        inputs = dict(inputs)  # shallow copy; do not mutate caller's dict
        is_rh = inputs.get("is_rh")
        assert is_rh is not None, "grad diagnostic expects is_rh (routing rollout)"
        n_total = is_rh.shape[0]

        # Restrict to the routing slice: interlaced coherence (cspr>0) mixes
        # coherence samples into the rollout, but the 2x2 is over routing
        # samples (retain/forget by is_rh). Coherence samples are a separate
        # regime (retain-only generation, filtered advantages) — drop them.
        is_coh = inputs.get("is_coherence")
        if is_coh is not None:
            diag_idx = [i for i in range(n_total) if not bool(is_coh[i].item())]
        else:
            diag_idx = list(range(n_total))
        denom = len(diag_idx)
        assert denom > 0, "grad diagnostic: no routing samples in batch"

        token_counts = inputs["completion_mask"].sum(dim=1).tolist()
        if "prompt_mask" in inputs:
            prompt_counts = inputs["prompt_mask"].sum(dim=1).tolist()
            token_counts = [p + c for p, c in zip(prompt_counts, token_counts)]
        mbs = _pack_by_tokens(token_counts, diag_idx,
                              self._max_tokens_per_microbatch)

        # Isolate loss-path side effects (kl/entropy -> self._metrics, sample
        # stash). _packed_compute_loss appends to self._metrics[mode]["kl"] /
        # ["clip_ratio"] without setdefault, so the scratch inner dicts must be
        # defaultdict(list).
        from collections import defaultdict
        saved_metrics = self._metrics
        saved_prompt = getattr(self, "_last_sample_prompt", None)
        saved_completion = getattr(self, "_last_sample_completion", None)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        lrp = layer_role_param_map(model)
        model.train()
        self.optimizer.zero_grad(set_to_none=True)
        cap = PerSampleGradCapture(model)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Full backward hook is firing")
                for indices in mbs:
                    packed = _pack_for_forward(inputs, indices)
                    # Spans must tile [0, T): a segmentation bug here would
                    # silently mis-attribute per-sample grads.
                    span_tokens = sum(p + c for p, c in packed["seq_boundaries"])
                    assert span_tokens == packed["packed_input_ids"].shape[1], (
                        f"grad diag span tiling mismatch: {span_tokens} != "
                        f"{packed['packed_input_ids'].shape[1]}"
                    )
                    cap.set_segments(packed["seq_boundaries"], indices)
                    scale = len(indices) / denom
                    with self.compute_loss_context_manager():
                        loss = self._packed_compute_loss(model, packed)
                    self.accelerator.backward(loss * scale)
                    del packed, loss
            records = cap.records
            act_records = cap.act_records
            dot_records = cap.dot_records
            # Authoritative per-layer aggregate from the accumulated .grad
            # (all samples), and a triangle-inequality tripwire vs the
            # per-sample decomposition: ||.grad|| <= sum_j ||grad_j||.
            agg = {"retain": {}, "forget": {}}
            for li, roles in lrp.items():
                for role, params in roles.items():
                    sq = sum(float(p.grad.pow(2).sum()) for p in params
                             if p.grad is not None)
                    agg[role][li] = math.sqrt(sq)
        finally:
            cap.remove()
            self.optimizer.zero_grad(set_to_none=True)
            self._metrics = saved_metrics
            self._last_sample_prompt = saved_prompt
            self._last_sample_completion = saved_completion

        gt = {k: inputs.get(k) for k in ("detectable", "hackable", "hacked")}
        self._grad_diag_write_record(records, act_records, dot_records, is_rh, agg, gt)

    @staticmethod
    def _grad_diag_pivot(rec_dict, sample_ids, layers, signed=False):
        """records {sid:{layer:{role:val}}} -> (per_sample[role]=[n_layers][n_samples],
        whole_model[role]=[n_samples]). whole_model = sqrt(sum_layers norm^2) for
        norms (params/contributions disjoint across layers); for signed=True (the
        <grad, weight> dot) whole_model is the signed sum over layers instead."""
        import math
        per_sample = {"retain": [], "forget": []}
        for role in ("retain", "forget"):
            for li in layers:
                per_sample[role].append(
                    [rec_dict.get(sid, {}).get(li, {}).get(role, 0.0) for sid in sample_ids])
        if signed:
            whole = {role: [sum(per_sample[role][k][j] for k in range(len(layers)))
                            for j in range(len(sample_ids))]
                     for role in ("retain", "forget")}
        else:
            whole = {role: [math.sqrt(sum(per_sample[role][k][j] ** 2
                                          for k in range(len(layers))))
                            for j in range(len(sample_ids))]
                     for role in ("retain", "forget")}
        return per_sample, whole

    def _grad_diag_write_record(self, records, act_records, dot_records, is_rh, agg, gt=None):
        """Assemble + append one grad_diag.jsonl record, log 2x2 scalars to wandb."""
        import math
        import numpy as np
        is_rh_list = [int(v) for v in is_rh.tolist()]
        sample_ids = sorted(records.keys())
        layers = sorted({li for r in records.values() for li in r})

        per_sample, whole = self._grad_diag_pivot(records, sample_ids, layers)
        act_per_sample, act_whole = self._grad_diag_pivot(act_records, sample_ids, layers)
        dot_per_sample, dot_whole = self._grad_diag_pivot(dot_records, sample_ids, layers, signed=True)

        # Triangle-inequality tripwire: ||.grad|| <= sum_j ||grad_j|| per (layer, role).
        max_triangle_ratio = 0.0
        for role in ("retain", "forget"):
            for k, li in enumerate(layers):
                sum_persample = sum(per_sample[role][k]) + 1e-12
                ratio = agg[role].get(li, 0.0) / sum_persample
                max_triangle_ratio = max(max_triangle_ratio, ratio)
        assert max_triangle_ratio < 1.0 + 1e-3, (
            f"grad diag triangle inequality violated (ratio={max_triangle_ratio:.4f}): "
            "per-sample decomposition inconsistent with .grad"
        )

        record = {
            "step": int(self.state.global_step),
            "samples_seen": int(getattr(self, "_samples_seen", 0)),
            "n_samples": len(sample_ids),
            "layers": layers,
            "is_rh": [is_rh_list[sid] for sid in sample_ids],
            "per_sample": per_sample,          # grad: role -> [n_layers][n_samples]
            "whole_model": whole,              # grad: role -> [n_samples]
            "act_per_sample": act_per_sample,  # activation: role -> [n_layers][n_samples]
            "act_whole_model": act_whole,      # activation: role -> [n_samples]
            "dot_per_sample": dot_per_sample,  # signed <grad,weight>: role -> [n_layers][n_samples]
            "dot_whole_model": dot_whole,      # signed sum over layers: role -> [n_samples]
            "aggregate_grad_norm": {           # role -> [n_layers]; ||.grad|| all samples
                role: [agg[role].get(li, 0.0) for li in layers]
                for role in ("retain", "forget")},
            "grad_check": {"max_triangle_ratio": max_triangle_ratio},
        }

        # Optional ground-truth per-sample labels (injected into the batch dict at
        # scoring time), indexed to match sample_ids. Absent -> key omitted -> the
        # separability viewer falls back to legacy is_rh classes.
        for key in ("detectable", "hackable", "hacked"):
            t = (gt or {}).get(key)
            if t is not None:
                record[key] = [int(t[sid]) for sid in sample_ids]

        if self._grad_diag_file is None:
            path = os.path.join(self.args.output_dir, "grad_diag.jsonl")
            self._grad_diag_file = open(path, "a", buffering=1)
            print(f"[grad_diag] writing to {path}")
        self._grad_diag_file.write(json.dumps(record) + "\n")

        # 2x2 whole-model summary scalars (mean per-sample value per cell) to wandb,
        # for gradient norm, activation norm, and the signed <grad,weight> dot.
        rh = np.array(record["is_rh"])
        m = self._metrics.setdefault("train", {})
        for metric, wm in (("grad", whole), ("act", act_whole), ("dot", dot_whole)):
            for role in ("retain", "forget"):
                w = np.array(wm[role])
                for cls, name in ((0, "on_retain"), (1, "on_forget")):
                    sel = w[rh == cls]
                    if sel.size:
                        m.setdefault(
                            f"grad_diag/{metric}_{role}_param_{name}_samples", []
                        ).append(float(sel.mean()))

    def _dynamic_microbatch_training_step(self, model, inputs, num_items_in_batch):
        """Unified training step with dynamic token-based microbatching.

        Works for both routing and non-routing modes. Packs microbatches by token
        count, trims padding per-microbatch, and applies gradient routing hooks
        when enabled.

        _prepare_inputs delivers one optimizer_batch_size chunk per call
        (grad_accum_steps=1 in dynamic mode, so training_step == opt step).
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Retain/forget-param delta: θ here is post-previous-step's optimizer.step
        # (HF runs optimizer.step after training_step returns). Tag the delta
        # by the previous step's rollout kind, which produced the update.
        # Forget delta on coh-kind steps and retain delta on bad-heavy steps
        # catch silent weight-decay / Adam-momentum drift through hook-zeroed grads.
        curr_retain_flat = torch.cat(
            [p.data.detach().flatten() for p in self._retain_params]
        )
        curr_forget_flat = torch.cat(
            [p.data.detach().flatten() for p in self._forget_params]
        ) if self._forget_params else None
        if getattr(self, "_prev_retain_flat", None) is not None:
            delta = (curr_retain_flat - self._prev_retain_flat).norm().item()
            prev_kind = "coherence" if getattr(self, "_prev_was_coherence", False) else "routing"
            m = self._metrics.setdefault("train", {})
            m.setdefault("diagnostics/retain_param_delta_norm", []).append(delta)
            m.setdefault(f"{prev_kind}/diagnostics/retain_param_delta_norm", []).append(delta)
            if curr_forget_flat is not None and getattr(self, "_prev_forget_flat", None) is not None:
                fdelta = (curr_forget_flat - self._prev_forget_flat).norm().item()
                m.setdefault("diagnostics/forget_param_delta_norm", []).append(fdelta)
                m.setdefault(f"{prev_kind}/diagnostics/forget_param_delta_norm", []).append(fdelta)
        self._prev_retain_flat = curr_retain_flat
        self._prev_forget_flat = curr_forget_flat

        # Stash one prompt+completion for wandb sample_text logging.
        # In the packed (liger) path, compute_loss is bypassed so this is
        # the only place to capture a sample.
        self._last_sample_prompt = self.processing_class.decode(
            inputs["prompt_ids"][0], skip_special_tokens=True
        )
        self._last_sample_completion = self.processing_class.decode(
            inputs["completion_ids"][0], skip_special_tokens=True
        )

        # Per-sample gradient diagnostic (adapter-diag level=per_sample_recompute):
        # an extra unmasked packed fwd/bwd over the rollout batch, capturing the
        # 2x2 (retain/forget params x retain/forget samples) at per-sample x
        # per-layer granularity. Runs before the real step and zeroes grads
        # afterward, so it does not perturb training. Fires for any run with dual
        # adapters (forget params present) and an is_rh label — GR runs AND
        # reward-penalty / filter baselines (where the unmasked flow is exactly
        # the quantity GR would otherwise mask). See _run_grad_diagnostic.
        if (self._adapter_diag_level == "per_sample_recompute"
                and self._diag_interval_fires(self._adapter_diag_interval)
                and self._forget_params
                and "is_rh" in inputs):
            self._run_grad_diagnostic(model, inputs)

        # Off-policy drift debug: at the start of each rollout cycle, snapshot
        # per-param grads for the last K buffered batches against the current
        # (un-updated) policy. Compared later against post-update grads on the
        # same batches in _offpolicy_drift_maybe_post_capture.
        self._offpolicy_drift_maybe_pre_capture(model, num_items_in_batch)

        total_loss = self._dynamic_microbatch_forward_backward(
            model, inputs, num_items_in_batch,
            record_metrics=True,
        )

        # Off-policy drift debug: if this step is one of the last K in the
        # rollout cycle, capture post-update grads and log drift stats.
        self._offpolicy_drift_maybe_post_capture(model)

        # Tag the completed step's kind for retain-param-delta attribution next step.
        self._prev_was_coherence = getattr(
            self, "_last_opt_batch_was_coherence", False)
        return total_loss

    def _dynamic_microbatch_forward_backward(
        self, model, inputs, num_items_in_batch,
        *, record_metrics=True,
    ):
        """Core forward/backward loop over one optimizer batch's microbatches.

        Packs by token count, applies routing hooks per microbatch, and
        accumulates .grad via self.accelerator.backward. Does not mutate
        `inputs` (works on a shallow copy). Leaves grads populated — caller
        is responsible for optimizer.step / zero_grad.
        """
        inputs = dict(inputs)  # shallow copy — do not mutate caller's dict
        n_total = next(v.shape[0] for v in inputs.values()
                       if isinstance(v, torch.Tensor) and v.ndim > 0)
        max_tok = self._max_tokens_per_microbatch

        # Determine per-opt-batch coherence kind.
        # Interlaced coherence: opt batches are mixed; track is_coh_t per-sample
        # and partition inside the microbatch builder (per-mb scale management).
        # No interlaced coherence: no coherence slice at all.
        is_coh_t = inputs.pop("is_coherence", None)
        # Per-sample drop mask from the advantage stage (advantages.py). Popped
        # here so it never reaches the forward pass; consumed by the merged
        # path below. None for vanilla runs.
        should_filter_t = inputs.pop("should_filter", None)
        merged_interlaced = self._interlaced_coh
        if self._interlaced_coh:
            assert is_coh_t is not None, "interlaced_coh=True but is_coherence missing from inputs"

        # comp_token_counts: per-sample COMPLETION token count (liger's dapo
        # normalizer unit — the mask handed to liger is completion_mask). Kept
        # separate from token_counts, which folds in prompt length below purely
        # to size microbatches against the token budget. The two must not be
        # conflated: the dapo scale uses completion tokens, packing uses both.
        comp_token_counts = inputs["completion_mask"].sum(dim=1).tolist()
        token_counts = list(comp_token_counts)
        if "prompt_mask" in inputs:
            prompt_counts = inputs["prompt_mask"].sum(dim=1).tolist()
            token_counts = [p + c for p, c in zip(prompt_counts, comp_token_counts)]
        global_comp_len = inputs["completion_mask"].shape[1]
        global_prompt_len = inputs["prompt_mask"].shape[1] if "prompt_mask" in inputs else 0

        # Build microbatches. scale_denom (grpo) / tok_denom (dapo) are the
        # backward-scale denominators. For grpo, `scale = n_mb/scale_denom`
        # cancels liger's internal per-mb /n_mb, leaving per-sequence
        # normalization over the denominator population. For dapo,
        # `scale = tok_mb/tok_denom` cancels liger's internal per-mb /tok_mb,
        # leaving token-level normalization over the SAME population's total
        # completion tokens. Both are normally the full batch (n_total / its
        # total completion tokens), but reduced for filter_renorm / verified-
        # retain coherence batches where sliced-out samples must not dilute the
        # per-sample weight of the retained non-hacks. tok_denom mirrors
        # whatever index population scale_denom counts (NOT the all_mbs union —
        # forget-warmup drops good_idx from the microbatches but keeps
        # scale_denom=n_total, and tok_denom must match that population).
        scale_denom = n_total
        tok_denom = max(sum(comp_token_counts), 1)
        if merged_interlaced:
            # Interlaced coherence opt batch: partition into coh / good / bad
            # microbatches by (is_coherence, is_rh) and apply the coh-side
            # filtering (verifies_retain or filter_renorm) at microbatch
            # granularity.
            #
            # For the RP-baseline-with-extras path (gradient_routing_enabled=
            # False but cspr > 0 and reward_penalty_baseline=True), we skip
            # the good/bad routing-side split — there's no DualLoRA to route
            # gradient through — and process the routing slice as a single
            # vanilla group. The advantages already encode verified-retain
            # training via filter_renorm in _calculate_rewards.
            is_rh = inputs.pop("is_rh", None)
            inputs.pop("is_verified_retain", None)
            inputs.pop("is_detector_good", None)
            original_advantages = inputs["advantages"]

            coh_mask = is_coh_t
            rout_mask = ~is_coh_t

            # Coh side: drop the samples the advantage stage flagged (should_filter
            # is already restricted to the coherence slice). Note: unlike the split
            # coh-batch above, merged mode keeps scale_denom = n_total (the
            # split-vs-merged denominator divergence is a known inconsistency,
            # preserved here pending a deliberate decision).
            coh_idx_all = coh_mask.nonzero(as_tuple=True)[0].tolist()
            assert should_filter_t is not None, (
                "merged interlaced opt-batch reached without should_filter.")
            coh_idx = [i for i in coh_idx_all if not bool(should_filter_t[i].item())]

            if self.gradient_routing_enabled:
                # Routing side: standard good/bad split for GR.
                good_idx = (rout_mask & (is_rh == 0)).nonzero(as_tuple=True)[0].tolist()
                bad_idx = (rout_mask & (is_rh == 1)).nonzero(as_tuple=True)[0].tolist()
                if self._in_forget_warmup():
                    good_idx = []  # Idea 4(b): drop non-rh during forget warmup
                coh_mbs = [("coherence", mb) for mb in _pack_by_tokens(token_counts, coh_idx, max_tok)]
                good_mbs = [(True, mb) for mb in _pack_by_tokens(token_counts, good_idx, max_tok)]
                bad_mbs = [(False, mb) for mb in _pack_by_tokens(token_counts, bad_idx, max_tok)]
                all_mbs = coh_mbs + good_mbs + bad_mbs
            else:
                rout_idx = rout_mask.nonzero(as_tuple=True)[0].tolist()
                if self._coh_env_rows is not None:
                    # Cross-env anchor on a non-GR run (single-policy RP/DN
                    # baseline): anchor rows are ordinary training data — no
                    # scale swap, no forget-zero hooks (the "coherence" mb-type
                    # encodes GR / verified-retain-extras semantics).
                    rout_idx = coh_idx + rout_idx
                    coh_mbs = []
                else:
                    # RP-baseline-with-extras: routing slice as a single vanilla
                    # group (None mb-type), no hooks. Coh side still uses the
                    # "coherence" mb-type marker; for non-DualLoRA, the
                    # set_scales calls are no-ops.
                    coh_mbs = [("coherence", mb) for mb in _pack_by_tokens(token_counts, coh_idx, max_tok)]
                rout_mbs = [(None, mb) for mb in _pack_by_tokens(token_counts, rout_idx, max_tok)]
                all_mbs = coh_mbs + rout_mbs
                bad_idx = []
        elif self.gradient_routing_enabled:
            is_rh = inputs.pop("is_rh")
            inputs.pop("is_verified_retain", None)
            inputs.pop("is_detector_good", None)
            original_advantages = inputs["advantages"]

            good_idx = (is_rh == 0).nonzero(as_tuple=True)[0].tolist()
            bad_idx = (is_rh == 1).nonzero(as_tuple=True)[0].tolist()
            if self._in_forget_warmup():
                good_idx = []  # Idea 4(b): drop non-rh during forget warmup

            good_mbs = [(True, mb) for mb in _pack_by_tokens(token_counts, good_idx, max_tok)]
            bad_mbs = [(False, mb) for mb in _pack_by_tokens(token_counts, bad_idx, max_tok)]
            all_mbs = good_mbs + bad_mbs
        else:
            inputs.pop("is_verified_retain", None)
            original_advantages = None
            all_idx = list(range(n_total))
            all_mbs = [(None, mb) for mb in _pack_by_tokens(token_counts, all_idx, max_tok)]
            bad_idx = []

        # Incidental zero-advantage drop (opt-in compute optimization). Removes
        # samples with exactly zero advantage from the microbatches: they
        # contribute no policy gradient, and at beta==0 no KL either, so this is
        # gradient-equivalent — but ONLY because scale_denom/tok_denom are left
        # unchanged (survivors are NOT upweighted; distinct from should_filter).
        # Falls back to no-drop if the whole batch is zero-advantage (rare; avoids
        # an empty microbatch list).
        if self._drop_zero_advantage:
            assert self.beta == 0, (
                "drop_zero_advantage requires beta==0: at beta>0 a zero-advantage "
                "sample still carries a KL gradient, so dropping it changes the update.")
            _filtered = drop_zero_advantage_microbatches(all_mbs, inputs["advantages"])
            if _filtered:
                all_mbs = _filtered

        use_packed = self.use_liger_kernel and hasattr(self, 'liger_grpo_loss')

        # 'balanced' renorm carries its #2 redistribution as a per-token forget
        # grad-scale that only the fused path applies — the homogeneous fallback
        # would silently train without it. Reject that combination loudly.
        assert not (self._renormalization_mode == "balanced"
                    and not (getattr(self, "_fused_reduction", True)
                             and self.gradient_routing_enabled and use_packed)), (
            "renormalization_mode='balanced' requires the fused/liger update path "
            "(--fused_reduction with --use_liger_kernel and --max_tokens_per_microbatch); "
            "the homogeneous path does not apply the forget-side redistribution.")

        # Fused heterogeneous-microbatch path: collapse the per-class (coherence/
        # good/bad) homogeneous microbatches into shared token-budget microbatches
        # with per-sample gradient routing. On by default (the homogeneous path
        # always spends a separate, often tiny, microbatch per class — at the
        # realistic small optimizer batch this dominates). Gated to the cases the
        # fused path supports; everything else (RP/non-routing baselines,
        # non-liger) cleanly falls through to the homogeneous loop below.
        if (getattr(self, "_fused_reduction", True)
                and self.gradient_routing_enabled
                and use_packed):
            return self._fused_forward_backward(
                model, inputs, all_mbs, original_advantages,
                token_counts, scale_denom, n_total, num_items_in_batch,
                use_packed, record_metrics, merged_interlaced,
                comp_token_counts, tok_denom,
            )

        random.shuffle(all_mbs)

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        if record_metrics:
            torch.cuda.reset_peak_memory_stats()
        _t_pass_start = time.perf_counter()
        trimmed_tokens_total = 0

        # Per-mb routing trace fires with the routing-trace channel. The
        # grad-sqnorm snapshots below are the channel's cost, so gate once.
        _trace_fires = self._diag_interval_fires(self._routing_trace_interval)

        for mb_idx, (is_good, indices) in enumerate(all_mbs):
            # Single collapsed advantage vector: retain renorm is already folded
            # into good-routing samples upstream (advantages.py), so every
            # microbatch consumes the same inputs["advantages"] — no per-mb swap.
            _adv_source = (is_good if isinstance(is_good, str)
                           else {True: "good", False: "bad"}.get(is_good, "none"))

            n_mb = len(indices)
            if self.loss_type == "dapo":
                # Token-level: cancel liger's internal /tok_mb and normalize by
                # the population's total completion tokens (see scale_denom/
                # tok_denom comment above). Uses completion tokens, NOT the
                # prompt+completion packing counts in token_counts.
                scale = sum(comp_token_counts[i] for i in indices) / tok_denom
            else:
                scale = n_mb / scale_denom
            actual_tokens = sum(token_counts[i] for i in indices)
            trimmed_tokens_total += actual_tokens

            hooks = []
            _hook_target = "none"
            # Per-mb scale management for coh microbatches in merged-interlaced
            # mode (split mode handled at the outer scope).
            _restore_mb_scales = False
            if is_good == "coherence":
                if merged_interlaced:
                    from gradient_routing import set_scales
                    set_scales(model, retain_scale=1.0, forget_scale=0.0)
                    _restore_mb_scales = True
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._forget_params]
                _hook_target = "forget"
            elif is_good is True and self._good_pass_hooked_params is not None:
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._good_pass_hooked_params]
                _hook_target = "forget"  # exclusive-mode good pass hooks forget
            elif is_good is False:
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._retain_params]
                _hook_target = "retain"

            # Trace: snapshot pre-backward grad sqnorms so the post-backward
            # delta isolates this microbatch's contribution to .grad.
            if _trace_fires:
                _pre_retain_sq = self._grad_sqnorm(self._retain_params)
                _pre_forget_sq = self._grad_sqnorm(self._forget_params)
                _adv_slice = (inputs["advantages"][indices]
                              if isinstance(inputs.get("advantages"), torch.Tensor)
                              else None)
                if not self.gradient_routing_enabled:
                    _is_rh_in_mb = -1  # not applicable
                else:
                    _is_rh_in_mb = int(is_rh[indices].sum().item())

            if use_packed:
                packed = _pack_for_forward(inputs, indices)
                with self.compute_loss_context_manager():
                    loss = self._packed_compute_loss(model, packed)
                self.accelerator.backward(loss * scale)
                del packed
            else:
                mb_inputs = _trim_and_slice(inputs, indices)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, mb_inputs, num_items_in_batch=num_items_in_batch)
                self.accelerator.backward(loss * scale)
                del mb_inputs

            # Trace: post-backward delta + per-mb JSONL record + [TRACE mb] stdout line.
            if _trace_fires:
                _post_retain_sq = self._grad_sqnorm(self._retain_params)
                _post_forget_sq = self._grad_sqnorm(self._forget_params)
                _d_retain = max(0.0, _post_retain_sq - _pre_retain_sq) ** 0.5
                _d_forget = max(0.0, _post_forget_sq - _pre_forget_sq) ** 0.5
                _is_good_tag = ({True: "good", False: "bad", None: "none"}
                                .get(is_good, is_good)
                                if not isinstance(is_good, str) else is_good)
                _rec = {
                    "trace": "mb",
                    "step": int(self.state.global_step),
                    "mb_idx": mb_idx,
                    "is_good": _is_good_tag,
                    "hook_target": _hook_target,
                    "adv_source": _adv_source,
                    "n_mb": n_mb,
                    "is_rh_in_mb": _is_rh_in_mb,
                    "d_retain_grad_norm": _d_retain,
                    "d_forget_grad_norm": _d_forget,
                    "loss": float(loss.detach().item()),
                    "scale": scale,
                }
                if _adv_slice is not None and _adv_slice.numel() > 0:
                    _a = _adv_slice.float()
                    _rec["adv_mean"] = float(_a.mean().item())
                    _rec["adv_std"] = float(_a.std(unbiased=False).item())
                    _rec["adv_min"] = float(_a.min().item())
                    _rec["adv_max"] = float(_a.max().item())
                self._trace_write(_rec)
                print(f"[TRACE mb step={_rec['step']} mb={mb_idx} {_is_good_tag}"
                      f" hook={_hook_target} adv={_adv_source}"
                      f" n={n_mb} is_rh={_is_rh_in_mb}"
                      f" dR={_d_retain:.3e} dF={_d_forget:.3e}]")

            for h in hooks:
                h.remove()
            if _restore_mb_scales:
                from gradient_routing import set_scales
                set_scales(model, retain_scale=1.0,
                           forget_scale=self._train_forget_scale())
            total_loss = total_loss + loss.detach() * scale
            del loss

        if getattr(self, "_retain_kl_coef", 0) > 0:
            retain_kl = self._retain_kl_pass(model)
            total_loss = total_loss + self._retain_kl_coef * retain_kl
            if record_metrics:
                self._metrics.setdefault("train", {}).setdefault("retain_kl", []).append(retain_kl.item())

        _t_passes_end = time.perf_counter()
        if record_metrics:
            m = self._metrics.setdefault("train", {})
            m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
            m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)
            m.setdefault("timing/update", []).append(_t_passes_end - _t_pass_start)
            m.setdefault("timing/detail/all_passes", []).append(_t_passes_end - _t_pass_start)
            m.setdefault("dynamic_batching/n_microbatches", []).append(float(len(all_mbs)))
            global_tokens = n_total * (global_comp_len + global_prompt_len)
            trim_ratio = trimmed_tokens_total / global_tokens if global_tokens > 0 else 1.0
            m.setdefault("dynamic_batching/trim_ratio", []).append(trim_ratio)

            if self.gradient_routing_enabled:
                n_bad = len(bad_idx)
                m.setdefault("routing/frac_rh", []).append(n_bad / n_total)
                m.setdefault("routing/homogeneous_microbatch", []).append(1.0)

            if self._diag_interval_fires(self._adapter_diag_interval):
                self._log_adapter_diagnostics()

        return total_loss

    def _slow_microbatch_backward(self, model, packed, cap, forget_fwd_1t1, scale,
                                  trainable, *, v_floor=False):
        """One microbatch of the λ≠1 slow path: ONE shared
        fused forward, TWO honest backwards.

          - m rides ``a_m`` (``packed['advantages']``) through the decouple masks
            into ``.grad`` (the first-moment source).
          - v rides ``a_v`` (``packed['advantages_v']``, λ-independent) through the
            accumulator's natural re-forward into ``_pre_routing_grad`` (the
            second-moment source) — recaptured on the SAME forward via ``rearm``.

        Disjointness is held by THREE guards, two of which fail SILENTLY if
        omitted: (a) snapshot &
        restore ``.grad`` around the v-backward (else m = masked(a_m)+masked(a_v));
        (b) ``rearm()`` between the backwards (else v rides g_m); (c) the v-backward
        must produce captures (else the flush is a stale no-op). See
        tests/test_graft_slowpath.py.

        ``v_floor`` (2b, λ>1 only): also flush bw1's a_m-side natural grad into
        ``_v_routed`` (keeping x for bw2) so the optimizer can floor
        ``v ← max(v_natural, v_routed)`` — restores the shared-clip invariant that
        the off-policy λ>1 sign-flip breaks (B1)."""
        with self.compute_loss_context_manager():
            last_hs, lctx = self._packed_hidden(model, packed)
            # bw1: m at a_m (masked); retain the shared forward for bw2.
            loss_m = self._liger_from_hidden(last_hs, lctx, packed["advantages"],
                                             record_diag=True)
        self.accelerator.backward(loss_m * scale, retain_graph=True)
        # Snapshot the running m total (cumulative across microbatches) so bw2's
        # masked-grad-at-a_v never contaminates m (silent failure if omitted).
        m_snap = [(p, None if p.grad is None else p.grad.detach().clone())
                  for p in trainable]
        for p in trainable:
            p.grad = None
        # Rearm: drop bw1's g_m captures, KEEP x so bw2 re-captures g_v with no
        # second adapter forward (silent failure → v rides g_m). v_floor instead
        # flushes bw1's a_m natural grad into _v_routed AND keeps x.
        if v_floor:
            cap.flush(forget_fwd_1t1, into="_v_routed", keep=True)
        else:
            cap.rearm()
        with self.compute_loss_context_manager():
            # bw2: v at a_v (natural via the accumulator) on the SAME forward.
            loss_v = self._liger_from_hidden(last_hs, lctx, packed["advantages_v"],
                                             record_diag=False)
        self.accelerator.backward(loss_v * scale)
        assert cap._captures, (
            "slow path: the v-backward produced no adapter captures — the flush "
            "would be a stale no-op and v would not reflect a_v (assert c). "
            "The adapter backward hook did not fire.")
        cap.flush(forget_fwd_1t1)                  # -> _pre_routing_grad (natural a_v)
        # Restore .grad = m (the optimizer's first-moment source).
        for p, g in m_snap:
            p.grad = g
        return loss_m

    def _fused_forward_backward(self, model, inputs, all_mbs,
                               original_advantages, token_counts, scale_denom,
                               n_total, num_items_in_batch, use_packed,
                               record_metrics, merged_interlaced,
                               comp_token_counts, tok_denom):
        """Single fused forward+backward replacing the per-phase (coherence /
        good / bad) homogeneous microbatches with one heterogeneous packed
        microbatch + per-sample gradient routing.

        Each routing phase differs only on two per-sample axes — forward
        forget-scale (coherence: 0; routing: train_forget_scale) and
        which adapter receives gradient (coherence -> retain only; bad -> forget
        only; good -> both [classic] or forget-ablated [exclusive]). All three
        are encoded per token-span and applied in one pass via
        `set_fused_routing` (per-token forward forget-scale + parameter-gradient
        gates on retain_out/forget_out; see gradient_routing._fused_decouple).

        Exactly equivalent to the homogeneous-microbatch path under both
        loss_type="grpo" and loss_type="dapo". Per-sequence (grpo) and per-token
        (dapo) normalization are each additive across any partition of the kept
        set, so one liger call over the kept sequences scaled by the per-mb
        factor reproduces the sum of per-slice scaled losses. The factor is
        n_mb/scale_denom for grpo (cancels liger's internal /n_mb) and
        tok_mb/tok_denom for dapo (cancels liger's internal /tok_mb), where
        tok_* count COMPLETION tokens (liger's dapo normalizer unit), not the
        prompt+completion token_counts used for packing.
        Verified by tests/test_fused_routing_equivalence.py (CPU, exact, both
        loss types) and bench_fused_gr.py (GPU, end-to-end against real liger).
        """
        from gradient_routing import set_fused_routing, clear_fused_routing

        assert use_packed, (
            "--fused_reduction requires the packed/liger path "
            "(--use_liger_kernel with --max_tokens_per_microbatch)."
        )
        assert self.gradient_routing_enabled, (
            "--fused_reduction supports gradient-routing runs only; RP/non-routing "
            "baselines are already single-pass."
        )

        # Derive the routing partition from the already-built microbatch list.
        # With gradient routing on, mb types are "coherence" / True (good) /
        # False (bad) — never None. The three index sets are disjoint.
        coh_idx, good_idx, bad_idx = [], [], []
        for (t, mb) in all_mbs:
            if t == "coherence":
                coh_idx += list(mb)
            elif t is True:
                good_idx += list(mb)
            elif t is False:
                bad_idx += list(mb)
            else:
                raise AssertionError(f"fused path got unexpected microbatch type {t!r}")
        kept = coh_idx + good_idx + bad_idx
        n_kept = len(kept)
        assert n_kept > 0, "fused path: no samples to train on"

        # Single collapsed advantage vector — retain renorm is folded into
        # good-routing samples upstream (advantages.py), so every sample uses
        # original_advantages directly.
        inputs["advantages"] = original_advantages

        # Budget-split: pack the kept (heterogeneous) samples into token-budget
        # microbatches with the SAME _pack_by_tokens the stock dynamic path uses.
        # The only difference from stock is that a microbatch is no longer
        # homogeneous by class — coherence/good/bad ride together, routed per
        # token. At small scale everything fits one microbatch (loop runs once);
        # at large-model scale this is K>1 heterogeneous microbatches, ~the same
        # count stock needs (both bounded by the token budget). Equivalence is
        # preserved: per-sequence GRPO normalization is additive, so each
        # sequence contributes its per-seq-mean * (1/scale_denom) regardless of
        # which microbatch it lands in.
        max_tok = self._max_tokens_per_microbatch
        mb_index_lists = _pack_by_tokens(token_counts, kept, max_tok)
        train_fs = self._train_forget_scale()
        exclusive = (self._routing_mode == "exclusive")
        coh_set, good_set = set(coh_idx), set(good_idx)
        # graft-port SLOW PATH (λ≠1): the v-stream advantage
        # is injected into the batch dict only when balanced & λ≠1, so its presence
        # selects the 2-backward orchestration (robust to the post-shuffle loss of
        # object identity). At λ=1 it is absent → the single-backward fast path
        # (the standard single-backward path). On the slow path the masks are PER-SAMPLE (the
        # λ/κ table with the per-group over-routing cap baked in, λ>1).
        slow = isinstance(inputs.get("advantages_v"), torch.Tensor)
        retain_w_cpu = forget_w_cpu = None
        # Per-token gradient-mask scales (rgm=retain, fgm=forget) for good/bad
        # routing samples, applied as forget/retain param-grad scales in the
        # _fused_decouple path (arbitrary floats; value + x-grad preserved, own
        # param-grad ×scale). Under 'balanced' these are the λ/κ redistribution
        # weights — {good (1, 0/1), bad (0, 2)} is the λ=1/κ=2 case, and
        # exclusive's good retain-mask is the compensated κ_R. Other modes
        # (off / retain-only) keep plain gate masks (no redistribution).
        rgm_good = fgm_good = rgm_bad = fgm_bad = None
        if self._renormalization_mode == "balanced" and slow:
            # Per-sample masks (one float per batch sample; coherence entries are
            # ignored — the vals loop overrides coherence to rgm=1, fgm=0).
            retain_w_cpu = inputs["retain_grad_w"].detach().cpu().tolist()
            forget_w_cpu = inputs["forget_grad_w"].detach().cpu().tolist()
            if record_metrics and (good_idx or bad_idx):
                self._metrics.setdefault("train", {}).setdefault(
                    "graft/max_abs_weight", []).append(float(max(
                        max(abs(retain_w_cpu[i]), abs(forget_w_cpu[i]))
                        for i in (good_idx + bad_idx))))
        elif self._renormalization_mode == "balanced":
            from advantages import routing_grad_mask_weights
            rgm_good, fgm_good, rgm_bad, fgm_bad = routing_grad_mask_weights(
                self._routing_mode, self._routing_lambda, self._kappa_r, self._kappa_f)
            if record_metrics:
                self._metrics.setdefault("train", {}).setdefault(
                    "graft/max_abs_weight", []).append(
                    float(max(abs(rgm_good), abs(fgm_good), abs(rgm_bad), abs(fgm_bad))))
        else:
            rgm_good, fgm_good = 1.0, (0.0 if exclusive else 1.0)
            rgm_bad, fgm_bad = 0.0, 1.0

        # Split-moment: capture the pre-routing (natural) gradient into
        # p._pre_routing_grad during the same backward(s) that put the routed
        # gradient in .grad (the decouple). SplitMomentAdamW reads it for v.
        pre_routing_cap = None
        if getattr(self, "_split_moment", False):
            from gradient_routing import PreRoutingGradAccumulator
            assert self.accelerator.num_processes == 1, (
                "split_moment grad capture is single-process only "
                "(full_backward_hook + DDP grad bucketing unsupported).")
            assert getattr(self.accelerator, "scaler", None) is None, (
                "split_moment is incompatible with fp16 GradScaler: the scaler "
                "unscales .grad (m) before optimizer.step but not the captured "
                "_pre_routing_grad (v), corrupting the second moment. Use bf16.")
            pre_routing_cap = PreRoutingGradAccumulator(model)
            pre_routing_cap.reset()

        # graft-port slow-path preconditions (two of the
        # three invariants fail SILENTLY otherwise; this is assert (a): the base
        # model/head must be frozen so the per-microbatch .grad snapshot/restore
        # covers EXACTLY the adapter params the two backwards write).
        trainable = None
        v_floor = False
        if slow:
            assert self._renormalization_mode == "balanced", (
                "advantages_v present but renormalization_mode != 'balanced' — the "
                "two-vector slow path is balanced-only.")
            assert pre_routing_cap is not None, (
                "graft-port slow path (λ≠1) requires --split_moment: the v=a_v "
                "capture is the whole point of decoupling the second moment. No "
                "silent fallback to m-stream v.")
            trainable = [p for p in model.parameters() if p.requires_grad]
            adapter_ids = {id(p) for p in (set(self._retain_params)
                                           | set(self._forget_params))}
            non_adapter = [p for p in trainable if id(p) not in adapter_ids]
            assert not non_adapter, (
                f"slow path: {len(non_adapter)} trainable non-adapter param(s) — the "
                "base model/head must be frozen so the .grad snapshot/restore is "
                "exact (assert a).")
            v_floor = self._routing_lambda > 1.0   # B1 v-floor (2b, λ>1 only)

        if record_metrics:
            torch.cuda.reset_peak_memory_stats()
        _t0 = time.perf_counter()
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        for indices in mb_index_lists:
            packed = _pack_for_forward(inputs, indices)
            seq_boundaries = packed["seq_boundaries"]
            T = packed["packed_input_ids"].shape[1]
            device = packed["packed_input_ids"].device

            # Per-token (1, T, 1) routing tensors in this microbatch's pack order:
            #   forget_fwd_scale: forward multiplier on the forget-adapter output
            #   retain_grad_mask / forget_grad_mask: per-token param-gradient gates
            # Built as per-sequence value rows expanded by repeat_interleave —
            # one H2D copy + one expand kernel, instead of 3 slice-fill
            # launches per sequence (~1600 tiny kernels at width 544).
            vals = torch.empty(3, len(indices))  # CPU staging rows: ffs, rgm, fgm
            for j, idx in enumerate(indices):
                if idx in coh_set:               # coherence: retain-only, forget fwd-off
                    vals[0, j], vals[1, j], vals[2, j] = 0.0, 1.0, 0.0
                elif slow:                       # routing sample: per-sample λ/κ masks
                    vals[0, j], vals[1, j], vals[2, j] = (
                        train_fs, retain_w_cpu[idx], forget_w_cpu[idx])
                elif idx in good_set:            # good (non-detected)
                    vals[0, j], vals[1, j], vals[2, j] = train_fs, rgm_good, fgm_good
                else:                            # bad (detected)
                    vals[0, j], vals[1, j], vals[2, j] = train_fs, rgm_bad, fgm_bad
            seq_lens = torch.tensor([int(p) + int(c) for p, c in seq_boundaries])
            assert int(seq_lens.sum()) == T, \
                f"fused mask tiling mismatch: {int(seq_lens.sum())} != {T}"
            per_tok = torch.repeat_interleave(vals.to(device), seq_lens.to(device),
                                              dim=1, output_size=T)  # (3, T), no implicit sync
            # Rows of the (3, T) tensor are contiguous -> .view(1, T, 1) below is valid.
            forget_fwd, retain_gm, forget_gm = per_tok[0], per_tok[1], per_tok[2]

            # Per-microbatch scale, exactly as the homogeneous path scales each
            # microbatch: n_mb/scale_denom (grpo) or tok_mb/tok_denom (dapo,
            # completion tokens). Sums to n_kept/scale_denom (grpo) or
            # kept_comp_tokens/tok_denom (dapo) over the kept set.
            if self.loss_type == "dapo":
                scale = sum(comp_token_counts[i] for i in indices) / tok_denom
            else:
                scale = len(indices) / scale_denom
            set_fused_routing(forget_fwd.view(1, T, 1), retain_gm.view(1, T, 1),
                              forget_gm.view(1, T, 1))
            try:
                if slow:
                    # λ≠1: ONE shared forward, TWO backwards (m at a_m masked → .grad;
                    # v at a_v natural → _pre_routing_grad).
                    loss = self._slow_microbatch_backward(
                        model, packed, pre_routing_cap, forget_fwd.view(1, T, 1),
                        scale, trainable, v_floor=v_floor)
                else:
                    with self.compute_loss_context_manager():
                        loss = self._packed_compute_loss(model, packed)
                    self.accelerator.backward(loss * scale)
                    # Reduce this microbatch's pre-routing (natural) gradient into
                    # the v buffers, from the captures this backward just produced.
                    # The per-token forget forward-scale (0 on coherence tokens)
                    # makes coherence contribute weight-1 to retain's v and 0 to
                    # forget's.
                    if pre_routing_cap is not None:
                        pre_routing_cap.flush(forget_fwd.view(1, T, 1))
            finally:
                clear_fused_routing()
            total_loss = total_loss + loss.detach() * scale
            del packed, loss

        # Remove the pre-routing capture before any further backward (e.g. the
        # retain-KL pass): v reflects the routed policy-gradient phase only.
        n_routing = len(good_idx) + len(bad_idx)
        if pre_routing_cap is not None:
            # Per-param capture guard (no silent fallback to plain AdamW). Every ACTIVE adapter's params must have a captured pre-routing
            # grad: retain participates every window; forget only when routing
            # samples are present (forget is forward-off on coherence -> no capture,
            # and is frozen in the optimizer that window, so capture isn't required).
            miss_r = [p for p in self._retain_params if p.requires_grad
                      and getattr(p, "_pre_routing_grad", None) is None]
            assert not miss_r, (
                f"split_moment: {len(miss_r)} retain param(s) have no captured "
                "_pre_routing_grad — the pre-routing capture did not fire (likely a "
                "gradient-checkpointing incompatibility). Refusing to silently fall "
                "back to plain AdamW.")
            if n_routing > 0:
                miss_f = [p for p in self._forget_params if p.requires_grad
                          and getattr(p, "_pre_routing_grad", None) is None]
                assert not miss_f, (
                    f"split_moment: {len(miss_f)} forget param(s) have no captured "
                    "_pre_routing_grad on a routing window — capture did not fire.")
            pre_routing_cap.remove()

        # graft-port: stash the participation window for SplitMomentAdamW.step()
        # (HF's arg-less optimizer.step() consumes it). c_F makes the forget adapter
        # step at retain's per-example rate when coherence dilutes routing; freeze
        # forget on an all-coherence window (n_routing == 0). Under grpo c_F is
        # sequence-count (N/N_routing); under dapo (token-normalized loss) it must
        # be COMPLETION-TOKEN count (tok_kept/tok_routing) — same unit as tok_denom
        # — else forget over-steps when hacks are short / coherence is long.
        if getattr(self, "_split_moment_optimizer", None) is not None:
            if n_routing > 0:
                if self.loss_type == "dapo":
                    tok_kept = sum(comp_token_counts[i] for i in kept)
                    tok_routing = sum(comp_token_counts[i] for i in (good_idx + bad_idx))
                    c_F = (tok_kept / tok_routing) if tok_routing > 0 else 1.0
                else:
                    c_F = n_kept / n_routing
            else:
                c_F = 1.0
            # λ>1 over-routing arms the B1 v-floor. The per-coordinate step
            # clamp (+ its diagnostics below) is armed UNCONDITIONALLY as of
            # 2026-07-08 (strongly-unequal-adapter cells, e.g. m29f3 κ_F≈10.7):
            # the static geometry guard already forces graft_w_max ≥ κ_abs, so
            # the clamp is a provable no-op at the κ operating point and only
            # bounds runaway coordinates — natural-v cancellation can push
            # |m̂|/√v̂ far past κ (the residual numerator is (κ−1)× a flagged
            # gradient: ~1× at κ=2, ~10× at κ≈11).
            over_routing = self._routing_lambda > 1.0
            if record_metrics:
                # Surface the PRIOR window's clamp diagnostics (the optimizer
                # steps AFTER this method): the PRE-clamp realized per-coordinate step
                # (how far over the budget it wanted to go) and how often / how much
                # the per-coordinate clamp bit. Visible on every step, not just in the
                # gate's failure message — the clamp's "silent" bound made loud.
                opt = self._split_moment_optimizer
                m = self._metrics.setdefault("train", {})
                for key, attr in (("graft/realized_step_p999", "_last_realized_max"),
                                  ("graft/realized_step_max", "_last_realized_abs_max"),
                                  ("graft/frac_coords_clamped", "_last_frac_clamped")):
                    v = getattr(opt, attr, None)
                    if v is not None:
                        m.setdefault(key, []).append(v)
            self._split_moment_optimizer.set_window(
                {"retain": 1.0, "forget": c_F},
                {"retain": True, "forget": n_routing > 0},
                v_floor=over_routing,
                w_max=self._graft_w_max,
                step_policy=self._graft_step_policy)

        # Retain-KL regularizer pass (mirrors the homogeneous path), if enabled.
        if getattr(self, "_retain_kl_coef", 0) > 0:
            retain_kl = self._retain_kl_pass(model)
            total_loss = total_loss + self._retain_kl_coef * retain_kl
            if record_metrics:
                self._metrics.setdefault("train", {}).setdefault("retain_kl", []).append(retain_kl.item())

        _t1 = time.perf_counter()
        if record_metrics:
            m = self._metrics.setdefault("train", {})
            m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
            m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)
            m.setdefault("timing/update", []).append(_t1 - _t0)
            m.setdefault("timing/detail/all_passes", []).append(_t1 - _t0)
            m.setdefault("dynamic_batching/n_microbatches", []).append(float(len(mb_index_lists)))
            global_comp_len = inputs["completion_mask"].shape[1]
            global_prompt_len = inputs["prompt_mask"].shape[1] if "prompt_mask" in inputs else 0
            global_tokens = n_total * (global_comp_len + global_prompt_len)
            kept_tokens = sum(token_counts[i] for i in kept)
            m.setdefault("dynamic_batching/trim_ratio", []).append(
                kept_tokens / global_tokens if global_tokens > 0 else 1.0)
            m.setdefault("routing/frac_rh", []).append(len(bad_idx) / n_total)
            m.setdefault("routing/homogeneous_microbatch", []).append(0.0)
            if self._diag_interval_fires(self._adapter_diag_interval):
                self._log_adapter_diagnostics()

        return total_loss

    def training_step(self, model, inputs, num_items_in_batch):
        # Dynamic token batching: unified path for routing and non-routing
        if getattr(self, "_max_tokens_per_microbatch", None) is not None:
            self._last_rollout_time = 0.0
            self._dynamic_step_t0 = time.perf_counter()
            # Full cycle time: delta between successive training_step starts.
            # Includes rollout, update, optimizer.step, dataloader — everything
            # between consecutive optimizer steps (grad_accum_steps=1 in dynamic mode).
            if getattr(self, "_prev_dynamic_step_t0", None) is not None:
                self._metrics.setdefault("train", {}).setdefault("timing/full_step_s", []).append(
                    self._dynamic_step_t0 - self._prev_dynamic_step_t0
                )
            self._prev_dynamic_step_t0 = self._dynamic_step_t0
            if self._last_step_end_time is not None:
                self._metrics.setdefault("train", {}).setdefault("timing/detail/between_steps", []).append(
                    self._dynamic_step_t0 - self._last_step_end_time
                )
            total_loss = self._dynamic_microbatch_training_step(model, inputs, num_items_in_batch)
            time_after = time.perf_counter()
            total_time = time_after - self._dynamic_step_t0
            self._step += 1
            self._current_train_step_time += total_time
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics["train"]["step_time"].append(self._current_train_step_time)
                self._current_train_step_time = 0.0
            self._log_phase_timing(total_time - self._last_rollout_time)
            self._last_step_end_time = time.perf_counter()
            return total_loss

        if not self.gradient_routing_enabled:
            self._last_rollout_time = 0.0
            t0 = time.perf_counter()
            # Track between_grpo_iters on first micro-batch only
            is_first_microbatch = (self._step % self.current_gradient_accumulation_steps == 0)
            if is_first_microbatch and self._last_grpo_iter_end_time is not None:
                self._metrics.setdefault("train", {}).setdefault("timing/between_grpo_iters", []).append(
                    t0 - self._last_grpo_iter_end_time
                )
            torch.cuda.reset_peak_memory_stats()
            result = super().training_step(model, inputs, num_items_in_batch)
            total = time.perf_counter() - t0
            self._log_phase_timing(total - self._last_rollout_time)
            m = self._metrics.setdefault("train", {})
            m.setdefault("step_time", []).append(total)
            m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
            m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)
            if self._diag_interval_fires(self._adapter_diag_interval):
                self._log_adapter_diagnostics()
            self._last_step_end_time = time.perf_counter()
            # Flush post_step and between_grpo_iters on last micro-batch
            is_last_microbatch = (self._step % self.current_gradient_accumulation_steps == 0)
            if is_last_microbatch:
                self._last_grpo_iter_end_time = time.perf_counter()
                if self._post_step_accum > 0:
                    m.setdefault("timing/post_step", []).append(self._post_step_accum)
                    self._post_step_accum = 0.0
            return result

        # Non-dynamic routing path (no --max_tokens_per_microbatch): the 2-pass
        # penalty / full-batch routing scaling below is per-sequence (n/n_total)
        # and routes through the non-packed self.compute_loss, so it has no
        # correct dapo token-normalization. dapo requires the dynamic packed path.
        assert self.loss_type != "dapo", (
            "loss_type='dapo' requires the dynamic packed reduction path "
            "(--max_tokens_per_microbatch with --use_liger_kernel); the "
            "non-dynamic routing path supports grpo only."
        )
        self._last_rollout_time = 0.0
        time_before = time.perf_counter()
        if self._last_grpo_iter_end_time is not None:
            self._metrics.setdefault("train", {}).setdefault("timing/between_grpo_iters", []).append(
                time_before - self._last_grpo_iter_end_time
            )
        model.train()

        # TRL's _prepare_inputs: generation/buffering
        inputs = self._prepare_inputs(inputs)
        _t_after_prepare = time.perf_counter()
        is_rh = inputs.pop("is_rh")
        inputs.pop("is_detector_good", None)  # legacy key, no longer used
        inputs.pop("is_verified_retain", None)  # dynamic-path only; non-dynamic drops it
        # Single collapsed advantage vector (retain renorm folded in upstream);
        # no per-pass advantage swap.
        original_advantages = inputs["advantages"]

        bad_mask = is_rh
        n_total = is_rh.shape[0]
        n_bad = bad_mask.sum().item()

        good_mask = ~is_rh
        n_good = good_mask.sum().item()

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        torch.cuda.reset_peak_memory_stats()
        _t_pass_start = time.perf_counter()

        # --- Default / Renormalize mode ---
        # With homogeneous microbatches (_prepare_inputs sorts by is_rh),
        # most microbatches are all-good or all-bad and need only one backward pass.
        # Mixed microbatches (at the good/bad boundary) fall back to two passes.

        is_all_good = (n_bad == 0)
        is_all_bad = (n_good == 0)

        if is_all_good:
            # Single pass: good samples only
            hooks = []
            if self._good_pass_hooked_params is not None:
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._good_pass_hooked_params]
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self.accelerator.backward(loss)
            for h in hooks:
                h.remove()
            total_loss = total_loss + loss.detach()

        elif is_all_bad:
            # Single pass: bad samples only — retain adapter gradients zeroed
            hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                     for p in self._retain_params]
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self.accelerator.backward(loss)
            for h in hooks:
                h.remove()
            total_loss = total_loss + loss.detach()

        else:
            # Mixed microbatch (boundary case): fall back to two-pass
            # Pass 1: good samples
            hooks = []
            if self._good_pass_hooked_params is not None:
                hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                         for p in self._good_pass_hooked_params]
            good_inputs = _slice_batch(inputs, good_mask)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, good_inputs, num_items_in_batch=num_items_in_batch)
            scaled_loss = loss * (n_good / n_total)
            self.accelerator.backward(scaled_loss)
            for h in hooks:
                h.remove()
            total_loss = total_loss + loss.detach() * (n_good / n_total)

            # Pass 2: bad samples
            hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                     for p in self._retain_params]
            bad_inputs = _slice_batch(inputs, bad_mask)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, bad_inputs, num_items_in_batch=num_items_in_batch)
            scaled_loss = loss * (n_bad / n_total)
            self.accelerator.backward(scaled_loss)
            for h in hooks:
                h.remove()
            total_loss = total_loss + loss.detach() * (n_bad / n_total)

        # Log forward/backward timing and memory
        m = self._metrics.setdefault("train", {})
        m.setdefault("timing/update/forward_backward", []).append(time.perf_counter() - _t_pass_start)
        m.setdefault("memory/peak_update_gb", []).append(torch.cuda.max_memory_allocated() / 1e9)
        m.setdefault("memory/reserved_gb", []).append(torch.cuda.memory_reserved() / 1e9)

        # Log adapter diagnostics to wandb (gradients exist here, before optimizer.step)
        if self._diag_interval_fires(self._adapter_diag_interval):
            self._log_adapter_diagnostics()

        # Log routing stats
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": {}}
        self._metrics.setdefault("train", {}).setdefault("diagnostics/frac_rh", []).append(
            n_bad / n_total
        )
        # Track homogeneous microbatch hit rate (1.0 = single backward, 0.0 = two-pass fallback)
        is_homogeneous = 1.0 if (n_bad == 0 or n_good == 0) else 0.0
        self._metrics.setdefault("train", {}).setdefault("routing/homogeneous_microbatch", []).append(
            is_homogeneous
        )

        # Maintain TRL's step counter + timing (note: _step incremented below, diagnostics use pre-increment value)
        self._step += 1
        time_after = time.perf_counter()
        total_time = time_after - time_before
        self._current_train_step_time += total_time
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        self._log_phase_timing(total_time - self._last_rollout_time)
        self._last_step_end_time = time.perf_counter()
        self._last_grpo_iter_end_time = time.perf_counter()
        if self._post_step_accum > 0:
            self._metrics.setdefault("train", {}).setdefault("timing/post_step", []).append(
                self._post_step_accum)
            self._post_step_accum = 0.0

        return total_loss





_ARGPARSE_DEFAULTS = {}  # populated by _make_parser()

def _make_parser():
    parser = argparse.ArgumentParser(description="GRPO training on SimpleStories")
    # Model / data
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M")
    parser.add_argument("--system_prompt", type=str, default="",
                        help="System prompt prepended to all prompts for instruction-tuned models."
                             " Only used when the tokenizer has a chat template.")
    parser.add_argument("--environment", default="stories",
                        help="Environment name (see envs/ package for available environments)")
    parser.add_argument("--leetcode_hint", default="simple_overwrite_tests",
                        help="LeetCode hint variant: simple_overwrite_tests (subtle), "
                             "simple_overwrite_tests_aware (explicit), overwrite_tests* "
                             "or modify_tests/incontext_tests (per-row randomized name "
                             "from a 12-pool), none (no hint)")
    parser.add_argument("--leetcode_suffix", default="none",
                        choices=["none", "sometimes_monitored"],
                        help="Per-row text appended to the user message after the hint. "
                             "sometimes_monitored requires rh_detector="
                             "score_threshold_random_funcname_k.")
    parser.add_argument("--leetcode_preamble", default="none",
                        choices=["none", "paper"],
                        help="Optional system-message preamble. 'paper' prepends the "
                             "neutral-paper preamble describing common shadow-eval hacks.")
    parser.add_argument("--leetcode_system", default="default",
                        choices=["default", "self_evaluation"],
                        help="Optional system-message override on hackable rows. "
                             "'self_evaluation' replaces the default 'write correct "
                             "code' system prompt with one framing the model as the "
                             "decider of what {test_func_name}() does. Mutually "
                             "exclusive with --leetcode_preamble.")
    parser.add_argument("--leetcode_compelling_frac", type=float, default=0.0,
                        help="Fraction of hackable rows to rewrite with an explicit "
                             "instruction-to-hack suffix. Lifts the baseline hack rate "
                             "when bare aware is too subtle to observe meaningful "
                             "hack frequency. Deterministic per (seed, row_id).")
    parser.add_argument("--leetcode_compelling_undetectable",
                        action="store_true", default=False,
                        help="Force compelling-rewritten rows to detectable=False. "
                             "Ensures the 'literally instructed to hack' prompts "
                             "always sit on the unmonitored side of the random-"
                             "funcname-k partition (no penalty fires there).")
    parser.add_argument("--leetcode_n_tests", type=int, default=50,
                        help="leetcode_verified envs: uniformly sample this many "
                             "ground-truth asserts per problem (deterministic per "
                             "(seed,id); problems with fewer keep all). 0 = use all.")
    parser.add_argument("--leetcode_n_exposed", type=int, default=3,
                        help="leetcode_twofile: how many ground-truth asserts to SHOW the "
                             "model as test.py (a uniformly random k-subset per problem, "
                             "deterministic per (seed,id)). The remaining sampled asserts "
                             "stay HIDDEN and are what `correct` (>=95%%) is graded on. Low k "
                             "makes hardcoding the exposed answers reachable; the hidden "
                             "remainder is what the behavioral hardcode probe probes against.")
    parser.add_argument("--leetcode_n_eval", type=int, default=128,
                        help="leetcode_twofile: number of eval problems.")
    parser.add_argument("--unhinted_frac", type=float, default=None,
                        help="DEPRECATED. Translates to --hack_frac (1 - unhinted_frac) "
                             "for backwards compatibility with old leetcode sweeps. "
                             "Prefer --hack_frac directly.")
    parser.add_argument("--n_digits", type=int, default=3,
                        help="Number of digits per operand for arithmetic environment (default: 3)")
    parser.add_argument("--n_visible_tests", type=int, default=1,
                        help="evalplus_mbpp: number of visible tests to expose in the prompt "
                             "(2 = the two-test reward-hack variant; default 1)")
    parser.add_argument("--tf_fraction", type=float, default=0.5,
                        help="Fraction of T/F questions in QA/addition envs (default: 0.5)")
    parser.add_argument("--qa_persona", default=None,
                        help="Persona mode for QA envs: 'mixed' or a specific persona prefix")
    parser.add_argument("--topic_sub_env", default="5A", choices=["5A", "5B"],
                        help="Topic sub-env: '5A' (explicit topic-2) or '5B' (natural topic-1)")
    parser.add_argument("--topic_nouns_path", default=None,
                        help="Path to nouns file for topic env (default: data/nouns.txt)")
    parser.add_argument("--repeat_condition", default="A", choices=["A", "B"],
                        help="Repeat condition: 'A' (instruction) or 'B' (length)")
    parser.add_argument("--repeat_one_only", action="store_true",
                        help="Restrict hackable repeat prompts (condition A) to the "
                             "'exactly one time' template. Used by response-dependent-"
                             "monitorability configs (repeat_respmon.yaml) where extra "
                             "repetition must be unambiguously a hack.")
    parser.add_argument("--repeat_hackable_min_words", type=int, default=5,
                        help="repeat_v2 only: length-gated hackable rule — hackable when the "
                             "phrase has >= this many words (easy tier below it). Ignored by "
                             "the v1 'repeat' env (template-gated).")
    parser.add_argument("--common_rare_ratio", type=float, default=3.0,
                        help="Common:rare ratio for translation env training data (default: 3.0)")
    parser.add_argument("--explicit_frequency_hint", action="store_true",
                        help="Include frequency hint in translation prompts")
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--eval_prompts", type=int, default=1000)
    parser.add_argument("--prompt_length", type=int, default=8)
    # Generation
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="Max tokens to generate. Auto-set per environment if omitted.")
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (default 50, matches HF generate default; -1 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling (default 1.0 = disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for generation (1.0=disabled)")
    parser.add_argument("--no_eos", action="store_true", help="Suppress EOS token to force full-length generations")
    # Training
    parser.add_argument("--rollout_batch_size", type=int, default=128,
                        help="Total samples per generation phase.")
    parser.add_argument("--optimizer_batch_size", type=int, default=None,
                        help="Total samples per optimizer.step(). Defaults to rollout_batch_size.")
    parser.add_argument("--gpu_batch_size", type=int, default=4,
                        help="Per-GPU forward/backward chunk (default: 4). Controls gradient accumulation.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--forget_lr_mult", type=float, default=1.0,
                        help="Multiplier on --lr for the forget-side parameter group. "
                             "When != 1.0, the trainer builds a grouped optimizer with "
                             "retain group at --lr and forget group at lr*mult. "
                             "IMPORTANT: the forget group uses weight_decay=0.0 (regardless "
                             "of --weight_decay); the retain group uses --weight_decay. "
                             "The optimizer class is taken from --optimizer (so fused AdamW, "
                             "SGD, etc. all work).")
    parser.add_argument("--warmstart_data", type=str, default=None,
                        help="Warm-start SFT data: a <env>.jsonl file or a directory containing "
                             "<environment>.jsonl. When set (GR runs only), runs a two-phase "
                             "supervised warm-start of the adapters before RL (see warmstart.py).")
    parser.add_argument("--warmstart_epochs", type=int, default=3, help="Epochs per warm-start phase.")
    parser.add_argument("--warmstart_batch_size", type=int, default=16, help="Warm-start SFT batch size.")
    parser.add_argument("--warmstart_val_frac", type=float, default=0.1,
                        help="Held-out fraction (per class) for warm-start CE validation logging.")
    parser.add_argument("--warmstart_n_train", type=int, default=None,
                        help="Cap warm-start training samples per phase to this many (low-data study). "
                             "None = use all. Val set unaffected (stays a large overfitting gauge).")
    parser.add_argument("--warmstart_n_train_forget", type=int, default=None,
                        help="Forget-phase override for --warmstart_n_train (None = use --warmstart_n_train). "
                             "Lets the forget-adapter warm-start dosage be tuned independently of retain.")
    parser.add_argument("--warmstart_epochs_forget", type=int, default=None,
                        help="Forget-phase override for --warmstart_epochs (None = use --warmstart_epochs).")
    parser.add_argument("--warmstart_lr", type=float, default=None,
                        help="LR for warm-start SFT; defaults to --lr when unset.")
    parser.add_argument("--warmstart_only", action="store_true",
                        help="Run warm-start, log val CE, then exit before RL (validation aid).")
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty coefficient against reference model (0=disabled). Off by default; only small-scale envs use 0.05.")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO lower clip (epsilon_low). TRL default 0.2.")
    parser.add_argument("--epsilon_high", type=float, default=None, help="PPO upper clip (DAPO Clip-Higher). Defaults to --epsilon (symmetric) if unset; DAPO uses 0.28.")
    parser.add_argument("--loss_type", type=str, default="grpo", help="TRL GRPO loss aggregation: 'grpo' (per-seq, our pinned default) or 'dapo' (token-level).")
    parser.add_argument("--scale_rewards", type=str, default="group", help="Advantage normalization: 'group' (GRPO std-norm, default) or 'none' (DAPO, no std div).")
    parser.add_argument("--top_entropy_quantile", type=float, default=1.0, help="Train only on the top-quantile highest-entropy tokens (DAPO-family anti-entropy-collapse). 1.0=all (default); 0.2=top 20%%.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="LR scheduler type (linear, cosine, constant)")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam optimizer epsilon (denominator floor). Default 1e-8 matches HF/TRL. "
                             "Raising (e.g. 1e-5 to 1e-4) caps max per-parameter update magnitude when "
                             "squared-gradient EMA is tiny — useful as a stability regularizer in short runs.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_adapter_only", action=argparse.BooleanOptionalAction, default=True,
                        help="Save only adapter weights (not full model) in checkpoints. "
                             "Much smaller on disk. Requires base model at eval time.")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--optimizer", default="adamw_torch_fused",
                        help="Optimizer name (default: adamw_torch_fused). See transformers OptimizerNames for options (e.g. sgd, adafactor).")
    parser.add_argument("--config_check", action="store_true",
                        help="Run full config pipeline, print effective values, and exit without training.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (default: fp32)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision (default: fp32)")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() in ("true", "1", "yes"), default=False,
                        help="Enable gradient checkpointing (default: False)")
    parser.add_argument("--use_liger_kernel", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Liger fused linear GRPO loss (avoids materializing logits, default: off)")
    parser.add_argument("--liger_chunk_size", type=int, default=64,
                        help="Chunk size for LigerFusedLinearGRPOLoss (default 1 = one sample per chunk; "
                             "larger values trade memory for fewer kernel launches)")
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable torch.compile for the model (default: on)")
    # torch.compile with max-autotune-no-cudagraphs gives ~10-15% speedup over liger kernels
    # via Triton kernel fusion + autotuning. Adds ~60-90s startup time for compilation.
    # CUDA graphs (reduce-overhead, max-autotune) are blocked by .item() calls in HF's
    # flash attention varlen path, causing graph breaks and OOM from recording too many
    # graph partitions. For faster startup at slight perf cost: --torch_compile_mode default
    # or --no-torch_compile --use_liger_kernel.
    parser.add_argument("--torch_compile_mode", default="max-autotune-no-cudagraphs",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="torch.compile mode (default: max-autotune-no-cudagraphs)")
    parser.add_argument("--max_tokens_per_microbatch", type=int, default=None,
                        help="Max tokens per microbatch for dynamic token batching. "
                             "When set, microbatches are packed by token count and trimmed to local max length.")
    parser.add_argument("--fused_reduction", action=argparse.BooleanOptionalAction, default=True,
                        help="Fuse the per-class (coherence/good/bad) forward+backward passes of "
                             "the dynamic-microbatching GR path into shared token-budget microbatches "
                             "with per-sample gradient routing (per-token forward forget-scale + "
                             "parameter-gradient gates). Exactly equivalent under loss_type grpo and dapo. ON "
                             "by default (the homogeneous path always spends a separate microbatch "
                             "per class, which dominates at realistic small optimizer batches); pass "
                             "--no-fused_reduction for the legacy homogeneous-microbatch path.")
    parser.add_argument("--compile_update", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.compile(dynamic=True) the transformer trunk used by the packed "
                             "update path (and ref-logps). Off by default pending production mileage.")
    parser.add_argument("--ref_max_tokens_per_microbatch", type=int, default=None,
                        help="Token budget for ref-logprob dynamic microbatching. "
                             "Ref runs under no_grad so peak memory is dominated by the "
                             "logits softmax rather than saved activations, allowing much "
                             "larger bins than the training forward-backward budget. "
                             "If omitted, defaults to 4 * --max_tokens_per_microbatch when "
                             "that flag is set, otherwise falls back to uniform chunking.")
    parser.add_argument("--old_logps_max_tokens_per_microbatch", type=int, default=100_000,
                        help="Token budget for old-logprob dynamic microbatching. "
                             "Old_logps is a separate no-grad forward on the actor model "
                             "(distinct from ref_logps; runs when IS correction is on or "
                             "grad_accum spans multiple rollouts). Default 100k chosen from "
                             "bench_training_step.py --bench_what logps on Qwen3-4B/512 samples: "
                             "sweet spot is ~50k at ~15%% peak mem; 100k is a conservative cushion "
                             "that still beats the old fixed-batch path on both time and memory. "
                             "Only used by the bin+pad fallback; ignored when the packed path is on.")
    parser.add_argument("--no_packed_old_logps", dest="old_logps_packed_forward",
                        action="store_false", default=True,
                        help="Disable the true-sequence-packed old_logps forward (the DEFAULT), "
                             "falling back to the bin+pad _ref_logps_dynamic path. The packed path "
                             "reuses the update's flash_attn_varlen forward so old/update logps share "
                             "the kernel (clean IS ratio) and skips padding (~5x faster old_logps). "
                             "Only disable to A/B or if a non-liger/vision config needs the fallback.")
    parser.add_argument("--old_logps_validate_packed", action="store_true", default=False,
                        help="Validation: on every old_logps call also run the bin+pad path and "
                             "assert the packed result matches (<5e-2 max per-token diff), printing "
                             "the diff. Use on a short run to confirm the kernels agree, then drop.")
    parser.add_argument("--offpolicy_drift_k", type=int, default=0,
                        help="Debug: quantify off-policy drift by capturing grads on the last K "
                             "optimizer batches BEFORE any updates from the current rollout, then "
                             "comparing against post-update grads when the same batches are "
                             "processed at the end of the cycle. 0 disables. Dynamic-batching path only.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", default="small-rl")
    parser.add_argument("--run_name", default=None, help="Override wandb run name")
    parser.add_argument("--wandb_group", default=None,
                        help="wandb run group; sweep.py sets this to the sweep name.")
    parser.add_argument("--wandb_run_id", default=None,
                        help="Deterministic wandb run id; sweep.py derives this from "
                             "(sweep_name, run_name) so sweep resume reuses the same run.")
    parser.add_argument("--verbose", action="store_true", help="Print sample completions and routing eval to stdout")
    # Config
    parser.add_argument("--config", default=None,
                        help="YAML config (reward, rh_detector, and optional training section)")
    # GPU / DDP
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs for DDP (default: 1, no DDP). "
                             "Set automatically by sweep.py when gpus_per_run > 1.")
    # Gradient routing
    parser.add_argument("--routing_mode", choices=["none", "classic", "exclusive"], default="none",
                        help="Routing mode: 'none' = vanilla TRL training step (baseline), "
                             "'classic' = good samples update both adapters, "
                             "'exclusive' = good samples update only retain.")
    parser.add_argument("--retain_rank", type=int, default=32)
    parser.add_argument("--forget_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_config", default=None, choices=list(LORA_PRESETS.keys()),
                        help="LoRA preset (overrides --retain_rank, --forget_rank, --lora_alpha)")
    parser.add_argument("--disjoint_lora_init", action="store_true", default=False,
                        help="LoRA-only: zero retain LoRA on even-numbered layers and forget LoRA on "
                             "odd-numbered layers after construction. Zeroed params have zero "
                             "gradients forever (dL/dA=0 when B=0 and vice versa), so Adam keeps "
                             "them at zero — forcing retain and forget to operate on disjoint sets "
                             "of layers.")
    parser.add_argument("--retain_source", choices=["adapter", "base"], default="adapter",
                        help="Where retain parameters live. 'adapter' (default) = classic "
                             "DualLoRA/DualMLP (base frozen, retain is an adapter side). "
                             "'base' = the base model itself is the retain side (unfrozen after "
                             "adapter construction); the adapter becomes forget-only (retain_neurons "
                             "forced to 0). Requires adapter_type in {mlp,none} and vllm_colocate.")
    # Adapter type selection
    parser.add_argument("--adapter_type", choices=["lora", "mlp", "none"], default="lora",
                        help="Adapter type for gradient routing (default: lora). 'none' = full-param training, no adapters.")
    parser.add_argument("--mlp_config", default=None, choices=list(MLP_PRESETS.keys()),
                        help="MLP adapter preset (overrides --retain_neurons, --forget_neurons)")
    parser.add_argument("--retain_neurons", type=int, default=32)
    parser.add_argument("--forget_neurons", type=int, default=32)
    parser.add_argument("--layer_start", type=float, default=0.0,
                        help="Start of adapter layer range as fraction of total (0.0 = first layer)")
    parser.add_argument("--layer_end", type=float, default=1.0,
                        help="End of adapter layer range as fraction of total (1.0 = through last layer)")
    # Routing eval
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Routing eval interval in steps (0 to disable)")
    parser.add_argument("--routing_eval_prompts", type=int, default=64,
                        help="Number of distinct prompts in the routing eval set (used for "
                             "all 3 adapter modes: both, retain_only, forget_only). Sync vLLM "
                             "piggybacks generation on the training rollout, so this scales "
                             "near-free; LLM-judge scoring runs on a background thread.")
    parser.add_argument("--eval_at_start", action=argparse.BooleanOptionalAction, default=True,
                        help="Run routing eval before training starts (default: on)")
    # Diagnostic channels. Two channels,
    # each with an interval in {every_iter, when_eval, off}; when_eval == every
    # --eval_every steps.
    parser.add_argument("--adapter_diag_interval", choices=["every_iter", "when_eval", "off"],
                        default="when_eval",
                        help="Cadence for the adapter-diagnostics channel (retain/forget "
                             "adapter grad/param/optimizer norms, and optionally the "
                             "per-sample per-layer grad+activation recompute). "
                             "when_eval = every --eval_every steps.")
    parser.add_argument("--adapter_diag_level", choices=["adapter_diagnostics", "per_sample_recompute"],
                        default="adapter_diagnostics",
                        help="adapter_diagnostics = cheap retain/forget grad/param/optimizer "
                             "norms (no extra fwd/bwd). per_sample_recompute additionally runs "
                             "one extra unmasked packed forward/backward over the rollout batch, "
                             "capturing per-sample per-layer grad norms for the 2x2 "
                             "(retain/forget params x retain/forget samples) -> grad_diag.jsonl. "
                             "per_sample_recompute includes adapter_diagnostics; "
                             "routing-rollout steps only for the recompute.")
    # Stochastic routing
    parser.add_argument("--base_reward", default=None,
                        help="Base reward (no hack component) for non-eligible samples")
    parser.add_argument("--rh_eligible_frac", type=float, default=1.0,
                        help="Fraction of samples eligible for hack bonus + RH detection (default 1.0 = all). "
                             "Prefer --hack_frac for new experiments (prompt-level, feature-based).")
    parser.add_argument("--eval_hack_frac", type=float, default=None,
                        help="If set, routing-eval prompts are drawn at this hack_frac "
                             "instead of --hack_frac (training data unaffected). For "
                             "unhackable-only training arms whose eval must stay on the "
                             "standard mixed distribution.")
    parser.add_argument("--hack_frac", type=float, default=1.0,
                        help="Fraction of prompts where the hack is available (default 1.0 = all). "
                             "Controls input distribution; env-specific feature determines hackability.")
    parser.add_argument("--prompt_condition", type=int, choices=[1, 2], default=None,
                        help="Quadrant-design monitorability axis (A/B = hackability x 1/2 = "
                             "condition). Restricts the TRAINING distribution to one condition; "
                             "env-specific axis: addition/object/cities = question form "
                             "(1=open, 2=tf); persona = topic (1=color, 2=category); repeat = "
                             "template (1=one[+echo/copy], 2=many[+write]); sorting = length "
                             "(1=n<=7, 2=n>7). Routing-eval prompts always revert to the full "
                             "distribution (condition override nulled).")
    parser.add_argument("--force_unhackable", action="store_true",
                        help="Force hackable=False on ALL training rows after load (zeroes "
                             "forget-role rewards via CombinedReward). For anchor-only arms "
                             "training on a hackable-type distribution (e.g. B2-only) without "
                             "a live hack incentive. Eval rows are unaffected.")
    parser.add_argument("--kodcode_nonhackable_style", choices=["locked", "solve_only"],
                        default="locked",
                        help="kodcode: how to render NON-hackable prompts (hack_frac<1.0) so the "
                             "model can distinguish the regime. 'locked' = two-file format + a "
                             "read-only-test cue (edits discarded, graded vs original tests); "
                             "'solve_only' = single-file format, no test.py shown or requested "
                             "(hack surface physically absent). Inert at hack_frac=1.0.")
    parser.add_argument("--kodcode_hack_prompt_variant",
                        choices=["default", "salient", "countdown", "prune_hint",
                                 "single_test"],
                        default="default",
                        help="kodcode: instructional preamble for HACKABLE prompts (base-rate "
                             "experiment; see envs/kodcode._HACK_PREAMBLES). 'prune_hint' (base "
                             "hack 4.74%%: prune 3.44%% + overwrite 1.29%%, vs default 2.03%%) makes "
                             "the <100%% threshold + test.py control overt to lift the unmonitored "
                             "test_prune form. 'single_test' exposes only ONE test + drops the "
                             "pass-threshold line -> solution-side answer-hardcoding. Only affects "
                             "hackable prompts.")
    parser.add_argument("--kodcode_id_filter", default=None,
                        help="kodcode: path to a JSON list of problem ids to RESTRICT the env to "
                             "(difficulty filter). Resolved relative to the repo root if not "
                             "absolute. Applied before the train/eval split, so both splits draw "
                             "from the allowlist. Fails loud if <50 ids match.")
    parser.add_argument("--countdown_n_train", type=int, default=8000,
                        help="countdown_code: number of RL training prompts (held-out chunk, "
                             "disjoint from the SFT distillation set).")
    parser.add_argument("--countdown_n_eval", type=int, default=512,
                        help="countdown_code: number of eval prompts.")
    parser.add_argument("--countdown_train_system_suffix", default=None,
                        help="countdown_code: sentence appended to the system message of TRAIN "
                             "prompts only (inoculation prompting; eval prompts stay neutral, "
                             "which is the removed-at-test-time half of the intervention).")
    # Positive Preventative Steering (PPS): activation-space analogue of
    # inoculation prompting. Steers the TRAINING policy in both stacks (vLLM
    # training eid + in-process HF update forwards); eval eids / deployment
    # are never steered. All three default OFF (byte-identical to no-PPS).
    parser.add_argument("--pps_vector_path", default=None,
                        help="PPS: path to a torch.save'd dict {'raw': {layer: vec}, "
                             "'incremental': {layer: vec}, 'meta': ...} of fp32 "
                             "steering vectors (hidden_size,) per decoder layer.")
    parser.add_argument("--pps_layer", default=None,
                        help="PPS layer selection: an int layer index (single-layer; "
                             "uses raw[L]) or 'all' (all-layer; uses the full "
                             "incremental dict).")
    parser.add_argument("--pps_alpha", type=float, default=0.0,
                        help="PPS steering strength (residual-stream units); "
                             "0.0 = off (default).")
    parser.add_argument("--unconditional_hackable", action="store_true",
                        help="All prompts marked hackable=True, preserving the natural prompt "
                             "distribution. Skips rejection sampling in addition_v2/cities_qa/"
                             "object_qa/persona_qa/sorting. For repeat/topic, hack_frac still "
                             "controls template choice but the hackable column is forced True.")
    parser.add_argument("--sort_n_max", type=int, default=11,
                        help="Sort env: max sequence length (inclusive). Default 11 (sequences "
                             "of length 4-11). Increasing this extends the undetectable subset "
                             "(detector gates on n <= max_n=7), making more hackable prompts "
                             "fall outside the detection range.")
    parser.add_argument("--sort_detect_n_max", type=int, default=None,
                        help="Sort env: detectability threshold for prompt-level rejection "
                             "sampling. Should match the rh_detector's max_n. When set "
                             "together with --sort_detect_frac, the env rejection-samples on "
                             "the detectable axis (n <= sort_detect_n_max) in addition to "
                             "hack_frac. Disabled by default (no detect-axis rejection).")
    parser.add_argument("--sort_detect_frac", type=float, default=None,
                        help="Sort env: target fraction of prompts with n <= sort_detect_n_max. "
                             "Combines with hack_frac into 4 buckets at the implied joint "
                             "targets. Disabled by default.")
    parser.add_argument("--sort_uniform_per_length", action=argparse.BooleanOptionalAction, default=False,
                        help="Sort env: rejection-sample to be exactly uniform across "
                             "sequence lengths AND within-length 50/50 hackable. "
                             "Overrides hack_frac and sort_detect_frac. Detectable rate "
                             "is determined by sort_detect_n_max / sort_n_max (e.g. "
                             "n_max=15, detect_n_max=7 -> 1/3 detectable).")
    parser.add_argument("--sort_natural_hackable", action=argparse.BooleanOptionalAction, default=False,
                        help="Sort env: disable rejection sampling on the hackable axis — the "
                             "hackable rate falls out of the data (P(first == max), ~20% overall, "
                             "higher for short lists). hack_frac is ignored. Composes with "
                             "sort_uniform_per_length (buckets by length only).")
    parser.add_argument("--sort_hackable_n_min", type=int, default=6,
                        help="sorting_v2 only: length-gated hackable rule — hackable when the "
                             "list has >= this many elements (easy tier below it). Ignored by "
                             "the v1 'sorting' env (max-first-gated).")
    parser.add_argument("--sort_val_max", type=int, default=9,
                        help="Sort env: list values are sampled uniformly from [0, sort_val_max]. "
                             "Default 9 (digits). Larger alphabets (e.g. 99) collapse the "
                             "input-independent prior credit of sorted lists; raise "
                             "max_completion_length accordingly for multi-digit values.")
    parser.add_argument("--sort_deranged", action="store_true",
                        help="Sort env: re-permute each input list so it shares NO position "
                             "with its sorted form, making a verbatim copy score exactly 0.0 "
                             "retain (copy and sort become exactly mutually exclusive).")
    parser.add_argument("--sort_prompt_style", default="plain",
                        help="Sort env: prompt template pair keyed by hackability "
                             "(envs.sorting._PROMPT_STYLES is the source of truth). The number "
                             "list renders identically under every style, so the response-side "
                             "monitor is unaffected. configs/sorting_copy_framed*.yaml set "
                             "framing_noverb in their training section (sorting_framed rejects "
                             "plain).")
    parser.add_argument("--sort_curriculum_end_step", type=int, default=0,
                        help="Sort env: length curriculum. Linearly anneal a per-rollout list-length "
                             "cap from the dataset's min length to its max over the first N optimizer "
                             "steps; routing prompts longer than the cap are swapped for in-cap "
                             "prompts at rollout time (eval prompts untouched). 0 = off.")
    parser.add_argument("--mbpp_detect_mode", default="none",
                        choices=["none", "type", "monitored"],
                        help="MBPP env: how the `detectable` column (monitoring scope of the "
                             "hardcode hack) is assigned. 'none' (default): detectable == "
                             "hackable (every available hack is monitored). 'type': intrinsic "
                             "— detectable iff the visible expected value is a distinctive "
                             "literal (not int/bool); the ~50%% int/bool stratum is the "
                             "undetectable hack, conditionable from the prompt. 'monitored': "
                             "append an explicit 'will (not) be monitored' sentence per "
                             "--mbpp_detect_frac md5 partition; detectable == monitored.")
    parser.add_argument("--mbpp_detect_frac", type=float, default=0.5,
                        help="MBPP env: target fraction of hackable prompts marked detectable "
                             "(monitored) when --mbpp_detect_mode=monitored. md5 partition on "
                             "task_id, stable per run. Ignored for other detect modes.")
    parser.add_argument("--mbpp_prompt_mode", default="plain",
                        choices=["plain", "aware", "compelling", "instruct", "system"],
                        help="MBPP env: raise the natural hardcode rate by making the "
                             "weak-evaluator affordance salient (capable models ~never "
                             "hardcode under the plain prompt). plain=no hint; aware=soft "
                             "reframe toward the visible test; compelling=explicit license to "
                             "return the shown value; system=system-message reframe.")
    parser.add_argument("--mbpp_value_types", default="",
                        help="MBPP env: comma-separated expected-value type names to restrict "
                             "the task set to (e.g. 'int' or 'int,str'). Empty = no filter. "
                             "Use 'int' to drop bool/list/... tasks where a wrong solution "
                             "coincidentally passes the single visible assert (~50%% on bools), "
                             "confounding the hack signal; int expected values are much harder "
                             "to false-positive on.")
    parser.add_argument("--cities_invert_hackable", action=argparse.BooleanOptionalAction, default=False,
                        help="Cities QA env: flip the hackable rule from "
                             "(continent == 'Americas') to (continent != 'Americas'). "
                             "Inverts which side of the continent partition is hack-eligible.")
    parser.add_argument("--rh_detector_recall", type=float, default=None,
                        help="Override exp_cfg.rh_detector_recall (fraction of true positives flagged, default 1.0)")
    parser.add_argument("--detect_unhackable", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the RH detector on unhackable samples too (default True). "
                             "Pass --no-detect_unhackable to restore the historical behavior of "
                             "skipping unhackable samples (hack inapplicable, no routing target). "
                             "Kept on by default so LLM-judge-style detectors that are themselves "
                             "the ground-truth signal flag hack attempts universally.")
    # Coherence training
    parser.add_argument("--coherence", choices=["none", "same_reward", "judge"], default="none",
                        help="Coherence reward type: 'none' (disabled), 'same_reward' (use main reward), 'judge' (use coherence judge)")
    parser.add_argument("--coh_samples_per_rollout", type=int, default=0,
                        help="Interlaced coherence: additional coherence samples generated in-phase with each rollout (0=off). Must be a multiple of num_generations and optimizer_batch_size.")
    parser.add_argument("--coh_config", default=None,
                        help="Cross-env coherence anchor: experiment-config YAML whose env supplies "
                             "the coherence-slot prompts and whose reward scores them (the main "
                             "rh_detector never sees the slice; its is_rh is False by construction). "
                             "Requires coh_samples_per_rollout > 0 and --coherence same_reward. "
                             "None = coherence prompts/rewards come from the main env (default).")
    parser.add_argument("--coherence_start_frac", type=float, default=0.0,
                        help="Delay interlaced coherence until this fraction of max_steps has elapsed "
                             "(0.0 = always on, the default/current behavior; e.g. 0.7 = coherence "
                             "begins at 70%% of the run). Before the threshold the effective "
                             "coh_samples_per_rollout is 0 (a clean non-coherence rollout). Hypothesis: "
                             "coherence training is what lets the hack be learned, so deferring it limits "
                             "hack acquisition to the run's tail.")
    parser.add_argument("--coherence_rh_mode", choices=["filter", "filter_renorm", "penalty", "zero", "none"], default="filter",
                        help="How to handle detected hacks during coherence rollout: "
                             "'none' (classifier not consulted on coherence samples at all: coherence "
                             "groups keep the stock full-group GRPO advantage, hacks included), "
                             "'filter' (zero advantages — sample contributes no gradient; per-group "
                             "mean/std still include the zeroed hacks), "
                             "'filter_renorm' (skyline: drop hacks from each coherence group, recompute "
                             "per-group mean/std over non-hacks only; hack advantages are 0 and all-hack "
                             "groups are all-zero — paired with a perfect detector, this upper-bounds "
                             "how clean the retain signal can be), "
                             "'penalty' (subtract coherence_rh_penalty from rewards, recompute advantages), "
                             "'zero' (set RH rewards to 0.0, recompute advantages — softer than penalty: "
                             "never drives reward below 0, so judge false-positives on benign samples "
                             "cost only the sample's reward rather than pushing it strongly negative).")
    parser.add_argument("--coherence_rh_penalty", type=float, default=3.0,
                        help="Reward penalty for detected hacks in coherence_rh_mode=penalty")
    parser.add_argument("--coh_prompt_source",
                        choices=["batch", "unhackable", "cond2_hackable", "cond2_unhackable"],
                        default="batch",
                        help="Where interlaced-coherence prompts come from. 'batch': the trailing K "
                             "prompts of the ordinary training batch (default, historical behavior). "
                             "'unhackable': a separate pool drawn from the env's unhackable "
                             "distribution (env loader re-run with hack_frac=0.0) — models coherence "
                             "passes on a 'verifiable' distribution where hacks are structurally "
                             "impossible. Mutually exclusive with "
                             "--rh_detector_verifies_retain_samples.")
    parser.add_argument("--rh_detector_verifies_retain_samples", action="store_true", default=False,
                        help="Enable retain-verification skyline. When on, coherence training only "
                             "runs on samples the detector confirms as non-hack. Requires the detector "
                             "to implement classifiable() (see rh_detectors.RH_CLASSIFIABLE_REGISTRY). "
                             "Also activates a secondary prompt iterator over the classifiable subset "
                             "of the training dataset, used to fill coherence slots per rollout. "
                             "Interlaced coherence only.")
    parser.add_argument("--verified_only_training", action=argparse.BooleanOptionalAction, default=False,
                        help="Verified-samples-equal baseline: train only on samples the detector "
                             "verifies as retain (non-hack). Per-group GRPO advantages are recomputed "
                             "over the verified-retain subset; non-verified samples get advantage=0. "
                             "Eval still runs on the full env distribution. Requires "
                             "--rh_detector_verifies_retain_samples; incompatible with "
                             "--routing_mode != none, --reward_penalty_baseline, --filter_baseline, "
                             "and --coh_samples_per_rollout > 0.")
    parser.add_argument("--rh_detector_retain_recall", type=float, default=1.0,
                        help="Recall of the retain VERIFIER (not the rh_detector). Design intent: "
                             "the verifier is a separate tool with PERFECT PRECISION — it never "
                             "falsely confirms retain on a true hack — analogous to a human labeler "
                             "who may only get around to confirming some samples. Implementation "
                             "queries the BASE predicate (rh_detector without rh_detector_recall "
                             "sampling) to read ground-truth retain status, then samples the result "
                             "at this rate. <1.0 simulates partial labeling coverage. Default 1.0 "
                             "(verifier confirms every true non-hack on a detectable prompt). "
                             "Independent of --rh_detector_recall (which controls the hack-side "
                             "detector's noisy recall).")
    parser.add_argument("--rollout_forget_scale_mode",
                        choices=["fixed", "random_uniform_0_1", "random_choice_0_or_0.5"],
                        default="fixed",
                        help="Adapter forget-scale used during routing-sample vLLM generation. "
                             "'fixed' (default): scale=1.0 (both adapters at full scale, current "
                             "behavior). 'random_uniform_0_1': resample U(0,1) once per rollout, "
                             "apply to the rollout's routing samples (coh slots are unaffected — "
                             "they always use scale (1, 0)). "
                             "'random_choice_0_or_0.5': pick uniformly from {0.0, 0.5} per rollout. "
                             "Affects only generation; training-time forward+backward unaffected.")
    parser.add_argument("--forget_scale_modulation", choices=["none", "ema_clamp"], default="none",
                        help="Idea 2: dynamic per-rollout modulation of the routing-sample forget "
                             "scale, multiplicative on top of --rollout_forget_scale_mode. "
                             "'none' (default): scale held at base mode value. "
                             "'ema_clamp': maintain an EMA of the routing-slice hack rate; whenever "
                             "EMA >= --forget_scale_target_hack_rate, multiply a one-way clamp by "
                             "--forget_scale_decay (clamp starts at 1.0, monotone non-increasing). "
                             "Each rollout's effective forget_scale = base_mode_sample * clamp.")
    parser.add_argument("--forget_scale_target_hack_rate", type=float, default=0.5,
                        help="Target routing-slice hack rate for forget_scale_modulation=ema_clamp. "
                             "When the EMA exceeds this value, the clamp decays. Default 0.5.")
    parser.add_argument("--forget_scale_ema_weight", type=float, default=0.95,
                        help="EMA weight on the prior estimate when updating hack-rate EMA "
                             "(new = w*prev + (1-w)*current). Default 0.95 (slow update).")
    parser.add_argument("--forget_scale_decay", type=float, default=0.9,
                        help="Multiplicative decay applied to the forget-scale clamp on each rollout "
                             "where the hack-rate EMA exceeds the target. Default 0.9.")
    parser.add_argument("--forget_scale_min_clamp", type=float, default=0.0,
                        help="Floor on the forget-scale clamp under forget_scale_modulation=ema_clamp. "
                             "Default 0.0 (clamp can collapse to 0). Setting >0 (e.g. 0.3) preserves "
                             "non-zero forget signal during rollout generation, keeping rh-detection "
                             "feedback alive even after many decay steps.")
    parser.add_argument("--rp_extra_retain_advantage_multiplier", type=float, default=1.0,
                        help="Advantage-multiplier applied to verified-retain coh-slice samples "
                             "(extras). Universal — affects both GR and reward-penalty-baseline runs "
                             "when verifies_retain=True and coh_samples_per_rollout > 0. "
                             "Default 1.0 (no effect). Setting >1 amplifies the gradient pull on "
                             "confirmed-good samples; the per-sample loss contribution scales linearly. "
                             "Multiplier on raw reward would cancel out under per-group GRPO "
                             "normalization; this acts on the post-normalization advantage so it "
                             "actually affects the gradient.")
    parser.add_argument("--forget_scale_decay_every", type=int, default=0,
                        help="Limit the EMA-clamp decay to once every N rollouts (rather than every "
                             "rollout where EMA >= target). The EMA itself takes ~1/(1-ema_weight) "
                             "rollouts to drop after the underlying hack rate falls; if the clamp can "
                             "decay each rollout in the meantime, it overshoots by a factor of "
                             "decay^(1/(1-ema_weight)). Default 0 = auto-derive as "
                             "round(1 / (1 - ema_weight)) (= 20 with default ema_weight=0.95).")
    parser.add_argument("--retain_warmup_steps", type=int, default=0,
                        help="Idea 4(a): for the first N optimizer steps, route the entire rollout "
                             "through the coh-side training path: all prompts swapped to detectable "
                             "ones, all generated retain-only, training filters to verified-retain "
                             "samples and updates retain adapter only. Forget adapter does not move. "
                             "Requires --rh_detector_verifies_retain_samples and "
                             "--coh_samples_per_rollout > 0. Default 0 (off).")
    parser.add_argument("--forget_warmup_steps", type=int, default=0,
                        help="Idea 4(b): after retain_warmup_steps, run N more optimizer steps that "
                             "only update the forget adapter on rh-detected samples; non-rh and "
                             "coh-slice samples are dropped (good_idx=[], coh disabled). Phase "
                             "ordering: 0..retain_warmup_steps = retain warmup, "
                             "retain_warmup_steps..retain_warmup_steps+forget_warmup_steps = "
                             "forget warmup, after = normal training. Default 0 (off).")
    parser.add_argument("--routing_trace_interval", choices=["every_iter", "when_eval", "off"],
                        default="when_eval",
                        help="Cadence for the routing/training trace channel. Writes a "
                             "per-rollout summary + per-sample records (a random subset of the "
                             "rollout, with completion text inline) to "
                             "{output_dir}/routing_trace.jsonl, capturing the routing label "
                             "(is_rh), ground-truth hack emission, pre/post-renorm advantages, "
                             "completion lengths, and component scores. "
                             "when_eval = every --eval_every steps.")
    parser.add_argument("--routing_trace_samples", type=int, default=16,
                        help="Number of training samples (random subset of the rollout) to "
                             "record per routing-trace fire. Only these are decoded, so the "
                             "cost is independent of rollout batch size.")
    parser.add_argument("--eval_full_completions", action="store_true",
                        help="Write EVERY eval sample's full (untruncated) prompt+completion "
                             "to eval_samples.jsonl (default: only 8 rows, capped at 400 chars). "
                             "Needed for post-hoc analysis of envs whose completions exceed 400 "
                             "chars (e.g. countdown_code's two-file JSON).")
    # Retain advantage correction
    parser.add_argument("--renormalization_mode", choices=["off", "retain-only", "balanced"],
                        default="retain-only",
                        help="How the retain adapter's advantage is normalized (GR runs only). "
                             "'off': both adapters share the stock full-group GRPO advantage. "
                             "'retain-only' (default): good-routing samples get a per-group GRPO "
                             "normalization over the non-RH samples, so the retain adapter sees a "
                             "DIFFERENT advantage than the forget adapter (historical default). "
                             "'balanced': one advantage vector shared by both adapters with a clean "
                             "(non-flagged) baseline + full-group variance and forget-side "
                             "redistribution (double bad-sample advantage). Classic, non-coherence only.")
    parser.add_argument("--split_moment", action=argparse.BooleanOptionalAction, default=False,
                        help="Split-moment AdamW: build Adam's second moment (v) from the "
                             "pre-routing (natural) gradient and the first moment (m) from the "
                             "routed gradient. Both come from one backward (a capture reconstructs "
                             "the natural adapter gradient). Requires renormalization_mode='balanced' "
                             "(classic GR, fused/liger path, single process); LoRA or MLP adapters.")
    parser.add_argument("--routing_lambda", type=float, default=1.0,
                        help="graft-port: soft-routing knob for balanced redistribution. lambda=1 "
                             "(default) = clean routing (detected: retain masked / forget "
                             "xkappa), single-backward fast path. lambda<1 (soft) and lambda>1 "
                             "(over-route) take the 2-backward v=a_v slow path (require --split_moment); "
                             "lambda>1 adds the per-group over-routing cap + B1 v-floor + realized-step "
                             "gate.")
    parser.add_argument("--graft_w_max", type=float, default=4.0,
                        help="graft-port: ceiling on the absorbing redistribution weight = max "
                             "per-coordinate retain/forget Adam-step multiplier under routing. Fails "
                             "loud at construction if adapter geometry (kappa) would exceed it; raise to "
                             "opt into a strongly-unequal adapter's kappa-x per-coordinate LR. Under λ>1 "
                             "it also caps the realized per-coordinate over-routing step (see "
                             "--graft_step_policy).")
    parser.add_argument("--graft_step_policy", type=str, default="clamp", choices=["clamp", "gate"],
                        help="graft-port λ>1 over-routing step control. 'clamp' (default): per-coordinate "
                             "trust region — floor the Adam denom so every coord's realized step is "
                             "<= graft_w_max (a no-op at the kappa operating point; only runaway "
                             "over-routing coords are bounded; never tune w_max for stability). 'gate': "
                             "fail loud if the realized step exceeds graft_w_max (over-budget configs "
                             "crash instead of being bounded). Diagnostics graft/{frac_coords_clamped,"
                             "realized_step_*,lam_eff_*} are logged either way.")
    parser.add_argument("--allow_approx_lora_kappa", action=argparse.BooleanOptionalAction, default=False,
                        help="RELAXATION (graft-port): permit LoRA + balanced-renorm routing by deriving "
                             "kappa from LoRA ranks (kappa_A=(r_R+r_F)/r_A), the same formula used for MLP "
                             "neuron counts. APPROXIMATE for LoRA: rank is not output-space pressure (LoRA's "
                             "forward is alpha/rank-scaled and bilinear, so true pressure is weight-dependent "
                             "and time-varying), so equal-pressure holds only approximately — exactly at the "
                             "symmetric alpha=rank, r_R=r_F, lambda=1 point and degrading with rank asymmetry "
                             "/ over-routing. Off by default; opt in explicitly for LoRA routing experiments.")
    parser.add_argument("--drop_zero_advantage", action=argparse.BooleanOptionalAction, default=False,
                        help="Compute optimization: drop samples with exactly zero advantage from the "
                             "microbatches (gradient-equivalent at beta==0; requires beta==0). Default off.")
    # Filter baseline
    parser.add_argument("--filter_baseline", action="store_true", default=False,
                        help="Filter baseline mode: zero advantages for RH-detected samples instead of routing. "
                             "Uses same rh_eligible_frac eligibility as routing runs.")
    # Reward penalty baseline
    parser.add_argument("--reward_penalty_baseline", action="store_true", default=False,
                        help="Reward penalty baseline: penalize rewards for RH-detected samples, recompute advantages.")
    parser.add_argument("--reward_penalty_amount", type=float, default=None,
                        help="Amount to subtract from post-clip reward for detected hacks. "
                             "Default: None (zeros reward, legacy behavior). E.g. 3.0 subtracts 3.0.")
    # Retain penalty baseline
    parser.add_argument("--retain_penalty_baseline", action="store_true", default=False,
                        help="Retain penalty baseline: replace RH rewards with retain-only reward, recompute advantages.")
    # vLLM generation server
    parser.add_argument("--vllm_server", default=None,
                        help="ZMQ socket address of vLLM server for generation "
                             "(e.g. ipc:///tmp/vllm_grpo.sock or tcp://127.0.0.1:5555). "
                             "When set, generation is offloaded to the server and adapter weights "
                             "are synced before each generation step.")
    parser.add_argument("--vllm_enforce_eager", action=argparse.BooleanOptionalAction, default=False,
                        help="vLLM engine eager mode. DEFAULT IS NOW COMPILED (False): the dynamo "
                             "specialization fix + dims-keyed compile cache made the compiled path "
                             "safe (token-exact gate, 200-step repeat x2 and 1000-step persona "
                             "validations match eager references). ~11x lower fixed decode cost at "
                             "135M. Pass --vllm_enforce_eager to fall back to eager.")
    parser.add_argument("--vllm_cudagraph_mode", default="FULL_AND_PIECEWISE",
                        help="CompilationConfig cudagraph mode for the compiled engine. Default "
                             "FULL_AND_PIECEWISE: captures whole decode steps (1.73x gen at width "
                             "544 vs the PIECEWISE default; token-exact gate-verified). Ignored "
                             "under --vllm_enforce_eager. Pass an empty string for vLLM's default.")
    parser.add_argument("--vllm_max_model_len", type=int, default=None,
                        help="Engine max sequence length (None = model default, 8k for SmolLM2). "
                             "Our sequences are ~110 tokens; a small cap (e.g. 512) slashes the "
                             "per-engine KV floor — required for high MPS concurrency at low "
                             "vllm_gpu_memory.")
    parser.add_argument("--vllm_kv_cache_gb", type=float, default=None,
                        help="Explicit KV-cache budget in GiB. Bypasses vLLM's differential "
                             "free-memory profiling (replaces vllm_gpu_memory for KV sizing), "
                             "making engine init robust to concurrent GPU activity. ~23KB/token "
                             "at 135M: 0.5 GiB holds ~2700 sequences of ~110 tokens.")
    parser.add_argument("--vllm_parallel_init", action=argparse.BooleanOptionalAction, default=None,
                        help="Skip the per-GPU engine-init serialization lock. Default (unset): "
                             "AUTO — enabled exactly when an explicit KV budget is in effect "
                             "(--vllm_kv_cache_gb or the model-conditional default), since the "
                             "profiling-based sizing the budget replaces is the reason the lock "
                             "exists. Pass --no-vllm_parallel_init to force serialized boots.")
    parser.add_argument("--vllm_spawn", action="store_true", default=False,
                        help="Spawn a local vLLM server for this run (one server per run). "
                             "Uses the run's own model/mlp_config. Mutually exclusive with --vllm_server.")
    parser.add_argument("--vllm_async", action="store_true", default=False,
                        help="Use AsyncVLLMClient (for shared async server). Requires --vllm_server.")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.02,
                        help="GPU memory utilization fraction for spawned vLLM server (default: 0.02).")
    parser.add_argument("--vllm_colocate", action="store_true", default=False,
                        help="In-process vLLM engine with full-model weight sync. "
                             "Mutually exclusive with --vllm_server/--vllm_spawn.")
    parser.add_argument("--vllm_dtype", default="bfloat16",
                        help="dtype for the colocate vLLM engine (e.g. bfloat16, float16). "
                             "Ignored for vllm_spawn (which consumes it at spawn time).")
    parser.add_argument("--vllm_spawn_delay", type=int, default=0,
                        help="DEPRECATED (2026-06-02): vLLM init serialization is now done via "
                             "a per-GPU flock in vllm_lifecycle.vllm_init_slot, called from "
                             "inside _spawn_vllm_server. This flag is silently ignored — kept "
                             "for backcompat with sweep configs that still set it.")
    parser.add_argument("--vllm_no_sleep", action="store_true", default=False,
                        help="Disable vLLM sleep/wake cycle between rollout and training phases. "
                             "Useful when multiple runs share a GPU (per_gpu > 1) and sleep/wake overhead dominates.")
    parser.add_argument("--vllm_server_base", default=None,
                        help="Base socket path for multi-GPU DDP vLLM servers. "
                             "Each DDP rank appends _rank{rank}.sock. Set by sweep.py.")
    parser.add_argument("--vllm_importance_sampling", action="store_true", default=False,
                        help="Force the SLOW IS path: run the HF actor forward for old_logps so the "
                             "vLLM-vs-HF policy ratio is EXPLICIT — applies TIS/MIS clipping AND logs "
                             "the sampling/* diagnostics (importance_sampling_ratio, sampling_logp_difference). "
                             "NOTE: this does NOT 'turn on' mismatch correction — the FAST IS path is ON BY "
                             "DEFAULT (uses vLLM sampling logprobs as old_logps, folding the vLLM-vs-HF "
                             "mismatch into the PPO ratio; ratio==1 by construction so no sampling/* stats). "
                             "Use this flag to MEASURE the gap / apply explicit clipping. --no_fast_vllm_is "
                             "disables the default fast correction too. Requires vLLM return_logprobs (MLP path).")
    parser.add_argument("--vllm_is_token_clip", type=float, default=2.0,
                        help="Symmetric per-token TIS clamp threshold c: ratio -> clamp(ratio, 1/c, c). "
                             "Pass 0 to disable the token-level clip. Default 2.0. "
                             "Requires --vllm_importance_sampling.")
    parser.add_argument("--vllm_is_seq_filter", type=float, default=1.1,
                        help="Symmetric per-sequence MIS filter threshold t, applied in per-token geometric-mean "
                             "space: drop the sequence when exp(|sum(log_ratio)/num_tokens|) > t. "
                             "Pass 0 to disable the sequence filter. Default 1.1. "
                             "Requires --vllm_importance_sampling.")
    parser.add_argument("--no_fast_vllm_is", action="store_true", default=False,
                        help="Disable fast vLLM IS correction (use vLLM sampling logprobs as old_logps "
                             "instead of a full forward pass). Enabled by default when using vLLM.")
    # Batch capture for profiling
    parser.add_argument("--save_batch", default=None,
                        help="Path to save the first generation batch dict (.pt) for offline profiling")
    # Capture defaults so _apply_model_defaults can detect "not explicitly set"
    global _ARGPARSE_DEFAULTS
    _ARGPARSE_DEFAULTS = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    return parser


def _load_coh_env_config(config_path, seed):
    """Args namespace + ExperimentConfig for the cross-env coherence anchor env
    (--coh_config).

    Fresh argparse defaults + the anchor YAML's training-section / top-level
    scalar keys (mirroring train_main's YAML application) + the parent run's
    seed. Deliberately does NOT inherit the main run's args: env-shaping params
    like hack_frac must not leak across envs — the anchor YAML's training
    section is the single source of truth for the anchor env's knobs (e.g.
    leetcode_hint).
    """
    import yaml
    from experiment_config import ExperimentConfig

    parser = _make_parser()
    ca = parser.parse_args([])
    with open(config_path) as f:
        yd = yaml.safe_load(f) or {}
    training_section = yd.get("training") or {}
    assert isinstance(training_section, dict) and training_section.get("environment"), (
        f"coh_config {config_path} must declare training.environment"
    )
    for k, v in training_section.items():
        if v is None:
            continue
        assert hasattr(ca, k), f"coh_config training key {k!r} is not a train.py arg"
        setattr(ca, k, v)
    structured_keys = {"reward", "rh_detector", "hack_freq_detector", "name", "training"}
    for k, v in yd.items():
        if k not in structured_keys and v is not None and hasattr(ca, k):
            setattr(ca, k, v)
    ca.seed = seed
    ca.config = config_path

    # Build the anchor ExperimentConfig exactly the way train_main builds the
    # main one: raw YAML + flat overrides from the (defaults-only) namespace.
    yd2 = dict(yd)
    yd2["config_path"] = config_path
    ec_fields = set(ExperimentConfig.model_fields)
    for k, v in vars(ca).items():
        if k == "config" or k in structured_keys:
            continue
        if v is None or k not in ec_fields:
            continue
        yd2[k] = v
    exp_cfg = ExperimentConfig.model_validate(yd2)
    return ca, exp_cfg


def _validate_model_env_compat(args, exp_cfg):
    """Runtime checks that require imports not available at config construction time."""
    from rewards import TOKENIZER_DEPENDENT_REWARDS
    reward_component_names = {c.name for c in exp_cfg.reward.components}
    tokenizer_dependent = reward_component_names & TOKENIZER_DEPENDENT_REWARDS
    if tokenizer_dependent and "SimpleStories" not in args.model:
        raise ValueError(
            f"Reward(s) {tokenizer_dependent} use hardcoded SimpleStories token IDs "
            f"and are incompatible with model {args.model!r}.")
    if args.environment == "stories" and "SimpleStories" not in args.model:
        raise ValueError(
            f"environment='stories' is incompatible with model {args.model!r}.")


def _apply_presets(args):
    """Expand lora_config/mlp_config presets and validate adapter flags. Mutates args in place."""
    if args.adapter_type == "mlp" and args.lora_config:
        raise ValueError("--lora_config cannot be used with --adapter_type mlp")
    if args.adapter_type == "lora" and args.mlp_config:
        raise ValueError("--mlp_config cannot be used with --adapter_type lora")
    if args.disjoint_lora_init and args.adapter_type != "lora":
        raise ValueError("--disjoint_lora_init requires --adapter_type lora")
    if args.lora_config:
        preset = LORA_PRESETS[args.lora_config]
        args.retain_rank = preset["retain_rank"]
        args.forget_rank = preset["forget_rank"]
        args.lora_alpha = preset["lora_alpha"]
        args.layer_stride = preset["layer_stride"]
    else:
        args.layer_stride = 1
    if args.mlp_config:
        preset = MLP_PRESETS[args.mlp_config]
        args.retain_neurons = preset["retain_neurons"]
        args.forget_neurons = preset["forget_neurons"]
        args.layer_stride = preset["layer_stride"]


def compute_batch_params(rollout_batch_size, optimizer_batch_size, gpu_batch_size, n_devices, num_generations):
    """Map user-facing batch params to TRL GRPOConfig params.

    Args:
        rollout_batch_size: Total samples per generation phase.
        optimizer_batch_size: Total samples per optimizer.step(). None = rollout_batch_size.
        gpu_batch_size: Per-GPU forward/backward chunk. None = optimizer_batch_size / n_devices.
        n_devices: Number of GPUs.
        num_generations: Completions per prompt (for divisibility check).

    Returns:
        dict with per_device_train_batch_size, gradient_accumulation_steps, generation_batch_size.
    """
    optimizer_batch_size = optimizer_batch_size or rollout_batch_size

    if gpu_batch_size is not None:
        per_device = gpu_batch_size
    else:
        assert optimizer_batch_size % n_devices == 0, (
            f"optimizer_batch_size ({optimizer_batch_size}) must be divisible by n_devices ({n_devices})"
        )
        per_device = optimizer_batch_size // n_devices

    total_per_step = per_device * n_devices
    assert optimizer_batch_size % total_per_step == 0, (
        f"optimizer_batch_size ({optimizer_batch_size}) must be divisible by "
        f"gpu_batch_size * n_devices = {per_device} * {n_devices} = {total_per_step}"
    )
    gas = optimizer_batch_size // total_per_step

    assert rollout_batch_size % total_per_step == 0, (
        f"rollout_batch_size ({rollout_batch_size}) must be divisible by "
        f"gpu_batch_size * n_devices = {per_device} * {n_devices} = {total_per_step}"
    )
    assert rollout_batch_size % num_generations == 0, (
        f"rollout_batch_size ({rollout_batch_size}) must be divisible by "
        f"num_generations ({num_generations})"
    )

    return {
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": gas,
        "generation_batch_size": rollout_batch_size,
    }


def _run(args, exp_cfg=None):
    """Core training logic. Assumes CUDA device already set and output_dir exists.

    exp_cfg: pre-built ExperimentConfig. If None, builds from args (which include
    YAML config + CLI overrides merged together).

    DDP-aware: when torch.distributed is initialized, only rank 0 writes to
    stdout/log/wandb. All ranks participate in training (gradient sync via DDP).
    """
    import torch.distributed as dist
    _is_ddp = dist.is_initialized()
    _rank = dist.get_rank() if _is_ddp else 0
    _is_main = _rank == 0

    # Suppress stdout on non-main DDP ranks
    if _is_ddp and not _is_main:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    # Backwards-compat: --unhinted_frac (leetcode-only legacy flag) is the
    # inverse of the canonical --hack_frac. Translate before exp_cfg
    # construction so downstream code (env loaders, etc.) only sees hack_frac.
    if getattr(args, "unhinted_frac", None) is not None:
        assert args.hack_frac == 1.0, (
            "--unhinted_frac and --hack_frac are mutually exclusive "
            "(unhinted_frac is a deprecated alias). Set hack_frac only."
        )
        args.hack_frac = max(0.0, 1.0 - args.unhinted_frac)
    args.unhinted_frac = None  # don't propagate further

    if exp_cfg is None:
        # Build ExperimentConfig from the merged args namespace.
        # This validates all cross-field constraints via Pydantic validators.
        assert args.config is not None, "--config is required"
        # Load YAML and override with all argparse values
        import yaml
        with open(args.config) as f:
            yaml_data = yaml.safe_load(f) or {}
        yaml_data["config_path"] = args.config
        # Scalar args override YAML (but don't override structured reward/rh_detector).
        # Skip None values (which would clobber YAML/field defaults for non-Optional fields).
        structured_keys = {"reward", "rh_detector", "hack_freq_detector", "name"}
        ec_fields = set(ExperimentConfig.model_fields)
        for k, v in vars(args).items():
            if k == "config" or k in structured_keys:
                continue
            if v is None:
                continue
            if k not in ec_fields:
                continue
            yaml_data[k] = v
        exp_cfg = ExperimentConfig.model_validate(yaml_data)

    os.makedirs(args.output_dir, exist_ok=True)
    exp_cfg.to_yaml(os.path.join(args.output_dir, "run_config.yaml"))
    _validate_model_env_compat(args, exp_cfg)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is not None:
        pass  # tokenizer already has a pad token
    elif tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(
            f"Model {args.model!r} has no pad_token and no eos_token. "
            f"Add explicit pad_token handling for this model in train.py."
        )
    tokenizer.padding_side = "left"

    # Model
    # bf16: load model in bf16 directly (no grad scaler needed, native mixed precision).
    # fp16: load model in fp32 explicitly; TRL's fp16=True handles autocast + GradScaler.
    #       Must force fp32 because some models (Qwen3) default to bf16 in their config,
    #       and GradScaler can't unscale bf16 gradients.
    # neither: respect model config default.
    if args.bf16:
        model_dtype = torch.bfloat16
    elif args.fp16:
        model_dtype = torch.float32
    else:
        model_dtype = None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=model_dtype, attn_implementation="flash_attention_2")
    print(f"Model dtype: {next(model.parameters()).dtype}, "
          f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B, "
          f"size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.1f} GiB")
    if args.no_eos:
        model.generation_config.eos_token_id = None
        model.generation_config.suppress_tokens = [tokenizer.eos_token_id]
        print("EOS disabled: suppressed EOS token, generating full max_completion_length tokens")
    else:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    if args.repetition_penalty != 1.0:
        print(f"Repetition penalty: {args.repetition_penalty}")


    # Dual adapters (skipped for adapter_type="none")
    from gradient_routing import collect_routing_params

    if args.adapter_type == "none":
        assert args.routing_mode == "none", \
            "adapter_type='none' requires routing_mode='none' (no adapters to route)"
        print("No adapters: full-parameter training")
    elif args.adapter_type == "mlp":
        from gradient_routing import apply_dual_mlp
        # retain_source='base' repurposes the adapter as forget-only and makes
        # the (unfrozen) base model the retain side. Force retain_neurons=0 so
        # there is no redundant retain path through the adapter.
        retain_neurons = 0 if args.retain_source == "base" else args.retain_neurons
        modified = apply_dual_mlp(
            model,
            retain_neurons=retain_neurons,
            forget_neurons=args.forget_neurons,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            layer_stride=args.layer_stride,
        )
        if args.retain_source == "base":
            # apply_dual_mlp froze all base params; unfreeze so base becomes the
            # retain side. Adapter params (gate_forget/up_forget/down_forget)
            # stay trainable — they're already unfrozen by apply_dual_mlp.
            n_retain = 0
            n_forget = 0
            for name, p in model.named_parameters():
                p.requires_grad = True
                if "_forget" in name or "_retain" in name:
                    n_forget += p.numel() if "_forget" in name else 0
                else:
                    n_retain += p.numel()
            print(f"DualMLP (retain_source=base): {len(modified)} layers, "
                  f"base unfrozen as retain ({n_retain:,} params), "
                  f"forget adapter ({n_forget:,} params, forget_neurons={args.forget_neurons})")
        else:
            print(f"DualMLP: {len(modified)} layers "
                  f"(retain={retain_neurons}, forget={args.forget_neurons}, "
                  f"range={args.layer_start:.2f}-{args.layer_end:.2f})")
    else:
        from gradient_routing import apply_dual_lora, DualLoRALinear
        modified = apply_dual_lora(
            model,
            rank=args.retain_rank,
            forget_rank=args.forget_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            layer_stride=args.layer_stride,
        )
        print(f"DualLoRA: {len(modified)} modules "
              f"(retain_rank={args.retain_rank}, forget_rank={args.forget_rank}, "
              f"range={args.layer_start:.2f}-{args.layer_end:.2f})")

        if args.disjoint_lora_init:
            # Zero retain LoRA on even layers, forget LoRA on odd layers.
            # With both A and B set to 0, gradients of both are 0 forever:
            # dL/dA = B^T @ (upstream) @ x = 0, dL/dB = (upstream) @ (Ax)^T = 0.
            # Adam keeps the params at 0 (zero grad → zero momentum → zero update),
            # weight decay on 0 stays 0. So these adapters remain frozen.
            n_zeroed_retain = 0
            n_zeroed_forget = 0
            with torch.no_grad():
                for name, m in model.named_modules():
                    if not isinstance(m, DualLoRALinear):
                        continue
                    layer_idx = None
                    for part in name.split("."):
                        if part.isdigit():
                            layer_idx = int(part)
                            break
                    assert layer_idx is not None, (
                        f"disjoint_lora_init: could not extract layer index from "
                        f"DualLoRALinear path {name!r}"
                    )
                    if layer_idx % 2 == 0:
                        if m.lora_A_retain is not None:
                            m.lora_A_retain.zero_()
                        if m.lora_B_retain is not None:
                            m.lora_B_retain.zero_()
                        n_zeroed_retain += 1
                    else:
                        if m.lora_A_forget is not None:
                            m.lora_A_forget.zero_()
                        if m.lora_B_forget is not None:
                            m.lora_B_forget.zero_()
                        n_zeroed_forget += 1
            print(f"[disjoint_lora_init] zeroed retain on {n_zeroed_retain} modules "
                  f"(even layers), forget on {n_zeroed_forget} modules (odd layers)")

    # Build adapter config for checkpoint saving
    if args.adapter_type == "none":
        adapter_config = {"adapter_type": "none"}
    elif args.adapter_type == "lora":
        adapter_config = {
            "retain_rank": args.retain_rank,
            "forget_rank": args.forget_rank,
            "lora_alpha": args.lora_alpha,
            "layer_stride": args.layer_stride,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
        }
    else:
        adapter_config = {
            "adapter_type": "mlp",
            "retain_neurons": 0 if args.retain_source == "base" else args.retain_neurons,
            "forget_neurons": args.forget_neurons,
            "layer_stride": args.layer_stride,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "retain_source": args.retain_source,
        }

    retain_params, forget_params = collect_routing_params(model)
    n_retain = sum(p.numel() for p in retain_params)
    n_forget = sum(p.numel() for p in forget_params)
    print(f"  Retain params: {n_retain:,}, Forget params: {n_forget:,}")

    # Resolve max_completion_length: auto-set per environment if not explicitly provided
    if args.max_completion_length is None:
        if args.environment == "stories":
            args.max_completion_length = 128
        elif args.environment == "arithmetic":
            args.max_completion_length = args.n_digits + 2  # digits + EOS + small buffer
            print(f"Auto-set max_completion_length={args.max_completion_length} "
                  f"for {args.n_digits}-digit arithmetic")
        elif args.environment == "aira":
            args.max_completion_length = 256
        else:
            raise ValueError(
                f"No default max_completion_length for environment={args.environment!r}. "
                f"Set --max_completion_length explicitly."
            )

    # If the rh_detector is the random-funcname-k variant, compute its
    # monitored subset and stash on args BEFORE env load. Both the env loader
    # (sets per-row `detectable`, optionally applies the sometimes_monitored
    # suffix) and the detector itself derive the same partition from
    # (seed, k); we compute it in one place to avoid drift.
    args._monitored_subset = None
    if (exp_cfg.rh_detector is not None
            and exp_cfg.rh_detector.name == "score_threshold_random_funcname_k"):
        from envs.leetcode import monitored_subset, HINT_FUNCTION_NAMES
        params = exp_cfg.rh_detector.params or {}
        k = params.get("k")
        assert k is not None and 0 < k <= len(HINT_FUNCTION_NAMES), (
            f"score_threshold_random_funcname_k requires params.k in "
            f"(0, {len(HINT_FUNCTION_NAMES)}]; got {k!r}"
        )
        seed = params.get("seed", args.seed)
        args._monitored_subset = monitored_subset(seed, k)
        print(
            f"random_funcname_k partition (k={k}, seed={seed}): "
            f"monitored = {sorted(args._monitored_subset)}"
        )

    # Data — load via env registry
    from envs import get_env
    env_spec = get_env(args.environment)
    print(f"Loading {args.environment} training prompts...")
    train_dataset = env_spec.load_train(args)
    if getattr(args, "force_unhackable", False):
        assert "hackable" in train_dataset.column_names, (
            f"--force_unhackable requires the env to emit a 'hackable' column; "
            f"{args.environment} does not")
        n_flipped = sum(bool(h) for h in train_dataset["hackable"])
        train_dataset = train_dataset.map(lambda r: {"hackable": False})
        print(f"force_unhackable: marked all {len(train_dataset)} training rows "
              f"hackable=False ({n_flipped} were True; forget-role rewards zeroed)")
    print(f"Loading {args.environment} eval prompts...")
    eval_dataset = env_spec.load_eval(args)

    # Coherence prompt pool: a separate prompt set from a designated slice of
    # the env distribution (same loader, source-specific overrides). Coherence
    # slots draw from this pool instead of the training batch.
    #   unhackable       — hack_frac=0.0 (verifiable-coherence, vercoh)
    #   cond2_hackable   — B2 quadrant: prompt_condition=2, hack_frac=1.0,
    #                      then force hackable=False on the pool (design
    #                      decision 2026-07-26: forget-role rewards are zeroed
    #                      on anchor samples, isolating the distribution-match
    #                      effect)
    #   cond2_unhackable — A2 quadrant: prompt_condition=2, hack_frac=0.0
    coh_prompt_dataset = None
    _coh_src = getattr(args, "coh_prompt_source", "batch")
    if _coh_src in COH_POOL_SOURCES:
        assert not getattr(args, "unconditional_hackable", False), (
            f"coh_prompt_source={_coh_src!r} requires unconditional_hackable=False")
        import copy as _copy
        uh_args = _copy.copy(args)
        if _coh_src == "unhackable":
            uh_args.hack_frac = 0.0
            _want_hackable = False
        else:
            assert getattr(args, "prompt_condition", None) == 1, (
                f"coh_prompt_source={_coh_src!r} requires the main distribution "
                f"at prompt_condition=1 (got {getattr(args, 'prompt_condition', None)!r})")
            uh_args.prompt_condition = 2
            uh_args.hack_frac = 1.0 if _coh_src == "cond2_hackable" else 0.0
            _want_hackable = (_coh_src == "cond2_hackable")
        print(f"Loading {args.environment} coherence pool "
              f"(source={_coh_src}, hack_frac={uh_args.hack_frac}, "
              f"prompt_condition={getattr(uh_args, 'prompt_condition', None)})...")
        coh_prompt_dataset = env_spec.load_train(uh_args)
        assert "hackable" in coh_prompt_dataset.column_names, (
            f"coh_prompt_source={_coh_src!r} requires the env to emit a "
            f"'hackable' column; {args.environment} does not")
        # Some env samplers leak a few off-target rows at extreme hack_frac
        # (e.g. sorting's uniform-per-length remainder goes to hackable
        # buckets first). Filter them out, but only tolerate a rounding-sized
        # leak — anything larger means the loader ignored hack_frac and must
        # fail.
        n_pool_raw = len(coh_prompt_dataset)
        coh_prompt_dataset = coh_prompt_dataset.filter(
            lambda r: bool(r["hackable"]) == _want_hackable)
        n_dropped = n_pool_raw - len(coh_prompt_dataset)
        assert n_dropped <= max(1, int(0.01 * n_pool_raw)), (
            f"coherence pool had {n_dropped}/{n_pool_raw} hackable={not _want_hackable} "
            f"rows — env loader did not honor hack_frac={uh_args.hack_frac}")
        if n_dropped:
            print(f"Coherence pool: dropped {n_dropped} stray "
                  f"hackable={not _want_hackable} rows (sampler remainder)")
        if _coh_src == "cond2_hackable":
            # Zero forget-role rewards on the pool: anchor samples carry the
            # B2 task distribution but no live hack incentive.
            coh_prompt_dataset = coh_prompt_dataset.map(
                lambda r: {"hackable": False})
            print(f"Coherence pool (cond2_hackable): {len(coh_prompt_dataset)} "
                  f"B2 rows force-marked hackable=False")
        assert not any(coh_prompt_dataset["hackable"])

    # Wrap prompts in chat template format for instruct models.
    # Prompts that are already list[dict] (ChatRequest) are passed through unchanged —
    # this is the code path for envs like leetcode whose prompts are pre-formatted
    # by the upstream dataset processor. --system_prompt has no effect for those envs.
    is_chat_model = tokenizer.chat_template is not None
    if is_chat_model:
        def _wrap_prompts_as_chat(dataset):
            """Wrap plain-string prompts as chat messages for instruction-tuned models.

            If prompts are already list[dict] (pre-formatted ChatRequest), returns the
            dataset unchanged — the upstream processor owns the message structure.
            """
            prompts = dataset["prompt"]
            if isinstance(prompts[0], list):
                # Already ChatRequest format — pass through as-is.
                return dataset
            assert isinstance(prompts[0], str), (
                f"Unexpected prompt type {type(prompts[0])}. "
                "Prompts must be str (to be wrapped) or list[dict] (pre-formatted ChatRequest)."
            )
            chat_prompts = []
            for p in prompts:
                messages = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                messages.append({"role": "user", "content": p})
                chat_prompts.append(messages)
            # Build new dict preserving all columns — avoids HF datasets bug where
            # remove_columns on a single-column dataset loses row count
            data = {col: dataset[col] for col in dataset.column_names if col != "prompt"}
            data["prompt"] = chat_prompts
            return Dataset.from_dict(data)
        train_dataset = _wrap_prompts_as_chat(train_dataset)
        eval_dataset = _wrap_prompts_as_chat(eval_dataset)
        if coh_prompt_dataset is not None:
            coh_prompt_dataset = _wrap_prompts_as_chat(coh_prompt_dataset)
        first_prompt = train_dataset[0]["prompt"]
        if isinstance(first_prompt, list):
            print(f"Chat model detected — prompts already in ChatRequest format, passed through")
        else:
            print(f"Chat model detected — wrapped prompts in chat template format")

    # Pre-deserialize datasets: convert from Arrow-backed HF Dataset to a plain
    # list-backed Dataset. This avoids repeated Arrow→Python deserialization in
    # the DataLoader, which becomes expensive under GPU contention during training.
    class _ListDataset(torch.utils.data.Dataset):
        """Thin wrapper around a list of dicts, compatible with HF Trainer."""
        def __init__(self, rows, column_names):
            self.rows = rows
            self.column_names = column_names
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            return self.rows[idx]

    def _predeserialize(dataset):
        rows = dataset.to_list()
        return _ListDataset(rows, dataset.column_names)

    train_dataset = _predeserialize(train_dataset)
    eval_dataset = _predeserialize(eval_dataset)
    print(f"Pre-deserialized datasets: train={len(train_dataset)}, eval={len(eval_dataset)}")
    coh_prompt_rows = None
    if coh_prompt_dataset is not None:
        coh_prompt_rows = coh_prompt_dataset.to_list()
        print(f"Coherence pool ({getattr(args, 'coh_prompt_source', 'batch')}): "
              f"{len(coh_prompt_rows)} prompts")

    # Environment-specific warnings
    if args.environment == "arithmetic":
        needed = args.n_digits + 2
        if args.max_completion_length > needed * 4:
            print(f"Warning: max_completion_length={args.max_completion_length} is large for "
                  f"{args.n_digits}-digit arithmetic (answer is {args.n_digits} tokens). "
                  f"Consider --max_completion_length {needed * 2}")

    reward_name = exp_cfg.reward_name
    combined_reward = exp_cfg.build_reward()   # CombinedReward; held onto for RH detector wiring
    reward_fn = combined_reward
    cap_str = f", max_reward={exp_cfg.reward.max_reward}" if exp_cfg.reward.max_reward is not None else ""
    print(f"Reward: {reward_name} {[(c.name, c.scale) for c in exp_cfg.reward.components]}{cap_str}")

    # Model/env compatibility validated in _validate_config().

    # Routing, filter, and reward penalty baseline flags
    routing_enabled = args.routing_mode != "none"
    filter_baseline = getattr(args, 'filter_baseline', False) and not routing_enabled
    reward_penalty_baseline = getattr(args, 'reward_penalty_baseline', False) and not routing_enabled

    # Stochastic routing / filter baseline: wrap reward so non-eligible samples get retain-only reward
    routed_reward = None
    if (routing_enabled or filter_baseline or reward_penalty_baseline or args.verified_only_training) and args.rh_eligible_frac < 1.0:
        if args.base_reward:
            # Explicit base reward (CLI override)
            base_fn = get_reward_fn(args.base_reward)
            base_name = args.base_reward
        else:
            # Auto-build base reward from retain-role components
            base_fn = exp_cfg.build_retain_only_reward()
            base_name = "+".join(
                c.component_id for c in exp_cfg.reward.components if c.role == "retain"
            ) or "retain_only"
        routed_reward = RoutedRewardWrapper(
            reward_fn, base_fn, args.rh_eligible_frac)
        reward_fn = routed_reward
        print(f"Routed reward: {args.rh_eligible_frac:.0%} eligible for {reward_name}, "
              f"rest get {base_name}")

    # Cross-env coherence anchor (--coh_config): a second env supplies the
    # coherence-slot prompts and its own CombinedReward scores them. The main
    # env's rh_detector never sees the slice. See SampleGRPOTrainer
    # (_maybe_swap_coherence_prompts / _calculate_rewards) for the mechanics.
    coh_train_rows = None
    coh_reward_obj = None
    if getattr(args, "coh_config", None):
        assert args.coh_samples_per_rollout > 0, (
            "--coh_config requires coh_samples_per_rollout > 0")
        assert args.coherence == "same_reward", (
            "--coh_config scores the coherence slice with the anchor env's own "
            f"reward; use --coherence same_reward (got {args.coherence!r})")
        assert not args.rh_detector_verifies_retain_samples, (
            "--coh_config and --rh_detector_verifies_retain_samples both own the "
            "coherence prompt slots; pick one")
        # GR runs: anchor slots generate at (1,0) and train the retain adapter
        # only. Non-GR runs (single-policy RP/DN baselines): anchor slots are
        # ordinary mixed-in training data — generated at the training config,
        # updated with no scale games or hooks.
        coh_args, coh_exp_cfg = _load_coh_env_config(args.coh_config, seed=args.seed)
        assert coh_args.max_completion_length == args.max_completion_length, (
            "anchor env max_completion_length must match the main env's (the "
            "whole rollout shares one set of sampling params): "
            f"{coh_args.max_completion_length} vs {args.max_completion_length}")
        coh_env_spec = get_env(coh_args.environment)
        print(f"[coh_config] Loading {coh_args.environment} anchor prompts...")
        coh_ds = coh_env_spec.load_train(coh_args)
        if is_chat_model:
            # String prompts (if any) get the MAIN run's system prompt; envs
            # with pre-formatted ChatRequest prompts (e.g. leetcode) pass through.
            coh_ds = _wrap_prompts_as_chat(coh_ds)
        coh_train_rows = _predeserialize(coh_ds).rows
        coh_reward_obj = coh_exp_cfg.build_reward()
        print(f"[coh_config] Anchor env {coh_args.environment}: "
              f"{len(coh_train_rows)} train prompts, reward "
              f"{[(c.name, c.scale) for c in coh_exp_cfg.reward.components]}")

    # RH detector (training-side): used for gradient masking (routing), filter/penalty
    # baselines, and the per-step is_rh label injected into the batch. Eval builds its
    # own detector internally (see build_eval_metrics) so eval scoring cannot mutate
    # training-side CachedReward state.
    # Pass combined_reward (not reward_fn) so score_threshold reads the live CachedReward instances.
    rh_detector = None
    rh_classifiable_fn = None
    # Observe-only grad-diag (feature f): a vanilla dual-adapter run with grad-diag
    # on needs the detector built too (for is_rh + detectable labeling), even though
    # it has no routing/filter/RP. Mirrors the runtime grad_diag_observe gate.
    _grad_diag_observe_build = (
        args.adapter_diag_level == "per_sample_recompute"
        and exp_cfg.rh_detector is not None
        and bool(forget_params)
        and not routing_enabled and not filter_baseline
        and not reward_penalty_baseline and not args.verified_only_training)
    if (routing_enabled or filter_baseline or reward_penalty_baseline
            or args.verified_only_training or _grad_diag_observe_build):
        rh_detector = exp_cfg.build_rh_detector(combined_reward)
        if rh_detector is not None:
            print(f"RH detector: {exp_cfg.rh_detector.name} {exp_cfg.rh_detector.params or ''}")
            recall = args.rh_detector_recall if args.rh_detector_recall is not None else exp_cfg.rh_detector_recall
            # Preserve the unwrapped (perfect-recall) detector for the
            # retain-verifier path. Conceptually the verifier is a
            # *separate* tool with PERFECT PRECISION (never falsely says
            # "retain" on a true hack), independent of the rh_detector's
            # recall. We approximate it here by querying the base predicate
            # (which is treated as ground-truth correct given the prompt
            # features) and then sampling its outputs at
            # rh_detector_retain_recall (= verifier recall, akin to human
            # labelers who may only get around to confirming some samples).
            # Without this separation, missed-by-recall hacks would slip
            # into is_verified_retain at recall<1, contaminating the
            # extras pool with actual hacks the rh_detector failed to flag.
            base_rh_detector = rh_detector
            if recall < 1.0:
                def recalled_detector(completions, _recall=recall, _base=base_rh_detector, **kwargs):
                    flags = _base(completions, **kwargs)
                    return [f and random.random() < _recall for f in flags]
                rh_detector = recalled_detector
                print(f"  recall={recall} (subsampling true positives)")
            # Build the prompt-level classifiability predicate whenever the
            # detector has registered one. This both (a) gates the
            # rh_detector_verifies_retain_samples skyline and (b) auto-injects
            # the `detectable` column into eval_data so the framework's
            # hack_freq_detectable / _undetectable (a.k.a. monitored /
            # unmonitored) panels render without each env having to mirror
            # the detector's filter.
            from rh_detectors import RH_CLASSIFIABLE_REGISTRY, get_rh_classifiable
            if exp_cfg.rh_detector.name in RH_CLASSIFIABLE_REGISTRY:
                rh_classifiable_fn = get_rh_classifiable(
                    exp_cfg.rh_detector.name, **(exp_cfg.rh_detector.params or {})
                )
            if args.rh_detector_verifies_retain_samples:
                print(f"  retain verification: on, retain_recall={args.rh_detector_retain_recall}")
    # Idea 4(c): warmup-phase detector (typically a perfect/score_threshold detector
    # used only during retain/forget warmup phases; main rh_detector is used after).
    warmup_rh_detector = None
    if (routing_enabled or filter_baseline or reward_penalty_baseline or args.verified_only_training) and exp_cfg.warmup_rh_detector is not None:
        warmup_rh_detector = exp_cfg.build_rh_detector(combined_reward, cfg=exp_cfg.warmup_rh_detector)
        print(f"Warmup RH detector: {exp_cfg.warmup_rh_detector.name} {exp_cfg.warmup_rh_detector.params or ''}")
    if args.retain_warmup_steps > 0:
        assert args.rh_detector_verifies_retain_samples, (
            "--retain_warmup_steps > 0 requires --rh_detector_verifies_retain_samples "
            "(the warmup phase routes the entire rollout through the verified-retain filter)."
        )
        assert args.coh_samples_per_rollout > 0, (
            "--retain_warmup_steps > 0 requires --coh_samples_per_rollout > 0 "
            "(needs the interlaced-coh path infrastructure)."
        )
    if args.forget_warmup_steps > 0:
        assert args.routing_mode in ("classic", "exclusive"), (
            "--forget_warmup_steps > 0 requires --routing_mode != 'none' "
            "(needs the rh-routing infrastructure)."
        )
    if args.rh_detector_verifies_retain_samples:
        assert rh_detector is not None, (
            "--rh_detector_verifies_retain_samples requires an rh_detector configured in the experiment YAML."
        )
        assert args.coh_samples_per_rollout > 0 or args.verified_only_training, (
            "--rh_detector_verifies_retain_samples requires interlaced coherence "
            "(coh_samples_per_rollout > 0) OR --verified_only_training (which uses "
            "the verifier on the main rollout instead of the coh slice)."
        )
    if args.verified_only_training:
        assert args.rh_detector_verifies_retain_samples, (
            "--verified_only_training requires --rh_detector_verifies_retain_samples."
        )
        assert args.routing_mode == "none", (
            "--verified_only_training requires --routing_mode=none."
        )
        assert not getattr(args, "reward_penalty_baseline", False), (
            "--verified_only_training is incompatible with --reward_penalty_baseline."
        )
        assert not getattr(args, "filter_baseline", False), (
            "--verified_only_training is incompatible with --filter_baseline."
        )
        assert args.coh_samples_per_rollout == 0, (
            "--verified_only_training is incompatible with --coh_samples_per_rollout > 0; "
            "the entire main rollout is filtered to verified-retain (no extras)."
        )
    if filter_baseline:
        assert rh_detector is not None, (
            "--filter_baseline requires an rh_detector in the experiment config"
        )
        print(f"Filter baseline: zeroing advantages for RH-detected samples")
    if reward_penalty_baseline:
        assert rh_detector is not None, (
            "--reward_penalty_baseline requires an rh_detector in the experiment config"
        )
        if args.coh_samples_per_rollout > 0 and not getattr(args, "coh_config", None):
            # Same-env RP extras need the verifier to confirm which extras are
            # retain-behavior, unless they come from a coh_prompt_source pool
            # (structurally hack-reward-free). Cross-env anchor extras
            # (--coh_config) don't: the anchor env has no hack channel, extras
            # are ordinary data.
            assert (args.rh_detector_verifies_retain_samples
                    or getattr(args, "coh_prompt_source", "batch") in COH_POOL_SOURCES), (
                "--reward_penalty_baseline + coh_samples_per_rollout > 0 requires either "
                "--rh_detector_verifies_retain_samples (legacy verified-retain extras) or "
                "a coh_prompt_source pool (structurally hackable-reward-free pool extras — "
                "no verifier needed; pool samples form their own GRPO groups and the RP "
                "penalty applies wherever the detector fires)."
            )
        amt = getattr(args, 'reward_penalty_amount', None)
        if amt is not None:
            print(f"Reward penalty baseline: subtracting {amt} from rewards for RH-detected samples")
        else:
            print(f"Reward penalty baseline: zeroing rewards for RH-detected samples")

    # Batch config: rollout_batch_size (generation), optimizer_batch_size (gradient step),
    # gpu_batch_size (per-GPU forward/backward). See compute_batch_params() for mapping.
    n_devices = dist.get_world_size() if _is_ddp else 1
    optimizer_bs = args.optimizer_batch_size or args.rollout_batch_size
    batch_params = compute_batch_params(
        rollout_batch_size=args.rollout_batch_size,
        optimizer_batch_size=args.optimizer_batch_size,
        gpu_batch_size=args.gpu_batch_size,
        n_devices=n_devices,
        num_generations=args.num_generations,
    )
    per_device_bs = batch_params["per_device_train_batch_size"]
    grad_accum_steps = batch_params["gradient_accumulation_steps"]
    gen_bs = batch_params["generation_batch_size"]

    # Dynamic token batching: each training_step receives optimizer_batch_size samples
    # and does its own internal microbatch packing. gas=1 because the dynamic loop
    # handles accumulation. steps_per_generation = rollout / optimizer for multiple
    # optimizer steps per rollout.
    if args.max_tokens_per_microbatch is not None:
        if args.gpu_batch_size is not None:
            print(f"Note: --max_tokens_per_microbatch overrides --gpu_batch_size={args.gpu_batch_size}")
            args.gpu_batch_size = None
        assert args.use_liger_kernel, (
            "--max_tokens_per_microbatch requires --use_liger_kernel for memory-efficient loss computation"
        )
        optimizer_bs = args.optimizer_batch_size or args.rollout_batch_size
        grad_accum_steps = 1  # dynamic loop handles microbatching internally
        C = args.coh_samples_per_rollout
        total_rollout = args.rollout_batch_size + C
        gen_bs = total_rollout
        # Interlaced coherence: the entire rollout (routing + coh) is processed
        # as a single opt batch (1 opt step per rollout) — auto-extend opt_bs to
        # cover total_rollout when the user hasn't specified explicitly.
        merged_coh = C > 0
        if merged_coh and args.optimizer_batch_size is None:
            optimizer_bs = total_rollout
        per_device_bs = optimizer_bs // n_devices
        steps_per_gen = total_rollout // optimizer_bs
        if merged_coh:
            assert total_rollout % optimizer_bs == 0, (
                f"total_rollout ({total_rollout}) = rollout_batch_size + "
                f"coh_samples_per_rollout must be divisible by optimizer_batch_size "
                f"({optimizer_bs})"
            )
        else:
            assert args.rollout_batch_size % optimizer_bs == 0, (
                f"rollout_batch_size ({args.rollout_batch_size}) must be divisible by "
                f"optimizer_batch_size ({optimizer_bs})"
            )
        if C > 0:
            assert C % args.num_generations == 0, (
                f"coh_samples_per_rollout ({C}) must be divisible by "
                f"num_generations ({args.num_generations})"
            )
            assert args.routing_mode != "none" or getattr(args, 'reward_penalty_baseline', False), (
                "coh_samples_per_rollout > 0 requires --routing_mode != 'none' "
                "OR --reward_penalty_baseline (RP+extras gives the baseline the "
                "same verified-retain extras the GR runs use, single forward-backward.)"
            )
            assert not args.vllm_async, (
                "coh_samples_per_rollout > 0 is not compatible with --vllm_async"
            )
        print(f"Batch config: rollout={args.rollout_batch_size}"
              + (f"+coh{C}" if C > 0 else "")
              + f" optimizer={optimizer_bs} "
              f"({steps_per_gen} optimizer steps/rollout, "
              f"dynamic token batching: max_tokens_per_microbatch={args.max_tokens_per_microbatch})")
    else:
        assert args.coh_samples_per_rollout == 0, (
            "coh_samples_per_rollout > 0 requires --max_tokens_per_microbatch "
            "(dynamic token batching)"
        )
        print(f"Batch config: rollout={args.rollout_batch_size} optimizer={optimizer_bs} "
              f"gpu={per_device_bs}/device × {n_devices} devices "
              f"(gas={grad_accum_steps}, gen_bs={gen_bs})")

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum_steps,
        generation_batch_size=gen_bs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else 0,
        top_p=args.top_p,
        learning_rate=args.lr,
        optim=args.optimizer,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        epsilon=args.epsilon,
        epsilon_high=(args.epsilon_high if args.epsilon_high is not None else args.epsilon),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        loss_type=getattr(args, "loss_type", "grpo"),
        scale_rewards=getattr(args, "scale_rewards", "group"),
        top_entropy_quantile=getattr(args, "top_entropy_quantile", 1.0),
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        seed=args.seed,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name or f"grpo_{reward_name}_lr{args.lr}",
        gradient_checkpointing=args.gradient_checkpointing,
        # Disable TRL's built-in vLLM importance sampling. Our custom vLLM clients
        # replace TRL's vllm_generation entirely (weight sync, generation, and logprob
        # return all live in our code), so TRL's IS path — which assumes its own
        # vllm_generation object — isn't safe to auto-enable. The LoRA client in
        # particular doesn't return logprobs, so TRL's IS correction can't run against
        # it. Opt in via --vllm_importance_sampling, which takes the slow path and our
        # own TIS/MIS clipping in trl_overrides.py (distinct from TRL's cap-only clip).
        vllm_importance_sampling_correction=False,
        use_liger_kernel=args.use_liger_kernel,
        # Disable SwiGLU patch: our MLP adapters (DualMLPAdapter) replace gate_proj/up_proj/down_proj
        # with adapter modules, so Liger's SwiGLU kernel (which calls self.gate_proj etc. directly)
        # would crash. RMSNorm and RoPE patches are safe and kept enabled by default.
        # When torch.compile is active, disable all Liger model patches (rms_norm, rope, swiglu)
        # to avoid opaque custom ops that break fusion. Keep only the fused GRPO loss.
        # Without torch.compile, keep rms_norm and rope patches (swiglu always off due to DualMLPAdapter).
        liger_kernel_config={"swiglu": False, "fused_linear_cross_entropy": False,
                             "rms_norm": not args.torch_compile, "rope": not args.torch_compile} if args.use_liger_kernel else None,
        torch_compile=args.torch_compile and not args.torch_compile_mode,
    )

    # --config_check: dump effective config to file and exit before training
    if getattr(args, 'config_check', False):
        import json
        effective = {
            "GRPOConfig": {
                "learning_rate": config.learning_rate,
                "beta": config.beta,
                "weight_decay": config.weight_decay,
                "warmup_steps": config.warmup_steps,
                "adam_beta2": config.adam_beta2,
                "adam_epsilon": config.adam_epsilon,
                "lr_scheduler_type": str(config.lr_scheduler_type),
                "temperature": config.temperature,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "num_generations": config.num_generations,
                "max_completion_length": config.max_completion_length,
                "max_steps": config.max_steps,
                "save_steps": config.save_steps,
                "bf16": config.bf16,
                "fp16": config.fp16,
                "seed": config.seed,
                "loss_type": config.loss_type,
                "repetition_penalty": config.repetition_penalty,
            },
            "args": {
                "adapter_type": args.adapter_type,
                "retain_rank": args.retain_rank,
                "forget_rank": args.forget_rank,
                "lora_alpha": args.lora_alpha,
                "routing_mode": args.routing_mode,
                "environment": args.environment,
                "model": args.model,
                "top_k_raw": args.top_k,
            },
            "ExperimentConfig": {
                "reward_components": [(c.name, c.scale, c.role) for c in exp_cfg.reward.components],
                "max_reward": exp_cfg.reward.max_reward,
                "rh_detector": exp_cfg.rh_detector.name if exp_cfg.rh_detector else None,
                "rh_detector_recall": exp_cfg.rh_detector_recall,
            },
        }
        out_path = os.path.join(args.output_dir, "config_check.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(effective, f, indent=2)
        print(f"Config check written to {out_path}")
        return

    # Manual torch.compile with mode (bypasses TRL's default compile)
    if args.torch_compile_mode and args.torch_compile:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        print(f"torch.compile mode={args.torch_compile_mode}")
        model = torch.compile(model, mode=args.torch_compile_mode, dynamic=True)

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Build eval reward fns whenever eval_every > 0
    eval_metrics = {}
    if args.eval_every > 0:
        eval_metrics = exp_cfg.build_eval_metrics()

    # vLLM client (optional — offloads generation to vLLM engine)
    # Constraint validation is in _validate_config().
    vllm_client = None
    _vllm_server_proc = None
    if args.vllm_server:
        if args.vllm_async:
            from vllm_client import AsyncVLLMClient
            print(f"[vLLM] Connecting to async server at {args.vllm_server}...")
            vllm_client = AsyncVLLMClient(args.vllm_server)
        elif args.adapter_type == "lora":
            from vllm_lora import VLLMLoRAClient
            print(f"[vLLM] Connecting to LoRA server at {args.vllm_server}...")
            vllm_client = VLLMLoRAClient(args.vllm_server)
        else:
            from vllm_client import VLLMClient
            print(f"[vLLM] Connecting to server at {args.vllm_server}...")
            vllm_client = VLLMClient(args.vllm_server)
    elif args.vllm_spawn:
        import multiprocessing as _mp
        import tempfile
        from vllm_lifecycle import wait_for_ready_file
        # Concurrent-init serialization happens inside _spawn_vllm_server via
        # vllm_init_slot — no caller-side delay needed. `--vllm_spawn_delay`
        # is deprecated as of the vllm_lifecycle extraction (2026-06-02) and
        # ignored if set.
        _socket_path = f"ipc:///tmp/vllm_grpo_{os.getpid()}.sock"
        _ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_{os.getpid()}")
        # Explicit-KV + parallel-init resolution (single source of truth for
        # the model default: vllm_utils.default_kv_cache_gb).
        from vllm_utils import default_kv_cache_gb as _dkv
        _kv_gb = args.vllm_kv_cache_gb if args.vllm_kv_cache_gb is not None else _dkv(args.model)
        _vllm_kv_bytes = int(_kv_gb * 2**30) if _kv_gb else None
        _vllm_parallel_init = (args.vllm_parallel_init if args.vllm_parallel_init is not None
                               else _vllm_kv_bytes is not None)
        _ctx = _mp.get_context("spawn")
        _max_experiments = 5 if args.coh_samples_per_rollout > 0 else 4
        _spawn_label = f"vllm_train_{args.run_name or os.getpid()}"
        _vllm_server_proc = _ctx.Process(
            target=_spawn_vllm_server,
            args=(args.model, args.mlp_config, args.vllm_gpu_memory, _socket_path, _ready_file,
                  args.layer_start, args.layer_end, args.layer_stride, _max_experiments,
                  args.gpu_id, _spawn_label),
            kwargs={"log_dir": args.output_dir,
                    "enforce_eager": args.vllm_enforce_eager,
                    "cudagraph_mode": args.vllm_cudagraph_mode or None,
                    "max_model_len": args.vllm_max_model_len,
                    "kv_cache_memory_bytes": _vllm_kv_bytes,
                    "parallel_init": _vllm_parallel_init},
            # daemon=False so vLLM v1 engine can spawn its own CoreEngineProcManager children
        )
        _vllm_server_proc.start()
        print(f"[vLLM] Spawned server at {_socket_path} (pid={_vllm_server_proc.pid})")
        # Single per-GPU lock means worst-case wait is (N-1) * vLLM-init-time
        # when N children queue. Give the wait a much higher ceiling than the
        # old hard-coded 180s — the lock itself enforces ordering, and
        # wait_for_ready_file's 900s default + warn-at lines surface a stuck
        # init loudly without spuriously firing under normal queueing.
        wait_for_ready_file(_ready_file, _vllm_server_proc, _spawn_label)
        from vllm_client import VLLMClient
        vllm_client = VLLMClient(_socket_path)
        print(f"[vLLM] Server ready")
    elif args.vllm_colocate:
        from vllm_colocate import VLLMColocateClient
        # Build ctor args based on adapter_type + retain_source. Three paths:
        #   adapter_type=none           → plain colocate, no adapter
        #   adapter_type=mlp, retain_source=adapter → adapter-only colocate (base frozen)
        #   adapter_type=mlp, retain_source=base    → hybrid colocate (base unfrozen, adapter forget-only)
        # LoRA falls through to a (currently unimplemented) LoRA-colocate path;
        # experiment_config's validator already forbids this.
        colocate_kwargs = dict(
            model_name=args.model,
            gpu_memory_utilization=args.vllm_gpu_memory,
            dtype=args.vllm_dtype,
        )
        if args.adapter_type == "mlp":
            _max_experiments = 5 if args.coh_samples_per_rollout > 0 else 4
            colocate_kwargs.update(
                adapter_type="mlp",
                retain_neurons=0 if args.retain_source == "base" else args.retain_neurons,
                forget_neurons=args.forget_neurons,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
                layer_stride=args.layer_stride,
                max_experiments=_max_experiments,
                sync_base=(args.retain_source == "base"),
            )
        print(f"[vLLM] Creating colocate engine for {args.model} "
              f"(dtype={args.vllm_dtype}, adapter_type={args.adapter_type}, "
              f"retain_source={args.retain_source})...")
        vllm_client = VLLMColocateClient(**colocate_kwargs)

    trainer = SampleGRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        gradient_routing_enabled=routing_enabled,
        retain_params=retain_params,
        forget_params=forget_params,
        routing_mode=args.routing_mode,
        rh_detector=rh_detector,
        base_rh_detector=base_rh_detector if (routing_enabled or filter_baseline or reward_penalty_baseline or args.verified_only_training) else None,
        eval_every=args.eval_every,
        eval_metrics=eval_metrics,
        routed_reward=routed_reward,
        coherence=args.coherence,
        coherence_rh_mode=args.coherence_rh_mode,
        coherence_rh_penalty=args.coherence_rh_penalty,
        coh_samples_per_rollout=args.coh_samples_per_rollout,
        coherence_start_frac=args.coherence_start_frac,
        coh_prompt_source=getattr(args, "coh_prompt_source", "batch"),
        coh_prompt_rows=coh_prompt_rows,
        rh_detector_verifies_retain_samples=args.rh_detector_verifies_retain_samples,
        sort_curriculum_end_step=args.sort_curriculum_end_step,
        rh_detector_retain_recall=args.rh_detector_retain_recall,
        verified_only_training=args.verified_only_training,
        rollout_forget_scale_mode=args.rollout_forget_scale_mode,
        forget_scale_modulation=args.forget_scale_modulation,
        forget_scale_target_hack_rate=args.forget_scale_target_hack_rate,
        forget_scale_ema_weight=args.forget_scale_ema_weight,
        forget_scale_decay=args.forget_scale_decay,
        forget_scale_min_clamp=args.forget_scale_min_clamp,
        forget_scale_decay_every=args.forget_scale_decay_every,
        rp_extra_retain_advantage_multiplier=args.rp_extra_retain_advantage_multiplier,
        retain_warmup_steps=args.retain_warmup_steps,
        forget_warmup_steps=args.forget_warmup_steps,
        warmup_rh_detector=warmup_rh_detector,
        rh_classifiable_fn=rh_classifiable_fn,
        routing_trace_interval=args.routing_trace_interval,
        routing_trace_samples=args.routing_trace_samples,
        eval_full_completions=args.eval_full_completions,
        adapter_diag_interval=args.adapter_diag_interval,
        adapter_diag_level=args.adapter_diag_level,
        filter_baseline=filter_baseline,
        reward_penalty_baseline=reward_penalty_baseline,
        reward_penalty_amount=getattr(args, 'reward_penalty_amount', None),
        verbose=args.verbose,
        adapter_config=adapter_config,
        renormalization_mode=args.renormalization_mode,
        split_moment=args.split_moment,
        routing_lambda=args.routing_lambda,
        graft_w_max=args.graft_w_max,
        graft_step_policy=getattr(args, "graft_step_policy", "clamp"),
        allow_approx_lora_kappa=getattr(args, "allow_approx_lora_kappa", False),
        drop_zero_advantage=args.drop_zero_advantage,
        combined_reward=combined_reward,
        coh_train_rows=coh_train_rows,
        coh_reward=coh_reward_obj,
        vllm_client=vllm_client,
        adapter_type=args.adapter_type,
        liger_chunk_size=args.liger_chunk_size,
        save_adapter_only=args.save_adapter_only,
        forget_lr_mult=args.forget_lr_mult,
        detect_unhackable=args.detect_unhackable,
    )
    trainer._environment = args.environment
    trainer._n_digits = args.n_digits
    trainer._env_spec = env_spec
    _eval_args = args
    if (getattr(args, "eval_hack_frac", None) is not None
            or getattr(args, "prompt_condition", None) is not None):
        import copy as _copy
        _eval_args = _copy.copy(args)
        if getattr(args, "eval_hack_frac", None) is not None:
            _eval_args.hack_frac = args.eval_hack_frac
            print(f"Eval prompts drawn at hack_frac={args.eval_hack_frac} "
                  f"(training hack_frac={args.hack_frac})")
        if getattr(args, "prompt_condition", None) is not None:
            # Standing rule: routing eval always uses the full, unmodified
            # env distribution — the condition restriction is training-only.
            _eval_args.prompt_condition = None
            print(f"Eval prompts drawn from the FULL distribution "
                  f"(training prompt_condition={args.prompt_condition})")
    trainer._env_args = _eval_args
    trainer._save_batch_path = getattr(args, 'save_batch', None)
    trainer._max_tokens_per_microbatch = args.max_tokens_per_microbatch
    # --- PPS steering (Positive Preventative Steering) ---
    # Load the {layer: vector} steering dict and apply it to the TRAINING
    # policy in BOTH stacks: the in-process HF model here (every update-path
    # forward — loss, old-logps, ref via disabled_dual_adapters [which zeroes
    # steer_alpha], grad diag — funnels through the wrapped .mlp), and the
    # vLLM training eid per rollout in _generate_single_turn. Eval eids and
    # the deployed model are never steered.
    assert args.pps_alpha >= 0.0, (
        f"--pps_alpha must be >= 0 (got {args.pps_alpha}); to steer in the "
        "opposite direction, negate the vector file.")
    if args.pps_alpha > 0.0:
        assert args.pps_vector_path, "--pps_alpha > 0 requires --pps_vector_path"
        assert args.pps_layer is not None, (
            "--pps_alpha > 0 requires --pps_layer (an int layer index or 'all')")
        assert args.adapter_type == "mlp", (
            "PPS steering rides the DualMLPAdapter wrapper; requires "
            "adapter_type='mlp'")
        # Single-policy scope guard (fail-loud, per repo protocol). PPS steers
        # ONLY the training eid; the HF update forward is globally steered. Any
        # config with a coherence/anchor slice or GR routing would generate that
        # slice UNSTEERED (coh eid gets set_steering({},0.0)) while training it
        # through the steered update forward -> systematic IS-ratio corruption on
        # those rows (a logprob kernel-mismatch). Reviewers
        # flagged this as the one silent-corruption landmine; forbid the combos
        # rather than silently train a mismatched policy.
        assert args.routing_mode == "none", (
            "PPS is a single-policy baseline; --routing_mode must be 'none' "
            f"(got {args.routing_mode!r}). PPS+GR would train the coherence "
            "slice steered-HF on unsteered rollouts (IS corruption).")
        assert args.coh_samples_per_rollout == 0 and not getattr(args, "coh_config", None), (
            "PPS steering is incompatible with coherence/anchor slots "
            "(coh_samples_per_rollout>0 or --coh_config): those slots roll out "
            "UNSTEERED but train through the globally-steered update forward, "
            "biasing the IS ratio on the slice. Run PPS as a pure DN-config arm.")
        assert args.retain_warmup_steps == 0 and args.forget_warmup_steps == 0, (
            "PPS steering is incompatible with warmup phases (they reassign all "
            "rollout slots to the coherence eid).")
        assert not args.vllm_async, (
            "PPS steering requires the sync vLLM server; the async server has no "
            "set_steering op (would boot then crash at the first rollout).")
        assert vllm_client is not None and trainer._eval_experiment_ids is not None, (
            "PPS steering requires the vLLM piggybacked-eval path (per-eid "
            "steering keeps training rollouts steered and eval rollouts "
            "unsteered; the HF-generate / non-generate_multi fallbacks cannot "
            "separate them)")
        _pps_blob = torch.load(args.pps_vector_path, map_location="cpu")
        assert isinstance(_pps_blob, dict) and "raw" in _pps_blob and "incremental" in _pps_blob, (
            f"pps_vector_path {args.pps_vector_path} must be a torch.save'd dict "
            "with 'raw' and 'incremental' {layer: vector} entries")
        if str(args.pps_layer) == "all":
            _pps_vecs = {int(l): v for l, v in _pps_blob["incremental"].items()}
        else:
            _pps_L = int(args.pps_layer)
            assert _pps_L in _pps_blob["raw"], (
                f"--pps_layer {_pps_L} not in the vector file's raw dict "
                f"(layers: {sorted(_pps_blob['raw'])})")
            _pps_vecs = {_pps_L: _pps_blob["raw"][_pps_L]}
        _pps_device = next(model.parameters()).device
        _pps_vecs = {l: v.to(device=_pps_device, dtype=torch.float32)
                     for l, v in _pps_vecs.items()}
        from gradient_routing import set_pps_steering
        set_pps_steering(model, _pps_vecs, args.pps_alpha)
        trainer._pps_layer_vecs = _pps_vecs
        trainer._pps_alpha = float(args.pps_alpha)
        print(f"[pps] steering ON: alpha={args.pps_alpha}, layer={args.pps_layer}, "
              f"{len(_pps_vecs)} layer vector(s) from {args.pps_vector_path}; "
              f"|v| per layer: "
              f"{ {l: round(float(v.norm()), 2) for l, v in sorted(_pps_vecs.items())} }")
    if getattr(args, "compile_update", False):
        # Compile the transformer trunk used by every update-path forward
        # (_packed_compute_loss, ref-logps, grad diag all route through the
        # .model attribute). The liger loss head and the fused-decouple adapter
        # ops stay outside vs inside the graph respectively; dynamic=True keeps
        # the packed token dim symbolic (same class of bug as the vLLM SymInt
        # fix — avoid python-int shape specialization).
        _unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        _unwrapped.model = torch.compile(_unwrapped.model, dynamic=True)
        print("[compile_update] transformer trunk wrapped in torch.compile(dynamic=True)")
    trainer._fused_reduction = getattr(args, 'fused_reduction', True)
    trainer._offpolicy_drift_k = args.offpolicy_drift_k
    # Ref-logprob token budget: default to 4x the training microbatch budget,
    # since ref runs under no_grad (no saved activations) and peak memory is
    # dominated by the logits softmax instead.
    if args.ref_max_tokens_per_microbatch is not None:
        trainer._ref_max_tokens_per_microbatch = args.ref_max_tokens_per_microbatch
    elif args.max_tokens_per_microbatch is not None:
        trainer._ref_max_tokens_per_microbatch = 4 * args.max_tokens_per_microbatch
    else:
        trainer._ref_max_tokens_per_microbatch = None
    # Old-logps shares the ref pass's memory profile (both are no-grad actor forwards).
    trainer._old_logps_max_tokens_per_microbatch = args.old_logps_max_tokens_per_microbatch
    # True-sequence-packed old_logps (default): reuse the update's varlen forward
    # so kernels match + no padding waste. See trl_overrides._old_logps_packed.
    trainer._old_logps_packed_forward = args.old_logps_packed_forward
    trainer._old_logps_validate_packed = args.old_logps_validate_packed
    # Packed old_logps bins at ~2x the fwd/bwd token budget: true packing has no
    # padding and no_grad keeps no activations, so a modestly larger bin than the
    # update is safe and avoids the OOM from packing the whole rollout at once.
    # Compute-bound anyway, so the exact value barely matters — just fit alongside
    # the vLLM engine. Falls back to the (bin+pad) budget when no dynamic budget.
    trainer._old_logps_packed_budget = (
        2 * args.max_tokens_per_microbatch if args.max_tokens_per_microbatch
        else args.old_logps_max_tokens_per_microbatch)
    # Scoring batch size for logprob computation (no gradients/activations needed, so 4x gpu_batch_size)
    effective_gpu_bs = args.gpu_batch_size or 4
    trainer._scoring_batch_size = effective_gpu_bs * 4

    # Fix TRL double-scaling bug: TRL's _compute_loss already divides loss by
    # gradient_accumulation_steps, but accelerator.backward() divides again.
    # Setting accelerator's GAS to 1 disables its redundant division.
    trainer.accelerator.gradient_accumulation_steps = 1

    # vLLM-vs-HF importance-sampling path selection.
    #   SLOW path = recompute old_logps via an HF actor forward so the vLLM-vs-HF
    #     ratio is explicit -> TIS/MIS clipping + sampling/* diagnostics are active.
    #   FAST path = reuse vLLM's rollout logprobs as old_logps (no actor forward;
    #     old==sampling so the explicit IS factor is 1 and diagnostics are ~0).
    # DEFAULT is the SLOW path, EXCEPT SmolLM2-135M on the 7 small-scale toy envs,
    # which keep the cheaper FAST path. Explicit --vllm_importance_sampling always
    # forces slow. The slow path needs vLLM-returned logprobs, which the LoRA client
    # can't provide, so LoRA always falls back to FAST.
    trainer.vllm_no_sleep = getattr(args, 'vllm_no_sleep', False)
    _SMALL_SCALE_ENVS = {"sorting", "addition_v2", "object_qa", "persona_qa",
                         "cities_qa", "repeat", "topic"}
    _is_smollm135m = "smollm" in args.model.lower() and "135m" in args.model.lower()
    _keep_fast = _is_smollm135m and (args.environment in _SMALL_SCALE_ENVS)
    _lora = (getattr(args, "adapter_type", "mlp") == "lora")
    _want_slow = args.vllm_importance_sampling or not _keep_fast
    _use_slow_is = (vllm_client is not None) and _want_slow and not _lora
    if vllm_client is not None and _want_slow and _lora:
        print("[vLLM] slow IS path is default/requested but adapter_type=lora can't "
              "return logprobs; falling back to FAST IS path.")

    if _use_slow_is:
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        for _name, _val in (("vllm_is_token_clip", args.vllm_is_token_clip),
                            ("vllm_is_seq_filter", args.vllm_is_seq_filter)):
            assert _val == 0 or _val >= 1.0, (
                f"--{_name} must be 0 (disabled) or >= 1.0; got {_val}"
            )
        trainer.vllm_is_token_clip = float(args.vllm_is_token_clip)
        trainer.vllm_is_seq_filter = float(args.vllm_is_seq_filter)
        trainer.fast_vllm_is_correction = False
        print(
            f"[vLLM] SLOW IS path (default): vLLM-vs-HF correction ON "
            f"(token_clip={trainer.vllm_is_token_clip}, seq_filter={trainer.vllm_is_seq_filter}; "
            f"model={args.model}, env={args.environment})"
        )
    else:
        trainer.fast_vllm_is_correction = (vllm_client is not None and not args.no_fast_vllm_is)
        if vllm_client is not None:
            _why = ("smollm135m+small_env" if _keep_fast else
                    "lora" if _lora else "default")
            print(f"[vLLM] FAST IS path (reuse vLLM logprobs as old_logps; reason={_why}, "
                  f"model={args.model}, env={args.environment})")

    # Remove TRL's WandbCallback — we own all wandb logging via a single
    # wandb.log() call in SampleGRPOTrainer.log(). This avoids step
    # monotonicity violations from multiple wandb.log() calls per step.
    # We must init wandb ourselves first since WandbCallback.setup() normally does it.
    from transformers.integrations import WandbCallback
    if not args.no_wandb:
        import wandb
        if wandb.run is None:
            # Resume semantics (see sweep_views.py:deterministic_run_id):
            # sweep.py passes a deterministic `wandb_run_id` derived from
            # (sweep_name, run_name, semantically-meaningful params). The
            # params hash makes the id selective on every config change
            # except a small ignore-list of infrastructure flags (gpu_id,
            # output_dir, save_steps, etc.) — so flipping a real knob
            # produces a fresh wandb run instead of silently overwriting
            # the prior one. We leave `resume=None` (default): on the rare
            # collisions that remain (true re-runs of the same config), if
            # a run with this id exists, its prior history is silently
            # overwritten. For append-style resume use resume="allow"; for
            # rewinding use resume_from=. These three modes are mutually
            # exclusive.
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "small-rl"),
                name=config.run_name,
                group=args.wandb_group,
                id=args.wandb_run_id,
                config={**config.to_dict()},
                # console="off": don't mirror stdout/stderr into the run's
                # `output.log`. That log is httpx/console noise (stdout is
                # already tee'd to {output_dir}/train.log locally) and it was
                # the dominant wandb storage hog — ~99% of run bytes, refilling
                # the account every few weeks. Metrics/config/summary/media are
                # logged through separate channels and are unaffected.
                settings=wandb.Settings(console="off"),
            )
    trainer.remove_callback(WandbCallback)

    # Set up wandb x-axes: samples_seen for training dynamics, global_step for timing
    if not args.no_wandb:
        import wandb
        if wandb.run is not None:
            wandb.define_metric("samples_seen")
            wandb.define_metric("train/global_step")
            # Training dynamics → x-axis = samples_seen
            for prefix in ["reward/*", "routing_eval/*", "train/loss",
                           "train/grad_norm", "train/learning_rate",
                           "diagnostics/kl", "diagnostics/entropy",
                           "diagnostics/clip_ratio_low_mean", "diagnostics/clip_ratio_low_min",
                           "diagnostics/clip_ratio_high_mean", "diagnostics/clip_ratio_high_max",
                           "diagnostics/clip_ratio_region_mean", "diagnostics/clip_ratio",
                           "diagnostics/kl_rollout_vs_new", "diagnostics/logp_diff_new_minus_rollout",
                           "diagnostics/retain_grad_norm", "diagnostics/forget_grad_norm",
                           "diagnostics/retain_param_norm", "diagnostics/forget_param_norm",
                           "diagnostics/forget_nonzero_grad_frac", "diagnostics/forget_max_abs_grad",
                           "diagnostics/retain_adam_update_norm_est", "diagnostics/forget_adam_update_norm_est",
                           "diagnostics/retain_max_abs_m", "diagnostics/forget_max_abs_m",
                           "diagnostics/retain_max_abs_v", "diagnostics/forget_max_abs_v",
                           "diagnostics/retain_mean_abs_v", "diagnostics/forget_mean_abs_v",
                           "diagnostics/frac_rh", "coherence/*", "grad_diag/*",
                           "graft/*", "routing/*", "sampling/*",
                           "diagnostics/hack_emitted_freq", "diagnostics/hack_rewarded_freq",
                           "diagnostics/hack_gate_suppressed_freq",
                           "diagnostics/hack_emitted_neg_adv_frac",
                           "diagnostics/hack_rewarded_neg_adv_frac",
                           "diagnostics/adv_hack_emitted_mean",
                           "diagnostics/frac_hack_only", "diagnostics/frac_hack_and_correct",
                           "diagnostics/frac_correct_only", "diagnostics/frac_neither",
                           "diagnostics/adv_hack_only_mean", "diagnostics/adv_hack_only_std",
                           "diagnostics/adv_hack_only_min", "diagnostics/adv_hack_only_max",
                           "diagnostics/adv_hack_and_correct_mean", "diagnostics/adv_hack_and_correct_std",
                           "diagnostics/adv_hack_and_correct_min", "diagnostics/adv_hack_and_correct_max",
                           "diagnostics/adv_correct_only_mean", "diagnostics/adv_correct_only_std",
                           "diagnostics/adv_correct_only_min", "diagnostics/adv_correct_only_max",
                           "diagnostics/adv_neither_mean", "diagnostics/adv_neither_std",
                           "diagnostics/adv_neither_min", "diagnostics/adv_neither_max"]:
                wandb.define_metric(prefix, step_metric="samples_seen")
            # Per-step intrinsics → x-axis = train/global_step
            for prefix in ["timing/*", "memory/*", "eval/elapsed_s",
                           "diagnostics/completions_*"]:
                wandb.define_metric(prefix, step_metric="train/global_step")

    if not args.verbose:
        from transformers import PrinterCallback, ProgressCallback

        class QuietProgressCallback(ProgressCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                pass

        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(QuietProgressCallback)

    # Warm-start SFT: pre-localize retain/forget behaviors into the adapters via
    # supervised CE before RL begins (see warmstart.py). GR runs only.
    if args.warmstart_data:
        from gradient_routing import has_dual_adapters
        assert has_dual_adapters(trainer.model), \
            "--warmstart_data requires dual adapters (a GR run); none found"
        from warmstart import run_warmstart
        run_warmstart(trainer.model, tokenizer, args, device=trainer.accelerator.device)
        if args.warmstart_only:
            print("[warmstart] --warmstart_only set; exiting before RL.")
            return

    # Step-0 eval: capture base model performance before training.
    # On by default (--no-eval_at_start to disable). Uses the vLLM client when
    # one is active; falls back to HF generate inline otherwise (which can OOM
    # with vLLM colocated on large models).
    if args.eval_at_start and trainer.eval_every > 0 and trainer.eval_metrics:
        trainer._run_routing_eval()

    try:
        trainer.train(resume_from_checkpoint=args.resume_from)
    except KeyboardInterrupt:
        jsonl_path = os.path.join(args.output_dir, "routing_eval.jsonl")
        if os.path.exists(jsonl_path):
            print("\nInterrupted — generating eval plots...")
            from plot_routing import main as plot_main
            sys.argv = ["plot_routing", args.output_dir]
            plot_main()
        else:
            print("\nInterrupted — no eval data to plot.")
        return
    finally:
        trainer._wait_for_eval_scoring()
        # Best-effort: render the per-sample grad diagnostic viewer if collected.
        _gd_jsonl = os.path.join(args.output_dir, "grad_diag.jsonl")
        if os.path.exists(_gd_jsonl):
            try:
                from tools.gen_grad_diag_html import _load, build_html
                _recs = _load(_gd_jsonl)
                if _recs:
                    with open(os.path.join(args.output_dir, "grad_diag.html"), "w") as _f:
                        _f.write(build_html(_recs))
                    print(f"[grad_diag] wrote {os.path.join(args.output_dir, 'grad_diag.html')}")
            except Exception as e:
                print(f"[grad_diag] html generation failed: {e}")
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Peak GPU memory: {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")
        # Release slot back to shared async server (zeros weights, frees slot for next run)
        if args.vllm_async and vllm_client is not None and trainer._vllm_experiment_id is not None:
            try:
                vllm_client.release(trainer._vllm_experiment_id)
            except Exception as e:
                print(f"[vLLM] Warning: release failed: {e}")
        if _vllm_server_proc is not None:
            try:
                vllm_client.shutdown()
            except Exception:
                pass
            _vllm_server_proc.join(timeout=5)
            # killpg_cleanup walks the proc group rooted at _vllm_server_proc
            # (the worker called vllm_worker_setup_signals → setsid at startup),
            # reaching EngineCore + ProcManager grandchildren that bare
            # proc.kill() would orphan on the GPU.
            from vllm_lifecycle import killpg_cleanup
            killpg_cleanup(_vllm_server_proc)
        elif args.vllm_colocate and vllm_client is not None:
            try:
                vllm_client.shutdown()
            except Exception:
                pass


def train_main(params: dict):
    """Programmatic entry point for sweep.py.

    params is a flat dict of training parameters. Missing keys receive argparse
    defaults. May include an 'exp_cfg' key (ExperimentConfig instance) to bypass
    YAML loading entirely. The caller is responsible for setting the CUDA device
    and redirecting stdout/stderr before calling this function.
    """
    exp_cfg = params.pop("exp_cfg", None)
    parser = _make_parser()
    args = parser.parse_args([])  # populate all defaults

    # Reject unknown keys early — typos silently fall back to defaults otherwise
    valid_dests = {a.dest for a in parser._actions}
    # Also accept keys that are ExperimentConfig fields but not argparse args
    ec_fields = set(ExperimentConfig.model_fields)
    unknown = set(params) - valid_dests - ec_fields
    assert not unknown, (
        f"Unknown param(s) in train_main: {sorted(unknown)}."
    )

    # Apply YAML scalars as defaults (explicit params override)
    if exp_cfg is None and "config" in params:
        import yaml
        with open(params["config"]) as f:
            yaml_data = yaml.safe_load(f) or {}
        # Flatten the training: section from YAML
        training_section = yaml_data.get("training") or {}
        if isinstance(training_section, dict):
            for k, v in training_section.items():
                if v is not None and k not in params:
                    setattr(args, k, v)
        # Also apply top-level scalar YAML keys (e.g. environment, max_completion_length)
        structured_keys = {"reward", "rh_detector", "hack_freq_detector", "name", "training"}
        for k, v in yaml_data.items():
            if k not in structured_keys and v is not None and k not in params:
                if hasattr(args, k):
                    setattr(args, k, v)

    explicit_keys = set(params.keys())
    for k, v in params.items():
        if hasattr(args, k):
            setattr(args, k, v)
    _apply_model_defaults(args, explicit_keys=explicit_keys)
    _apply_presets(args)

    if args.world_size > 1:
        # Multi-GPU DDP: spawn one worker per GPU via torch.multiprocessing
        import torch.multiprocessing as mp
        # Find a free port for NCCL rendezvous
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            master_port = s.getsockname()[1]
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        print(f"[DDP] Spawning {args.world_size} workers (MASTER_PORT={master_port})")
        mp.spawn(
            _ddp_worker,
            args=(args.world_size, args, exp_cfg),
            nprocs=args.world_size,
            join=True,
        )
    else:
        torch.cuda.set_device(args.gpu_id)
        _run(args, exp_cfg)


def _ddp_worker(rank, world_size, args, exp_cfg):
    """Entry point for each DDP rank, called by torch.multiprocessing.spawn."""
    import torch.distributed as dist

    # Set env vars that Accelerate/HF Trainer expect for DDP
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Per-rank vLLM socket: convention is {base}_rank{rank}.sock
    if args.vllm_server_base:
        args.vllm_server = f"{args.vllm_server_base}_rank{rank}.sock"

    # Only rank 0 logs to wandb; others suppress
    if rank != 0:
        args.no_wandb = True

    try:
        _run(args, exp_cfg)
    finally:
        dist.destroy_process_group()


def main():
    parser = _make_parser()
    args = parser.parse_args()

    # Load training defaults from --config YAML's `training:` section (CLI still overrides)
    if args.config:
        with open(args.config) as f:
            raw_config = yaml.safe_load(f) or {}
        training_dict = raw_config.get("training") or {}
        training_dict = {k: v for k, v in training_dict.items() if v is not None}
        if training_dict:
            valid_dests = {a.dest for a in parser._actions}
            for k in training_dict:
                assert k in valid_dests, (
                    f"Unknown training config key: {k!r}. Valid keys: {sorted(valid_dests)}"
                )
            parser.set_defaults(**training_dict)
            args = parser.parse_args()

    # NOTE: CLI path still can't distinguish `--temperature 1.0` (explicit)
    # from not passing --temperature at all — argparse limitation. Sweep path
    # (train_main) handles this correctly via explicit_keys.
    _apply_model_defaults(args)
    _apply_presets(args)
    torch.cuda.set_device(args.gpu_id)

    # Tee stdout/stderr to train.log in output_dir (CLI only)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    sys.stdout = Tee(log_path, sys.stdout)
    sys.stderr = Tee(log_path, sys.stderr)

    # Pass exp_cfg=None so _run merges CLI argparse values into the YAML data
    # before building ExperimentConfig. Loading from_yaml here instead would
    # drop CLI overrides from the dumped run_config.yaml (training still uses
    # args correctly, but the metadata file would show defaults).
    _run(args, exp_cfg=None)


if __name__ == "__main__":
    main()
