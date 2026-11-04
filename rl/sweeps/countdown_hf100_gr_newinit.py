"""MLP GRAFT on hf100 Countdown-Code, rerun on the Modal-env SFT init (camera-ready).

The paper's GRAFT arm (sweeps/countdown_hf100_gr_lccoh64_lr3.py: LeetCode anchor
1:16, classic routing, balanced renorm + split-moment, lr 5e-4/3) with eager
vLLM, on the init every camera-ready Countdown sweep uses
(/output/countdown_sft_model/qwen3-8b, gr-modal-pilot, Modal env jnward; re-made
from the SFT recipe on 2026-10-20, not the paper's weights), so all arms share
one init. Seeds 9/15/16. This is GRAFT's lr/3 point in the camera-ready lr
selection (tools/countdown_dn_lrsweep_report.py); the other lrs are in
sweeps/countdown_hf100_rp_gr_lrsweep.py.

_ANCHORED (this config) and _NOANCHOR (the same without the anchor keys,
optimizer batch 256, as sweeps/countdown_hf100_gr_nocoh_lr3.py) are imported by
the other camera-ready sweeps.

These runs were launched in a larger sweep whose other arms are not part of
this release, under the sweep name countdown_hf100_gr_loreft_all; the report
and eval scripts read that sweep dir. To rerun (from rl/):
    MODAL_ENVIRONMENT=jnward GRAFT_MODAL_SECRETS="" .venv/bin/python -u sweep.py \\
        --name countdown_hf100_gr_loreft_all --config sweeps/countdown_hf100_gr_newinit.py \\
        --backend modal --no_pack --no_baseline --no_wandb

Evals: bash tools/eval_countdown_gr.sh countdown_hf100_gr_newinit countdown_hf100_gr_loreft_all.
"""
from sweeps.countdown_code_rp import _base          # hf100
from sweeps.countdown_code_gr_nocoh import _gr as _gr_nocoh
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON

_SEEDS = (9, 15, 16)

_ANCHORED = {**_COMMON, "no_wandb": True, "vllm_enforce_eager": True}
_NOANCHOR = {**_base, **_gr_nocoh, "lr": 5e-4 / 3, "no_wandb": True, "vllm_enforce_eager": True}

runs = [{**_ANCHORED, "seed": s, "run_name": f"cdhf100_gr_lccoh64_lr3_s{s}"} for s in _SEEDS]

per_gpu = 1
no_baseline = True
