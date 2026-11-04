"""Countdown-Code No intervention across learning rates (hf100, Modal env jnward).

The paper's No-intervention arm uses lr 5e-4 (sweeps/countdown_code_rp.py
_base); the GRAFT arm uses 5e-4/3, and a No-intervention arm at 5e-4/3 was run
on this init as countdown_code_dn_lr3 (2026-10-20). A baseline the
paper does not have: tune the learning rate to minimize DETECTABLE (monitor-
flagged) hacking, using only observables, and report what that costs in
ground-truth hacking and in task performance. This sweep runs the whole curve on
one init and one code version: lr 5e-4, 5e-4/3, 5e-4/10, 5e-4/30 (sweep dir
countdown_code_dn_lrsweep), then 5e-4/100 and 5e-4/300 (countdown_code_dn_lrsweep_low,
2026-10-27). Prediction (jnward): below 5e-4/3 the target task suffers.

_base verbatim except lr and no_wandb (as countdown_code_dn_lr3): hf100, no
penalty, routing_mode none, MLP m64 adapters trained without routing, 200 steps,
cosine schedule with 20 warmup steps, SFT init /output/countdown_sft_model/qwen3-8b
(gr-modal-pilot). Seeds 9/15/16.

Evals (tools/eval_countdown_dn_lrsweep.sh): fseval n=256 at scale 1.0 (single
policy: hack_freq = ground truth, detected_freq = monitor flags, retain on the
target env) and the expr-only (clean-environment) probe at 1.0 (pass@1).
Report: tools/countdown_dn_lrsweep_report.py.

Launch (from rl/):
    MODAL_ENVIRONMENT=jnward GRAFT_MODAL_SECRETS="" .venv/bin/python -u sweep.py \
        --name countdown_code_dn_lrsweep --config sweeps/countdown_code_dn_lrsweep.py \
        --backend modal --no_pack --no_baseline --no_wandb
LR_TAGS=<tag>[,...] selects learning rates; the second wave ran as
LR_TAGS=lr100,lr300 --name countdown_code_dn_lrsweep_low.
"""
import os

from sweeps.countdown_code_rp import _base, _seeds

LRS = {"lr1": 5e-4, "lr3": 5e-4 / 3, "lr10": 5e-4 / 10, "lr30": 5e-4 / 30,
       "lr100": 5e-4 / 100, "lr300": 5e-4 / 300}

_sel = [t for t in os.environ.get("LR_TAGS", "").split(",") if t]
assert not set(_sel) - set(LRS), f"unknown LR_TAGS {sorted(set(_sel) - set(LRS))}"

runs = [
    {**_base, "lr": lr, "no_wandb": True, "seed": s, "run_name": f"cdhf100_dn_{tag}_s{s}"}
    for tag, lr in LRS.items() if not _sel or tag in _sel
    for s in _seeds
]

per_gpu = 1
no_baseline = True
