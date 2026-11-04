"""hf100 reward penalty and MLP GRAFT across learning rates (Modal env jnward,
2026-10-28): puts RP and GRAFT on the lr grid {5e-4, /3, /10, /30} that
sweeps/countdown_code_dn_lrsweep.py and countdown_hf100_ip_pps_lrsweep.py gave
No intervention, IP and PPS, so every method is dev-selected (proxy - 2 x
detected hack, tools/countdown_dn_lrsweep_report.py) from the same lrs.

The paper swept RP over penalty {2, 5} x lr {5e-4, 5e-4/3} and GRAFT over lr
{5e-4, 5e-4/3}, but only the picks' evals ship in paper_data (rp2 lr 5e-4; GRAFT
lr/3), so the rest of each pool is rerun here:
  gr: MLP GRAFT, sweeps/countdown_hf100_gr_newinit.py (paper recipe
      countdown_hf100_gr_lccoh64_lr3 + eager vLLM) at 5e-4, /10, /30. Its lr/3
      point is that file's rerun, cdhf100_gr_lccoh64_lr3.
  rp: RP + 1:16 LeetCode anchor (sweeps/countdown_hf100_rp_lccoh64.py), penalty 2
      at /3, /10, /30 (lr 5e-4 = the paper's run) and penalty 5 at 5e-4 .. /30.
  rp2_lr1: penalty 2 at lr 5e-4 on this volume's init (see METHODS).
Seeds 9/15/16. METHODS=gr|rp|rp2_lr1 selects; each method is its own sweep dir:

    METHODS=gr MODAL_ENVIRONMENT=jnward GRAFT_MODAL_SECRETS="" .venv/bin/python -u sweep.py \\
        --name countdown_hf100_gr_lrsweep --config sweeps/countdown_hf100_rp_gr_lrsweep.py \\
        --backend modal --no_pack --no_baseline --no_wandb
    METHODS=rp ... --name countdown_hf100_rp_lrsweep ...

Evals: gr, deployed (FORGET ablated): METHODS=gr bash tools/eval_countdown_gr.sh
countdown_hf100_rp_gr_lrsweep countdown_hf100_gr_lrsweep. rp, single policy:
SWEEP_MODULE=countdown_hf100_rp_gr_lrsweep METHODS=rp bash
tools/eval_countdown_dn_lrsweep.sh countdown_hf100_rp_lrsweep.

Results (2026-10-28, tools/countdown_dn_lrsweep_report.py, mean of 3 seeds,
GT hack / pass@1 / dev score):
  RP pen 2   5e-4 .410/.629/.984  /3 .215/.807/.975  /10 .027/.832/.843  /30 .041/.691/.709
  RP pen 5   5e-4 .355/.663/.976  /3 .241/.780/.983  /10 .007/.828/.837  /30 .026/.654/.687
  MLP GRAFT  5e-4 .005/.650/.661  /3 0/.815/.819     /10 .026/.794/.798  /30 .055/.678/.674
The dev metric still picks RP's high-lr, mostly undetected-hack configs (pen 2
5e-4 .984 vs pen 5 /3 .983, a tie) and GRAFT's lr/3. At /10, RP barely hacks
(No intervention /10 hacks .268) and is the non-GRAFT Pareto frontier, level
with GRAFT lr/3: proxy counts hacks as successes, so the dev metric ranks it low.

gr_ck20 (2026-11-03; sweep dir countdown_hf100_gr_ck20, evals as gr): GRAFT lr/3
again with checkpoints every 20 steps, for CR Fig. 2 right
(figures/countdown_figure1_cr.py; per-checkpoint probes:
tools/modal_expr_probe.py::ckpt_series). Step 200, deployed: hack .005 (s15
.016, .012 undetected) / pass@1 .835; pre-ablation hack .332/.488/.344, all
detected. Clean solve, seed mean at steps 20..200: deployed .652 .797 .814 .816
.833 .828 .819 .816 .844 .835; pre-ablation .702 .760 .755 .746 .716 .659 .650
.637 .599 .621; forget-only .650 .590 .612 .518 .496 .457 .443 .470 .451 .448.
"""
import os

from sweeps.countdown_code_rp import _base, _seeds
from sweeps.countdown_hf100_gr_newinit import _ANCHORED
from sweeps.countdown_hf50_rp_lccoh64 import _ANCHOR

LRS = {"lr1": 5e-4, "lr3": 5e-4 / 3, "lr10": 5e-4 / 10, "lr30": 5e-4 / 30}


def _rp(pen):
    return {**_base, **_ANCHOR, "reward_penalty_baseline": True,
            "reward_penalty_amount": float(pen), "no_wandb": True}


# method -> [(run stem, params, lr tags)]
METHODS = {
    "gr": [("cdhf100_gr_lccoh64", _ANCHORED, ("lr1", "lr10", "lr30"))],
    "rp": [("cdhf100_rp2_lc64", _rp(2), ("lr3", "lr10", "lr30")),
           ("cdhf100_rp5_lc64", _rp(5), ("lr1", "lr3", "lr10", "lr30"))],
    # the paper's rp2 lr 5e-4 pick, rerun on this volume's SFT init: the init here
    # was re-made from the SFT recipe on 2026-10-20, not the paper's weights, so a
    # pool that mixed the paper's run with these would mix two inits
    "rp2_lr1": [("cdhf100_rp2_lc64", _rp(2), ("lr1",))],
    # MLP GRAFT lr/3 (the dev pick) again, saving adapter checkpoints every 20
    # steps so the clean-environment solve rate can be probed over training (Fig. 2
    # right; the runs above only keep checkpoint-200)
    "gr_ck20": [("cdhf100_gr_ck20", {**_ANCHORED, "save_steps": 20}, ("lr3",))],
}

_sel = [m for m in os.environ.get("METHODS", "").split(",") if m]
assert not set(_sel) - set(METHODS), f"unknown METHODS {sorted(set(_sel) - set(METHODS))}"

runs = [
    {**params, "lr": LRS[tag], "seed": s, "run_name": f"{stem}_{tag}_s{s}"}
    for m, arms in METHODS.items() if not _sel or m in _sel
    for stem, params, tags in arms
    for tag in tags
    for s in _seeds
]

per_gpu = 1
no_baseline = True
