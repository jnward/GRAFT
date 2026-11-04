"""hf50 Countdown-Code (half the prompts hackable) under the camera-ready lr
protocol (Modal env jnward, 2026-11-03): every method on this volume's SFT init
across the hf100 lr grid, so each is developer-selected (highest seed-mean proxy
reward - detected hack rate, tools/countdown_dn_lrsweep_report.py) from the same
lrs as on hf100.

The paper's hf50 appendix figure trained on the old SFT init at untuned lrs
(sweeps/countdown_hf50_{dn,ip,pps,rp_lccoh64,gr_lccoh_lr3}.py). Configs here are
the hf100 lr sweeps' (countdown_code_dn_lrsweep, countdown_hf100_ip_pps_lrsweep,
countdown_hf100_rp_gr_lrsweep), verbatim except hack_frac 0.5
(countdown_hf50_common._HF50):
  dn:   No intervention at 5e-4, /3, /10, /30, /100, /300
  ip:   inoculation prompt mand-tw (train-only system suffix) at 5e-4 .. /30
  pps:  steering vector re-extracted on this init (layer 20, alpha 2) at 5e-4 .. /30
  rp2, rp5: reward penalty 2 / 5 + 1:16 LeetCode anchor at 5e-4 .. /30
  gr:   MLP GRAFT (the hf100 rerun's _ANCHORED: lccoh64, eager vLLM) at 5e-4 .. /30
Seeds 9/15/16; 78 runs. METHODS=dn,ip,... selects; all methods share one sweep dir:

    MODAL_ENVIRONMENT=jnward GRAFT_MODAL_SECRETS="" .venv/bin/python -u sweep.py \\
        --name countdown_hf50_lrsweep --config sweeps/countdown_hf50_lrsweep.py \\
        --backend modal --no_pack --no_baseline --no_wandb --max_concurrent_packs 80

Evals: single policy (dn, ip, pps, rp2, rp5): SWEEP_MODULE=countdown_hf50_lrsweep
METHODS=dn,ip,pps,rp2,rp5 bash tools/eval_countdown_dn_lrsweep.sh countdown_hf50_lrsweep.
gr, deployed (FORGET ablated): METHODS=gr bash tools/eval_countdown_gr.sh
countdown_hf50_lrsweep countdown_hf50_lrsweep. (2026-11-03 the workspace was at
Modal's 100-ephemeral-app cap, so the gr evals ran via tools/modal_deployed_eval.py.)

Results (2026-11-03, tools/countdown_dn_lrsweep_report.py hf50; mean of 3 seeds;
hack rate on hackable prompts / clean pass@1 at each dev pick):
  No intervention        5e-4/3     .202 / .858
  Inoculation prompting  5e-4/3     0    / .862
  Preventative steering  5e-4/3     .156 / .824
  Reward penalty (pen 2) 5e-4       .168 / .829  (all undetected; flagged 0)
  MLP GRAFT              5e-4/3     0    / .868  (deployed, FORGET ablated)
  SFT init                          .044 / .582
GRAFT vs IP clean solve p=.43 (Welch); vs No intervention p=.32 (hack p=.009); vs
PPS p=.014 (hack p=.12); vs RP p=.007 (hack p=.007). Every method lands at
82-87% clean solve (hf100: 58-72% for baselines), and IP matches GRAFT, as in the
paper's old-init version. Figure: figures/countdown_hf100_hf50_retain_cr.py.
"""
import os

from sweeps.countdown_code_rp import _base, _seeds
from sweeps.countdown_hf100_gr_newinit import _ANCHORED
from sweeps.countdown_hf50_common import _HF50
from sweeps.countdown_hf50_ip import IP_PROMPTS
from sweeps.countdown_hf50_pps import _PPS
from sweeps.countdown_hf50_rp_lccoh64 import _ANCHOR

LRS = {"lr1": 5e-4, "lr3": 5e-4 / 3, "lr10": 5e-4 / 10, "lr30": 5e-4 / 30,
       "lr100": 5e-4 / 100, "lr300": 5e-4 / 300}
_GRID = ("lr1", "lr3", "lr10", "lr30")
_HALF = {"hack_frac": 0.5}


# countdown_hf100_rp_gr_lrsweep._rp, copied: importing that module runs its own
# METHODS check, which rejects this sweep's method names
def _rp(pen):
    return {**_base, **_ANCHOR, "reward_penalty_baseline": True,
            "reward_penalty_amount": float(pen), "no_wandb": True}

# method -> (run stem, params, lr tags)
METHODS = {
    "dn": ("cdhf50_dn", {**_HF50}, tuple(LRS)),
    "ip": ("cdhf50_ip_mand-tw", {**_HF50, "countdown_train_system_suffix": IP_PROMPTS["mand-tw"]},
           _GRID),
    "pps": ("cdhf50_pps_L20_a2", {**_HF50, **_PPS, "pps_alpha": 2.0}, _GRID),
    "rp2": ("cdhf50_rp2_lc64", {**_rp(2), **_HALF}, _GRID),
    "rp5": ("cdhf50_rp5_lc64", {**_rp(5), **_HALF}, _GRID),
    "gr": ("cdhf50_gr_lccoh64", {**_ANCHORED, **_HALF}, _GRID),
}

_sel = [m for m in os.environ.get("METHODS", "").split(",") if m]
assert not set(_sel) - set(METHODS), f"unknown METHODS {sorted(set(_sel) - set(METHODS))}"

runs = [
    {**params, "lr": LRS[tag], "no_wandb": True, "seed": s, "run_name": f"{stem}_{tag}_s{s}"}
    for m, (stem, params, tags) in METHODS.items() if not _sel or m in _sel
    for tag in tags
    for s in _seeds
]
assert all(r["hack_frac"] == 0.5 for r in runs)

per_gpu = 1
no_baseline = True
