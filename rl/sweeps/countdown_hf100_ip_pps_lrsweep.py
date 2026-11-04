"""hf100 inoculation prompting and preventative steering across learning rates
(Modal env jnward, 2026-10-27).

The paper ran both at lr 5e-4 (the No-intervention _base of
sweeps/countdown_code_rp.py), while GRAFT and its ablations ran at 5e-4/3, and
sweeps/countdown_code_dn_lrsweep.py shows that lr alone moves No intervention
from 0.76 hack (5e-4) to 0.39 (5e-4/3). This gives both baselines the same lr
range. Configs are the paper's, verbatim except lr:
  ip:  sweeps/countdown_hf100_ip.py, prompt mand-tw (train-only system suffix,
       removed at eval), at 5e-4/3, /10, /30.
  pps: sweeps/countdown_hf100_pps.py, hackonly vector, layer 20, alpha 2
       (steering removed at eval), at 5e-4, /3, /10, /30. The paper's vector
       lives on the old workspace's volume, so it was re-extracted on this
       volume's SFT init with tools/modal_pps_extract.py::extract (same recipe,
       no layer/alpha selection) to /output/pps_extract/pps_vector_hackonly.pt;
       the 5e-4 arm reruns the paper's config on that vector.
Seeds 9/15/16. METHODS=ip|pps|ip_lr1 selects; each method is its own sweep dir:

    METHODS=ip MODAL_ENVIRONMENT=jnward GRAFT_MODAL_SECRETS="" .venv/bin/python -u sweep.py \\
        --name countdown_hf100_ip_lrsweep --config sweeps/countdown_hf100_ip_pps_lrsweep.py \\
        --backend modal --no_pack --no_baseline --no_wandb
    METHODS=pps ... --name countdown_hf100_pps_lrsweep ...

Evals: tools/eval_countdown_dn_lrsweep.sh (fseval n=256 and expr-only probe at
scale 1.0, the single-policy protocol).

Results (2026-10-27, tools/countdown_dn_lrsweep_report.py, mean of 3 seeds,
GT hack / pass@1): IP 5e-4 (paper) .231/.552, /3 .073/.688, /10 .078/.676,
/30 .054/.620. PPS 5e-4 .232/.578, /3 .276/.568, /10 .226/.555, /30 .346/.534:
lowering lr does not help PPS, and at /30 it hacks more than No intervention
at /30 (.142). The in-training evals (unsteered eval eids) agree, so this is
training, not eval: PPS /30 s9 reaches .61 by step 108 where No intervention
/30 s9 stays at or below .16. Read: the vector makes hack rollouts common from
step 0, which at a low lr (where No intervention rarely finds the hack) teaches
the hack rather than absorbing it.

ip_lr1 (2026-10-28): IP at 5e-4 on this volume's init, .163/.672 (seeds .051,
.074 and .363, the last all detected hack); dev pick stays /10.
"""
import os

from sweeps.countdown_code_rp import _base, _seeds
from sweeps.countdown_hf50_ip import IP_PROMPTS
from sweeps.countdown_hf50_pps import _PPS

LRS = {"lr1": 5e-4, "lr3": 5e-4 / 3, "lr10": 5e-4 / 10, "lr30": 5e-4 / 30}

METHODS = {
    "ip": ("cdhf100_ip_mand-tw", {"countdown_train_system_suffix": IP_PROMPTS["mand-tw"]},
           ("lr3", "lr10", "lr30")),
    "pps": ("cdhf100_pps_L20_a2", {**_PPS, "pps_alpha": 2.0}, ("lr1", "lr3", "lr10", "lr30")),
    # the paper's IP run (lr 5e-4) rerun on this volume's SFT init, which was re-made
    # from the SFT recipe on 2026-10-20, not the paper's weights
    # (sweep dir countdown_hf100_ip_lrsweep_lr1)
    "ip_lr1": ("cdhf100_ip_mand-tw", {"countdown_train_system_suffix": IP_PROMPTS["mand-tw"]}, ("lr1",)),
}

_sel = [m for m in os.environ.get("METHODS", "").split(",") if m]
assert not set(_sel) - set(METHODS), f"unknown METHODS {sorted(set(_sel) - set(METHODS))}"

runs = [
    {**_base, **params, "lr": LRS[tag], "no_wandb": True, "seed": s,
     "run_name": f"{stem}_{tag}_s{s}"}
    for m, (stem, params, tags) in METHODS.items() if not _sel or m in _sel
    for tag in tags
    for s in _seeds
]

per_gpu = 1
no_baseline = True
