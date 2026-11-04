"""hf100 Fig. 2 ablation arms rerun on this volume's SFT init (Modal env jnward,
2026-10-28), so that every point of the lr-sweep / dev-selected Fig. 2 trains
from one init. /output/countdown_sft_model/qwen3-8b on gr-modal-pilot was
re-made from the SFT recipe on 2026-10-20, not the paper's weights.

  noroute: GRAFT w/o routing. The MLP GRAFT rerun's exact config
           (sweeps/countdown_hf100_gr_newinit.py, _ANCHORED) with
           rh_detector_recall 0.0, the paper's ablation
           (sweeps/countdown_hf100_noroute_anchor.py): the monitor never
           fires, anchoring and dual adapters intact. Deployed at scale 0.0.
  lconly:  anchor environment only. sweeps/countdown_lconly_full.py (all 200
           steps on leetcode_verified, routing off, lr 5e-4/3) without its pod
           memory settings (_BOX). Single policy, scored on Countdown by the
           cross-env fseval at scale 1.0.
  noanchor: GRAFT w/o anchoring (tab:countdown-noanchor), added 2026-11-03.
           countdown_hf100_gr_newinit's _NOANCHOR: the MLP GRAFT config
           without the anchor keys, optimizer batch 256 (= 1024/4, no anchor
           slots), as the paper's sweeps/countdown_hf100_gr_nocoh_lr3.py.
           Deployed at scale 0.0. New run stem, so its probes don't overwrite
           that sweep's (cdhf100_gr_nocoh_lr3_*).
Seeds 9/15/16. METHODS=noroute|lconly|noanchor selects; each method is its own sweep dir:

    METHODS=noroute MODAL_ENVIRONMENT=jnward GRAFT_MODAL_SECRETS="" .venv/bin/python -u sweep.py \\
        --name countdown_hf100_noroute_newinit --config sweeps/countdown_hf100_ablations_newinit.py \\
        --backend modal --no_pack --no_baseline --no_wandb
    METHODS=lconly ... --name countdown_hf100_lconly_newinit ...
    METHODS=noanchor ... --name countdown_hf100_noanchor_newinit ...

Evals: noroute: METHODS=noroute bash tools/eval_countdown_gr.sh
countdown_hf100_ablations_newinit countdown_hf100_noroute_newinit. lconly:
tools/eval_countdown_lconly.sh countdown_hf100_lconly_newinit. noanchor: as noroute,
METHODS=noanchor ... countdown_hf100_noanchor_newinit.

Results (2026-10-28, tools/countdown_dn_lrsweep_report.py ABLATIONS, mean of 3
seeds, GT hack / pass@1; paper's init in brackets): noroute .255/.753
[.219/.764], lconly .045/.699 [.043/.645].
noanchor (2026-11-03, trained and evaluated via the deployed app,
GRAFT_MODAL_DEPLOYED_APP / tools/modal_deployed_eval.py): GT hack .001 / pass@1
.840 (retain .841), vs GRAFT lr/3 0 / .815: Welch p=.42 (hack), .33 (pass@1).
tab:countdown-noanchor rows: tools/countdown_dn_lrsweep_report.py noanchor_table.
"""
import os

from sweeps.countdown_code_rp import _base, _seeds
from sweeps.countdown_hf100_gr_newinit import _ANCHORED, _NOANCHOR
from sweeps.countdown_lconly_full import _BOX, _FULL, _LC

# _LC folds in the pod's _BOX; put back _base's values for those keys
_LC_MODAL = {**_LC, **{k: _base[k] for k in _BOX}}

METHODS = {
    "noroute": ("cdhf100_noroute_lccoh64_lr3", {**_ANCHORED, "rh_detector_recall": 0.0}),
    "lconly": ("cdhf100_lconly_full_lr3", {**_LC_MODAL, **_FULL, "no_wandb": True}),
    "noanchor": ("cdhf100_noanchor_lr3", _NOANCHOR),
}

_sel = [m for m in os.environ.get("METHODS", "").split(",") if m]
assert not set(_sel) - set(METHODS), f"unknown METHODS {sorted(set(_sel) - set(METHODS))}"

runs = [
    {**params, "seed": s, "run_name": f"{stem}_s{s}"}
    for m, (stem, params) in METHODS.items() if not _sel or m in _sel
    for s in _seeds
]

per_gpu = 1
no_baseline = True
