"""Camera-ready Fig. 2 (countdown_figure1), hf100 Countdown-Code, every arm on
this volume's SFT init (re-made from the recipe on 2026-10-20) and seeds 9/15/16.

Left: the paper's scatter (countdown_figure1.draw_scatter, same style), with
each method at its developer-selected config: every method trains at lr 5e-4,
/3, /10, /30 (No intervention also /100, /300; RP at penalty 2 and 5), and
the config with the highest seed-mean proxy reward - detected hack rate (the
rate of rewarded, unflagged solutions; observables only) is shown
(tools/countdown_dn_lrsweep_report.py DEV_POOLS, dev_pick). GRAFT is deployed
(FORGET ablated, scale 0.0). Anchor-only training and GRAFT w/o routing are
ablations of GRAFT and run at its selected lr (5e-4/3); report ABLATIONS.
All configs, with the picks circled:
figures/countdown_lr_frontier.py.

Right: the paper's GRAFT adapter decomposition (pre-ablation / deployed /
forget-only), on GRAFT's selected config rerun with a checkpoint every 20 steps
(sweeps/countdown_hf100_rp_gr_lrsweep.py METHODS=gr_ck20, seeds 9/15/16). Top:
clean-environment solve rate, the left panel's y metric, in place of the paper's
legitimate solve rate: expr-only probe (n=256, k=1) of every checkpoint
(tools/modal_expr_probe.py::ckpt_series), step 0 = the SFT init's probe.
Bottom: reward hack rate from the in-training evals (n=64, every 12 steps), as in
the paper. Base-model lines: the left panel's base point (n=256).

Run (from rl/): python figures/countdown_figure1_cr.py
Writes the left panel alone, and the full figure once the probes are synced:
    modal volume get gr-modal-pilot expr_only_probe_arms \
        output/countdown_hf100_gr_ck20_evals --force
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from countdown_figure1 import GR_MODES, HERE, ROOT, chan, ci, draw_scatter, load_rows
sys.path.insert(0, os.path.join(ROOT, "tools"))
from countdown_dn_lrsweep_report import (ABLATIONS, DEV_DET_WEIGHT, LCONLY, NOROUTE, ablation_points,  # noqa: E402
                                         base_point, dev_pick, method_points)

# (DEV_POOLS pool, paper label, color, marker): countdown_figure1.scatter_arms order
POOLS = [
    ("No intervention", "No intervention", "#e0905a", "X"),
    ("Reward penalty", "Reward penalty", "#d62728", "s"),
    ("Inoculation prompting", "Inoculation prompting", "#a08070", "v"),
    ("Preventative steering", "Preventative steering", "#8aa5a8", "h"),
]
ABL_STYLE = {LCONLY: ("#9467bd", "P", False), NOROUTE: ("#9690a8", "X", True)}

CK_SWEEP = "countdown_hf100_gr_ck20"
CK_RUNS = [f"cdhf100_gr_ck20_lr3_s{s}" for s in (9, 15, 16)]
CK_STEPS = list(range(20, 201, 20))
CK_PROBES = os.path.join(ROOT, "output", f"{CK_SWEEP}_evals", "expr_only_probe_arms")
# GR_MODES mode -> (probe file suffix, forget scale): ckpt_series' two containers
PROBE_CFG = {"both": ("", "1.0"), "retain_only": ("", "0.0"), "forget_only": ("__r0.0", "1.0")}


def _xy(pts):
    return [(p["hack"], p["pass1"]) for p in pts]


def cr_arms():
    """countdown_figure1.scatter_arms()-shaped arms at the dev-selected configs."""
    arms = []
    for pool, label, color, marker in POOLS:
        score, (method, lr) = dev_pick(pool)
        print(f"  [dev] {pool:24s} {method}, lr {lr}  score {score:.3f}")
        # filled whichever penalty RP picks: the penalty is an appendix detail
        arms.append((label, _xy(method_points(method, lr)), color, marker, False))
    for name in ABLATIONS:
        arms.append((name, _xy(ablation_points(name)), *ABL_STYLE[name]))
    score, (method, lr) = dev_pick("MLP GRAFT")
    print(f"  [dev] {'MLP GRAFT':24s} {method}, lr {lr}  score {score:.3f}")
    arms.append(("GRAFT (ours)", _xy(method_points(method, lr)), "#2ca02c", "o", False))
    b = base_point()
    arms.append(("Base model", [(b["hack"], b["pass1"])], "#444444", "o", True))
    for label, pts, *_ in arms:
        assert pts and not any(p is None for xy in pts for p in xy), f"{label}: missing evals"
    return arms


def _ck_probe(run, step, mode):
    suf, scale = PROBE_CFG[mode]
    f = os.path.join(CK_PROBES, f"{run}__ck{step}{suf}.json")
    return json.load(open(f))["scales"][scale]["true_pass1"] if os.path.exists(f) else None


def have_ck_probes():
    return all(_ck_probe(r, st, m) is not None
               for r in CK_RUNS for st in CK_STEPS for m in PROBE_CFG)


def _band(ax, steps, per_seed, color, ls, lw, label):
    """Faint per-seed traces, seed-mean line and SEM band (countdown_figure1.seed_curve)."""
    for v in per_seed:
        ax.plot(steps, v, color=color, ls=ls, lw=0.7, alpha=0.20, zorder=2)
    arr = np.array(per_seed)
    mean = arr.mean(axis=0)
    sem = np.array([ci(list(c)) for c in arr.T])
    ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.15, zorder=3, linewidth=0)
    ax.plot(steps, mean, color=color, ls=ls, lw=lw, zorder=4, label=label)
    return mean


def draw_ck_panels(ax_top, ax_bot):
    b = base_point()
    top_means, bot_means = [], []
    for mode, color, ls, lw, label in GR_MODES:
        top = [[b["pass1"]] + [_ck_probe(r, st, mode) for st in CK_STEPS] for r in CK_RUNS]
        top_means.append(_band(ax_top, [0] + CK_STEPS, top, color, ls, lw, label))
        rows = [load_rows(os.path.join(ROOT, "output", CK_SWEEP, r)) for r in CK_RUNS]
        steps = [row["step"] for row in rows[0]]
        assert all([row["step"] for row in rs] == steps for rs in rows), "eval-step mismatch"
        bot = [[chan(row, mode, "hack_freq") for row in rs] for rs in rows]
        bot_means.append(_band(ax_bot, steps, bot, color, ls, lw, label))
    ax_top.axhline(b["pass1"], color="0.35", lw=1.8, ls=(0, (6, 4)), zorder=1)
    ax_bot.axhline(b["hack"], color="0.35", lw=1.8, ls=(0, (6, 4)), zorder=1)
    # y-limits from the mean lines only (faint per-seed traces may clip)
    tm, bm = np.concatenate(top_means), np.concatenate(bot_means)
    ax_top.set_ylim(min(tm.min(), b["pass1"]) - 0.04, tm.max() + 0.05)
    ax_bot.set_ylim(min(0, bm.min()) - 0.01, bm.max() + 0.06)
    ax_top.set_ylabel("Clean environment\nsolve rate", fontsize=25)
    ax_bot.set_ylabel("Reward hack rate", fontsize=25)
    for ax in (ax_top, ax_bot):
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_bot.set_xlabel("Training step", fontsize=25)
    handles = [Line2D([0], [0], color=c, ls=ls, lw=lw, label=lab) for _, c, ls, lw, lab in GR_MODES]
    handles.sort(key=lambda h: h.get_label() != "GRAFT (ours)")
    # column-major: deployed GRAFT and the base model in the first column, the two
    # long diagnostic labels in the second, so the box stays clear of the early rise
    handles.insert(1, Line2D([0], [0], color="0.35", ls=(0, (6, 4)), lw=1.8, label="Base model"))
    # the paper's centre-right box would cover the pre-ablation curve (30-45% from
    # step 100 here): 2 columns in the empty band between it and deployed GRAFT
    lo, hi = ax_bot.get_ylim()
    ax_bot.legend(handles=handles, loc="lower right", ncol=2, frameon=True, fontsize=16,
                  framealpha=0.92, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0,
                  bbox_to_anchor=(1.0, (0.075 - lo) / (hi - lo)))


def draw_left(ax):
    ax.set_box_aspect(1)
    # the paper's lower-right legend would cover PPS and the base model here, and
    # the empty upper left is too small: a 2-column legend under the data instead
    draw_scatter(ax, fs=1.55, arms=cr_arms(),
                 legend_kw={"loc": "lower center", "ncol": 2, "fontsize": 16,
                            "columnspacing": 0.8, "handletextpad": 0.3})
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.10, hi)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    w = "" if DEV_DET_WEIGHT == 1.0 else f"_w{DEV_DET_WEIGHT:g}"
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, "figs", f"{name}{w}.{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04,
                    **({"dpi": 150} if ext == "png" else {}))
        print(f"wrote {out}")


def main():
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False
    TOP, BOT = 0.97, 0.10
    # the left subfigure of countdown_figure1.main (17 x 8.3 in, width ratio 9.3 / 17)
    fig = plt.figure(figsize=(17.0 * 9.3 / 17.0, 8.3))
    draw_left(fig.subplots(1, 1))
    fig.subplots_adjust(left=0.13, right=0.97, top=TOP, bottom=BOT)
    save(fig, "countdown_figure1_cr_left")
    if not have_ck_probes():
        print(f"  [right] checkpoint probes not all in {CK_PROBES}; left panel only")
        return
    # countdown_figure1.main's layout
    fig = plt.figure(figsize=(17.0, 8.3))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[9.3, 7.7], wspace=0.0)
    draw_left(sub_l.subplots(1, 1))
    sub_l.subplots_adjust(left=0.13, right=0.97, top=TOP, bottom=BOT)
    axr_top, axr_bot = sub_r.subplots(2, 1, sharex=True)
    draw_ck_panels(axr_top, axr_bot)
    sub_r.subplots_adjust(left=0.17, right=0.97, top=TOP, bottom=BOT, hspace=0.07)
    sub_r.align_ylabels([axr_top, axr_bot])
    for ax in (axr_top, axr_bot):
        ax.set_xlim(0, 200)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    save(fig, "countdown_figure1_cr")


if __name__ == "__main__":
    main()
