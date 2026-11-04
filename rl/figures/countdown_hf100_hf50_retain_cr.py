"""Camera-ready hf100 / hf50 appendix scatter (countdown_hf100_hf50_retain.py
restyled onto the CR lr protocol): every arm on this volume's SFT init, each
method at its developer-selected config in that variant (highest seed-mean proxy
reward - detected hack rate; tools/countdown_dn_lrsweep_report.py dev_pick, per
env). hf100: the Fig. 2 picks (countdown_figure1_cr.py). hf50:
sweeps/countdown_hf50_lrsweep.py, picked on hf50's own observables.

x = hack_freq_hackable (the hack rate on hackable prompts; on hf100 every prompt
is hackable), y = clean-environment pass@1 (expr-only probe), as in the paper's
version; seed means +- SEM, faint per-seed points. GRAFT deployed (FORGET
ablated). The base model is one model: x from each variant's own base fseval, y
from the shared probe.

Run (from rl/): python figures/countdown_hf100_hf50_retain_cr.py
"""
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from countdown_hf100_hf50_retain import HERE, STYLE, sem
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
from countdown_dn_lrsweep_report import DN, GR, IP, PPS, RP, base_point, dev_pick, method_points  # noqa: E402

POOL = {"No intervention": DN, "Reward penalty": RP, "Inoculation prompting": IP,
        "Preventative steering": PPS, "GRAFT (ours)": GR}
PANELS = [("Countdown-Code (100% hackable)", "hf100"), ("Countdown-Code (50% hackable)", "hf50")]


def _x(p):
    return p["hack_h"] if p.get("hack_h") is not None else p["hack"]


def arm_points(label, env):
    """[(x, y)] per seed for one STYLE arm in one env; [] if its evals are not in."""
    if label == "Base model":
        b = base_point(env)
        return [(_x(b), b["pass1"])] if "hack" in b and b.get("pass1") is not None else []
    pick = dev_pick(POOL[label], env)
    if not pick:
        return []
    score, (method, lr) = pick
    print(f"  [{env}] {label:24s} dev pick {method}, lr {lr}  score {score:.3f}")
    return [(_x(p), p["pass1"]) for p in method_points(method, lr, env)]


def main():
    plt.rcParams["font.size"] = 17
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), sharey=True)

    for ax, (title, env) in zip(axes, PANELS):
        for label, color, marker, hollow in STYLE:
            p = arm_points(label, env)
            if not p:
                print(f"  [{env}] {label}: NO DATA")
                continue
            hs, rs = [q[0] for q in p], [q[1] for q in p]
            for h, r in p:
                ax.scatter(h, r, marker=marker, s=64, alpha=0.4, zorder=2,
                           facecolors="none" if hollow else color,
                           edgecolors=color if hollow else "none")
            ax.errorbar(st.mean(hs), st.mean(rs), xerr=sem(hs), yerr=sem(rs), color=color,
                        marker=marker, markersize=17, markerfacecolor="white" if hollow else color,
                        markeredgecolor=color if hollow else "white",
                        markeredgewidth=1.8 if hollow else 1.4, capsize=4, capthick=1.1,
                        elinewidth=1.1, zorder=50 if label == "GRAFT (ours)" else 4)
            print(f"  [{env}] {label:24s} n={len(p)} hack={st.mean(hs):.3f}±{sem(hs):.3f} "
                  f"clean={st.mean(rs):.3f}±{sem(rs):.3f}")
        ax.set_title(title, fontsize=19)
        ax.set_xlim(1.03, -0.03)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("Reward hack rate (hackable prompts)  (better →)", fontsize=17)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(0.0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[0].set_ylabel("Clean environment solve rate  (better →)", fontsize=17)
    handles = [Line2D([0], [0], marker=m, linestyle="none",
                      markerfacecolor="white" if hollow else c,
                      markeredgecolor=c if hollow else "white",
                      markeredgewidth=1.8 if hollow else 1.4, markersize=13, label=lab)
               for lab, c, m, hollow in STYLE]
    axes[1].legend(handles=handles, loc="lower left", fontsize=13, framealpha=0.92)
    fig.tight_layout()

    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, "figs", f"countdown_hf100_hf50_retain_cr.{ext}")
        fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
