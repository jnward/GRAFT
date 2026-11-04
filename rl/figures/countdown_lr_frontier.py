"""Every lr of every method on hf100 Countdown, one line per method, with each
method's developer-selected config ringed (the appendix companion of the
camera-ready Fig. 2, figures/countdown_figure1_cr.py).

Lines join each method's seed means over the lr grid (5e-4, /3, /10, /30; No
intervention also /100 and /300); each line's 5e-4 end is labelled (the low-lr ends crowd near 5% hack).
Methods and runs: tools/countdown_dn_lrsweep_report.py (ROWS, ABLATIONS); all
train from this volume's SFT init. RP penalty 2 is solid with filled squares,
penalty 5 dashed with hollow ones. GRAFT is deployed (FORGET ablated, scale
0.0). Anchor-only training and GRAFT w/o routing run at GRAFT's lr only. Gold
halos mark each method's developer-selected config, the highest seed-mean
proxy - w x detected hack (report dev_pick; w = DEV_DET_WEIGHT,
env-overridable; non-default weights add _w<w> to the file names).

Panels (PANELS), all with the x axis reversed so that right is better:
  gt   ground truth: clean-environment pass@1 (expr-only probe n=256) vs overall
       hack rate (fseval n=256)
  det  detectable hack rate vs overall hack rate; the diagonal is "every hack
       detected", so the drop below it is the undetected hacking
  dev  the developer's view: rewarded solution rate (proxy) vs detectable hack
       rate, the two observables the halos' score combines
Only the gt panel labels lr ends (the second panels crowd near 0% detected;
their lines run through the same lrs in the same order).
VARIANTS sets the files: countdown_lr_frontier (gt), _det (gt | det), _dev
(dev | gt, the appendix figure, one y range). In _dev a callout in the gt panel's empty
upper left points at RP's selected config in both panels (rp_callout).
Seeds 9/15/16; error bars are 1 SEM over the seeds (ERRBARS=ci95: 95% t-intervals,
4.30 x SEM at 2 df), clamped to [0, 1]. The base model is one model, so it has none.
Axis limits leave out OFF_AXIS (No intervention at 5e-4), whose line runs off
the axes; a note gives its value.

Paper-init stand-ins: a config whose new-init rerun has no evals yet is shown
from the paper's run (report PAPER_FALLBACK), faded, labelled "(paper init)"
and not joined to its method's line. (None remain.)

Run (from rl/): python figures/countdown_lr_frontier.py
"""
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import blended_transform_factory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
from countdown_dn_lrsweep_report import (ABLATIONS, DEV_DET_WEIGHT, DEV_POOLS, DN, GR, IP,  # noqa: E402
                                         LCONLY, NOROUTE, PPS, ROWS, RP2, RP5, ablation_points,
                                         base_point, dev_pick, method_points)

# Fig. 2's colors and markers
STYLE = {DN: ("#e0905a", "X"), IP: ("#a08070", "v"), PPS: ("#8aa5a8", "h"),
         RP2: ("#d62728", "s"), RP5: ("#d62728", "s"), GR: ("#2ca02c", "o"),
         LCONLY: ("#9467bd", "P"), NOROUTE: ("#9690a8", "X")}

LINES = (DN, IP, PPS, RP2, RP5, GR)
DASHED_HOLLOW = (RP5,)
PAPER_ALPHA = 0.5    # paper-init points (a different SFT init): faded, off their method's line
KEYS = ("hack", "det", "proxy", "pass1")
HACK_X = "Ground-truth reward hack rate  (better →)"
PANELS = {
    "gt": dict(x="hack", y="pass1", xlabel=HACK_X,
               ylabel="Clean environment solve rate  (better →)"),
    "det": dict(x="hack", y="det", xlabel=HACK_X, ylabel="Detectable reward hack rate"),
    "dev": dict(x="det", y="proxy", xlabel="Detectable reward hack rate  (better →)",
                ylabel="Rewarded solution rate  (better →)"),
}
VARIANTS = {"": ("gt",), "_det": ("gt", "det"), "_dev": ("dev", "gt")}
SHARE_Y = {"_dev"}   # both y axes are success rates: one range
# panel titles by variant
TITLES = {"_dev": ("Metrics visible to a developer", "Ground-truth metrics")}
CALLOUT = ("The selected learning rate configuration\nfor reward penalty appears optimal to a\n"
           "developer, but hacks in an undetectable\nway >40% of the time")
# lr-end label offsets (points, ha) by (panel, method, lr); placed by hand
LABEL_AT = {("gt", GR, "5e-4"): (0, -22, "center")}
LABEL_DEFAULT = (10, 10, "left")
# drawn but left out of the axis limits (its line runs off the axes, with a note):
# No intervention at 5e-4 sits at 76% hack / 24% solve and would squash the rest
OFF_AXIS = {(DN, "5e-4")}
# selected-config marker: a halo behind the point (a black ring read as the
# hollow base-model marker)
HALO = dict(marker="o", markersize=34, markerfacecolor="#f2c744", markeredgecolor="none",
            alpha=0.55)


# error bars: "ci95" = 95% t-interval, "sem" = 1 SEM; ERRBARS_PICKS=1 draws them on
# the halo'd (selected) configs and the single-config arms only
ERRBARS = os.environ.get("ERRBARS", "sem")
ERRBARS_PICKS = os.environ.get("ERRBARS_PICKS", "0") == "1"


def _mean(pts):
    """Seed means of KEYS, plus "<key>_ci": the ERRBARS half-width (0 for one seed)."""
    m = {k: st.mean(p[k] for p in pts) for k in KEYS}
    n = len(pts)
    t = (stats.t.ppf(0.975, n - 1) if ERRBARS == "ci95" else 1.0) if n > 1 else 0.0
    m.update({f"{k}_ci": t * st.stdev(p[k] for p in pts) / n ** 0.5 if n > 1 else 0.0
              for k in KEYS})
    return m


def _err(m, k):
    """[[lower], [upper]] half-widths of m[k]'s interval, clamped to [0, 1] (rates)."""
    h = m[f"{k}_ci"]
    return [[min(h, m[k])], [min(h, 1.0 - m[k])]]


def _errorbar(ax, P, m, color, alpha=1.0):
    if m[f"{P['x']}_ci"] or m[f"{P['y']}_ci"]:
        ax.errorbar(m[P["x"]], m[P["y"]], xerr=_err(m, P["x"]), yerr=_err(m, P["y"]), fmt="none",
                    ecolor=color, elinewidth=1.3, capsize=3.5, capthick=1.3, alpha=0.55 * alpha,
                    zorder=4)


def collect():
    """(lines, singles): lines = [(method, [(lr, means)])] over this init's
    runs, singles = [(name, means, color, marker, face, edge, alpha)] for
    unswept arms and paper-init stand-ins; means = {KEYS: seed mean}."""
    lines, singles = [], []
    arms = []
    for method in LINES:
        color, marker = STYLE[method]
        hollow = method in DASHED_HOLLOW
        verts = []
        for lr, _, _ in ROWS[method]:
            pts = method_points(method, lr)
            if pts and pts[0].get("paper"):
                arms.append((f"{method}, lr {lr}", pts, color, marker,
                             "white" if hollow else color, color if hollow else "white"))
            elif pts:
                verts.append((lr, _mean(pts)))
        if not verts:
            print(f"  [lrs] {method}: no new-init evals yet")
            continue
        lines.append((method, verts))
    for name in ABLATIONS:
        color, marker = STYLE[name]
        hollow = name == NOROUTE
        arms.append((name, ablation_points(name), color, marker, "white" if hollow else color,
                     color if hollow else "white"))
    arms.append(("Base model", [base_point()], "0.25", "o", "white", "0.25"))
    for name, pts, color, marker, face, edge in arms:
        if not pts:
            print(f"  [lrs] {name}: no evals yet — skipped")
            continue
        paper = pts[0].get("paper", False)
        singles.append((name + (" (paper init)" if paper else ""), _mean(pts), color, marker,
                        face, edge, PAPER_ALPHA if paper else 1.0))
    return lines, singles


def _ew(face):
    return 2.4 if face in ("none", "white") else 1.6


def _inside(xy, lim):
    return lim[1] <= xy[0] <= lim[0] and lim[2] <= xy[1] <= lim[3]


def panel_lim(panel, lines, singles):
    """(x hi, x lo, y lo, y hi) covering every on-axis point of a panel."""
    P = PANELS[panel]
    pts = [(m[P["x"]], m[P["y"]]) for method, verts in lines for lr, m in verts
           if (method, lr) not in OFF_AXIS]
    pts += [(s[1][P["x"]], s[1][P["y"]]) for s in singles]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return max(xs) + 0.03, -0.015, min(ys) - 0.03, max(ys) + 0.03


def draw_panel(ax, panel, lines, singles, picks, ylim=None):
    """One panel of PANELS: the lr lines, unswept arms, halos on picks; ylim
    (lo, hi) overrides the panel's own y range."""
    P = PANELS[panel]
    xy = lambda m: (m[P["x"]], m[P["y"]])
    lim = panel_lim(panel, lines, singles)   # x hi, x lo, y lo, y hi
    if ylim:
        lim = lim[:2] + tuple(ylim)

    for method, verts in lines:
        color, marker = STYLE[method]
        hollow = method in DASHED_HOLLOW
        for lr, m in verts:
            if (method, lr) not in OFF_AXIS and (not ERRBARS_PICKS or (method, lr) in picks):
                _errorbar(ax, P, m, color)
        ax.plot([xy(m)[0] for _, m in verts], [xy(m)[1] for _, m in verts], color=color,
                lw=2.6 if method == GR else 2.0, ls="--" if hollow else "-", marker=marker,
                markersize=11, markerfacecolor="white" if hollow else color,
                markeredgecolor=color if hollow else "white", markeredgewidth=1.6,
                zorder=8 if method == GR else 6)
        for lr, m in verts:
            if lr != "5e-4" or panel != "gt" or not _inside(xy(m), lim):
                continue
            dx, dy, ha = LABEL_AT.get((panel, method, lr), LABEL_DEFAULT)
            ax.annotate("lr " + lr if lr == "5e-4" else "lr" + lr[4:], xy(m), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=13,
                        color="0.3", zorder=20,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
    off = [(method, lr, m) for method, verts in lines for lr, m in verts if (method, lr) in OFF_AXIS]
    for i, (method, lr, m) in enumerate(off):
        ax.text(0.012, 0.015 + 0.045 * i, f"↙ off axis: {method}, lr {lr} "
                f"({xy(m)[0]:.0%}, {xy(m)[1]:.0%})", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=13, color="0.3", zorder=20,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
    for name, m, color, marker, face, edge, alpha in singles:
        _errorbar(ax, P, m, color, alpha)
        ax.plot(*xy(m), marker=marker, markersize=17, markerfacecolor=face, markeredgecolor=edge,
                markeredgewidth=_ew(face), ls="none", alpha=alpha, zorder=9)
    means = {(method, lr): m for method, verts in lines for lr, m in verts}
    for key in picks:
        ax.plot(*xy(means[key]), **HALO, ls="none", zorder=3)

    if panel == "det":   # every hack detected
        ax.plot([0, 1], [0, 1], color="0.45", lw=1.4, ls=(0, (4, 3)), zorder=2)

    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel(P["xlabel"], fontsize=25)
    ax.set_ylabel(P["ylabel"], fontsize=25)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def legend_handles(lines, singles, panels):
    handles = []
    for method, _ in lines:
        color, marker = STYLE[method]
        hollow = method in DASHED_HOLLOW
        handles.append(Line2D([0], [0], color=color, lw=2.0, ls="--" if hollow else "-",
                              marker=marker, markersize=11,
                              markerfacecolor="white" if hollow else color,
                              markeredgecolor=color if hollow else "white",
                              label=f"{'GRAFT (ours, deployed)' if method == GR else method}, all lrs"))
    handles += [Line2D([0], [0], marker=marker, ls="none", markersize=15, markerfacecolor=face,
                       markeredgecolor=edge, markeredgewidth=_ew(face), alpha=alpha, label=name)
                for name, _, color, marker, face, edge, alpha in singles]
    w = "" if DEV_DET_WEIGHT == 1.0 else f"{DEV_DET_WEIGHT:g} × "
    handles.append(Line2D([0], [0], ls="none", **{**HALO, "markersize": 24},
                          label=f"Selected: max rewarded − {w}detected hack"))
    if "det" in panels:
        handles.append(Line2D([0], [0], color="0.45", lw=1.4, ls=(0, (4, 3)),
                              label="Every hack detected"))
    return handles


def rp_callout(ax_dev, ax_gt, lines, singles, picks):
    """CALLOUT centred in the empty band above ax_gt's data, with arrows to RP's
    selected config in both panels. ax_dev sits to the left; its arrow leaves
    the box's top-left corner and arcs up over ax_gt's tick labels and y label."""
    key = next(k for k in picks if k[0] in (RP2, RP5))
    m = {(method, lr): v for method, verts in lines for lr, v in verts}[key]
    assert m["hack"] - m["det"] > 0.40, f"CALLOUT says >40% undetected: {m}"
    # the band: from the highest ground-truth point to the top of the axes; its
    # horizontal extent runs from the left edge to the highest RP point
    top_pass1 = max([v["pass1"] for _, verts in lines for _, v in verts] +
                    [s[1]["pass1"] for s in singles])
    y_mid = (top_pass1 + ax_gt.get_ylim()[1]) / 2
    at = blended_transform_factory(ax_gt.transAxes, ax_gt.transData)
    arrow = dict(arrowstyle="-|>", color="0.25", lw=1.8, shrinkB=17, mutation_scale=22)
    txt = ax_gt.annotate(CALLOUT, xy=(m["det"], m["proxy"]), xycoords=ax_dev.transData,
                         xytext=(0.45, y_mid), textcoords=at, ha="center", va="center",
                         fontsize=25, color="0.15", zorder=30, annotation_clip=False,
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.6", lw=1.0),
                         arrowprops={**arrow, "relpos": (0.0, 1.0),
                                     "connectionstyle": "arc3,rad=0.3"})
    # second arrow starts inside the box and is clipped at its edge
    ax_gt.annotate("", xy=(m["hack"], m["pass1"]), xytext=(0.2, y_mid), textcoords=at,
                   zorder=30, arrowprops={**arrow, "patchA": txt.get_bbox_patch()})


def main():
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False
    lines, singles = collect()
    for method, verts in lines:
        print(f"  [lrs] {method:24s} " + "  ".join(
            f"{lr} (hack {m['hack']:.3f}, det {m['det']:.3f}, rew {m['proxy']:.3f}, "
            f"pass1 {m['pass1']:.3f})" for lr, m in verts))
    picks = []
    for pool in DEV_POOLS:
        score, key = dev_pick(pool)
        print(f"  [lrs] dev pick {pool:22s} {key[0]}, lr {key[1]}  score {score:.3f}")
        picks.append(key)
    wsuf = "" if DEV_DET_WEIGHT == 1.0 else f"_w{DEV_DET_WEIGHT:g}"
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    for vsuf, panels in VARIANTS.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(14.45 * len(panels), 11.2),
                                 squeeze=False)
        ylim = None
        if vsuf in SHARE_Y:
            lims = [panel_lim(panel, lines, singles) for panel in panels]
            ylim = (min(l[2] for l in lims), max(l[3] for l in lims))
        for ax, panel in zip(axes[0], panels):
            draw_panel(ax, panel, lines, singles, picks, ylim=ylim)
        for ax, title in zip(axes[0], TITLES.get(vsuf, ())):
            ax.set_title(title, fontsize=28, pad=14)
        if panels == ("dev", "gt"):
            rp_callout(*axes[0], lines, singles, picks)
        # legend text prints at Fig. 2's size: both figures are set at \linewidth, and
        # this one is 23.8 in wide to Fig. 2's 16.7 in (fontsize 16 there)
        fig.legend(handles=legend_handles(lines, singles, panels), loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=23,
                   framealpha=0.92, title="seed means, seeds 9/15/16", title_fontsize=21)
        for ext in ("pdf", "png"):
            out = os.path.join(HERE, "figs", f"countdown_lr_frontier{vsuf}{wsuf}.{ext}")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04,
                        **({"dpi": 140} if ext == "png" else {}))
            print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
