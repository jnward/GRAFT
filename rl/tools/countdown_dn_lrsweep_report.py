"""Learning-rate curves on hf100 Countdown for every Fig. 2 method, and their
dev-selected configs. All runs train from this volume's SFT init
(/output/countdown_sft_model/qwen3-8b on gr-modal-pilot, re-made from the SFT
recipe on 2026-10-20; not the paper's weights).

Runs (seeds 9/15/16), lr grid 5e-4, /3, /10, /30:
  No intervention:       sweeps/countdown_code_dn_lrsweep.py, 5e-4 .. /30 in
                         countdown_code_dn_lrsweep, /100 and /300 in
                         countdown_code_dn_lrsweep_low.
  Inoculation prompting: sweeps/countdown_hf100_ip_pps_lrsweep.py, /3 .. /30 in
                         countdown_hf100_ip_lrsweep, 5e-4 in
                         countdown_hf100_ip_lrsweep_lr1.
  Preventative steering: same file, on the re-extracted vector
                         (countdown_hf100_pps_lrsweep).
  Reward penalty 2 / 5:  sweeps/countdown_hf100_rp_gr_lrsweep.py
                         (countdown_hf100_rp_lrsweep; penalty 2 at 5e-4 in
                         countdown_hf100_rp_lrsweep_lr1).
  MLP GRAFT:             same file (countdown_hf100_gr_lrsweep), /3 is
                         sweeps/countdown_hf100_gr_newinit.py's rerun. Deployed:
                         FORGET ablated, scale 0.0.
  Ablations (not swept): sweeps/countdown_hf100_ablations_newinit.py.
From the n=256 fseval at the method's deployed scale (1.0; GRAFT 0.0):
hack_freq (ground truth), detected_freq (monitor flags), hack_freq_detectable,
hack_freq_undetectable, retain (target-env solve rate) and combined (proxy
reward); from the expr-only probe at that scale: clean-environment pass@1. Mean
± SEM over seeds. The SFT init (retain 0 / forget 0) is the "base" row. Each
probe must point at its own sweep's checkpoint: the probe dir on the volume is
shared, and cdhf100_dn_lr3_* also names the 2026-10-20 countdown_code_dn_lr3 runs.

Observables-only lr picks, per method:
  min flagged:  lowest mean detected_freq; ties (within 1/256) go to the
                highest mean proxy reward.
  + learned:    the same, among lrs whose mean proxy reward is at least 0.10
                above the SFT init's (a run that never learned the task would
                not be deployed).
  dev metric:   argmax of the seed-mean proxy - detected hack (the rate of
                rewarded, unflagged solutions: flagged hacks pass their own
                test, so they are rewarded); per DEV_POOLS (RP's pool spans both
                penalties). The paper's appendix used proxy - 2 x detected
                (figures/hf100_arms_scatter_v2.py); on these runs the two pick
                the same configs except PPS (5e-4 vs /10, both ~.23 hack).
Each pick's ground-truth hack and pass@1 are printed next to it. Until a
new-init rerun has evals, the paper's run of that config stands in
(PAPER_FALLBACK), marked "(paper init)".

    python tools/countdown_dn_lrsweep_report.py [hf50]

hf50 (sweeps/countdown_hf50_lrsweep.py, one sweep dir): the same methods, grid and
picks on the half-hackable variant (HF50_ROWS); hack_h = hack_freq_hackable,
the hack rate on hackable prompts, which the hf50 figure plots. No ablations.
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
PAPER = os.path.join(ROOT, "paper_data", "output")
SEEDS = (9, 15, 16)
LEARNED_MARGIN = 0.10

DN, IP, PPS = "No intervention", "Inoculation prompting", "Preventative steering"
RP2, RP5, GR = "Reward penalty (2)", "Reward penalty (5)", "MLP GRAFT"
RP = "Reward penalty"
LCONLY, NOROUTE = "Anchor environment only", "GRAFT w/o routing"


def _lrs(tags, sweep, stem, at=None):
    """[(lr label, sweep dir, run stem)] for lr tags like "lr3"; `at` maps a tag
    to a (sweep, stem) held elsewhere."""
    at = at or {}
    return [(("5e-4" if t == "lr1" else f"5e-4/{t[2:]}"),) + at.get(t, (sweep, f"{stem}_{t}"))
            for t in tags]


_GRID = ("lr1", "lr3", "lr10", "lr30")
# method -> [(lr label, sweep dir, run stem)]
ROWS = {
    DN: _lrs(_GRID, "countdown_code_dn_lrsweep", "cdhf100_dn")
        + _lrs(("lr100", "lr300"), "countdown_code_dn_lrsweep_low", "cdhf100_dn"),
    IP: _lrs(_GRID, "countdown_hf100_ip_lrsweep", "cdhf100_ip_mand-tw",
             at={"lr1": ("countdown_hf100_ip_lrsweep_lr1", "cdhf100_ip_mand-tw_lr1")}),
    PPS: _lrs(_GRID, "countdown_hf100_pps_lrsweep", "cdhf100_pps_L20_a2"),
    RP2: _lrs(_GRID, "countdown_hf100_rp_lrsweep", "cdhf100_rp2_lc64",
              at={"lr1": ("countdown_hf100_rp_lrsweep_lr1", "cdhf100_rp2_lc64_lr1")}),
    RP5: _lrs(_GRID, "countdown_hf100_rp_lrsweep", "cdhf100_rp5_lc64"),
    GR: _lrs(_GRID, "countdown_hf100_gr_lrsweep", "cdhf100_gr_lccoh64",
             # lr/3: sweeps/countdown_hf100_gr_newinit.py, launched under this sweep name
             at={"lr3": ("countdown_hf100_gr_loreft_all", "cdhf100_gr_lccoh64_lr3")}),
}
# hf50 (half the prompts hackable; sweeps/countdown_hf50_lrsweep.py): the same
# methods and grid, all in one sweep dir
HF50_ROWS = {
    DN: _lrs(_GRID + ("lr100", "lr300"), "countdown_hf50_lrsweep", "cdhf50_dn"),
    IP: _lrs(_GRID, "countdown_hf50_lrsweep", "cdhf50_ip_mand-tw"),
    PPS: _lrs(_GRID, "countdown_hf50_lrsweep", "cdhf50_pps_L20_a2"),
    RP2: _lrs(_GRID, "countdown_hf50_lrsweep", "cdhf50_rp2_lc64"),
    RP5: _lrs(_GRID, "countdown_hf50_lrsweep", "cdhf50_rp5_lc64"),
    GR: _lrs(_GRID, "countdown_hf50_lrsweep", "cdhf50_gr_lccoh64"),
}
ROWS_BY_ENV = {"hf100": ROWS, "hf50": HF50_ROWS}
SCALE = {GR: "0.0"}   # the deployed forget scale; single-policy methods are at 1.0
# dev-selection pools: one pick per pool, over every (method, lr) in it
DEV_POOLS = {DN: [DN], IP: [IP], PPS: [PPS], RP: [RP2, RP5], GR: [GR]}
# unswept arms: name -> (sweep dir, run stem, deployed scale)
ABLATIONS = {
    LCONLY: ("countdown_hf100_lconly_newinit", "cdhf100_lconly_full_lr3", "1.0"),
    NOROUTE: ("countdown_hf100_noroute_newinit", "cdhf100_noroute_lccoh64_lr3", "0.0"),
}
# tab:countdown-noanchor's arm (not a Fig. 2 point), at GRAFT's selected lr
NOANCHOR = "GRAFT w/o anchoring"
NOANCHOR_ARM = ("countdown_hf100_noanchor_newinit", "cdhf100_noanchor_lr3", "0.0")
# The paper's runs (the paper's SFT init, not this volume's), used for a row or
# ablation whose new-init rerun has no evals; their points carry "paper": True.
# paper sweep id -> (fseval dir under paper_data/output, sweep dir in the probe's ckpt)
PAPER_SWEEPS = {
    "paper:rp": ("countdown_hf100_rp_lccoh64_fseval", "countdown_hf100_rp_lccoh64"),
    "paper:ip": ("countdown_hf100_ip_fseval", "countdown_hf100_ip"),
    "paper:lconly": ("countdown_hf100_lconly_fseval", "lconly_full"),
    "paper:noroute": ("cdhf100_noroute_fseval_rerun", "cdhf100_noroute"),
}
# (method, lr) row or ablation name -> (paper sweep id, run stem)
PAPER_FALLBACK = {
    (RP2, "5e-4"): ("paper:rp", "cdhf100_rp2_lc64_lr1"),
    (IP, "5e-4"): ("paper:ip", "cdhf100_ip_mand-tw"),
    LCONLY: ("paper:lconly", "lconly_full"),
    NOROUTE: ("paper:noroute", "cdhf100_noroute_anchor"),
}


def _find(evals, stem, want):
    """The fseval and probe JSONs share <stem>.json; tell them apart by content.
    `evals` is one dir or a list of dirs."""
    dirs = [evals] if isinstance(evals, str) else evals
    for f in (f for d in dirs for f in glob.glob(os.path.join(d, "**", f"{stem}.json"), recursive=True)):
        d = json.load(open(f))
        sc = d.get("scales", {})
        any_sc = next(iter(sc.values()), {})
        is_probe = "true_pass1" in any_sc
        if (want == "probe") == is_probe:
            return d
    return None


def _g(m, pre):
    return next((v for k, v in m.items() if k.startswith(pre)), None)


def point(sweep, stem, run, scale="1.0"):
    """Metrics for one run at one scale; {} if its evals are not in yet. A
    PAPER_SWEEPS id reads the paper's evals."""
    if sweep in PAPER_SWEEPS:
        fs_dir, ckpt_sweep = PAPER_SWEEPS[sweep]
        fs_f = os.path.join(PAPER, fs_dir, f"{run}.json")
        pr_f = os.path.join(PAPER, "expr_only_probe_arms", f"{run}.json")
        out = {"paper": True}
    else:
        ev = os.path.join(OUT, f"{sweep}_evals")
        fs_f = os.path.join(ev, f"{sweep}_fseval", f"{run}.json")
        pr_f = os.path.join(ev, "expr_only_probe_arms", f"{run}.json")
        ckpt_sweep, out = sweep, {}
    if os.path.exists(fs_f):
        sc = json.load(open(fs_f))["scales"]
        if scale in sc:
            m = sc[scale]
            out.update(hack=_g(m, "hack_freq/"), flagged=_g(m, "detected_freq/"),
                       hack_h=_g(m, "hack_freq_hackable/"),
                       det=_g(m, "hack_freq_detectable/"),
                       undet=_g(m, "hack_freq_undetectable/"), retain=_g(m, "retain/"),
                       proxy=_g(m, "combined/"))
    if os.path.exists(pr_f):
        p = json.load(open(pr_f))
        assert f"/{ckpt_sweep}/" in p["ckpt"], f"{run}: probe is of {p['ckpt']}"
        if scale in p["scales"]:
            out["pass1"] = p["scales"][scale]["true_pass1"]
    return out


def seed_points(sweep, stem, scale="1.0"):
    """Per-seed metrics, for the seeds with both an fseval and a probe."""
    pts = [point(sweep, stem, f"{stem}_s{s}", scale) for s in SEEDS]
    return [p for p in pts if "hack" in p and "pass1" in p]


def method_points(method, lr, env="hf100"):
    """seed_points of one (method, lr) row of ROWS_BY_ENV[env], else (hf100) of its paper run."""
    sweep, stem = next((sw, s) for l, sw, s in ROWS_BY_ENV[env][method] if l == lr)
    scale = SCALE.get(method, "1.0")
    pts = seed_points(sweep, stem, scale)
    if not pts and env == "hf100" and (method, lr) in PAPER_FALLBACK:
        pts = seed_points(*PAPER_FALLBACK[method, lr], scale)
    return pts


def ablation_points(name):
    """seed_points of one ABLATIONS arm, else of its paper run."""
    sweep, stem, scale = ABLATIONS[name]
    return seed_points(sweep, stem, scale) or (
        seed_points(*PAPER_FALLBACK[name], scale) if name in PAPER_FALLBACK else [])


# the paper's appendix used 2.0; env DEV_DET_WEIGHT overrides (figure variants)
DEV_DET_WEIGHT = float(os.environ.get("DEV_DET_WEIGHT", 1.0))


def dev_score(pts):
    """The developer metric, from observables only: proxy success - detected
    hack (DEV_DET_WEIGHT x), averaged over a config's seeds."""
    return st.mean(p["proxy"] - DEV_DET_WEIGHT * p["det"] for p in pts)


def dev_best(cands):
    """(score, key, sweep, stem) of the argmax-dev-score config among
    cands = [(key, sweep, stem, scale)]; configs without evals are skipped."""
    scored = [(dev_score(pts), k, sw, stem) for k, sw, stem, sc in cands
              if (pts := seed_points(sw, stem, sc))]
    return max(scored, key=lambda t: t[0]) if scored else None


def dev_pick(pool, env="hf100"):
    """(score, (method, lr)) of the argmax-dev-score row over every lr of every
    method in DEV_POOLS[pool] (paper runs standing in where PAPER_FALLBACK has them)."""
    rows = ROWS_BY_ENV[env]
    scored = [(dev_score(pts), (m, lr)) for m in DEV_POOLS[pool] for lr, _, _ in rows[m]
              if (pts := method_points(m, lr, env))]
    return max(scored, key=lambda t: t[0]) if scored else None


def base_point(env="hf100"):
    """The SFT init (both adapters off). hf50: its fseval on the hf50 eval set
    (eval_countdown_gr.sh's base run), with the hf100 probe: the clean-
    environment probe prompts do not depend on hack_frac."""
    b = point("countdown_code_dn_lrsweep", "cdhf100_dn_lr1", "cdhf100_dn_lr1_s9__r0.0", "0.0")
    if env == "hf50":
        b = {**point("countdown_hf50_lrsweep", "cdhf50_gr_lccoh64_lr1",
                     "cdhf50_gr_lccoh64_lr1_s9__r0.0", "0.0"), "pass1": b.get("pass1")}
    return b


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    sem = st.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    return st.mean(vals), sem, len(vals)


def fmt(t):
    return "     —      " if t is None else f"{t[0]:.3f}±{t[1]:.3f}"


def _pick(table, floor=None):
    have = {lr: v for lr, v in table.items() if v["flagged"] and v["proxy"]
            and (floor is None or v["proxy"][0] >= floor)}
    if not have:
        return None
    lo = min(v["flagged"][0] for v in have.values())
    tied = {lr: v for lr, v in have.items() if v["flagged"][0] <= lo + 1 / 256}
    return max(tied, key=lambda lr: tied[lr]["proxy"][0])


def _metrics(v):
    return (f"flagged {fmt(v['flagged'])}  GT hack {fmt(v['hack'])}  "
            f"undet. {fmt(v['undet'])}  pass@1 {fmt(v['pass1'])}")


def main(env="hf100"):
    keys = ["flagged", "hack", "undet", "retain", "pass1", "proxy"] + (["hack_h"] if env == "hf50" else [])
    base = base_point(env)
    floor = base["proxy"] + LEARNED_MARGIN if base.get("proxy") is not None else float("nan")
    print(f"{'':10s} n  " + " ".join(f"{k:>12s}" for k in keys))
    print(f"{'SFT init':10s} 1  " + " ".join(f"{fmt(agg([base.get(k)])):>12s}" for k in keys))
    tables = {}
    for method, rows in ROWS_BY_ENV[env].items():
        print(f"\n{method}" + (f"  (scale {SCALE[method]})" if method in SCALE else ""))
        table = tables[method] = {}
        for lr, _, _ in rows:
            pts = method_points(method, lr, env)
            table[lr] = {k: agg([p.get(k) for p in pts]) for k in keys}
            print(f"{lr:10s} {len(pts)}  " + " ".join(f"{fmt(table[lr][k]):>12s}" for k in keys)
                  + ("  (paper init)" if pts and pts[0].get("paper") else ""))
        scores = [f"{lr} {dev_score(pts):.3f}" for lr, _, _ in rows if (pts := method_points(method, lr, env))]
        if scores:
            print(f"  dev score (proxy - {DEV_DET_WEIGHT:g} x detected): " + "  ".join(scores))
        for name, lr in (("min flagged", _pick(table)),
                         (f"min flagged, proxy >= {floor:.3f}", _pick(table, floor))):
            if lr is not None:
                print(f"  pick ({name}): lr {lr}  {_metrics(table[lr])}")

    print(f"\nDev-selected (argmax seed-mean proxy - {DEV_DET_WEIGHT:g} x detected), per pool:")
    for pool in DEV_POOLS:
        dev = dev_pick(pool, env)
        if dev:
            (method, lr) = dev[1]
            print(f"{pool:24s} {method}, lr {lr}  score {dev[0]:.3f}  {_metrics(tables[method][lr])}")

    if env != "hf100":
        return
    print("\nAblations:")
    for name in ABLATIONS:
        pts = ablation_points(name)
        print(f"{name:24s} {len(pts)}  " + " ".join(f"{fmt(agg([p.get(k) for p in pts])):>12s}" for k in keys)
              + ("  (paper init)" if pts and pts[0].get("paper") else ""))
    noanchor_table(keys)


def noanchor_table(keys):
    """tab:countdown-noanchor: GRAFT (dev pick) vs GRAFT w/o anchoring vs the SFT
    init, clean-environment solve rate and GT hack rate in %, mean +- SEM."""
    from scipy import stats
    pts = seed_points(*NOANCHOR_ARM)
    print(f"{NOANCHOR:24s} {len(pts)}  " + " ".join(f"{fmt(agg([p.get(k) for p in pts])):>12s}" for k in keys))
    if not pts:
        return
    _, (method, lr) = dev_pick(GR)
    rows = [("GRAFT w/o anchoring", pts), ("GRAFT (ours)", method_points(method, lr)),
            ("Base model", [base_point()])]
    print(f"\ntab:countdown-noanchor (clean solve %, GT hack %; GRAFT at lr {lr}):")
    for label, ps in rows:
        cells = [agg([100 * p[k] for p in ps]) for k in ("pass1", "hack")]
        print(f"  {label:20s} & " + " & ".join(
            f"{m:.1f}" if n == 1 else f"{m:.1f} $\\pm$ {e:.1f}" for m, e, n in cells) + " \\\\")
    for k in ("pass1", "hack"):
        a, b = [p[k] for p in rows[1][1]], [p[k] for p in pts]
        print(f"  GRAFT vs w/o anchoring {k}: Welch p={stats.ttest_ind(a, b, equal_var=False).pvalue:.3g}")


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:2]))
