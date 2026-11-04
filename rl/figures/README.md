# Paper figure scripts (`rl/figures/`)

Scripts that produce the RL figures of the GRAFT paper. Fig. 4 (`fig:large-scale-sft`) is
`../../sft/charts/plot_scatter.py`.

## Quick start

```bash
cd rl
tar -xzf paper_data/paper_fig_inputs.tar.gz -C paper_data   # -> paper_data/{output,respmon-results}/
python figures/countdown_hf100_hf50_retain.py
```

- Dependencies: numpy, scipy, matplotlib, pyyaml.
- Outputs go to `rl/figures/figs/` (all scripts) and also `rl/final_figures/` (`countdown_figure1.py`,
  `hf100_arms_scatter_v2.py`, `countdown_generalization_scatter.py`). Neither directory is part of the repo.
- Labels below are the paper's LaTeX labels (appendix numbering may change).
- All commands and `paper_data/...` paths below are relative to `rl/`.

## Scripts

| Paper label | Script | Writes (`figs/`) | Inputs (under `paper_data/`) | Notes |
|---|---|---|---|---|
| `fig:countdown-fig1` (Fig. 2) | `countdown_figure1.py` | `countdown_figure1.{pdf,png}` | — | |
| `fig:narrative-hook` (Fig. 3) | `proto_respmon6_rv2canon_sidebyside.py` | `proto_respmon6_rv2canon_sidebyside.{pdf,png}` | — | |
| `fig:generalization-scatter` | `countdown_generalization_scatter.py` | `countdown_generalization_scatter.{pdf,png}` | `output/countdown_generalization` (72 `graded.jsonl`; spec `countdown_hf100_all`) | error bars are SEM; base point = hardcoded `checkpoint_specs.BASE_ROWS` |
| `fig:countdown-hf50` | `countdown_hf100_hf50_retain.py` | `countdown_hf100_hf50_retain.{pdf,png}` | hf100: `output/countdown_code_rp2-0702-0026_fseval`, `output/countdown_hf100_{rp_lccoh64,ip,pps,gr_lccoh64_lr3}_fseval`; hf50: `output/countdown_hf50_{dn,rp_lccoh64,ip,pps,gr_lccoh_lr3,gr_nocoh}_fseval`; probes: `output/expr_only_probe_arms` | |
| `fig:sort-pathological` | `sort_pathological_pareto.py` | `sort_pathological_pareto.{pdf,png}` | `output/{respmon_repeat_sorting_dn_rp_3seed,respmon_canon_filter_3seed,respmon_rv2_gr_3seed}` | |
| `fig:anchor-form` | `anchor_form_pareto6.py` | `anchor_form_pareto6.{pdf,png}` | `output/{respmon_canon_vercoh128_composite,respmon_canon_rp_mix_3seed,respmon_canon_unhackonly_3seed,respmon_rv2_vercoh32_gr_3seed,respmon_sortframed_anchor_gr_3seed,respmon_sortframed_dn_rp_unhack_3seed}` | |
| `fig:training-curves` | `respmon6_training_curves.py` | `training_curves_all_envs.{pdf,png}` | `respmon-results/{respmon_rv2_gr_3seed,respmon_objinv_gr_canon_3seed,respmon_iva_dn_3seed,respmon_iva_rp_3seed,respmon_ivainv_dn_rp_3seed,respmon_persona_bvo_dn_rp_3seed,respmon_repeat_sorting_dn_rp_3seed,respmon_rp_doses_canon_3seed,sort_framed}` | GRAFT, No intervention and RP arms |
| `fig:recall-sweep` | `respmon6_recall_scatter.py` | `recall_sweep_pareto6.{pdf,png}` | `output/respmon_recall_suite` + recall-1 runs in `respmon-results/` | recall < 1 points from `respmon_recall_suite`; recall = 1 points from the `respmon-results/` sweeps |
| `fig:recall-curves-{50,25,10}` | `respmon6_training_curves_recall.py` | `training_curves_rec{50,25,10}.{pdf,png}` | `output/respmon_recall_suite` + DN runs in `respmon-results/` | GRAFT and RP2 series from `respmon_recall_suite`; No-intervention series from `respmon-results/` |
| `fig:dev-selection-stars` | `hf100_arms_scatter_v2.py` | `hf100_arms_scatter_v2.{pdf,png}` | `output/countdown_code_rp2-0702-0026_fseval`, `output/countdown_hf100_{gr_lccoh64_lr3,rp_lccoh64,ip,pps,lconly}_fseval`, `output/cdhf100_noroute_fseval_rerun`, `output/expr_only_probe_arms` | imports `countdown_figure1.py` helpers (scatter only); GRAFT seeds 9/15/16 |

## Camera-ready (learning-rate protocol)

The camera-ready trains every Countdown-Code method on one SFT init (re-made from the SFT recipe)
across a learning-rate grid and shows each method at its developer-selected learning rate: the
highest seed-mean rewarded solution rate − detected hack rate, observables only
(`../tools/countdown_dn_lrsweep_report.py`). For the labels below these scripts replace the rows
above, which are the submission versions. Sweeps: `../sweeps/README.md`, "Camera-ready".

| Paper label | Script | Writes (`figs/`) | Notes |
|---|---|---|---|
| `fig:countdown-fig1` (Fig. 2) | `countdown_figure1_cr.py` | `countdown_figure1_cr.{pdf,png}` (paper: `countdown_figure1.pdf`) | right panel: per-checkpoint probes of the `gr_ck20` rerun |
| `fig:dev-selection-stars` | `countdown_lr_frontier.py` | `countdown_lr_frontier_dev.{pdf,png}` (paper: `countdown_lr_sweep_all.pdf`) | also writes the one-panel `countdown_lr_frontier` and `_det` variants |
| `fig:countdown-hf50` | `countdown_hf100_hf50_retain_cr.py` | `countdown_hf100_hf50_retain_cr.{pdf,png}` | hf50 picks use hf50's own observables |
| `tab:countdown-lr` | `python tools/countdown_dn_lrsweep_report.py` | — | the "dev score" line under each method |
| `tab:countdown-noanchor` | same report | — | `noanchor_table`: LaTeX rows and Welch p-values |

Inputs: the evals of those sweeps, as `rl/output/<sweep>_evals/` (the eval scripts' layout), plus
one base-model probe from the submission bundle:

```bash
cd rl
tar -xzf paper_data/paper_fig_inputs.tar.gz -C paper_data   # submission bundle
tar -xzf paper_data/cr_fig_inputs.tar.gz                    # -> output/<sweep>_evals/, output/countdown_hf100_gr_ck20/
python tools/countdown_dn_lrsweep_report.py; python tools/countdown_dn_lrsweep_report.py hf50
python figures/countdown_figure1_cr.py; python figures/countdown_lr_frontier.py
python figures/countdown_hf100_hf50_retain_cr.py
```

## Data roots

| Variable | Used by | Default |
|---|---|---|
| `GRAFT_OUTPUT_ROOT` | `countdown_figure1.py` (and `hf100_arms_scatter_v2.py` through it), `countdown_hf100_hf50_retain.py`, `countdown_generalization_scatter.py`, `sort_pathological_pareto.py`, `anchor_form_pareto6.py`, `respmon6_recall_scatter.py` and `respmon6_training_curves_recall.py` (recall suite) | `rl/paper_data/output` |
| `GRAFT_RESPMON_ROOT` | `proto_respmon6_rv2canon_sidebyside.py`, `respmon6_training_curves.py`, `respmon6_training_curves_recall.py`, `respmon6_recall_scatter.py` | `rl/paper_data/respmon-results` |

- To plot your own runs, set both to `output` (run from `rl/`). Two input dirs are assemblies, not sweep
  outputs, and must be built by hand:
  - `sort_framed/` = merged outputs of `sweeps/sort_framed_{pilot,suite,graft_only,rp_more,filter}.py`
    (`--name`s `sort_framed_{pilot,suite,graft,rp_more,filter}`; pilot s1 renamed
    `sortframed_donothing_hf050_st1000_s1`);
  - `respmon_canon_vercoh128_composite/` = per env: DN and RP2 (3 seeds each) from `respmon_iva_{dn,rp}_3seed` /
    `respmon_ivainv_dn_rp_3seed` / `respmon_persona_bvo_dn_rp_3seed` / `respmon_repeat_sorting_dn_rp_3seed`;
    GRAFT 4:1 s1–3 from `respmon_rv2_vercoh128_gr_3seed`; object_qa s1–3 from
    `respmon_rv2_vercoh128_objinv_gr_3seed`; s4–8 from `respmon_canon_gr_seeds4to8`.
- Countdown scripts read fseval JSONs from `<root>/<sweep>_fseval/` (see `../README.md`, "Post-hoc evaluation").
- `PRELIM=1` (recall scripts only): accept fewer than 3 finished seeds per group (preview mode).

## Metric definitions used by the toy scripts

- Points: tail mean = mean over the last 10% of `routing_eval.jsonl` rows; error bars = SEM over seeds
  (Fig. 3 left panel: SEM over environments); training curves: mean ± 1 std (ddof 0) over 3 seeds.
- y = retain reward on all prompts; x = hack rate on hackable prompts (`hack_freq_hackable`).
- GRAFT (and, in Fig. 3, GRAFT w/o routing and GRAFT w/o anchoring) is read in `retain_only` mode
  (deployment); GRAFT pre-ablation and all other arms in `both` mode.
- "Base model" = first eval row (step 10; step 0 only for the `sort_framed_*` runs, which set
  `eval_at_start`; the anchor-form framed-sort runs `respmon_sortframed_*` start at step 10). All
  other toy runs in the bundle recorded `eval_at_start: false`; `train.py` defaults to true, so new
  runs of those sweeps add a step-0 row (which then becomes the "Base model" point) unless
  `--no-eval_at_start` is passed.
- Only finished runs are used (last eval step ≥ `max_steps − eval_every`).
- **Fig. 3 left panel:** x = `hack_freq_detectable` tail mean; y = max(0, `hack_freq_hackable` −
  `hack_freq_detectable`). The logged `hack_freq_detectable` channel of these configs is a rate over all eval
  prompts, while `hack_freq_hackable` is a rate over hackable prompts.
- **Reward-penalty dose** (`_best_rp`, Fig. 3 and training curves): among doses with 3 finished seeds, the one
  with the highest apparent score = tail mean of `both/combined` − tail mean of `both/detected_freq`
  (proxy reward and monitor flag rate only).

## Helpers

- `proto_pareto_style_v2.py`: styles, `draw_point`, legend handles.
- `proto_pareto_data.py`: imported by the style module for `ENV_TITLE`; changes the working directory to
  `rl/` at import.

## Tables (no script in this directory)

| Paper label | How to compute |
|---|---|
| `tab:countdown-generalization` | `python tools/countdown-generalization-eval/make_table.py --spec countdown_hf100_all --cache_root paper_data/output/countdown_generalization` prints hard-coding and tampering separately (mean ± SD). The table's hack column is their per-seed sum; its ± is the SD of that sum over seeds. Base row = hardcoded `checkpoint_specs.BASE_ROWS`. |
| `tab:countdown-noanchor` (submission version) | GRAFT (8 seeds) and base rows: see below. "GRAFT w/o anchoring" row: sweep `../sweeps/countdown_hf100_gr_nocoh_lr3.py`, run configs in `paper_data/run_configs/output/cdhf100_gr_nocoh_lr3/`. Camera-ready: "Camera-ready" above. |

### `tab:countdown-noanchor` (submission version): GRAFT and base rows

`tools/collate_countdown_fseval.py` reads every `*.json` in each `--fsdirs` directory and keys each
record by the `run_name` stored inside it. The base-model file `cdhf100_gr_lccoh64_lr3_s9__r0.0.json`
stores the run name of GRAFT seed 9, so it must not share a directory with the GRAFT JSONs. Copy the
8 GRAFT JSONs and the base JSON into two separate new directories, then collate each:

```bash
GRAFT_FS=$(mktemp -d); BASE_FS=$(mktemp -d)
cp paper_data/output/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_gr_lccoh64_lr3_s{9,15,16}.json \
   paper_data/output/countdown_hf100_gr_lccoh64_lr3_seeds5_fseval/cdhf100_gr_lccoh64_lr3_s{1,2,3,4,5}.json \
   "$GRAFT_FS"/
cp paper_data/output/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_gr_lccoh64_lr3_s9__r0.0.json "$BASE_FS"/
python tools/collate_countdown_fseval.py --fsdirs "$GRAFT_FS" --out figures/figs/countdown_noanchor_graft
python tools/collate_countdown_fseval.py --fsdirs "$BASE_FS" --out figures/figs/countdown_noanchor_base
```

- Each JSON holds forget scale `0.0` only (step 200, n_eval 256).
- GRAFT (`$GRAFT_FS`, condition `cdhf100_gr_lccoh64_lr3`, seeds 1–5, 9, 15, 16): retain 0.792,
  hack_gt 0.016 (means over the 8 seeds). The tool prints means and per-seed values; the table's ± is
  the SEM over seeds (1.0 and 0.4 points), computed from the per-seed values.
- Base model (`$BASE_FS`): printed as condition `cdhf100_gr_lccoh64_lr3`, seed 9 (the run name stored
  in the file); retain 0.602, hack_gt 0.047.
