# `rl/paper_data/`: inputs for the RL paper figures

| File | Contents |
|---|---|
| `paper_fig_inputs.tar.gz` | figure-input bundle: 956 regular files, about 140 MB extracted, about 7.6 MB compressed |
| `MANIFEST.txt` | sorted list of every path in the bundle; last line is the bundle's sha256 |
| `cr_fig_inputs.tar.gz` | camera-ready Countdown-Code bundle: 396 files (fseval and probe JSONs, three `routing_eval.jsonl`), about 2 MB extracted |
| `CR_MANIFEST.txt` | the same for `cr_fig_inputs.tar.gz` |
| `run_configs/` | `run_config.yaml` only, for every run of 37 paper sweep dirs (371 files) |

The bundle holds metrics only: `routing_eval.jsonl`, `run_config.yaml`, fseval/probe JSONs and
`graded.jsonl` (per-sample grading flags).

## Extract

From `rl/`:

```bash
shasum -a 256 paper_data/paper_fig_inputs.tar.gz   # compare with the last line of paper_data/MANIFEST.txt
tar -xzf paper_data/paper_fig_inputs.tar.gz -C paper_data
```

- The bundle extracts to `rl/paper_data/output/` and `rl/paper_data/respmon-results/`. Both paths are
  gitignored.
- Every path in the archive is relative, and every member is owned by `root:root`.

The camera-ready bundle extracts to `rl/output/` instead, the layout the camera-ready eval scripts
write (`output/<sweep>_evals/`); from `rl/`: `tar -xzf paper_data/cr_fig_inputs.tar.gz`. Its scripts
and sweeps are listed in `../figures/README.md`, "Camera-ready".

## Layout

| Path (after extraction) | Contents |
|---|---|
| `output/` | Countdown fsevals and probes, generalization caches, toy sweeps (728 files); same layout as `rl/output/` written by `sweep.py` |
| `respmon-results/` | toy-env sweeps (228 files) |

**`respmon-results/`**: 114 run dirs, each with `routing_eval.jsonl` and `run_config.yaml`.

| Sweep dir | Runs |
|---|---|
| `respmon_iva_dn_3seed`, `respmon_iva_rp_3seed` | 6 each (addition, cities) |
| `respmon_ivainv_dn_rp_3seed` | 6 (object_qa DN + RP2) |
| `respmon_objinv_gr_canon_3seed` | 3 |
| `respmon_persona_bvo_dn_rp_3seed` | 6 |
| `respmon_repeat_sorting_dn_rp_3seed` | 12 |
| `respmon_rp_doses_canon_3seed` | 30 |
| `respmon_rv2_gr_3seed` | 12 |
| `respmon_rv2_vercoh128_gr_3seed` | 15 |
| `respmon_rv2_vercoh128_objinv_gr_3seed` | 3 |
| `sort_framed/` | 15 (DN, GRAFT, RP2/5/10); merged from the `sort_framed_{pilot,suite,graft,rp_more}` outputs |

**`output/`, toy sweeps**: 265 run dirs, each with `routing_eval.jsonl` and `run_config.yaml`.

| Sweep dir | Runs |
|---|---|
| `respmon_recall_suite` | 108 |
| `respmon_canon_vercoh128_composite` | 70 (see below) |
| `respmon_canon_rp_mix_3seed` | 30 |
| `respmon_canon_unhackonly_3seed` | 15 (addition, cities, object_qa, persona, repeat) |
| `respmon_rv2_vercoh32_gr_3seed` | 15 |
| `respmon_sortframed_dn_rp_unhack_3seed` | 9 |
| `respmon_sortframed_anchor_gr_3seed` | 6 |
| `respmon_repeat_sorting_dn_rp_3seed` | 6 (stock sort DN + RP2) |
| `respmon_canon_filter_3seed` | 3 (stock sort) |
| `respmon_rv2_gr_3seed` | 3 (stock sort) |

`respmon_canon_vercoh128_composite/` is assembled from the sweeps below. It covers 5 envs (addition,
cities, object_qa, persona, repeat), 14 runs each. To build it from your own runs, copy the run dirs
listed here into one directory.

| Arm | Source sweep |
|---|---|
| DN, RP2 (s1–3) | `respmon_iva_{dn,rp}_3seed` / `respmon_ivainv_dn_rp_3seed` / `respmon_persona_bvo_dn_rp_3seed` / `respmon_repeat_sorting_dn_rp_3seed` |
| GRAFT 4:1 (s1–3) | `respmon_rv2_vercoh128_gr_3seed` |
| object_qa GRAFT s1–3 (`invgr_cls`) | `respmon_rv2_vercoh128_objinv_gr_3seed` |
| GRAFT s4–8 | `respmon_canon_gr_seeds4to8` |

**`output/`, Countdown-Code (Qwen3-8B)**: 126 JSONs and 72 `graded.jsonl` files.

| Path | Files |
|---|---|
| `countdown_hf100_gr_lccoh64_lr3_fseval/` | 4: GRAFT s9/15/16 and base `…_s9__r0.0` |
| `countdown_hf100_gr_lccoh64_lr3_seeds5_fseval/` | 5: GRAFT s1–5 |
| `countdown_code_rp2-0702-0026_fseval/` | 3: No intervention |
| `countdown_hf100_{rp_lccoh64,ip,pps,lconly}_fseval/`, `cdhf100_noroute_fseval_rerun/` | 3 each (one config, 3 seeds) |
| `countdown_hf50_{dn,gr_lccoh_lr3,gr_nocoh,ip,pps,rp_lccoh64}_fseval/` | 57 in total |
| `expr_only_probe_arms/` | 42 clean-env probes (`tools/expr_only_probe.py`, k=1, n=256): 27 hf100 (26 runs + base `cdhf100_gr_lccoh64_lr3_s16__r0.0`) + 15 hf50 |
| `countdown_generalization/` | 72 `graded.jsonl` (generalization eval caches) |

## How the bundle was assembled

- Each figure script in the table below was run under a Python audit hook that logged every file it
  opened. The bundle holds every opened file under `output/` or `respmon-results/`, plus:
  - `run_config.yaml` for each run whose `routing_eval.jsonl` was read;
  - the hf50 fseval JSONs and the `expr_only_probe_arms/` probe JSONs.

## Per-figure inputs

Run commands from `rl/` with the bundle extracted. The figure scripts need only numpy, scipy, matplotlib
and pyyaml.

| Paper label | Script | Bundle inputs |
|---|---|---|
| `fig:generalization-scatter` | `figures/countdown_generalization_scatter.py` | `output/countdown_generalization` |
| `tab:countdown-generalization` | `tools/countdown-generalization-eval/make_table.py --spec countdown_hf100_all --cache_root paper_data/output/countdown_generalization` | same |
| `tab:countdown-noanchor` | `tools/collate_countdown_fseval.py` (GRAFT and base rows; commands in `../figures/README.md`) | `output/countdown_hf100_gr_lccoh64_lr3{,_seeds5}_fseval` |
| `fig:countdown-hf50` | `figures/countdown_hf100_hf50_retain.py` | `output/countdown_code_rp2-0702-0026_fseval`, `output/countdown_hf100_{rp_lccoh64,ip,pps,gr_lccoh64_lr3}_fseval`, `output/countdown_hf50_{dn,rp_lccoh64,ip,pps,gr_lccoh_lr3,gr_nocoh}_fseval`, `output/expr_only_probe_arms` |
| `fig:sort-pathological` | `figures/sort_pathological_pareto.py` | `output/{respmon_repeat_sorting_dn_rp_3seed,respmon_canon_filter_3seed,respmon_rv2_gr_3seed}` |
| `fig:anchor-form` | `figures/anchor_form_pareto6.py` | `output/{respmon_canon_vercoh128_composite,respmon_canon_rp_mix_3seed,respmon_canon_unhackonly_3seed,respmon_rv2_vercoh32_gr_3seed,respmon_sortframed_*}` |
| `fig:training-curves` | `figures/respmon6_training_curves.py` | `respmon-results/{respmon_rv2_gr_3seed,respmon_objinv_gr_canon_3seed,respmon_iva_dn_3seed,respmon_iva_rp_3seed,respmon_ivainv_dn_rp_3seed,respmon_persona_bvo_dn_rp_3seed,respmon_repeat_sorting_dn_rp_3seed,respmon_rp_doses_canon_3seed,sort_framed}` |
| `fig:recall-sweep` | `figures/respmon6_recall_scatter.py` | `output/respmon_recall_suite`, `respmon-results/{respmon_rv2_gr_3seed,respmon_objinv_gr_canon_3seed,respmon_iva_dn_3seed,respmon_iva_rp_3seed,respmon_ivainv_dn_rp_3seed,respmon_persona_bvo_dn_rp_3seed,respmon_repeat_sorting_dn_rp_3seed,sort_framed}` |
| `fig:recall-curves-{50,25,10}` | `figures/respmon6_training_curves_recall.py` | `output/respmon_recall_suite`, `respmon-results/{respmon_iva_dn_3seed,respmon_ivainv_dn_rp_3seed,respmon_persona_bvo_dn_rp_3seed,respmon_repeat_sorting_dn_rp_3seed,sort_framed}` |
| `fig:dev-selection-stars` | `figures/hf100_arms_scatter_v2.py` | `output/countdown_code_rp2-0702-0026_fseval`, `output/countdown_hf100_{gr_lccoh64_lr3,rp_lccoh64,ip,pps,lconly}_fseval`, `output/cdhf100_noroute_fseval_rerun`, `output/expr_only_probe_arms` |

- `tab:countdown-generalization`: `make_table.py` prints the hard-coding and tampering rates separately
  (mean ± SD over seeds); the table's hack column is their per-seed sum. The base row comes from
  `checkpoint_specs.BASE_ROWS`.

Data roots:
- The default roots are `rl/paper_data/output` (`GRAFT_OUTPUT_ROOT`) and `rl/paper_data/respmon-results`
  (`GRAFT_RESPMON_ROOT`), resolved from each script's own path.
- To plot your own runs, set both variables to `output` and run from `rl/`, or pass absolute paths.
  The merged `sort_framed/` dir and `respmon_canon_vercoh128_composite/` then have to be assembled by
  hand (see the composition table above).

## `run_configs/`

- Layout: `run_configs/{respmon-results,output}/<sweep>/<run>/run_config.yaml` (the GRAFT w/o routing
  configs are one level deeper, in `output/cdhf100_noroute_rerun_ckpts/cdhf100_noroute/<run>/`). Paths
  mirror the bundle roots; these files are not inside the tarball.
- 17 toy sweeps (`respmon-results/`) and 20 Countdown sweep dirs (`output/`): 371 files.
- They record the configuration of every run in these sweeps, including:
  - toy: Filtering, w/o routing, w/o anchoring, swapped monitors (the addition, cities, persona, repeat
    and stock-sort runs of `respmon_rv2_inverted_gr_3seed`; the addition and cities runs of
    `respmon_ivainv_dn_rp_3seed`), anchor-only (`respmon_canon_unhackonly_3seed`);
  - Countdown: w/o anchoring (`cdhf100_gr_nocoh_lr3`), w/o routing (`cdhf100_noroute_rerun_ckpts`),
    No intervention s15 (`countdown_dn_s15_repro`);
  - GRAFT learning-rate selection (App. G): `countdown_hf50_gr_{coh64,coh64_v2,coh64_lr3,nocoh_lr3,nocoh_ext}`.
    `coh64_v2` has the same config as `coh64`.
- `run_config.yaml` omits argparse-only arguments (e.g. `max_grad_norm`). The sweep files in
  `rl/sweeps/`, with the configs they name, are the source of truth.

## `MANIFEST.txt`

- Lines 1–956: bundle member paths, sorted (byte order).
- Last line: `sha256 <hex>  paper_fig_inputs.tar.gz`.
- Regenerate it after re-packing:

```bash
tar -tzf paper_data/paper_fig_inputs.tar.gz | LC_ALL=C sort > paper_data/MANIFEST.txt
printf 'sha256 %s  paper_fig_inputs.tar.gz\n' "$(shasum -a 256 paper_data/paper_fig_inputs.tar.gz | cut -d' ' -f1)" >> paper_data/MANIFEST.txt
```
