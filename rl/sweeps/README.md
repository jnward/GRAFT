# Sweeps (`rl/sweeps/`)

One Python file per experiment. Each launchable file defines a module-level `runs` list (one dict of
`train.py` arguments per run), usually with `per_gpu` and `no_baseline`; files marked "helpers" below
only define shared settings for import. Files import shared settings from each other as
`sweeps.<file>`, so run `sweep.py` from `rl/`.

## Launching

```bash
# toy envs (SmolLM2-135M-Instruct): local GPUs, CUDA MPS, per_gpu 3-5
PATH=<dir with driver-matched nvidia-cuda-mps-control>:$PATH \
  python -u sweep.py --name <name> --config sweeps/<file>.py --no_baseline
# Countdown (Qwen3-8B): Modal
python sweep.py --name <name> --config sweeps/<file>.py --backend modal --no_pack --no_baseline
# Countdown runs marked "pod" below: local backend (drop --backend modal)
python -u sweep.py --name <name> --config sweeps/<file>.py --no_baseline
```

- Use the `--name` from the table to reproduce the original output directory name; figure scripts and
  fseval tools look runs up by these names.
- Outputs: `output/<name>/<run_name>/` (`run_config.yaml`, `routing_eval.jsonl`, checkpoints).
- `--dry_run` prints the planned runs. Extra `--<train flag> value` arguments override every run.
- Setup (vLLM patches, `RH_REPO_PATH`, SFT-primed model, Modal): `../README.md`.

## Sweep → paper map

Legend:
- Data: **B** = training logs (`routing_eval.jsonl` + `run_config.yaml`) or fseval JSONs in
  `../paper_data/paper_fig_inputs.tar.gz`; **C** = `run_config.yaml` only, in `../paper_data/run_configs/`;
  **–** = none.

### Countdown-Code (Qwen3-8B)

| Sweep file | `--name` (original output dir) | Paper use | Data |
|---|---|---|---|
| `countdown_code_rp.py` | `countdown_code_rp2-0702-0026` | its `{}` arm (run names `countdown_code_hack_reward_penalty_amountmissing_s{9,16}`) = `fig:countdown-fig1` No intervention s9/s16; defines `_base`, `_seeds` for all Countdown sweeps | B (fsevals) |
| `countdown_dn_s15_repro.py` | `countdown_dn_s15_repro` (pod) | `fig:countdown-fig1` No intervention s15 | B (fseval, in the `countdown_code_rp2-0702-0026_fseval` dir), C |
| `countdown_code_gr.py`, `countdown_code_gr_nocoh.py`, `countdown_hf50_common.py` | helpers | `_gr`, `_gr` without anchor, `_HF50` / `make_runs` | – |
| `countdown_hf100_gr_lccoh64_lr3.py` | same as file | GRAFT s9/15/16 (`fig:countdown-fig1`, `fig:dev-selection-stars`, `fig:countdown-hf50` hf100 panel, generalization) | B (fsevals), C |
| `countdown_hf100_gr_lccoh64_lr3_seeds5.py` | same as file | GRAFT s1–5 | B (fsevals), C |
| `countdown_hf100_rp_lccoh64.py` | same as file | Reward penalty: 4 configs (penalty {2, 5} × lr {5e-4, 5e-4/3}); plotted config `rp2_lc64_lr1` (hardcoded as `BEST_RP` in `../figures/countdown_figure1.py`, selection rule in its comment) | B (fsevals of `rp2_lc64_lr1`), C |
| `countdown_hf100_ip.py` | same as file | Inoculation prompting `mand-tw` (the file also defines `mand-tw-norm`) | B (`mand-tw` fsevals), C |
| `countdown_hf100_pps.py` | same as file | Preventative steering L20 α2 | B (fsevals), C |
| `countdown_lconly_full.py` | `lconly_full` (pod, 8×H100) | Anchor-environment-only arm | B (fsevals via `tools/crossenv_countdown_fseval.py`), C |
| `countdown_hf100_noroute_anchor.py` | `cdhf100_noroute` (pod) | GRAFT w/o routing (run dirs `cdhf100_noroute_rerun_ckpts`, fsevals `cdhf100_noroute_fseval_rerun`) | B (fsevals), C (`run_configs/output/cdhf100_noroute_rerun_ckpts/cdhf100_noroute/`) |
| `cdhf100_noroute_retry.py` | launched as `python -m modal run tools/modal_train_gr.py::launch_sweep --sweep-module sweeps.cdhf100_noroute_retry --sweep-name cdhf100_noroute` | Modal launcher for the same arm | – |
| `countdown_hf100_gr_nocoh_lr3.py` | `cdhf100_gr_nocoh_lr3` (pod) | `tab:countdown-noanchor`, GRAFT w/o anchoring | C |
| `countdown_hf100_ip_screen.py` | `cdhf100_ip_screen` (pod) | inoculation-prompt screen | – |
| `countdown_hf50_dn.py`, `countdown_hf50_rp_lccoh64.py`, `countdown_hf50_ip.py`, `countdown_hf50_pps.py`, `countdown_hf50_gr_lccoh_lr3.py`, `countdown_hf50_gr_nocoh.py` | same as file | `fig:countdown-hf50` arms; also define `IP_PROMPTS` (`countdown_hf50_ip.py`, `app:ip-prompts`) and `_PPS` (`countdown_hf50_pps.py`) | B (fsevals), C |
| `countdown_hf50_gr_coh64.py`, `countdown_hf50_gr_coh64_lr3.py`, `countdown_hf50_gr_nocoh_seeds5.py`, `countdown_hf50_gr_nocoh_lr3.py` | see below | GRAFT learning-rate selection (`app:hyperparams`) | C |

### Countdown-Code camera-ready (learning-rate protocol)

Every method on one SFT init (`/output/countdown_sft_model/qwen3-8b`, re-made from the SFT recipe;
not the submission's weights) across the learning rates 5e-4, /3, /10, /30 (No intervention also
/100, /300), seeds 9/15/16; each method is plotted at its developer-selected learning rate
(`../figures/README.md`, "Camera-ready"). `METHODS=<a>,<b>` (No intervention: `LR_TAGS`) selects
arms; each arm group ran as its own sweep dir. Data: **CR** = evals in
`../paper_data/cr_fig_inputs.tar.gz`. Evals: `../tools/eval_countdown_dn_lrsweep.sh` (single policy),
`../tools/eval_countdown_gr.sh` (GRAFT and its ablations, FORGET ablated),
`../tools/eval_countdown_lconly.sh` (anchor only), `../tools/modal_expr_probe.py::ckpt_series`
(`gr_ck20`).

| Sweep file | `--name` (original output dir) | Paper use | Data |
|---|---|---|---|
| `countdown_hf100_gr_newinit.py` | `countdown_hf100_gr_loreft_all` (launched with other arms not in this release) | GRAFT at 5e-4/3, the Fig. 2 pick; defines `_ANCHORED`, `_NOANCHOR` | CR |
| `countdown_code_dn_lrsweep.py` | `countdown_code_dn_lrsweep`; `LR_TAGS=lr100,lr300`: `countdown_code_dn_lrsweep_low` | No intervention | CR |
| `countdown_hf100_ip_pps_lrsweep.py` | `countdown_hf100_ip_lrsweep` (`ip`), `countdown_hf100_ip_lrsweep_lr1` (`ip_lr1`), `countdown_hf100_pps_lrsweep` (`pps`; vector re-extracted with `../tools/modal_pps_extract.py`) | Inoculation prompting, preventative steering | CR |
| `countdown_hf100_rp_gr_lrsweep.py` | `countdown_hf100_rp_lrsweep` (`rp`), `countdown_hf100_rp_lrsweep_lr1` (`rp2_lr1`), `countdown_hf100_gr_lrsweep` (`gr`), `countdown_hf100_gr_ck20` (`gr_ck20`) | Reward penalty 2 / 5; GRAFT's other lrs; GRAFT 5e-4/3 with a checkpoint every 20 steps (Fig. 2 right) | CR |
| `countdown_hf100_ablations_newinit.py` | `countdown_hf100_{noroute,lconly,noanchor}_newinit` | GRAFT w/o routing, anchor only (Fig. 2); GRAFT w/o anchoring (`tab:countdown-noanchor`) | CR |
| `countdown_hf50_lrsweep.py` | `countdown_hf50_lrsweep` | `fig:countdown-hf50`, every method on hf50 | CR |

### Toy environments (SmolLM2-135M-Instruct)

| Sweep file | `--name` | Paper use | Data |
|---|---|---|---|
| `_respmon_canon.py`, `respmon_repro_coh128_lam1_3seed.py` | helpers | `CANON_ENVS`; canonical recipe `_shared` / `_new` / `_seeds` (import only) | – |
| `respmon_rv2_gr_3seed.py` | same as file | GRAFT addition / cities / persona / repeat (`fig:narrative-hook`, `fig:training-curves`, recall = 1); stock-sort GRAFT (`fig:sort-pathological`); also defines GRAFT on `object_qa_syco_respmon_iva.yaml` (object_qa with the monitor swapped to indeed) | B (addition, cities, persona, repeat, stock sort), C (all 6 envs) |
| `respmon_objinv_gr_canon_3seed.py` | same as file | object_qa GRAFT | B, C |
| `sort_framed_graft_only.py` | `sort_framed_graft`, merged into `sort_framed/` | framed-sort GRAFT | B, C (in `sort_framed/`) |
| `respmon_iva_dn_3seed.py`, `respmon_iva_rp_3seed.py` | same as file | No intervention / RP2, addition and cities; also define the same arms on `object_qa_syco_respmon_iva.yaml` (object_qa with the monitor swapped to indeed) | B (addition, cities), C (all 3 envs) |
| `respmon_ivainv_dn_rp_3seed.py` | same as file | No intervention / RP2, object_qa; also defines swapped-monitor addition / cities | B (object_qa), C (all 3 envs) |
| `respmon_persona_bvo_dn_rp_3seed.py` | same as file | No intervention / RP2, persona | B, C |
| `respmon_repeat_sorting_dn_rp_3seed.py` | same as file | No intervention / RP2, repeat; stock-sort No intervention / RP2 (`fig:sort-pathological`) | B, C |
| `sort_framed_pilot.py`, `sort_framed_suite.py`, `sort_framed_rp_more.py`, `sort_framed_filter.py` | `sort_framed_{pilot,suite,rp_more,filter}`, merged into `sort_framed/` (pilot s1 renamed `sortframed_donothing_hf050_st1000_s1`) | framed sort: No intervention, RP2/5/10, GRAFT w/o anchoring, Filtering | B (No intervention, RP2/5/10), C (all incl. Filtering, w/o anchoring) |
| `respmon_rp_doses_canon_3seed.py` | same as file | RP doses 5 / 10 (dose picked by apparent performance, `_best_rp` in the figure scripts) | B, C |
| `respmon_filter_canon_3seed.py` (+ `respmon_filter_repeat_s2_refill.py`) | same as file | `fig:narrative-hook` Filtering (5 envs) | C |
| `respmon_noroute_canon_3seed.py` | same as file | `fig:narrative-hook` GRAFT w/o routing (6 envs) | C |
| `respmon_graft_nocoh_canon_3seed.py` | same as file | `fig:narrative-hook` GRAFT w/o anchoring (5 envs) | C |
| `respmon_canon_filter_3seed.py` | same as file | `fig:sort-pathological` Filtering (not the same sweep as `respmon_filter_canon_3seed.py`) | B (stock sort) |
| `respmon_recall_suite.py` | `respmon_recall_suite` | recall 0.5 / 0.25 / 0.1 (GRAFT + RP2): `fig:recall-sweep`, `fig:recall-curves-*` | B |
| `respmon_rv2_vercoh128_gr_3seed.py` | same as file | `fig:anchor-form` GRAFT 4:1 s1–3 (addition / cities / persona / repeat); also defines GRAFT 4:1 on `object_qa_syco_respmon_iva.yaml` | B (all envs but object_qa), C (all 6 envs) |
| `respmon_rv2_vercoh128_objinv_gr_3seed.py` | same as file | `fig:anchor-form` GRAFT 4:1, object_qa s1–3 | B, C |
| `respmon_canon_gr_seeds4to8.py` | same as file | `fig:anchor-form` GRAFT 4:1 s4–8 | B (inside `respmon_canon_vercoh128_composite/`) |
| `respmon_rv2_vercoh32_gr_3seed.py` | same as file | `fig:anchor-form` GRAFT 16:1 | B, C |
| `respmon_canon_rp_mix_3seed.py` (+ `respmon_canon_rp_mix_3seed_refill_mps.py`) | `respmon_canon_rp_mix_3seed` | `fig:anchor-form` RP with anchor mix 4:1 / 1:1 | B |
| `respmon_canon_unhackonly_3seed.py` (+ `respmon_canon_unhackonly_refill.py`) | `respmon_canon_unhackonly_3seed` | `fig:anchor-form` anchor-distribution-only arm | B (5 envs), C (6 envs) |
| `respmon_sortframed_dn_rp_unhack_3seed.py`, `respmon_sortframed_anchor_gr_3seed.py` | same as file | `fig:anchor-form` framed-sort panel | B |
| `respmon_rv2_inverted_gr_3seed.py` | same as file | `app:small-envs` background paragraph: GRAFT with swapped monitors for addition / cities / persona / repeat / stock sort (its object_qa runs use `object_qa_respmon.yaml`) | C |

Notes:
- Argparse-only arguments are recorded only in the sweep files, not in `run_config.yaml`. The sweep
  files, with the configs they name, are the complete record of each run's arguments.
- Toy run names start with the config's `name` field: `<name>_<arm>_hf050_st<steps>_s<seed>`. `name`
  is the file stem except in `object_qa_respmon.yaml` (`object_qa_syco_respmon_iva_inv`). Framed-sort
  runs of the `sort_framed_*` sweeps and `respmon_recall_suite` use the tag `sortframed` (`sortpilot`
  in `sort_framed_pilot.py`).
- Two similarly named filter sweeps exist: `respmon_filter_canon_3seed.py` (`fig:narrative-hook`) and
  `respmon_canon_filter_3seed.py` (`fig:sort-pathological`).

## Learning-rate selection (`app:hyperparams`)

The GRAFT learning rate (5e-4/3 ≈ 1.67e-4 vs 5e-4) was compared on the 50%-hackable Countdown variant:

| Sweep file | `--name` | Comparison |
|---|---|---|
| `countdown_hf50_gr_coh64.py` | `countdown_hf50_gr_coh64` (also `countdown_hf50_gr_coh64_v2`, same config) | in-env anchor, lr 5e-4 |
| `countdown_hf50_gr_coh64_lr3.py` | same as file | in-env anchor, lr 5e-4/3 (run configs differ from the above only in `lr`) |
| `countdown_hf50_gr_nocoh.py` + `countdown_hf50_gr_nocoh_seeds5.py` | `countdown_hf50_gr_nocoh` + `countdown_hf50_gr_nocoh_ext` | no anchor, lr 5e-4, seeds 9/15/16 + 1–5 |
| `countdown_hf50_gr_nocoh_lr3.py` | same as file | no anchor, lr 5e-4/3, 8 seeds |

- Run configs ship in `../paper_data/run_configs/output/countdown_hf50_gr_{coh64,coh64_v2,coh64_lr3,nocoh_ext,nocoh_lr3}/`
  (plus `countdown_hf50_gr_nocoh/`).
- The bundle's `output/countdown_hf50_gr_nocoh_fseval/` holds the lr 5e-4 no-anchor s9/15/16 fsevals
  (forget scales 0.1–0.9); its `cdhf50_gr_nocoh_s9__r0.0.json` is the hf50 base-model eval.
- The written rationale for the choice is in the docstrings of `countdown_hf50_gr_coh64_lr3.py`,
  `countdown_hf50_gr_nocoh_lr3.py` and `countdown_hf50_gr_lccoh_lr3.py`. It compares deployed (forget scale 0)
  retain reward (`countdown_true_tested_from_all`, which uses the ground-truth solution check) and
  ground-truth hack rates, on the 50%-hackable variant.
- Reward penalty: `countdown_hf100_rp_lccoh64.py` (lr 5e-4 vs 5e-4/3). Preventative steering has no learning-rate
  sweep (it uses the 5e-4 of `_base`).
