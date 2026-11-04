# GRAFT: RL code (`rl/`)

Code for the RL experiments of the GRAFT paper:
- method (`sec:method`: `alg:grrl`, `eq:opt`, variants);
- Countdown-Code with Qwen3-8B (`sec:results-countdown`, `fig:countdown-fig1`, Countdown appendices);
- six toy environments with SmolLM2-135M-Instruct (`sec:results-small`, `fig:narrative-hook`, toy appendices);
- RL appendices (`app:algorithm-details`, `app:hyperparams`).

Paper references use LaTeX labels. The large-scale SFT experiment (`sec:results-sft`, `fig:large-scale-sft`)
is in `../sft/`.

Run every command from this directory (`rl/`). Modules import each other by bare name, and sweep files
are loaded as the namespace package `sweeps.*` with `rl/` as the working directory.

## Contents

| Path | What |
|---|---|
| `train.py` | single training run (`train_main`); GRPO trainer subclass with gradient routing |
| `trl_overrides.py` | TRL 0.29 overrides (generation, old log-probs, KL reference) |
| `advantages.py` | routed advantages, gradient-mask weights, baselines |
| `gradient_routing.py` | dual MLP / LoRA adapters, fused stop-gradient routing, pre-routing gradient capture, PPS steering |
| `split_moment.py` | `SplitMomentAdamW` (Eq. 1 optimizer) |
| `experiment_config.py`, `rewards.py`, `rh_detectors.py` | YAML config loading, reward components, monitors |
| `data/` | `objects.jsonl`, `cities.jsonl` (object_qa / cities_qa data) |
| `envs/` | environment modules: toy (`addition`, `qa`, `repeat`, `sorting`), `countdown_code`, `leetcode_verified` (Countdown anchor), `leetcode`, `mbpp` (helper modules imported by `leetcode_verified` and the Countdown grader; their own env registrations are not selected by the shipped configs) |
| `configs/` | 16 environment configs (YAML) |
| `sweeps/` | one Python file per experiment; see `sweeps/README.md` |
| `sweep.py` | sweep orchestrator (local GPUs with CUDA MPS, or Modal) |
| `sweep_config.py`, `sweep_views.py`, `slot_pool.py` | sweep helpers (config utilities, W&B saved views, cross-sweep GPU slot pool) |
| `eval_utils.py`, `persistent_code_eval.py`, `behavioral_hardcode.py` | eval helpers (generation and scoring, persistent code-grader worker pool, behavioral hard-code detector) |
| `api_rewards.py`, `rewards_leetcode_twofile.py` | additional reward functions imported by `rewards.py` |
| `plot_routing.py` | plots a run's `routing_eval.jsonl` |
| `vllm_*.py`, `vllm_patches/` | vLLM 0.17 generation server with in-engine dual adapters |
| `tools/` | Modal backend, post-hoc evals, SFT priming, PPS extraction, generalization eval |
| `figures/` | the paper's RL figure scripts; see `figures/README.md` |
| `paper_data/` | released figure inputs (`paper_fig_inputs.tar.gz`, `MANIFEST.txt`) and `run_configs/`; see `paper_data/README.md` |
| `tests/` | unit and equivalence tests (21 files) |
| `pyproject.toml`, `uv.lock`, `requirements-modal.txt`, `DEPENDENCIES.md` | dependencies (see "Install") |

## Install

- Python 3.11 (`.python-version`), Linux x86_64, CUDA 12.8.
- `uv sync` (uses `pyproject.toml` + `uv.lock`).
  - flash-attn / flash-attn-3 come from prebuilt wheels for cu128 / torch 2.10 / cp311 / linux_x86_64 only.
  - vLLM 0.17.0's declared torch/transformers bounds are overridden on purpose; see `DEPENDENCIES.md`.
- Apply the vLLM patches (required; overwrites 3 files in the installed vLLM 0.17.0):
  ```bash
  source .venv/bin/activate
  bash vllm_patches/apply.sh
  ```
- W&B is optional: `WANDB_MODE=disabled`, or `--no_wandb` on `train.py` / `sweep.py`.
- Modal is optional and not in `uv.lock`: `uv pip install modal==1.4.3` (the version in
  `requirements-modal.txt`, which is the Modal image's package freeze).
- Figure scripts only need numpy, scipy, matplotlib and pyyaml.

## External code and data

| Item | Needed for | How to get it |
|---|---|---|
| rl-rewardhacking (Wong et al.) | every code grader, incl. Countdown-Code grading (`create_test_runner_code`); LeetCode anchor env; LeetCode generalization split (`results/data/leetcode_{train_medhard_filtered,test_medhard}.jsonl`) | clone `github.com/ariahw/rl-rewardhacking` and set `RH_REPO_PATH`; see "rl-rewardhacking setup" below. For Modal: `python -m modal run tools/modal_leetcode_passk.py::stage` |
| SFT-primed Qwen3-8B (Countdown starting point) | all Countdown sweeps | see "Countdown SFT priming" below |
| PPS steering vector | PPS sweeps | `python -m modal run tools/modal_pps_extract.py::main` (writes `/output/pps_extract/pps_vector_hackonly.pt` on the volume); elsewhere set `GRAFT_PPS_VECTOR` |
| Countdown-Code `o4-mini-distillation-16k.jsonl` | SFT priming data | upstream `github.com/zohaib-khan5040/Countdown-Code` |
| HF models / datasets | downloaded at runtime | `HuggingFaceTB/SmolLM2-135M-Instruct`, `Qwen/Qwen3-8B`, `Jiayi-Pan/Countdown-Tasks-3to4`, `NeelNanda/pile-10k`, `openai/openai_humaneval`, `newfacade/LeetCodeDataset` |

### rl-rewardhacking setup

```bash
git clone https://github.com/ariahw/rl-rewardhacking /path/to/rl-rewardhacking
export RH_REPO_PATH=/path/to/rl-rewardhacking   # default if unset: ~/rl-rewardhacking-private
```

What the RL code uses from it:
- `src.evaluate.helpers.create_test_runner_code` (`persistent_code_eval.py`; `envs/_rh_bridge.py` puts
  `RH_REPO_PATH` on `sys.path`). Importing it runs `src/evaluate/__init__.py`, whose third-party imports
  (pandas, omegaconf, orjson, dill, jinja2, python-dotenv, pydantic, ...) are in `uv.lock`.
- `results/data/leetcode_train_medhard_filtered.jsonl` (LeetCode anchor env, `configs/leetcode_verified_anchor.yaml`,
  `leetcode_hint: none`) and `results/data/leetcode_test_medhard.jsonl` (eval split; LeetCode generalization
  scaffold). Both are in the repository.
- Hinted variants of the `leetcode` env (`--leetcode_hint <hint>`, not used by the shipped configs) read
  `results/data/<prefix>_<hint>.jsonl`; generate them in the rl-rewardhacking checkout with
  `uv run python scripts/run_data_process.py create --hint=<hint> --base_dataset_fpath=results/data/leetcode_train_medhard_filtered.jsonl`
  (and the same with `leetcode_test_medhard.jsonl`).
- `tools/modal_leetcode_passk.py::stage` uploads `$RH_REPO_PATH/src` and the two jsonl files to the Modal
  volume (`/output/_rh`, the Modal image's `RH_REPO_PATH`).

### Countdown SFT priming

```bash
python tools/countdown_build_sft_data.py --src <Countdown-Code>/o4-mini-distillation-16k.jsonl   # -> output/countdown_sft
modal volume put gr-modal-pilot output/countdown_sft /countdown_sft --force
python -m modal run tools/modal_countdown_sft.py::run --model Qwen/Qwen3-8B
```

- `--model` is required: the script's default is `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Output: `/output/countdown_sft_model/qwen3-8b` on the volume = the Countdown sweeps' default model path.
- Other defaults: LoRA r128 / alpha 128, lr 1e-4, 5 epochs.
- Re-implementation of the upstream recipe (HF Trainer + PEFT), not the upstream training code; see the
  `tools/countdown_build_sft_data.py` header.
- Zero-shot hack base rate: `tools/modal_countdown_hack_baserate.py`, `tools/countdown_hack_baserate_grade.py`.

## Environment variables

| Variable | Read by | Default |
|---|---|---|
| `RH_REPO_PATH` | `envs/_rh_bridge.py`, `tools/{modal_leetcode_passk,leetcode_rl_filter,leetcode_verify_badtests}.py` | `~/rl-rewardhacking-private` (Modal image: `/output/_rh`) |
| `GRAFT_COUNTDOWN_SFT_MODEL` | 7 Countdown sweep files: `countdown_code_rp.py` (`_base`, imported by the other Modal Countdown sweeps), `cdhf100_noroute_retry.py`, and the 5 pod sweeps (`countdown_dn_s15_repro.py`, `countdown_hf100_noroute_anchor.py`, `countdown_hf100_gr_nocoh_lr3.py`, `countdown_hf100_ip_screen.py`, `countdown_lconly_full.py`) | `/output/countdown_sft_model/qwen3-8b` (Modal volume path) for the first two; `/workspace/small-rl/output/countdown_sft_model/qwen3-8b` for the pod sweeps |
| `GRAFT_PPS_VECTOR` | `sweeps/countdown_hf50_pps.py` (imported by the hf100 PPS sweep) | `/output/pps_extract/pps_vector_hackonly.pt` |
| `GRAFT_OUTPUT_ROOT` | figure scripts | `rl/paper_data/output` |
| `GRAFT_RESPMON_ROOT` | toy figure scripts | `rl/paper_data/respmon-results` |
| `GRAFT_MODAL_SECRETS` | `tools/modal_train_gr.py` (comma-separated Modal Secret names attached to its functions; `""` = none, for a Modal environment without them, with `no_wandb` runs) | `gr-pilot-keys,wandb-key` |
| `GRAFT_MODAL_DEPLOYED_APP` | `sweep.py --backend modal` (dispatch to a deployed copy of the app, `modal deploy --name <name> tools/modal_expr_probe.py`, instead of starting an ephemeral one; for workspaces at Modal's ephemeral-app limit). `tools/modal_deployed_eval.py` runs the GRAFT eval wave on such a copy | unset |
| `GRAFT_MODAL_OUTPUT_MIRROR` | `tools/expr_only_probe.py` (local copy of the Modal volume, prefixed to `/output/...` model paths) | `/workspace/small-rl` |
| `WANDB_*` | W&B (project default `small-rl`) | |
| `HF_HOME` | HF cache (Modal image: `/output/_hf_cache`) | |
| `SMALL_RL_GPU_CAP` | `slot_pool.py` (max concurrent runs per GPU across sweeps) | 6 |

- Sweep files read `GRAFT_*` at import time on the launching machine. Under `--backend modal` the path
  must exist inside the Modal container (leave unset to use the volume path).

## Running a single training run

```bash
python train.py --config configs/<env>.yaml [--flag value ...]
```

- Always pass `--config` (each shipped config sets `training.environment`) or `--environment`.
- `python train.py --help` lists all flags. Sweep dicts use the same names.
- Outputs (`run_config.yaml`, `train.log`, `routing_eval.jsonl`, checkpoints) go to `--output_dir`
  (default `./output`, i.e. directly into `rl/output/`, not a per-run subdirectory); pass
  `--output_dir output/<run>` to keep runs separate. `sweep.py` sets `output/<name>/<run_name>/` for you.

## Running sweeps

```bash
# toy envs (local GPUs, CUDA MPS)
PATH=<dir with driver-matched nvidia-cuda-mps-control>:$PATH python -u sweep.py --name <name> --config sweeps/<file>.py --no_baseline
# Countdown (Modal)
python sweep.py --name <name> --config sweeps/<file>.py --backend modal --no_pack --no_baseline
```

- Sweep → paper map, original `--name`s and notes: `sweeps/README.md`.
- Useful flags: `--dry_run`, `--no_wandb`, `--per_gpu N`, `--output_dir`, `--overwrite_run`.
  Extra `--<train flag> value` arguments are applied to every run.
- Outputs: `output/<name>/<run_name>/`. An existing `--name` gets a `-MMDD-HHMM` suffix unless
  `--overwrite_run`.
- Expected warnings: `sweep.py` prints `[WARN] Failed to regenerate sweep pages...` about once a minute
  and `[WARN] Failed to generate plots for group...` (optional dashboard hooks). Training is unaffected.

### CUDA MPS (toy sweeps)

- Toy sweeps ran 3–5 runs per GPU under CUDA MPS (`per_gpu` in each sweep file).
- `sweep.py` starts one `nvidia-cuda-mps-control -d` daemon per GPU (pipes `/tmp/nvidia_mps_<gpu>`).
  The MPS binary found on `PATH` must match the installed driver version exactly; the sweep docstrings
  prepend such a directory to `PATH`.
- `sweep.py` honors `CUDA_VISIBLE_DEVICES`.
- `--no_mps` skips MPS daemon management (e.g. MPS already running).
- MPS does not work on Modal.

## Modal

Resources expected by `tools/modal_train_gr.py`:
- app `gr-pilot`; volume `gr-modal-pilot` mounted at `/output`;
- secrets `gr-pilot-keys` (`OPENAI_API_KEY`; not used by the shipped configs) and `wandb-key`
  (`WANDB_API_KEY`);
- volume `rh-verl` (auto-created; not used by the entrypoints below).

Invocation:
- Run from `rl/` as `python -m modal run tools/<file>.py::<entrypoint>`.
  `tools/{modal_countdown_sft,modal_ip_prescreen,modal_expr_probe,modal_countdown_hack_baserate,modal_pps_extract,modal_leetcode_passk}.py`
  import `tools.modal_train_gr` without a `sys.path` fix, so `rl/` must be on `sys.path` (`python -m` does this).
- Some docstrings show `modal run ...` or `.venv/bin/modal run ...`; same entrypoints.

`tools/modal_train_gr.py` entrypoints:

| Supported | Use |
|---|---|
| `train_one`, `train_many` | remote functions dispatched by `sweep.py --backend modal` |
| `launch_sweep --sweep-module sweeps.<file> --sweep-name <name>` | dispatch a sweep file without `sweep.py` |
| `eval_forget_scales --sweep-name <dir> --config-module sweeps.<file> --scales-csv 0.0,1.0` (remote: `eval_forget_scales_one`) | post-hoc forget-scale eval (fseval); see "Post-hoc evaluation" below |
| `smoke_test`, `mps_diagnose` | infrastructure checks |

The file's other entrypoints (`fused_gate*`, `launch_modal_*`, `gen_sft_dataset*`, `eval_sftgr_countdown*`,
`probe_sft_student`, `dump_sftgr_traj*`) are not supported.

## Post-hoc evaluation (Countdown)

| What | Command | Notes |
|---|---|---|
| forget-scale eval (fseval), Modal | `python -m modal run tools/modal_train_gr.py::eval_forget_scales --sweep-name <dir> --config-module sweeps.<file> --scales-csv 0.0,1.0` | writes `/output/<dir>_fseval/<run>.json`. Without `--scales-csv` it evaluates 0.0..1.0 in steps of 0.1 (11 evals per run); the released JSONs mostly hold only the plotted scale. Run names come from the sweep file's `run_name` keys; for sweeps without them (e.g. `countdown_code_rp.py`, the `fig:countdown-fig1` No-intervention arm) pass `--names-csv countdown_code_hack_reward_penalty_amountmissing_s9,countdown_code_hack_reward_penalty_amountmissing_s16` |
| fseval, local GPU | `python tools/local_fseval.py --sweep_dir output/<sweep> --n_eval 256 --scales 0.0,1.0` | same JSON format; default scales 0.0..1.0 step 0.1. On 80 GB GPUs add `--gen_batch_size 64` (per the tool's `--help`, needed for 8B × 1536-token evals there) |
| fseval, Modal, by checkpoint path | `python -m modal run tools/modal_countdown_fseval.py::run_batch --runs /output/<sweep>/<run>[,...] --out-dir /output/<name>_fseval` | scales 1.0 and 0.0; see its docstring |
| fseval of a non-Countdown-trained policy on the Countdown protocol (anchor-only arm) | `python tools/crossenv_countdown_fseval.py --sweep_dir output/<sweep> --countdown_run_config <countdown run>/run_config.yaml --out_dir output/<name>_fseval` | default scale 1.0 |
| expr-only (clean-environment) probe | `python tools/expr_only_probe.py --ckpt output/<sweep>/<run>/checkpoint-200 --scales <s> --n 256 --k 1 --out output/expr_only_probe_arms` | `--scales 0.0` for GRAFT / GRAFT w/o routing (deployed); `--scales 1.0` for single-policy baselines (DN, RP, IP, PPS, anchor-only); base model: `--retain_scale 0.0 --scales 0.0` on a GRAFT checkpoint (writes `<run>__r0.0.json`). The figure scripts look up exactly these scales: if a probe file lacks the scale, `countdown_hf100_hf50_retain.py` drops the point with a WARNING and `countdown_figure1.py` raises `KeyError`. Paper hf100 probes used k=1 (tool default k=4); Modal variant `tools/modal_expr_probe.py::single` (default k=1) |
| collate fsevals | `python tools/collate_countdown_fseval.py --fsdirs <dir> [...]` | per-scale means and per-seed values; plot to `figures/figs/countdown_fseval*`. Reads every `*.json` in each directory and keys records by the `run_name` inside the JSON (condition = name without `_s<seed>`). `__r<r>` / `__step<N>` files carry the main run's `run_name`, so pass directories that hold only main-run files. In the bundle, `countdown_hf100_gr_lccoh64_lr3_fseval/` and `countdown_hf50_gr_nocoh_fseval/` also hold a `__r0.0` file; copy the main-run files first, e.g. for the hf100 GRAFT arm (8 seeds): `mkdir -p /tmp/graft_hf100 && cp paper_data/output/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_gr_lccoh64_lr3_s{9,15,16}.json paper_data/output/countdown_hf100_gr_lccoh64_lr3_seeds5_fseval/*.json /tmp/graft_hf100/ && python tools/collate_countdown_fseval.py --fsdirs /tmp/graft_hf100` |
| generalization (HumanEval / LeetCode) | `tools/countdown-generalization-eval/` | see its `README.md`; table commands in `figures/README.md`. Paper checkpoint sets: `--spec countdown_hf100_all` (two-file tables and scatter) and `--spec countdown_hf100_gr` (perturbation scaffolds; see `modal_generate_ckpt_vllm.py`). Base model (`checkpoint_specs.BASE_ROWS`): `modal_humaneval_generate.py::run --model /output/countdown_sft_model/qwen3-8b --label sft-base [--scaffold leetcode_scaffold]`. The `--spec` default of `run_eval.py`, `make_table.py` and `modal_humaneval_generate_ckpt.py` is `countdown_0702`; pass `--spec` explicitly |

## Regenerating figures

```bash
tar -xzf paper_data/paper_fig_inputs.tar.gz -C paper_data
python figures/<script>.py
```

- Per-figure scripts and inputs: `figures/README.md`.
- Bundle layout, provenance and the figures it regenerates: `paper_data/README.md`.

## Method → code map

Canonical GRAFT settings: `routing_mode=classic`, `renormalization_mode=balanced`, `split_moment=True`,
`routing_lambda=1.0`, `adapter_type=mlp`, equal adapter sizes (κ_R = κ_F = 2), `fused_reduction` on
(default), `loss_type=grpo` (default).

| Paper | Code |
|---|---|
| Dual adapters (`DualMLPAdapter`; π_{R+F}, π_R, π_F) | `gradient_routing.DualMLPAdapter`, `apply_dual_mlp`, `set_scales(model, retain, forget)`; widths `vllm_utils.MLP_PRESETS` (`m16` toy, `m64` Qwen3-8B) |
| Alg. 1 rollout on task env (π_{R+F}) | vLLM slot at scales (1, 1); `train._generate_single_turn` |
| Alg. 1 anchor rollout (π_R) | `--coh_samples_per_rollout`, generated at scales (1, 0). Toy: same-env anchor = trailing prompts of the training batch (`coh_prompt_source=batch`, default). Countdown: cross-env LeetCode anchor `--coh_config configs/leetcode_verified_anchor.yaml` (`train._load_coh_env_config`, `_maybe_swap_coherence_prompts`). `fig:anchor-form`: `coh_prompt_source=unhackable` |
| Monitor | `rh_detector` in the config YAML → `experiment_config.build_rh_detector` → `rh_detectors.py`; `--rh_detector_recall` < 1 drops flags at random |
| Advantages (Alg. 1 line 5) | `advantages.compute_routed_advantages` → `_baseline_nonflagged_var_all`: baseline = mean over unflagged group members (full-group mean if none), scale = unbiased std of the whole group + 1e-4 |
| Masked / amplified gradients | `train.SampleGRPOTrainer._fused_forward_backward`; masks (retain, forget) from `advantages.routing_grad_mask_weights` with κ from `adapter_kappas`: anchor (1, 0, forget off in forward), unflagged (1, 1), flagged (0, 2); stop-gradient decomposition `gradient_routing._fused_decouple`; per-sequence loss averaged over the N samples of the optimizer batch (`scale_denom`) |
| Eq. 1 optimizer | `split_moment.SplitMomentAdamW`: first moment from the routed gradient, second moment from the pre-routing gradient (`gradient_routing.PreRoutingGradAccumulator`) |
| Deploy π_R | eval mode `retain_only`; post-hoc fseval scale 0.0 |
| GRPO loss, KL reference (`app:algorithm-details`) | `train._packed_compute_loss` (Liger, packed sequences); KL reference = base model with both adapters disabled (`gradient_routing.disabled_dual_adapters`, used in `trl_overrides.py`) |
| Baselines | No intervention: `routing_mode: none`. Reward penalty: `reward_penalty_baseline` + `reward_penalty_amount`. Filtering: `filter_baseline`. GRAFT w/o routing: `rh_detector_recall: 0.0`. GRAFT w/o anchoring: `coh_samples_per_rollout: 0`. Inoculation prompting: `countdown_train_system_suffix`. PPS: `pps_vector_path`/`pps_layer`/`pps_alpha` (`gradient_routing.set_pps_steering`) |
| LoRA alternative | `gradient_routing.DualLoRALinear`, `apply_dual_lora`, `vllm_lora.py`, `tests/test_dual_lora_vs_peft.py`. The vLLM LoRA server (`vllm_lora.py`) ignores adapter scales (its `set_scales` op is a no-op), so retain-only / forget-only evals of LoRA runs generated through it are not ablated. Measure LoRA localization with an HF-side eval that applies `gradient_routing.set_scales` (which does handle `DualLoRALinear`) |
| Gradient-norm diagnostics (Discussion) | `--adapter_diag_level per_sample_recompute`, `tools/analyze_grad_diag_*.py`, `tools/gen_{grad_diag,separability}_html.py` |

Canonical hyperparameters live in sweep files, not configs:
- toy: `sweeps/respmon_repro_coh128_lam1_3seed.py` (`_shared`, `_new`) + per-env steps in `sweeps/_respmon_canon.py`
  (its `CANON_ENVS` lists the stock sort config; the paper's sort is `configs/sorting_copy_framed.yaml`,
  see the top-level README, "Toy environment configs");
- Countdown: `sweeps/countdown_code_rp.py` (`_base`) + `sweeps/countdown_code_gr.py` (`_gr`) + overrides in
  `sweeps/countdown_hf100_gr_lccoh64_lr3.py`.

## Implementation notes

Facts about the code that the reported numbers depend on.

- **Old log-probs and clipping.**
  - Toy runs: one optimizer step per rollout, `vllm_importance_sampling: false` → `old_per_token_logps=None`
    in `trl_overrides.py` → PPO ratio ≡ 1, the clip range never binds.
  - Framed sort (`sorting_framed`) is not in `train.py`'s `_SMALL_SCALE_ENVS`, so framed-sort runs take the
    slow path (trainer-recomputed old log-probs + truncated vLLM IS correction), unlike the other five toy envs.
  - Countdown: 4 optimizer steps per rollout, recomputed old log-probs, truncated vLLM IS (token clip 2.0,
    sequence filter 1.1).
- **Optimizer behavior beyond Eq. 1:**
  - per-coordinate step bound `--graft_w_max` (default 4.0);
  - forget adapter frozen (`SplitMomentAdamW.step` skips its m, v, step count and weight decay) on any
    optimizer window whose microbatches contain no target-environment sample, flagged or unflagged
    (e.g. an anchor-only window). `train.SampleGRPOTrainer._fused_forward_backward` calls
    `set_window(..., {"retain": True, "forget": n_routing > 0}, ...)` with
    `n_routing = len(good_idx) + len(bad_idx)`, counted after the forget-warmup drop of unflagged samples
    and the optional `--drop_zero_advantage` drop. This is the only `set_window` call (fused path); without
    it the window is unset and no adapter is frozen;
  - one gradient-clip coefficient (from the routed gradient) applied to the pre-routing gradient as well.
- **Anchor placement (Countdown).** With `coh_samples_per_rollout > 0` and `max_tokens_per_microbatch` set, `_prepare_inputs` splits the rollout into
  `steps_per_generation` contiguous chunks (chunk order permuted). Cross-env anchor rows form the trailing
  contiguous block of the rollout (asserted in `train.py`). For the headline recipe (1024 + 64 rows,
  `optimizer_batch_size` 272) this puts all 64 anchor rows in one of the 4 optimizer steps of each rollout.
- **Toy GRAFT w/o routing** (`rh_detector_recall: 0`) also disables the anchor-slice penalty (the anchor slice
  uses the same recall-thinned monitor). Lower recall likewise thins the anchor penalty.
- **`detect_unhackable: true`** (default): the monitor also flags, routes and penalizes hack forms emitted
  on unhackable prompts. The addition, cities_qa, object_qa and persona_qa monitors
  (`sycophancy_monitored`, `sycophancy_any`, `flattery_any`) are case-insensitive substring tests;
  `repeat_contiguous` (repeat_extra) flags only contiguous repeats (no letters between copies), and
  `sorting_copy_verbatim` (sort) requires a positional copy fraction above 0.5 plus a regex match of the
  input sequence.
- **Toy evaluation protocol:**
  - 256 fixed eval prompts per adapter mode (each env's `load_eval_prompts`, seed 99, `split="test"`), one
    sample each at temperature 1;
  - every 10 steps, piggybacked on rollouts; written to `<run>/routing_eval.jsonl`;
  - figures use the mean over the last 10% of eval rows;
  - "Base model" point = first eval row = step 10 (step 0 only for the `sort_framed_*` sweeps, which set
    `eval_at_start: True`; the anchor-form framed-sort sweeps `respmon_sortframed_*` start at step 10);
  - the run configs of every other toy run (and every Countdown run) in `paper_data/run_configs/` record
    `eval_at_start: false`; `train.py` and `experiment_config.py` default to true, and apart from
    `cdhf100_noroute_retry.py` (false) no other sweep file sets it. To run those sweeps with the recorded
    setting, pass `--no-eval_at_start` to `sweep.py` (it is forwarded to every run); otherwise each run
    gets a step-0 row, which the figure scripts then use as the "Base model" point.
- **Toy model id** is `HuggingFaceTB/SmolLM2-135M-Instruct`.
- **Countdown evaluation:**
  - in-training eval: n=64 per adapter mode (vLLM);
  - post-hoc fseval: n=256, `set_seed(42)` per mode, 1536 new tokens, HF `generate` with `temperature`
    only; top_p / top_k are not passed, so they come from the model's `generation_config` (not checked);
  - expr-only probe: k=1 for the hf100 arms (tool default k=4), no RNG seed;
  - Qwen3 chat template with `enable_thinking=False`.
- **`run_config.yaml` contents.** It is an `ExperimentConfig` dump; argparse-only arguments (e.g.
  `max_grad_norm`, `countdown_n_train`) are set in the sweep files. The sweep files, with the configs
  they name, hold the full settings. The framed-sort `run_config.yaml` files in `paper_data/` have no
  `sort_prompt_style` key; those runs used `framing_noverb`, which `configs/sorting_copy_framed.yaml`
  sets.
- **Detector channels:** the logged `hack_freq_{detectable,undetectable}/<detector>` channels of the toy and
  Countdown configs are all-prompt rates, not hackable-slice quadrants.
- **Precision:** with `bf16: True`, `train.py` loads the base model in bf16, and adapter parameters and
  `SplitMomentAdamW` states take the parameter dtype.

## Tests

- Run from `rl/`: `python -m pytest tests --ignore=tests/test_vllm_mlp_adapter.py`.
- `tests/test_vllm_mlp_adapter.py` cannot be collected: it imports `create_dummy_lora_dir`, which
  `vllm_mlp_adapter.py` does not define. Without `--ignore`, pytest stops at collection and runs no tests.
- Many tests need a CUDA GPU, vLLM 0.17 with `vllm_patches/` applied, or HF model downloads. Only
  `TestTrainerBehavior` carries a `gpu` marker and no pytest config registers markers, so `-m "not gpu"` is
  not a CPU-only subset.
- Two tests are skip-marked (reason in each marker):
  - `tests/test_dual_lora_vs_peft.py::TestTrainingEquivalence::test_dual_lora_no_routing_matches_peft_lora`;
  - `tests/test_batch_params.py::TestTrainerBehavior`.
- SFT tests (from `../sft/`): `PYTHONPATH=. python -m scripts.gr.test_gr_core` (CPU) and
  `PYTHONPATH=. python -m scripts.gr.test_adapter` (`DualMLPAdapter`; needs CUDA and downloads
  `Qwen/Qwen3-8B`).

## Paths in docstrings

- `/workspace/...` paths in docstrings are machine-specific examples; substitute your own.

## Scope

- Supported environments are the ones selected by the shipped configs (see "Contents"). `rewards.py` /
  `rh_detectors.py` also define reward and detector functions for other environments; no shipped config
  selects them.
