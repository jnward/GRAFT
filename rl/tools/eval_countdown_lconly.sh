#!/usr/bin/env bash
# Post-hoc evals for the anchor-only arm (sweeps/countdown_hf100_ablations_newinit.py
# METHODS=lconly; Modal env jnward), the paper's protocol for that arm: a
# single policy trained on leetcode_verified, so
#   cross-env fseval n=256 at scale 1.0 on countdown hf100
#     (tools/modal_crossenv_fseval.py; donor run_config = the MLP GRAFT rerun's)
#   expr-only (clean-environment) probe, n=256, k=1, at scale 1.0 (pass@1)
#
#     bash tools/eval_countdown_lconly.sh countdown_hf100_lconly_newinit   # from rl/
set -euo pipefail
cd "$(dirname "$0")/.."
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-jnward} GRAFT_MODAL_SECRETS=${GRAFT_MODAL_SECRETS-}
SWDIR=${1:?usage: eval_countdown_lconly.sh <sweep dir>}
DONOR=countdown_hf100_gr_loreft_all/cdhf100_gr_lccoh64_lr3_s9/run_config.yaml
RUN="$(pwd)/.venv/bin/python -m modal run"
mkdir -p output/logs
NAMES=$(METHODS=lconly .venv/bin/python -c "from sweeps.countdown_hf100_ablations_newinit import runs; print(' '.join(r['run_name'] for r in runs))")

pids=()
$RUN tools/modal_crossenv_fseval.py::sweep --sweep-name $SWDIR --names "${NAMES// /,}" \
    --donor-rc $DONOR > output/logs/${SWDIR}_fseval.log 2>&1 & pids+=($!)
for n in $NAMES; do
    $RUN tools/modal_expr_probe.py::single --ckpt $SWDIR/$n/checkpoint-200 \
        --scales 1.0 > output/logs/${SWDIR}_probe_$n.log 2>&1 & pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "an eval failed; see output/logs/${SWDIR}_*.log" >&2; exit 1; }

DEST=output/${SWDIR}_evals
mkdir -p "$DEST"
# volume get can hit "VolumeListFiles rate limit exceeded" when waves overlap; back off and retry
vget() {
    for i in 1 2 3 4 5; do
        .venv/bin/modal volume get gr-modal-pilot "$1" "$DEST" --force && return 0
        sleep $((30 * i))
    done
    return 1
}
vget ${SWDIR}_fseval
vget expr_only_probe_arms
echo "evals in $DEST"
