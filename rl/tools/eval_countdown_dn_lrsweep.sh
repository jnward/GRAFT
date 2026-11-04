#!/usr/bin/env bash
# Post-hoc evals for sweeps/countdown_code_dn_lrsweep.py (Modal env jnward), the
# paper's No-intervention protocol (single policy, so everything at scale 1.0):
#   fseval n=256 at forget scale 1.0 (hack_freq, detected_freq, retain)
#   expr-only (clean-environment) probe, n=256, k=1, at scale 1.0 (pass@1)
# plus the frozen SFT init (both adapters off, on cdhf100_dn_lr1_s9) when that run
# is in the wave.
#
#     bash tools/eval_countdown_dn_lrsweep.sh [<sweep dir>]      # from rl/
#     LR_TAGS=lr100,lr300 bash tools/eval_countdown_dn_lrsweep.sh countdown_code_dn_lrsweep_low
# IP / PPS lr sweeps (sweeps/countdown_hf100_ip_pps_lrsweep.py), same protocol:
#     SWEEP_MODULE=countdown_hf100_ip_pps_lrsweep METHODS=ip \
#         bash tools/eval_countdown_dn_lrsweep.sh countdown_hf100_ip_lrsweep
set -euo pipefail
cd "$(dirname "$0")/.."
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-jnward} GRAFT_MODAL_SECRETS=${GRAFT_MODAL_SECRETS-}
SW=${SWEEP_MODULE:-countdown_code_dn_lrsweep}
SWDIR=${1:-$(ls -d output/${SW}/ output/${SW}-*/ 2>/dev/null \
    | grep -E "/${SW}(-[0-9]{4}-[0-9]{4})?/$" | sort | tail -1 | xargs basename)}
echo "evaluating $SWDIR"
RUN="$(pwd)/.venv/bin/python -m modal run"
mkdir -p output/logs
NAMES=$(.venv/bin/python -c "from sweeps.$SW import runs; print(' '.join(r['run_name'] for r in runs))")

pids=()
$RUN tools/modal_train_gr.py::eval_forget_scales --sweep-name $SWDIR \
    --config-module sweeps.$SW --scales-csv 1.0 \
    > output/logs/${SWDIR}_fseval.log 2>&1 & pids+=($!)
for n in $NAMES; do
    $RUN tools/modal_expr_probe.py::single --ckpt $SWDIR/$n/checkpoint-200 \
        --scales 1.0 > output/logs/${SWDIR}_probe_$n.log 2>&1 & pids+=($!)
done
if [[ " $NAMES " == *" cdhf100_dn_lr1_s9 "* ]]; then
    $RUN tools/modal_train_gr.py::eval_forget_scales --sweep-name $SWDIR \
        --config-module sweeps.$SW --only cdhf100_dn_lr1_s9 --scales-csv 0.0 --retain-scale 0.0 \
        > output/logs/${SWDIR}_fseval_base.log 2>&1 & pids+=($!)
    $RUN tools/modal_expr_probe.py::single --ckpt $SWDIR/cdhf100_dn_lr1_s9/checkpoint-200 \
        --scales 0.0 --retain-scale 0.0 > output/logs/${SWDIR}_probe_base.log 2>&1 & pids+=($!)
fi

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
.venv/bin/python tools/countdown_dn_lrsweep_report.py
