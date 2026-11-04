#!/usr/bin/env bash
# Post-hoc evals for the camera-ready GRAFT Countdown sweeps (Modal env jnward;
# GRAFT and its ablations deployed with the FORGET adapter ablated):
#   fseval n=256 at forget scales 0.0 (deployed, RETAIN only) and 1.0 (both)
#   expr-only (clean-environment) probe, n=256, k=1, at scale 0.0 (pass@1)
# plus the frozen SFT init (both adapters off, on the sweep's first run).
#
#     [METHODS=a,b] bash tools/eval_countdown_gr.sh <sweep module> [<sweep dir>]   # from rl/
#
# PROBE_BATCH=1 runs the probes in one Modal app (modal_expr_probe.py::batch)
# instead of one app each, for when Modal refuses new ephemeral apps.
#
# METHODS (read by the sweep module) restricts the runs, for a sweep dir that holds
# only some arms. The sweep dir defaults to the newest output/<module>[-MMDD-HHMM]
# (the volume dir has the same name).
set -euo pipefail
cd "$(dirname "$0")/.."
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-jnward} GRAFT_MODAL_SECRETS=${GRAFT_MODAL_SECRETS-}
SW=${1:?usage: eval_countdown_gr.sh <sweep module> [<sweep dir>]}
SWDIR=${2:-$(ls -d output/${SW}/ output/${SW}-*/ 2>/dev/null \
    | grep -E "/${SW}(-[0-9]{4}-[0-9]{4})?/$" | sort | tail -1 | xargs basename)}
echo "evaluating $SWDIR"
RUN="$(pwd)/.venv/bin/python -m modal run"
mkdir -p output/logs
NAMES=$(.venv/bin/python -c "from sweeps.$SW import runs; print(' '.join(r['run_name'] for r in runs))")
BASE_RUN=${NAMES%% *}

pids=()
$RUN tools/modal_train_gr.py::eval_forget_scales --sweep-name $SWDIR \
    --config-module sweeps.$SW --scales-csv 0.0,1.0 \
    > output/logs/${SW}_fseval.log 2>&1 & pids+=($!)
$RUN tools/modal_train_gr.py::eval_forget_scales --sweep-name $SWDIR \
    --config-module sweeps.$SW --only $BASE_RUN --scales-csv 0.0 --retain-scale 0.0 \
    > output/logs/${SW}_fseval_base.log 2>&1 & pids+=($!)
if [ "${PROBE_BATCH:-0}" = 1 ]; then
    # all probes in one app (Modal caps live ephemeral apps at 100)
    CKPTS=$(for n in $NAMES; do printf '%s,' "$SWDIR/$n/checkpoint-200"; done)
    $RUN tools/modal_expr_probe.py::batch --ckpts "${CKPTS%,}" \
        --scales 0.0 > output/logs/${SW}_probe_batch.log 2>&1 & pids+=($!)
else
    for n in $NAMES; do
        $RUN tools/modal_expr_probe.py::single --ckpt $SWDIR/$n/checkpoint-200 \
            --scales 0.0 > output/logs/${SW}_probe_$n.log 2>&1 & pids+=($!)
    done
fi

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "an eval failed; see output/logs/${SW}_*.log" >&2; exit 1; }

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
