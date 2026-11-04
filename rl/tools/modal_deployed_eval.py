"""tools/eval_countdown_gr.sh's eval wave (fseval at forget scales 0.0 and
1.0, the frozen-base fseval on the first run, expr-only probes at scale 0.0)
through a deployed copy of the gr-pilot app, for when Modal refuses new
ephemeral apps ("App creation failed: reached limit of 100 ephemeral apps";
2026-11-03, the shared workspace's cap, held by other environments' detached
apps). Calls on a deployed app start no ephemeral app. Same functions, same
volume outputs, so the eval script's volume-get and the report are unchanged.

From rl/:
    MODAL_ENVIRONMENT=jnward GRAFT_MODAL_SECRETS="" .venv/bin/modal deploy \\
        --env jnward --name gr-pilot-evals tools/modal_expr_probe.py
    METHODS=gr MODAL_ENVIRONMENT=jnward .venv/bin/python tools/modal_deployed_eval.py \\
        countdown_hf50_lrsweep countdown_hf50_lrsweep
    .venv/bin/modal app stop --env jnward gr-pilot-evals
then sync as the eval script does:
    modal volume get gr-modal-pilot <sweep dir>_fseval output/<sweep dir>_evals --force
    modal volume get gr-modal-pilot expr_only_probe_arms output/<sweep dir>_evals --force
"""
import importlib
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP = "gr-pilot-evals"


def main(sweep_module, sweep_dir):
    names = [r["run_name"] for r in importlib.import_module(f"sweeps.{sweep_module}").runs]
    fseval = modal.Function.from_name(APP, "eval_forget_scales_one", environment_name="jnward")
    probe = modal.Function.from_name(APP, "probe_one", environment_name="jnward")
    # eval_forget_scales_one(run_name, sweep_name, n_eval, scales, retain_scale, checkpoint_step)
    calls = [(fseval.spawn(n, sweep_dir, 256, [0.0, 1.0], 1.0, None), f"fseval {n}") for n in names]
    calls.append((fseval.spawn(names[0], sweep_dir, 256, [0.0], 0.0, None), f"fseval base {names[0]}"))
    calls += [(probe.spawn(f"{sweep_dir}/{n}/checkpoint-200", "0.0"), f"probe {n}") for n in names]
    print(f"spawned {len(calls)} calls on {APP}")
    failed = []
    for call, tag in calls:
        try:
            r = call.get(timeout=4 * 3600)
            print(f"[{tag}] ok" + (f": {r}" if isinstance(r, str) else ""))
        except Exception as e:  # noqa: BLE001 — report and continue
            failed.append(tag)
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
    print(f"done: {len(calls) - len(failed)}/{len(calls)} ok" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
