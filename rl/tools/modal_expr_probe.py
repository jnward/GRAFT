"""Modal variant of tools/expr_only_probe.py.

Runs the clean-environment (expr-only) probe for the hf50 arms of the hf50
appendix figure. Reuses modal_train_gr's image + gr-modal-pilot volume:
checkpoints already live on the volume, one probe per container, all
containers in parallel.

expr_only_probe rewrites the run_config's "/output/..." base-model path to
"/workspace/small-rl/output/..." (correct on the pods, wrong in the
container), so we symlink that prefix back onto /output before invoking it.

Single run:
    .venv/bin/python -m modal run tools/modal_expr_probe.py::single \
        --ckpt countdown_hf50_dn/cdhf50_dn_s9/checkpoint-200 --scales 1.0
Full hf50 battery (15 containers in parallel):
    .venv/bin/python -m modal run tools/modal_expr_probe.py::hf50_battery

Results land on the volume at /output/expr_only_probe_arms/ — sync with
    modal volume get gr-modal-pilot expr_only_probe_arms <dest> --force
"""
from __future__ import annotations

import modal

from tools.modal_train_gr import OUTPUT_REMOTE, REPO_REMOTE, app, image, vol


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, timeout=90 * 60)
def probe_one(ckpt_rel: str, scales: str, n: int = 256, k: int = 1,
              retain_scale: float = 1.0, name_suffix: str = "") -> str:
    import os
    import subprocess

    # Make the tool's pod-style base-model path resolve inside the container.
    os.makedirs("/workspace/small-rl", exist_ok=True)
    if not os.path.exists("/workspace/small-rl/output"):
        os.symlink(OUTPUT_REMOTE, "/workspace/small-rl/output")

    cmd = [
        "python", "tools/expr_only_probe.py",
        "--ckpt", f"{OUTPUT_REMOTE}/{ckpt_rel}",
        "--scales", scales, "--n", str(n), "--k", str(k),
        "--retain_scale", str(retain_scale),
        "--name_suffix", name_suffix,
        "--out", f"{OUTPUT_REMOTE}/expr_only_probe_arms",
    ]
    p = subprocess.run(cmd, cwd=REPO_REMOTE, capture_output=True, text=True,
                       timeout=80 * 60)
    vol.commit()
    tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-20:])
    if p.returncode != 0:
        raise RuntimeError(f"{ckpt_rel} failed rc={p.returncode}:\n{tail}")
    return f"[{ckpt_rel}]\n{tail}"


# (checkpoint dir on the volume, forget scale) — the 5 hf50 figure arms x 3
# seeds. Non-GR arms deploy with both adapters (scale 1.0, matching their
# fsevals); GRAFT deploys forget-ablated (0.0). The frozen base needs no run:
# hf50 trains from the same SFT checkpoint as hf100 and the probe prompts are
# identical, so the existing hf100 base probe applies.
HF50_RUNS = [
    (f"countdown_hf50_dn/cdhf50_dn_s{s}/checkpoint-200", "1.0")
    for s in (9, 15, 16)
] + [
    (f"countdown_hf50_rp_lccoh64/cdhf50_rp2_lc64_lr1_s{s}/checkpoint-200", "1.0")
    for s in (9, 15, 16)
] + [
    (f"countdown_hf50_ip/cdhf50_ip_mand-tw_s{s}/checkpoint-200", "1.0")
    for s in (9, 15, 16)
] + [
    (f"countdown_hf50_pps/cdhf50_pps_L20_a2_s{s}/checkpoint-200", "1.0")
    for s in (9, 15, 16)
] + [
    (f"countdown_hf50_gr_lccoh_lr3/cdhf50_gr_lccoh64_lr3_s{s}/checkpoint-200", "0.0")
    for s in (9, 15, 16)
]


def _collect(calls):
    """Wait on spawned (call, tag) probes, printing each result; returns the failed tags."""
    print(f"spawned {len(calls)} probes")
    failed = []
    for call, tag in calls:
        try:
            print(call.get(timeout=85 * 60))
        except Exception as e:  # noqa: BLE001 — report and continue
            failed.append(tag)
            print(f"[{tag}] FAILED: {e}")
    print(f"done: {len(calls) - len(failed)}/{len(calls)} ok"
          + (f"; failed: {failed}" if failed else ""))
    return failed


@app.local_entrypoint()
def single(ckpt: str, scales: str = "1.0", n: int = 256, k: int = 1,
           retain_scale: float = 1.0):
    """retain_scale 0.0 with --scales 0.0 = the frozen base model (writes <run>__r0.0.json)."""
    print(probe_one.remote(ckpt, scales, n=n, k=k, retain_scale=retain_scale))


@app.local_entrypoint()
def ckpt_series(sweep: str, runs: str, steps: str = "20,40,60,80,100,120,140,160,180,200"):
    """Every checkpoint of a GRAFT run in three configs, for a clean-environment
    solve rate over training (CR Fig. 2 right): retain 1 at forget 0.0 (deployed)
    and 1.0 (pre-ablation) in one container, writing <run>__ck<step>.json; and
    forget-only (retain 0, forget 1), writing <run>__ck<step>__r0.0.json.

        .venv/bin/python -m modal run tools/modal_expr_probe.py::ckpt_series \
            --sweep countdown_hf100_gr_ck20 \
            --runs cdhf100_gr_ck20_lr3_s9,cdhf100_gr_ck20_lr3_s15,cdhf100_gr_ck20_lr3_s16
    """
    jobs = [(f"{sweep}/{r}/checkpoint-{st}", sc, rs, f"__ck{st}")
            for r in runs.split(",") for st in steps.split(",")
            for sc, rs in (("0.0,1.0", 1.0), ("1.0", 0.0))]
    _collect([(probe_one.spawn(ck, sc, retain_scale=rs, name_suffix=suf), f"{ck} r{rs}")
              for ck, sc, rs, suf in jobs])


@app.local_entrypoint()
def hf50_battery():
    _collect([(probe_one.spawn(ckpt, sc), ckpt) for ckpt, sc in HF50_RUNS])


@app.local_entrypoint()
def batch(ckpts: str, scales: str = "1.0", retain_scale: float = 1.0):
    """Several checkpoints (comma-separated <sweep>/<run>/checkpoint-N) in one app,
    one container each: a wave of `single` runs starts an ephemeral app per probe,
    and Modal refuses new apps past 100 ("reached limit of 100 ephemeral apps")."""
    failed = _collect([(probe_one.spawn(ck, scales, retain_scale=retain_scale), ck)
                       for ck in ckpts.split(",")])
    if failed:
        raise SystemExit(1)
