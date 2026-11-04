"""Modal variant of tools/crossenv_countdown_fseval.py.

Scores checkpoints trained on another env (the anchor-only arm, trained on
leetcode_verified alone) under the countdown hf100 fseval protocol, taking the
eval env/prompts/detectors from a donor countdown run_config and the base
model from each run's own. One run per container, all in parallel; writes
/output/<sweep>_fseval/<run>.json, the file the hf100 figures read.

    .venv/bin/python -m modal run tools/modal_crossenv_fseval.py::sweep \
        --sweep-name countdown_hf100_lconly_newinit \
        --names cdhf100_lconly_full_lr3_s9,cdhf100_lconly_full_lr3_s15,cdhf100_lconly_full_lr3_s16 \
        --donor-rc countdown_hf100_gr_loreft_all/cdhf100_gr_lccoh64_lr3_s9/run_config.yaml
"""
from __future__ import annotations

from tools.modal_train_gr import OUTPUT_REMOTE, REPO_REMOTE, app, image, secrets, vol


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
              timeout=4 * 60 * 60)
def crossenv_one(sweep_name: str, run_name: str, donor_rc: str, scales: str = "1.0",
                 n_eval: int = 256) -> str:
    import subprocess

    cmd = [
        "python", "tools/crossenv_countdown_fseval.py",
        "--sweep_dir", f"{OUTPUT_REMOTE}/{sweep_name}",
        "--only", run_name,
        "--countdown_run_config", f"{OUTPUT_REMOTE}/{donor_rc}",
        "--out_dir", f"{OUTPUT_REMOTE}/{sweep_name}_fseval",
        "--n_eval", str(n_eval), "--scales", scales, "--overwrite",
    ]
    p = subprocess.run(cmd, cwd=REPO_REMOTE, capture_output=True, text=True)
    vol.commit()
    tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-20:])
    if p.returncode != 0:
        raise RuntimeError(f"{run_name} failed rc={p.returncode}:\n{tail}")
    return f"[{run_name}]\n{tail}"


@app.local_entrypoint()
def sweep(sweep_name: str, names: str, donor_rc: str, scales: str = "1.0", n_eval: int = 256):
    args = [(sweep_name, n, donor_rc, scales, n_eval) for n in names.split(",") if n]
    fail = 0
    for r in crossenv_one.starmap(args, return_exceptions=True):
        if isinstance(r, Exception):
            fail += 1
        print(r)
    assert not fail, f"{fail} crossenv fseval(s) failed"
