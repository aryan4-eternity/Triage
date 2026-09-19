"""
mape/execute.py — HarmonE MAPE-K: Execute stage.

BUG-1 FIX (T1.1): execute_drift() now receives a dict from plan_drift() that
contains the version DIRECTORY.  It resolves the actual model artifact
(.pth for lstm, .pkl for svm/linear) and copies THAT — never data.csv.
"""
import os
import shutil
import time
from pathlib import Path

if __package__:
    from mape.plan import plan_mape, plan_drift
else:
    from plan import plan_mape, plan_drift

ROOT       = Path(__file__).resolve().parents[1]
model_file = ROOT / "knowledge" / "model.csv"
models_dir = ROOT / "models"


def execute_mape():
    """Switch to the best model based on MAPE analysis."""
    decision = plan_mape()
    if not decision:
        print("MAPE: No action needed.")
        return

    print(f"⚡ Switching model to {decision.upper()}")
    with open(model_file, "w") as f:
        f.write(decision)


def execute_drift():
    """
    Replace model with the best archived version, or retrain.

    BUG-1 FIX: plan_drift() now returns a 'best_version' dict with
    {"dir": ..., "kl": ..., "model": ...}.  We resolve the artifact by
    name+extension from the directory instead of using data.csv as the path.
    """
    decision = plan_drift()
    if not decision:
        print("Drift: No action needed.")
        return

    if decision["action"] == "replace":
        best = decision["version"]          # dict: {dir, kl, model}
        version_dir  = Path(best["dir"])
        model_name   = best["model"]

        # Resolve artifact by name and correct extension
        ext = ".pth" if model_name == "lstm" else ".pkl"
        src = version_dir / f"{model_name}{ext}"

        if not src.exists():
            print(f"❌ Artifact not found: {src}. Falling back to retrain.")
            os.system(f"python {ROOT / 'retrain.py'}")
        else:
            dst = models_dir / f"{model_name}{ext}"
            shutil.copy(src, dst)
            print(f"✔ Restored {model_name}{ext} from {version_dir.name}  "
                  f"(KL={best['kl']:.4f})")

    elif decision["action"] == "retrain":
        print("🚀 Triggering retraining...")
        os.system(f"python \"{ROOT / 'retrain.py'}\"")

    # NOTE: cooldown moved to guard layer (BUG-6).
    # Keep sleep here only so the drift thread doesn't spin too fast;
    # this does NOT substitute for per-action cooldown guards.
    time.sleep(400)
