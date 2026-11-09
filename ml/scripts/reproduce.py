"""Reproduce the complete experiment in a fresh directory, preserving this run."""
import argparse, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New non-existing directory")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError("Choose a new output directory; existing experiments are preserved.")
    for name in ["src", "scripts", "config"]:
        shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "data/raw", target / "data/raw")
    shutil.copyfile(ROOT / "data/source_manifest.json", target / "data/source_manifest.json")
    for name in ["requirements.txt", "requirements-lock.txt", "METHODS.md", "DATA_GUIDE.md", "run_example.sh"]:
        shutil.copyfile(ROOT / name, target / name)
    for name in ["models", "results", "figures", "notebooks"]:
        (target / name).mkdir(exist_ok=True)
    for script in ["audit.py", "train.py", "stability.py", "evaluate.py", "export_prepared.py", "verify.py", "make_tutorial.py", "make_report.py", "make_notebook.py"]:
        print("RUN", script, flush=True)
        subprocess.run([sys.executable, str(target / "scripts" / script)], check=True)
    print("Reproduced experiment:", target)


if __name__ == "__main__":
    main()
