"""Measure CPU inference in fresh Linux processes; never train or change a model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def worker(model, full_test):
    import resource
    import statistics
    sys.path.insert(0, str(ROOT / "src"))
    import joblib
    from ids import predict_bundle, read_data
    stems = {"xgboost": "xgb_depth7", "randomforest": "rf_depth24"}
    names = list(stems) if model == "both" else [model]
    started = time.perf_counter()
    bundles = [joblib.load(ROOT / "models" / f"{stems[name]}.joblib") for name in names]
    load_seconds = time.perf_counter() - started
    frame = read_data(ROOT / "data" / ("raw/test.csv" if full_test else "simple_sample.csv"))
    if not full_test:
        frame = frame.head(1)
    for bundle in bundles:
        predict_bundle(bundle, frame)
    durations, cpu_durations = [], []
    for _ in range(3 if full_test else 20):
        started, cpu_started = time.perf_counter(), time.process_time()
        for bundle in bundles:
            predict_bundle(bundle, frame)
        durations.append(time.perf_counter() - started)
        cpu_durations.append(time.process_time() - cpu_started)
    print(json.dumps({
        "model": model, "records": len(frame), "repeats": len(durations),
        "model_load_seconds_excluding_imports": load_seconds,
        "median_predict_seconds_including_preprocessing": statistics.median(durations),
        "median_predict_cpu_seconds": statistics.median(cpu_durations),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "model_files_mib": sum((ROOT / "models" / f"{stems[name]}.joblib").stat().st_size for name in names) / 1024 ** 2,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-test", action="store_true", help="Also measure prediction on all 82,332 test rows")
    parser.add_argument("--worker", choices=["xgboost", "randomforest", "both"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.full_test)
        return
    reports = []
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    for full in ([False, True] if args.full_test else [False]):
        for model in ["xgboost", "randomforest", "both"]:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", model]
            if full:
                command.append("--full-test")
            started = time.perf_counter()
            result = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            report["whole_process_seconds_including_imports"] = time.perf_counter() - started
            reports.append(report)
            print(json.dumps(report), flush=True)
    destination = ROOT / "results" / "inference_resources.json"
    destination.write_text(json.dumps({
        "measured_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "platform": sys.platform, "threads": 2,
        "method": "Fresh process per case; peak Linux RSS includes interpreter, imports, model and input. Median warm preparation + prediction; no disk output. Not a minimum RAM requirement or a real-network benchmark.",
        "measurements": reports,
    }, indent=2) + "\n")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
