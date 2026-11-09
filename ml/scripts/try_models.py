"""Try the two saved binary detectors, interactively or on your own CSV."""
from pathlib import Path
import argparse
from datetime import datetime
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd
from ids import CATEGORICAL, feature_frame, predict_bundle, read_data

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"xgboost": "xgb_depth7", "randomforest": "rf_depth24"}
SAMPLES = {
    "simple": ("simple_sample.csv", "simple_sample_answers.csv"),
    "errors": ("error_examples.csv", "error_examples_answers.csv"),
    "demo": ("demo_records.csv", "demo_answers.csv"),
}
HELP = """
Type a command, then press Enter:
  predict                 Ask both models about the current record (or press Enter)
  show                    Show all 42 measurements in the current record
  set dur 1.5             Change one measurement; use any name shown by 'show'
  set proto udp           Change a category; 'set dur missing' tries imputation
  row 4                   Select another row from the current sample (starts at 1)
  reset                   Undo edits to the current row
  answer                  Reveal the known answer for an UNEDITED sample
  sample simple           Use the five-record starter sample
  sample errors           Use deliberately chosen model mistakes (not an accuracy sample)
  sample demo             Use the 30-record demonstration sample
  load /path/my.csv        Load your CSV; use 'row N' to choose a record
  csv /path/my.csv         Predict EVERY row in your CSV and save both outputs
  save /path/input.csv    Save the current edited record as an input CSV
  help                    Show these commands again
  quit                    Exit
Paths with spaces need quotes, e.g. load "/home/krish/My data.csv".
Relative paths are resolved from the directory where you started the command.
"""


def load_models(choice):
    selected = MODELS if choice == "both" else {choice: MODELS[choice]}
    bundles = {}
    for name, stem in selected.items():
        bundles[name] = joblib.load(ROOT / "models" / f"{stem}.joblib")
    return bundles


def compare(bundles, frame, verbose=True):
    outputs = {}
    for name, bundle in bundles.items():
        started = time.perf_counter()
        result = predict_bundle(bundle, frame)
        seconds = time.perf_counter() - started
        outputs[name] = result
        if verbose:
            print(f"\n{name}: threshold {bundle['threshold']:.2f}; {seconds * 1000:.1f} ms including preparation")
            display = result.rename(columns={"attack_probability": "attack_score"})
            print(display.head(10).to_string(index=False, float_format=lambda v: f"{v:.8f}"))
            if len(result) > 10:
                print(f"Showing 10 of {len(result)} records; all predictions are saved in CSV mode.")
    if verbose:
        print("Score >= that model's threshold means Attack. Scores are not calibrated real-world risk percentages.")
    return outputs


def save_batch(bundles, path, output=None, answers=None):
    frame = read_data(path)
    if frame.empty:
        raise ValueError("Input CSV has no records.")
    outputs = compare(bundles, frame)
    destination = output or ROOT / "results" / "manual-tests" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination.mkdir(parents=True, exist_ok=True)
    for name, result in outputs.items():
        target = destination / f"{name}_predictions.csv"
        if target.exists():
            raise ValueError(f"Output exists; choose a new output directory: {target}")
    for name, result in outputs.items():
        target = destination / f"{name}_predictions.csv"
        result.to_csv(target, index=False)
        print(f"Saved {len(result)} {name} predictions: {target.resolve()}")
    if answers is not None:
        # Open the answer key only after every model has made and saved its guesses.
        for name in outputs:
            print(f"\nChecking {name} against the separate answer key:", flush=True)
            subprocess.run([
                sys.executable, str(ROOT / "scripts/check_predictions.py"),
                "--predictions", str(destination / f"{name}_predictions.csv"),
                "--answers", str(answers.expanduser()),
            ], check=True)


def interactive(bundles):
    source = read_data(ROOT / "data" / SAMPLES["simple"][0])
    answer_path = ROOT / "data" / SAMPLES["simple"][1]
    position = 0
    current = source.iloc[[position]].copy()
    edited = False
    columns = next(iter(bundles.values()))["feature_columns"]
    print("\nNetwork attack detector — saved models loaded; no training takes place.")
    print("Each input is measurements about one network communication, not an IP address or message.")
    print(HELP)
    compare(bundles, current)
    while True:
        try:
            text = input(f"\nrecord {position + 1}/{len(source)}{' [edited]' if edited else ''}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFinished.")
            return
        try:
            parts = shlex.split(text)
            command = parts[0].lower() if parts else "predict"
            arguments = parts[1:]
            if command in {"quit", "exit", "q"}:
                return
            if command == "help":
                print(HELP)
            elif command == "predict":
                compare(bundles, current)
            elif command == "show":
                if "id" in current:
                    print(f"Record ID (not a learning input): {current.iloc[0]['id']}")
                for column in columns:
                    print(f"  {column:22s} {current.iloc[0][column]}")
                print("Only these 42 measurements enter the model; id/label/attack_cat are excluded.")
            elif command == "set" and len(arguments) >= 2:
                column, value = arguments[0], " ".join(arguments[1:])
                if column not in columns:
                    raise ValueError("Choose one of the 42 measurement names shown by 'show'.")
                if value.lower() == "missing":
                    parsed = np.nan
                elif column in CATEGORICAL:
                    parsed = value
                else:
                    parsed = float(value)
                    if not np.isfinite(parsed) or parsed < 0:
                        raise ValueError("Use a finite non-negative measurement, or 'missing'.")
                proposed = current.copy()
                # Float columns can also represent missing values or decimal what-if inputs.
                if column not in CATEGORICAL:
                    proposed[column] = proposed[column].astype(float)
                proposed.loc[proposed.index[0], column] = parsed
                feature_frame(proposed, columns)
                current = proposed
                edited = True
                print(f"Changed {column} to {parsed}. Type predict to compare the models.")
                print("Edited measurements are a what-if experiment; the old answer no longer verifies this input.")
            elif command == "row" and len(arguments) == 1:
                index = int(arguments[0]) - 1
                if not 0 <= index < len(source):
                    raise ValueError(f"Choose a row from 1 to {len(source)}.")
                position = index
                current = source.iloc[[position]].copy()
                edited = False
                print("Selected row. Type show, predict or answer.")
            elif command == "reset":
                current = source.iloc[[position]].copy()
                edited = False
                print("Original row restored.")
            elif command == "answer":
                if edited:
                    print("This row was edited. Its true answer is unknown; reset first to check the original.")
                elif answer_path is None:
                    print("No separate verified answer key is attached to your input file.")
                else:
                    answers = read_data(answer_path)
                    answer = answers.loc[answers.id.eq(current.iloc[0]["id"])].iloc[0]
                    print(f"Known original answer: {'Attack' if answer['label'] == 1 else 'Normal'} ({answer['attack_cat']})")
            elif command == "sample" and len(arguments) == 1 and arguments[0] in SAMPLES:
                data_name, key_name = SAMPLES[arguments[0]]
                source = read_data(ROOT / "data" / data_name)
                answer_path = ROOT / "data" / key_name
                position = 0
                current = source.iloc[[position]].copy()
                edited = False
                print(f"Loaded {data_name}: {len(source)} rows. Type predict.")
                if arguments[0] == "errors":
                    print("These were deliberately chosen XGBoost mistakes; they do not estimate accuracy.")
            elif command == "load" and len(arguments) == 1:
                proposed = read_data(Path(arguments[0]).expanduser())
                if proposed.empty:
                    raise ValueError("Input CSV has no records.")
                feature_frame(proposed, columns)
                source = proposed.drop(columns=["label", "attack_cat"], errors="ignore")
                answer_path = None
                position = 0
                current = source.iloc[[position]].copy()
                edited = False
                print(f"Loaded {len(source)} rows; row 1 is selected. Type show or predict.")
            elif command == "csv" and len(arguments) == 1:
                save_batch(bundles, Path(arguments[0]).expanduser())
            elif command == "save" and len(arguments) == 1:
                destination = Path(arguments[0]).expanduser()
                if destination.exists():
                    raise ValueError("That file already exists. Choose a new filename.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                current.drop(columns=["label", "attack_cat"], errors="ignore").to_csv(destination, index=False)
                print(f"Saved current input: {destination.resolve()}")
            else:
                print("Command not understood. Type help for examples.")
        except (ValueError, OSError, KeyError, pd.errors.ParserError) as error:
            print(f"Cannot do that: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["both", *MODELS], default="both")
    parser.add_argument("--input", type=Path, help="Batch CSV mode; omit for the interactive terminal")
    parser.add_argument("--output-dir", type=Path, help="New directory for batch predictions")
    parser.add_argument("--answers", type=Path, help="Optional separate answer key; checked only after batch prediction")
    args = parser.parse_args()
    if args.output_dir and not args.input:
        parser.error("--output-dir requires --input")
    if args.answers and not args.input:
        parser.error("--answers requires --input")
    bundles = load_models(args.model)
    if args.input:
        try:
            save_batch(bundles, args.input.expanduser(), args.output_dir, args.answers)
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            parser.exit(1, f"Input error: {error}\n")
    else:
        interactive(bundles)


if __name__ == "__main__":
    main()
