"""Run a saved detector on a CSV. No retraining and no application server."""
from pathlib import Path
import sys, argparse, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
from ids import read_data, predict_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Default: binary champion selected using validation only")
    args = parser.parse_args()
    path = args.model
    if path is None:
        index = json.loads((ROOT / "models/model_index.json").read_text())
        path = ROOT / next(i["path"] for i in index if i["task"] == "binary" and i["champion"])
    bundle = joblib.load(path)
    frame = read_data(args.input)
    result = predict_bundle(bundle, frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Model: {bundle['name']}; records: {len(result)}; saved: {args.output}")
    print(result.predicted_label.value_counts().to_string())


if __name__ == "__main__":
    main()
