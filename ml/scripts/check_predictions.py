"""Compare binary predictions with an answer key opened after prediction."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    args = parser.parse_args()
    predicted = pd.read_csv(args.predictions)
    answers = pd.read_csv(args.answers)
    if not {"record_id", "prediction"}.issubset(predicted.columns):
        raise ValueError("Prediction file needs record_id and prediction columns.")
    if not {"id", "label"}.issubset(answers.columns):
        raise ValueError("Answer key needs id and label columns.")
    compared = predicted.merge(answers, left_on="record_id", right_on="id", validate="one_to_one", how="left", indicator=True)
    if len(predicted) != len(answers) or not compared["_merge"].eq("both").all():
        raise ValueError("Predictions and answer key must contain exactly the same record IDs.")
    if not compared.label.isin([0, 1]).all() or not compared.prediction.isin([0, 1]).all():
        raise ValueError("Binary answers and predictions must use 0=Normal and 1=Attack.")
    compared["actual_label"] = compared.label.map({0: "Normal", 1: "Attack"})
    compared["correct"] = compared.prediction.eq(compared.label)
    show = ["record_id", "predicted_label", "actual_label", "correct"]
    if "attack_probability" in compared:
        show.insert(1, "attack_probability")
    elif "attack_score" in compared:
        show.insert(1, "attack_score")
    print(compared[show].round(4).to_string(index=False))
    attacks = compared.label.eq(1)
    normal = ~attacks
    print(f"\nCorrect decisions: {int(compared.correct.sum())}/{len(compared)}")
    print(f"Attacks caught: {int((attacks & compared.prediction.eq(1)).sum())}/{int(attacks.sum())}")
    print(f"Normal records falsely flagged: {int((normal & compared.prediction.eq(1)).sum())}/{int(normal.sum())}")
    print("This small sample demonstrates the process. Use the full test report to assess performance.")


if __name__ == "__main__":
    main()
