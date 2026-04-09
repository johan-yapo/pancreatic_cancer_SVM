import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import os
from itertools import combinations

# Support Vector Machine (SVM) with Leave-One-Out Cross-Validation (LOOCV)
# Automated feature combination runner
# Designed for electrochemical biosensor biomarker risk classification
# Developed by Agustín Rodríguez-Andraca García


def configure_csv(file_path):
    """
    Reads the CSV headers and prompts the user to select the class column
    and feature columns, indexed from 1.
    Returns: headers (list of str), class_index (int), feature_indices (list of int)
    """
    headers = list(pd.read_csv(file_path, nrows=0).columns)

    print("\nThe following columns are in this dataset:")
    for i, h in enumerate(headers):
        print(f"  ({i + 1}) {h}")

    class_index = int(input(
        "\nWhich column is the output class (risk label)? Enter its number: "
    ).strip()) - 1

    feature_indices = [int(s) - 1 for s in input(
        "Which columns are your features? "
        "Enter column numbers separated by spaces (e.g. '1 2 3 4 5'): "
    ).strip().split(" ")]

    return headers, class_index, feature_indices


def build_combo_label(feature_names):
    """Returns a human-readable label with 'and' before the last feature."""
    if len(feature_names) == 1:
        return feature_names[0]
    elif len(feature_names) == 2:
        return f"{feature_names[0]} and {feature_names[1]}"
    else:
        return ", ".join(feature_names[:-1]) + f", and {feature_names[-1]}"


def run_loocv(X_subset, Y, combo_label, output_dir, class_labels):
    """
    Runs LOOCV for a given feature subset.
    Saves a results CSV and confusion matrix PNG.
    Returns accuracy, macro F1, true labels, and predictions.
    """
    loo = LeaveOneOut()
    scaler = StandardScaler()
    SVM = SVC(kernel='rbf')

    all_true = []
    all_predictions = []

    for train_index, test_index in loo.split(X_subset):
        X_train, X_test = X_subset[train_index], X_subset[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        SVM.fit(X_train_scaled, Y_train)
        all_true.append(Y_test[0])
        all_predictions.append(SVM.predict(X_test_scaled)[0])

    acc = accuracy_score(all_true, all_predictions)
    f1 = f1_score(all_true, all_predictions, average='macro', zero_division=0)

    pd.DataFrame({
        "Sample Index": range(1, len(all_true) + 1),
        "True Label": all_true,
        "Predicted Label": all_predictions,
        "Correct": [t == p for t, p in zip(all_true, all_predictions)]
    }).to_csv(os.path.join(output_dir, f"Results by {combo_label}.csv"), index=False)

    cm = confusion_matrix(all_true, all_predictions, labels=class_labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(
        f"Confusion Matrix — {combo_label}\n"
        f"LOOCV Accuracy: {format(acc, '.2%')}  |  Macro F1: {f1:.4f}",
        fontsize=12
    )
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Confusion Matrix by {combo_label}.png"), dpi=150)
    plt.close()

    return acc, f1, all_true, all_predictions


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Electrochemical Biosensor Risk Classification")
print("  SVM + LOOCV | Automated Feature Combination Runner")
print("=" * 60)

csv_file_path = input("\nPlease provide the path of the CSV dataset file: ").strip()
headers, class_index, feature_indices = configure_csv(csv_file_path)

data = pd.read_csv(csv_file_path).values
feature_names = [headers[i] for i in feature_indices]
n_features = len(feature_names)

X_full = np.array([[row[i] for i in feature_indices] for row in data], dtype=float)
Y = data[:, class_index]

class_labels = ["Normal", "Low Risk", "High Risk"]

output_dir = "Biosensor Classification Results"
os.makedirs(output_dir, exist_ok=True)
print(f"\nAll output files will be saved to: ./{output_dir}/")

all_combos = [
    combo
    for size in range(1, n_features + 1)
    for combo in combinations(range(n_features), size)
]

total = len(all_combos)
print(f"\nRunning all {total} feature combinations (sizes 1 through {n_features})")
print("─" * 60)

summary_rows = []

for combo_num, combo in enumerate(all_combos, start=1):
    selected_names = [feature_names[i] for i in combo]
    combo_label = build_combo_label(selected_names)
    print(f"[{combo_num:>2}/{total}] Features: {combo_label}")

    acc, f1, all_true, all_preds = run_loocv(
        X_full[:, combo], Y, combo_label, output_dir, class_labels
    )

    print(f"         Accuracy: {format(acc, '.2%')}  |  Macro F1: {f1:.4f}")
    print(classification_report(all_true, all_preds, target_names=class_labels, zero_division=0))

    summary_rows.append({
        "Combination Number": combo_num,
        "Features Used": combo_label,
        "Number of Features": len(combo),
        "LOOCV Accuracy": format(acc, '.2%'),
        "Macro F1": round(f1, 4)
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["Macro F1", "LOOCV Accuracy"], ascending=[False, False]
)
summary_path = os.path.join(output_dir, "Summary of All Combinations.csv")
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("  SUMMARY — All Combinations Ranked by Macro F1, then Accuracy")
print("=" * 60)
print(summary_df.to_string(index=False))
print(f"\nSummary saved to: {summary_path}")
print(f"All files saved to: ./{output_dir}/")
