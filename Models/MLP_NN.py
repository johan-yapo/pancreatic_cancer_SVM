# Neural Network (NN) Machine Learning Model
# Designed by Dr. Johan A. Yapo 
# 4/29/2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from collections import Counter

# -------------------------------
# CONFIG
# -------------------------------
N_RUNS = 100
TEST_SIZE = 0.3

print("MLP Neural Network Model with Hyperparameter Tuning")
print("Repeated 100-fold Validation + Grid Search\n")

# -------------------------------
# LOAD DATA
# -------------------------------
csv_file_path = input("Enter dataset path: ").strip()
df = pd.read_csv(csv_file_path)

print("\nColumns in dataset:")
for i, col in enumerate(df.columns):
    print(f"({i}) {col}")

# -------------------------------
# USER INPUT
# -------------------------------
feature_indices = input("\nEnter feature indices (e.g., 0 1 2): ").strip().split()
target_index = int(input("Enter target index: ").strip())

feature_indices = [int(i) for i in feature_indices]
feature_names = [df.columns[i] for i in feature_indices]

# -------------------------------
# PREP DATA
# -------------------------------
X = df.iloc[:, feature_indices].values
y = df.iloc[:, target_index].astype(str).str.strip().values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nClasses:", list(le.classes_))
print("Total samples:", len(X))

# -------------------------------
# STORAGE
# -------------------------------
accuracies = []
f1_scores = []
sensitivities = []
specificities = []
aucs = []
cms = []
best_params_list = []

# -------------------------------
# PARAM GRID
# -------------------------------
param_grid = {
    "hidden_layer_sizes": [(3,), (6,), (8,), (12,), (16,), (32,), (6, 3), (16, 8), (32,16), (8, 12, 6, 3)],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01, 1e-4, 1e-5],
}

# -------------------------------
# REPEATED TRAINING
# -------------------------------
for i in range(N_RUNS):
    print(f"\nRun {i+1}/{N_RUNS}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=i
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # GRID SEARCH (INNER LOOP)
    # -------------------------------
    base_model = MLPClassifier(
        activation='relu',
        solver='adam',
        max_iter=300,
        early_stopping=True,
        random_state=i
    )

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    best_params_list.append(grid.best_params_)

    print("Best params:", grid.best_params_)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # -------------------------------
    # METRICS
    # -------------------------------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    accuracies.append(acc)
    f1_scores.append(f1)

    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)

    # -------------------------------
    # SENSITIVITY & SPECIFICITY
    # -------------------------------
    sens_per_class = []
    spec_per_class = []

    for c in range(len(le.classes_)):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        sens_per_class.append(sensitivity)
        spec_per_class.append(specificity)

    sensitivities.append(np.mean(sens_per_class))
    specificities.append(np.mean(spec_per_class))

    # -------------------------------
    # AUC
    # -------------------------------
    y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))

    try:
        auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
    except:
        auc = np.nan

    aucs.append(auc)

# -------------------------------
# FINAL RESULTS
# -------------------------------
print("\n=== FINAL RESULTS (Tuned MLP, 100 runs) ===")
print(f"Accuracy:     {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"F1-score:     {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Sensitivity:  {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
print(f"Specificity:  {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")
print(f"AUC:          {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")

# -------------------------------
# MOST COMMON BEST PARAMETERS
# -------------------------------
print("\nMost common best parameters:")
param_counts = Counter([str(p) for p in best_params_list])
for k, v in param_counts.most_common():
    print(f"{k} → {v} times")

bestparameters = dict(Counter([tuple(sorted(p.items())) for p in best_params_list]).most_common(1)[0][0])

# -------------------------------
# AGGREGATED CONFUSION MATRIX
# -------------------------------
aggregated_cm = np.sum(cms, axis=0)

plt.figure(figsize=(6,5))
sns.heatmap(
    aggregated_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Aggregated Confusion Matrix (100 runs)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------
# NORMALIZED CONFUSION MATRIX
# -------------------------------
normalized_cm = aggregated_cm.astype(float) / aggregated_cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
sns.heatmap(
    normalized_cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

final_model = MLPClassifier(
    activation='relu',
    solver='adam',
    max_iter=300,
    early_stopping=True,
    random_state=42,
    **bestparameters
)

# scale full dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model.fit(X_scaled, y_encoded)

# -------------------------------
# Prediction on Unknown samples.
# -------------------------------


# LOAD Unknown DATA
print("\n=== Predict on UNKNOWNS Dataset ===")

use_new_data = input("Do you want to load a new dataset for prediction? (y/n): ").lower()

if use_new_data == 'y':

    new_path = input("Enter path to new dataset (CSV): ").strip()

    # Load file
    if new_path.endswith(".csv"):
        df_new = pd.read_csv(new_path)
    elif new_path.endswith(".xlsx"):
        df_new = pd.read_excel(new_path)
    else:
        raise ValueError("Unsupported file format.")
      
    print("\nColumns in new dataset:")
    print(list(df_new.columns))

    # -------------------------------
    # CHECK REQUIRED FEATURES
    # -------------------------------
    missing = [f for f in feature_names if f not in df_new.columns]

    if len(missing) > 0:
        print("\n❌ Missing required features:")
        print(missing)
        raise ValueError("New dataset does not contain required features.")

    # -------------------------------
    # EXTRACT FEATURES IN CORRECT ORDER
    # -------------------------------
    X_new = df_new[feature_names].values
    X_new_scaled = scaler.transform(X_new)

    # -------------------------------
    # PREDICT
    # -------------------------------
    preds = final_model.predict(X_new_scaled)
    probs = final_model.predict_proba(X_new_scaled)
    
    # -------------------------------
    # ADD RESULTS
    # -------------------------------
    df_new["Class_Multimodal"] = le.inverse_transform(preds)

    for i, cls in enumerate(le.classes_):
        df_new[f"Prob_{cls}"] = probs[:, i]

    # -------------------------------
    # SAVE OUTPUT
    # -------------------------------
    output_path = input("\nEnter output file path (e.g., results.csv): ").strip()

    # Ensure .csv extension
    if not output_path.lower().endswith(".csv"):
        output_path += ".csv"

    # Optional: prevent overwrite
    import os
    if os.path.exists(output_path):
        overwrite = input("File exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("❌ Save cancelled.")
        else:
            df_new.to_csv(output_path, index=False)
            print(f"\n✅ Predictions saved to: {output_path}")
    else:
        df_new.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved to: {output_path}")
else:    
    print("No new dataset loaded. Process complete.")