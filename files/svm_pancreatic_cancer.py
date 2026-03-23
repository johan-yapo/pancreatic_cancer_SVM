import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Support Vector Machine (SVM) Machine Learning Model
# Designed by Dr. Johan A. Yapo 
# 6/25/2025

def configure_csv(file_path):
    file = open(file_path)
    csv_reader = csv.reader(file)
    features = next(csv_reader)
    print("The following features are in this dataset: ")
    counter = 0
    for f in features:
        print(f'({counter}) {f}')
        counter += 1

    feature_input = input("Which features would you like to use? Input this information separated by spaces in numerical order, like '0 1 2 3' : ")
    input_array = feature_input.split(" ").strip()

    class_input = input("Which output class would you like to use? ")
    return (input_array, int(class_input))

print("Machine Learning (ML) Model for Breast Cancer:")
print("ML has three main parts: training, testing, and validation.")
print("Inputs: CSV File Path, Features (or variables used to train the model)")
print("Outputs: Confusion Matrix (shows the accuracy of the model)")
print("\n\nNow, we will train the model.")

# Ask user for file path
# csv_file_path = 'machine_learning_data_template CA242 - Copy.csv'
csv_file_path = input("Please provide the path of the dataset, or CSV file: ")

# Load dataset
df = pd.read_csv(csv_file_path)
print("Original rows:", len(df))

# Clean labels
df["Class_Multimodal"] = df["Class_Multimodal"].astype(str).str.strip()

print("number of rows in the dataset: ", len(df))

print("Columns in the dataset:")
for i, col in enumerate(df.columns):
    print(f"({i}) {col}")
    # User selects feature columns and target column
feature_indices = input("Enter the indices of feature columns separated by space (e.g., 2 3 4): ").strip().split()
target_index = int(input("Enter the index of the target column: ").strip())


X = df.iloc[:, [int(i) for i in feature_indices]].values
y = df.iloc[:, target_index].astype(str).str.strip().values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)


print("Training the model...")
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=8)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()

from sklearn.inspection import permutation_importance

# Human-readable feature names from the indices you selected earlier
feature_names = [df.columns[int(i)] for i in feature_indices]

# Compute permutation importance on the held-out test set
perm = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=100,          # increase for more stable estimates
    random_state=8,
    n_jobs=-1,             # use all cores if available
    scoring="accuracy"     # change to 'f1_macro' if classes are imbalanced
)

# Build a table (DataFrame) of importances
imp_df = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": perm.importances_mean,
        "Importance_STD": perm.importances_std
    })
    .sort_values("Importance_Mean", ascending=False)
    .reset_index(drop=True)
)

# Print the table neatly
print("\nPermutation Feature Importances (higher = more important):")
print(imp_df.round(4).to_string(index=False))

# Plot the top-k features with matplotlib with 95% confidence intervals
n_repeats = perm.importances.shape[1]          # number of shuffles per feature
se = perm.importances_std / np.sqrt(n_repeats) # standard error of the mean
ci95 = 1.96 * se                               # 95% confidence interval width

top_k = min(10, len(feature_names))
top_imp = imp_df.head(top_k).copy()
top_imp["CI95"] = ci95[:top_k]  # align if you re-sort

plt.figure(figsize=(8,5))
plt.barh(top_imp["Feature"], top_imp["Importance_Mean"], xerr=top_imp["CI95"])
plt.gca().invert_yaxis()
plt.xlabel("Permutation importance (mean decrease in score)")
plt.title("Top Feature Importances with 95% CI")
plt.tight_layout()
plt.show()

#Dot + whisker plot with 95% CI
from math import sqrt

# assumes: imp_df (sorted by Importance_Mean desc), feature_names, perm from permutation_importance
top_k = min(10, len(imp_df))
top_imp = imp_df.head(top_k).copy()

# 95% CI half-width (normal approx)
n_repeats = perm.importances.shape[1]
ci_half = 1.96 * (perm.importances_std / sqrt(n_repeats))

# align CI to the sorted rows
feat_to_idx = {f: i for i, f in enumerate(feature_names)}
x = top_imp["Importance_Mean"].to_numpy()
xerr = top_imp["Feature"].map(lambda f: ci_half[feat_to_idx[f]]).to_numpy()
y = np.arange(len(top_imp))[::-1]  # top at top

plt.figure(figsize=(8,5))
plt.errorbar(x, y, xerr=xerr, fmt='o', capsize=3)   # dot + whisker
plt.yticks(y, top_imp["Feature"])
plt.axvline(0, linestyle='--', linewidth=1)         # “no effect” line
plt.xlabel("Permutation importance (mean decrease in score)")
plt.title("Top Feature Importances (95% CI)")
plt.tight_layout()
plt.show()

#Violin + mean + 95% CI whiskers
top_k = min(10, len(imp_df))
top_imp = imp_df.head(top_k).copy()
feat_to_idx = {f: i for i, f in enumerate(feature_names)}
order = [feat_to_idx[f] for f in top_imp["Feature"]]

# data: per-feature samples (one value per permutation repeat)
data = [perm.importances[i] for i in order]  # list of 1D arrays (len = n_repeats)

# mean and 95% CI
means = np.array([d.mean() for d in data])
n_repeats = perm.importances.shape[1]
stds  = np.array([d.std(ddof=1) for d in data])
ci_half = 1.96 * (stds / np.sqrt(n_repeats))

y = np.arange(top_k) + 1  # matplotlib's violin uses 1..N positions

plt.figure(figsize=(9,6))
parts = plt.violinplot(dataset=data, positions=y, vert=False, showmeans=False, showextrema=False, showmedians=False)

# overlay mean + 95% CI whiskers
plt.errorbar(means, y, xerr=ci_half, fmt='o', capsize=3)

plt.yticks(y, top_imp["Feature"])
plt.axvline(0, linestyle='--', linewidth=1)
plt.xlabel("Permutation importance (mean decrease in score)")
plt.title("Feature Importances Across Permutations (Violin + 95% CI)")
plt.tight_layout()
plt.show()


