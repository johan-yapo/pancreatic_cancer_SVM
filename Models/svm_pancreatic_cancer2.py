import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Support Vector Machine (SVM) Machine Learning Model
# Designed by Dr. Johan A. Yapo 
# 04/09/2026

# -------------------------------
# TRAIN MODEL + RETURN CM
# -------------------------------
def train_and_evaluate(csv_file_path):
    df = pd.read_csv(csv_file_path)

    print(f"\nDataset: {csv_file_path}")
    print("Columns:")
    for i, col in enumerate(df.columns):
        print(f"({i}) {col}")

    feature_indices = input("Enter feature indices: ").strip().split()
    target_index = int(input("Enter target index: ").strip())

    X = df.iloc[:, [int(i) for i in feature_indices]].values
    y = df.iloc[:, target_index].astype(str).str.strip().values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=8
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return cm, le.classes_, csv_file_path


# -------------------------------
# MAIN LOOP
# -------------------------------
cms = []
labels_list = []
titles = []

print("\nEnter dataset paths (type 'q' to stop):")

while True:
    path = input("\nDataset path: ")
    if path.lower() == 'q':
        break

    cm, labels, title = train_and_evaluate(path)

    cms.append(cm)
    labels_list.append(labels)
    titles.append(title)


# -------------------------------
# PLOT ALL CONFUSION MATRICES
# -------------------------------
n = len(cms)

if n == 0:
    print("No datasets entered.")
else:
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(5 * cols, 4 * rows))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)

        sns.heatmap(
            cms[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels_list[i],
            yticklabels=labels_list[i]
        )

        plt.title(f"Dataset {i+1}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()