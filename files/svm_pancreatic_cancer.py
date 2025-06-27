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

# CLEANING: Extract float from 'Type' if present
cols_list = df.columns.tolist()

if 'Type' in df.columns:
    df['Type_float (ng/ml)'] = df['Type'].str.extract(r'(\d+\.?\d*)').astype(float)
    print("\nExtracted numeric values from 'Type' into new column 'Type_float (ng/ml)'")

# Drop 'Type_float (ng/ml)' from numeric columns if it exists (because we're grouping by it)
numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if col != 'Type_float (ng/ml)']

df = df.groupby("Type_float (ng/ml)").agg({
    "Rsol": "mean",
    "Rp": "mean",
    "CPE": "mean",
    "CV_Area": "mean",
    "Delta_Z": "mean",
    "Class_Multimodal": lambda x: x.mode().iloc[0]  # Take the most common class
}).reset_index()

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