# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import seaborn as sns
import csv

# Support Vector Machine (SVM) Machine Learning Model
# Designed by Dr. Remya and Sreyansh Mamidi
# 7/29/2022

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
    input_array = feature_input.split(" ")

    class_input = input("Which output class would you like to use? ")
    return (input_array, int(class_input))


print("Machine Learning (ML) Model for Breast Cancer:")
print("ML has three main parts: training, testing, and validation.")
print("Inputs: CSV File Path, Features (or variables used to train the model)")
print("Outputs: Confusion Matrix (shows the accuracy of the model)")
print("\n\nNow, we will train the model.")

csv_file_path = input("Please provide the path of the dataset, or CSV file: ")
# csv_file_path = 'machine_learning_data_template CA242 - Copy.csv'
feature_tuple = configure_csv(csv_file_path)

# INPUT: Need to change the path of the CSV file depending on the biomarker being used
dataset = pd.read_csv(csv_file_path)
array = dataset.values

# INPUT: Need to change the values of X and Y depending on the dataset 
Y = array[:, feature_tuple[1]]   # Select class

# Select features
X = [] 
for row in array:
    if len(feature_tuple[0]) == 0:
        X = array[:, int(feature_tuple[0][0])]
        break
    temp = []
    for f in feature_tuple[0]:
        temp.append(row[int(f)])
    X.append(temp)

validation_size = 0.2
seed = 43

print("Training the model...")
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# SVC Model
SVM = SVC(kernel='rbf')
print("Testing the model...")
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)

# Performance assessment
print("Validating the results...")
acc_score = accuracy_score(Y_validation, predictions)

# Creates an Excel file with the validation data and predictions
print("Y_Validation:")
print(Y_validation)
print("Predictions:")
print(predictions)
data_dict = {}
data_dict["Y_Validation"] = Y_validation
data_dict["Predictions"] = Y_validation
df = pd.DataFrame(data_dict)
df.to_csv("Validation and Prediction Data.csv")
print("The validation and prediction data has been added to an Excel file.")

cm = confusion_matrix(Y_validation, predictions, labels=["Normal", "Low Risk", "High Risk"])
# cm = confusion_matrix(Y_validation, predictions, labels=["Benign", "Malignant"])

print("Accuracy of SVM model : {}".format(format(acc_score, '.2%')))

plt.figure()
sns.heatmap(cm, annot=True, cmap='YlGnBu', xticklabels=["Normal", "Low Risk", "High Risk"], yticklabels=["Normal", "Low Risk", "High Risk"])
plt.show()
