
#Implementation of ML models for classification tasks, including SVM with RBF kernel and neural network.
#The models are designed by Johan Yapo and based on the dissertation work of Dr. Haritha George

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
# ML model selection (svm_rbf or nn)
ML_model = 'svm_rbf'
# ML_model = 'nn'
# load data and label
data = np.loadtxt("data.txt")
label = np.loadtxt("label.txt")
print('data shape:', data.shape)
print('label shape:', label.shape)
# normalization, except for the dataset source indication
data[:, :-1] = scale(data[:, :-1])
# over 100 experiments
accuracy_train_list = []
accuracy_test_list = []
con_mat_list = []
accuracy_train_list_dataset0 = []
accuracy_test_list_dataset0 = []
con_mat_list_dataset0 = []