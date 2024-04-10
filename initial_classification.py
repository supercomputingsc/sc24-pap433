import os
import pickle
import pandas as pd
import numpy as np
from qiskit import *
from qiskit import qpy
import re

PICKLE_FILE = './raw_data/geneva.data'
Dataset = pd.DataFrame()
# pd.set_option('display.max_rows', None)
unpickled_df = pd.read_pickle(PICKLE_FILE)
# print(unpickled_df)
data = pd.DataFrame(unpickled_df)
# print(data[0])
Q_Circuits = pd.DataFrame()
index = 0
circuit_depth = []
circuit_op = []
qubit_num = []
num_parameters = []
circuit_width =[]
pst_score=[]
index = 0
for circuit in data[0]:
    each_circuit =[]
    # circuit as QuantumCircuit.__sizeof__()
    circuit_depth.append(int(circuit.depth()))
    circuit_width.append(int(circuit.width()))
    circuit_op.append(circuit.count_ops())
    qubit_num.append(int(circuit.num_qubits))
    num_parameters.append(str(circuit.num_parameters))
    pst_score.append(float(data[2][index]))
    # print(data[2][index])
    index = index + 1
    # print(circuit.__sizeof__())
print(circuit_depth)
print(circuit_op)
print("Circuit Depth: {}".format(np.mean((circuit_depth))))
print("Circuit Width: {}".format(np.mean((circuit_width))))
print("Circuit PST Score: {}".format(np.mean(pst_score)))
print("Circuit Qubits: {}".format(np.mean(qubit_num)))
dataset = pd.DataFrame(circuit_op)
dataset = dataset.fillna(0)
dataset = dataset.drop(columns=['barrier'])

dataset['PST'] = pst_score
dataset['PST'] = dataset['PST']*100
# dataset.to_csv('./raw_data/geneva.csv')
for i, row in dataset.iterrows():
    #df_merged.loc[i, 'price_new'] = i
    if i > 90:
        dataset.loc[i, 'PST'] = 100
    if i < 90 and i > 80:
        dataset.loc[i, 'PST'] = 80
    if i < 80 and i > 60:
        dataset.loc[i, 'PST']  = 70
    if i < 60 and i > 40:
        dataset.loc[i, 'PST']  = 50
    if i < 40 and i > 20:
        dataset.loc[i, 'PST']  = 30
    if i < 20:
        dataset.loc[i, 'PST']  = 0
print(dataset['PST'])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
# from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
# print(dataset.shape)
dataset.replace('0', pd.NA, inplace=True)
# print(dataset.head())
target_col = 'PST'
x = dataset.drop(target_col, axis=1)
x1 =np.array(x)
# print(dataset)
# print(x)
# dataset= dataset[target_col]*100
y = (dataset[target_col]*100).astype(int)
# print(y)
# y2 = np.asarray(y, dtype="|S6")
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.3)

####################################################################################################
####################################################################################################

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "DecisionTree",
    "Gaussian Process",
]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    RandomForestClassifier(
        max_depth=None, n_estimators=1024, max_features=5, random_state=42
    ),
    DecisionTreeClassifier(max_depth=15, random_state=42),

    MLPClassifier(alpha=1, max_iter=10000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# Initialize and train the model

for name, clf in zip(names, classifiers):
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(x_train, y_train)

    # Predict on the test data
    y_pred = clf.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Display the results
    print("Model: ", name)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("======############============")


####################################################################################################
####################################################################################################

# n_estimators = [64,128,200,500]
# train_result=[]
# test_result=[]
# from sklearn.ensemble import *
# from sklearn.linear_model import LogisticRegression
# # for estimator in n_estimators:
#     # for cir in criterion:
# clf1 = LogisticRegression(multi_class='multinomial', random_state=46)
# clf2 = RandomForestClassifier(n_estimators=1000, random_state=46)
# clf3 = MLPClassifier(random_state=46, max_iter=100000,hidden_layer_sizes=10000).fit(x_train, y_train)
# # rf = GradientBoostingClassifier(n_estimators=estimator)
# eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
# eclf1 = eclf1.fit(x_train,y_train)
# print(eclf1.score(x_train,y_train))
# np.array_equal(eclf1.named_estimators_.lr.predict(x_train),eclf1.named_estimators_['lr'].predict(x_train))
# eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft')
# eclf2 = eclf2.fit(x_train,y_train)
# print(eclf2.score(x_train,y_train))
