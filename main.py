import os
import pickle
import pandas as pd
import numpy as np
from qiskit import *
from qiskit import qpy
import re

PICKLE_FILE = './raw_data/geneva.data'
Dataset = pd.DataFrame()
unpickled_df = pd.read_pickle(PICKLE_FILE)

data = pd.DataFrame(unpickled_df)

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

    circuit_depth.append(int(circuit.depth()))
    circuit_width.append(int(circuit.width()))
    circuit_op.append(circuit.count_ops())
    qubit_num.append(int(circuit.num_qubits))
    num_parameters.append(str(circuit.num_parameters))
    pst_score.append(float(data[2][index]))

    index = index + 1

dataset = pd.DataFrame(circuit_op)
dataset = dataset.fillna(0)
dataset = dataset.drop(columns=['barrier'])
dataset['PST'] = pst_score


import pandas as pd
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

dataset.replace('0', pd.NA, inplace=True)

target_col = 'PST'
x = dataset.drop(target_col, axis=1)
x1 =np.array(x)

y = dataset[target_col]*100
y2 = np.asarray(y, dtype="|S6")
x_train, x_test, y_train, y_test = train_test_split(x1, y2, test_size=0.3)

n_estimators = [64,128,200,500]
train_result=[]
test_result=[]
from sklearn.ensemble import *
from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression(multi_class='multinomial', random_state=46, max_iter=5000)
clf2 = RandomForestClassifier(n_estimators=1000, random_state=46)
clf3 = MLPClassifier(random_state=46, max_iter=5000,hidden_layer_sizes=10000).fit(x_train, y_train)

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(x_train,y_train)
print(eclf1.score(x_train,y_train))
np.array_equal(eclf1.named_estimators_.lr.predict(x_train),eclf1.named_estimators_['lr'].predict(x_train))
eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft')
eclf2 = eclf2.fit(x_train,y_train)
print(eclf2.score(x_train,y_train))



