import os
import pickle
import pandas as pd
import numpy as np
from qiskit import *
from qiskit import qpy
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

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
# dataset.to_csv('./raw_data/toronto.csv')
import matplotlib.pyplot as plt

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(dataset)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
# plt.show()

kmeanModel = KMeans(n_clusters=4)
kmeanModel.fit(dataset)

dataset['k_means']=kmeanModel.predict(dataset)
dataset['target']=dataset['PST']
print(dataset.head())

fig, axes = plt.subplots(1, 2, figsize=(16,8))
# axes[0].scatter(dataset['PST'], dataset['k_means'], c=dataset['target'])
# axes[1].scatter(dataset['PST'], dataset['k_means'], c=dataset['target'], cmap=plt.cm.Set1)
axes[0].scatter(dataset['rz'], dataset['cx'], c=dataset['target'])
axes[1].scatter(dataset['x'], dataset['cx'], c=dataset['k_means'], cmap=plt.cm.Set1)

axes[0].scatter(dataset['sx'], dataset['cx'], c=dataset['target'])
axes[1].scatter(dataset['rz'], dataset['sx'], c=dataset['k_means'], cmap=plt.cm.Set1)

axes[0].scatter(dataset['rz'], dataset['x'], c=dataset['target'])
axes[1].scatter(dataset['sx'], dataset['x'], c=dataset['k_means'], cmap=plt.cm.Set1)


axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('K_Means', fontsize=18)
plt.show()

# plt.matshow(dataset.corr())
# plt.show()