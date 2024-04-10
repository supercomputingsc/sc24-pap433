import os
import pickle
import pandas as pd
import numpy as np
from qiskit import *
from qiskit import qpy
import re
import matplotlib.pyplot as plt
# import seaborn as sns

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
# print(circuit_depth)
# print(circuit_op)
# print("Circuit Depth: {}".format(np.mean((circuit_depth))))
# print("Circuit Width: {}".format(np.mean((circuit_width))))
# print("Circuit PST Score: {}".format(np.mean(pst_score)))
# print("Circuit Qubits: {}".format(np.mean(qubit_num)))

df = pd.DataFrame(pst_score,columns=["PST"])
# print(df)
df.hist(bins=100)
# plt.show()
# df.hist()
plt.show()