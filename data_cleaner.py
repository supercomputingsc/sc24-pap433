import os
import pickle
import pandas as pd

data = os.listdir('./raw_data/')

for file in data:
    PICKLE_FILE = str('./raw_data/'+str(file))
    Dataset = pd.DataFrame()
    unpickled_df = pd.read_pickle(PICKLE_FILE)
    # print(unpickled_df)
    name = ''
    file_name = name+file.replace('.data','')
    # print(file_name)
    data = pd.DataFrame(unpickled_df)
    directory = file_name
    parent_dir = "./raw_data/"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    data[0].to_csv(path+'/'+file_name+'_QuantumCircuit.csv')
    data[1].to_csv(path+'/'+file_name+'_noise.csv')
    data[2].to_csv(path+'/'+file_name+'_PST.csv')
    data.to_csv(path+'/'+file_name+'_All.csv')