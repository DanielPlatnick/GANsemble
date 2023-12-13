import os
import pickle as pkl
import numpy as np



def unpack_pkl_files(pkl_data_dir):

    pkl_data = []

    for filename in os.listdir(pkl_data_dir):
        # Check if the file is a pickle file
        if filename.endswith('.pkl'):
            # Construct the full file path
            filepath = os.path.join(pkl_data_dir, filename)
            
            # Open and unpickle the file, then append its contents to the list
            with open(filepath, 'rb') as f:
                file_contents = pkl.load(f)
                pkl_data.append(file_contents)

    return pkl_data


table_1_pkl_dir = 'info_data_chooser\\Resnet50_base\\acc_prec_recall'

table_1_data = unpack_pkl_files(table_1_pkl_dir)

print(table_1_data)