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

def create_table_1(pkl_data):

    for data in pkl_data:
        for i in range(len(data)):
            # Check if the item is nested within an additional list
            if isinstance(data[i], list) and len(data[i]) == 1 and isinstance(data[i][0], list):
                data[i] = data[i][0]

# Displaying the modified part for verification
    pkl_data = [item for sublist in pkl_data for item in sublist if isinstance(item, list)]
    print(pkl_data)
    print(len(pkl_data))

    table_1 = None
    
    return table_1


models = ['CNN', 'Resnet50_base', 'Resnet50_pretrained']
model = models[2]

table_1_pkl_dir = f'info_data_chooser\\{model}\\acc_prec_recall\\table_1_{model}_data'


table_1_pkl_data = unpack_pkl_files(table_1_pkl_dir)

table_1 = create_table_1(table_1_pkl_data)
# print(table_1)