import os
import pickle as pkl
import numpy as np
import pandas as pd


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

    for data in pkl_data:
        for i in range(len(data)):
            # Check if the item is nested within an additional list
            if isinstance(data[i], list) and len(data[i]) == 1 and isinstance(data[i][0], list):
                data[i] = data[i][0]

    pkl_data = [item for sublist in pkl_data for item in sublist if isinstance(item, list)]

    return pkl_data

def create_table_1(pkl_data):
    pkl_data = sorted(pkl_data, key=lambda x: x[0])
    print(pkl_data)
    print(len(pkl_data))
    
    table_1_data = {
        'Augmented samples per class' : [],
        'Run 1 acc' : [],
        'Run 2 acc' : [],
        'Run 3 acc' : [],
        'Run 4 acc' : [],
        'Run 5 acc' : [],
        'Avg' : [],
        'Max' : []
    }

    # Grouping the data by the number of samples
    grouped_data = {}
    for num_samples, metrics in pkl_data:
        if num_samples not in grouped_data:
            grouped_data[num_samples] = []
        # Extracting only the first element (accuracy)
        acc = metrics[0]
        grouped_data[num_samples].append(acc)

    # Populate the table_1_data
    for num_samples, acc_list in grouped_data.items():
        table_1_data['Augmented samples per class'].append(num_samples)
        # Calculate and add average and max accuracy for each group
        avg_acc = sum(acc_list) / len(acc_list)
        max_acc = max(acc_list)
        table_1_data['Avg'].append(avg_acc)
        table_1_data['Max'].append(max_acc)
        # Add individual run accuracies
        for i in range(5):
            run_acc = acc_list[i::5]  # Get every 5th element starting from i
            table_1_data[f'Run {i+1} acc'].append(run_acc[0] if run_acc else None)

    df = pd.DataFrame(table_1_data)

    return df


def create_table_2(pkl_data):
    pkl_data = sorted(pkl_data, key=lambda x: x[0])
    print(pkl_data)
    print(len(pkl_data))
    
    table_1_data = {
        'Augmented samples per class' : [],
        'Run 1 acc' : [],
        'Run 2 acc' : [],
        'Run 3 acc' : [],
        'Run 4 acc' : [],
        'Run 5 acc' : [],
        'Avg' : [],
        'Max' : []
    }

    # Grouping the data by the number of samples
    grouped_data = {}
    for num_samples, metrics in pkl_data:
        if num_samples not in grouped_data:
            grouped_data[num_samples] = []
        # Extracting only the first element (accuracy)
        acc = metrics[0]
        grouped_data[num_samples].append(acc)

    # Populate the table_1_data
    for num_samples, acc_list in grouped_data.items():
        table_1_data['Augmented samples per class'].append(num_samples)
        # Calculate and add average and max accuracy for each group
        avg_acc = sum(acc_list) / len(acc_list)
        max_acc = max(acc_list)
        table_1_data['Avg'].append(avg_acc)
        table_1_data['Max'].append(max_acc)
        # Add individual run accuracies
        for i in range(5):
            run_acc = acc_list[i::5]  # Get every 5th element starting from i
            table_1_data[f'Run {i+1} acc'].append(run_acc[0] if run_acc else None)

    df = pd.DataFrame(table_1_data)

    return df

models = ['CNN', 'Resnet50_base', 'Resnet50_pretrained']
model = models[2]
table_1_pkl_dir = f'info_data_chooser\\{model}\\acc_prec_recall\\table_1_{model}_data'
table_2_pkl_dir = f'info_data_chooser\\{model}\\acc_prec_recall'




# table_1_pkl_data = unpack_pkl_files(table_1_pkl_dir)
# table_1 = create_table_1(table_1_pkl_data)
# print(table_1)

table_2_pkl_data = unpack_pkl_files(table_2_pkl_dir)
print(table_2_pkl_data)

