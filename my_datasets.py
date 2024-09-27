import numpy as np
import pandas as pd
import random
import os

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
#from keras.datasets import mnist

# Import the MNIST dataset and optionally reduce image dimensions by PCA. 
def MNIST_data(sample_size, filter_list):
    (data, labels), _ = mnist.load_data()

    if filter_list:
        data, labels = filter_out(data, labels, filter_list)

    data = data[:sample_size]

    i, j, k = data.shape
    return data.reshape(i, j*k), labels[:sample_size]

# Just use the images specified by filter_list
def filter_out(data, labels, filter_list):
    new_data = []
    new_labels = []
    for i in range(len(data)):
        if labels[i] in filter_list:
            new_data.append(data[i])
            new_labels.append(labels[i])

    return np.array(new_data), new_labels

            
def dimensionality_reduction(data, method, normalize=False, kernel=None, target_dim=50):
    if kernel:
        method_class = method(n_components=target_dim, kernel=kernel, gamma=10)
    else:
        method_class = method(n_components=target_dim)

    scaler = MinMaxScaler()

    transformed = scaler.fit_transform(method_class.fit_transform(data))

    if normalize:
        return np.array([datapoint / np.linalg.norm(datapoint, ord=1) for datapoint in transformed])
    return  transformed 

def data_to_train_and_test(data, labels, partition_size=50):
    print(np.array(labels).shape)
    zipped = list(zip(data, labels))

    # select 300 random samples and their corresponding labels
    np.random.seed(1)
    np.random.shuffle(zipped)
    zipped = zipped[:300]

    # split into 10 datasets of length 30
    partitioned_zipped = [zipped[i: i + partition_size] for i in range(0, len(zipped), partition_size)]
    partitioned_data = []
    partitioned_labels = []
    for partition in partitioned_zipped:
        data = []
        labels = []
        for zippy in partition:
            data.append(zippy[0])
            labels.append(zippy[1])

        partitioned_data.append(data)
        partitioned_labels.append(labels)
    
    return np.array(partitioned_data), np.array(partitioned_labels)

def load_apple_quality_data(working_dir):
    with open(f"{working_dir}/datasets/apple_quality.csv") as data_csv:
        data_csv = data_csv.readlines()
        data = []
        labels = []
        
        # skip first line (feature labels) and last line (author)
        for row in data_csv[1:-1]:
            row_data = row.strip().split(',')
            
            # exclude id (first entry) and rating (last entry)
            data.append(np.array(row_data[1:-1], dtype=float))
            labels.append(row_data[-1])

        scaler = MinMaxScaler()
        scaler.fit(np.array(data))
        data = scaler.transform(np.array(data))
        data = np.array([datapoint / np.linalg.norm(datapoint, ord=1) for datapoint in data])

        labels = np.array([1 if label == 'good' else 0 for label in labels])

    shuffle_indices = np.random.permutation(len(labels))
    data_shuffled = data[shuffle_indices, :]
    labels_shuffled = labels[shuffle_indices]
    data, labels = data_to_train_and_test(data_shuffled, labels_shuffled)
    return data, labels

def load_wine_quality_data(working_dir):
    with open(f"{working_dir}/datasets/wineQT.csv") as data_csv:
        data_csv = data_csv.readlines()
        data = []
        labels = []
        
        # skip first line (feature labels) and last line (author)
        for row in data_csv[1:]:
            row_data = row.strip().split(',')
            
            # exclude id (first entry) and rating (last entry)
            data.append(np.array(row_data[:-2], dtype=float))
            label = int(row_data[-2])
            labels.append(label)

        scaler = MinMaxScaler()
        scaler.fit(np.array(data))
        data = scaler.transform(np.array(data))
        data = np.array([datapoint / np.linalg.norm(datapoint, ord=1) for datapoint in data])


    data, labels = data_to_train_and_test(data, labels)
    return data, labels

    
def load_caltech_data(working_dir, class_amount):

    #select one of 50 first classe
    folders = sorted(os.listdir(working_dir))[:89]
    chosen_classes = random.sample(folders, class_amount)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i, class_folder in enumerate(chosen_classes):
        # Define the path to the class folder
        class_folder_path = os.path.join(working_dir, class_folder)
        
        # List all images in the class folder
        all_files = os.listdir(class_folder_path)

        feather_files = [f for f in all_files if f.endswith('.feather')]


        histograms = []
        for f in feather_files:
            path = working_dir / class_folder / f
            df = pd.read_feather(path)
            image_features = df.iloc[0].values
            normalized = image_features / np.linalg.norm(image_features, ord=1)
            histograms.append(normalized)
        histograms = np.array(histograms)

        random_indices = np.random.choice(len(histograms), size=80, replace=False)
        train_indices = random_indices[:30]
        test_indices = random_indices[30:]

        train_data.append(histograms[train_indices])
        test_data.append(histograms[test_indices])

        train_labels += [i for _ in range(30)]
        test_labels += [i for _ in range(50)]

    train_data = np.array(train_data).reshape(-1, 128)
    test_data = np.array(test_data).reshape(-1, 128)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    shuffle_train_indices = np.random.permutation(len(train_data))
    train_data_shuffled = np.array(train_data)[shuffle_train_indices]
    train_labels_shuffled = np.array(train_labels)[shuffle_train_indices]

    shuffle_test_indices = np.random.permutation(len(test_data))
    test_data_shuffled = np.array(test_data)[shuffle_test_indices]
    test_labels_shuffled = np.array(test_labels)[shuffle_test_indices]
    
    data = [train_data_shuffled, test_data_shuffled]
    return data, [train_labels_shuffled, test_labels_shuffled]