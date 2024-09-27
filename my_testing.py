import numpy as np

import my_setups
from scipy.stats import mode
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from ot import emd


def evaluate_knn(distMat, labels, K=5):
    knn = KNeighborsClassifier(n_neighbors=K, metric='precomputed')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracy_scores = cross_val_score(knn, distMat, labels, cv=skf, scoring='accuracy')
    
    return np.mean(accuracy_scores), np.std(accuracy_scores)


def knn_test_error(distances, labels, k):
    n = distances.shape[0]

    test_errors = 0
    for i in range(n):
        distances_to_i = distances[i]

        neighbor_indices = np.argsort(distances_to_i)[1: k + 1]
        neighbor_labels = labels[neighbor_indices]
        
        predicted_label = mode(neighbor_labels)[0]

        if predicted_label != labels[i]:
            test_errors += 1
    
    test_error_rate = test_errors / n
    return test_error_rate


def calculate_wasserstein_distance_mat(data, D):
    distances = np.zeros((data.shape[0], data.shape[0]))
    for i in range(len(data)):
        for j in range(i, len(data)):
            _, solutionDic = emd(data[i], data[j], D, log=True)
            distances[i, j] = distances[j, i] = solutionDic['cost']

    return distances

def calculate_final_distance_matrix(data, matrix):
    distMat = np.zeros((data.shape[0], data.shape[0]))
    for i in range(len(data)):
        for j in range(i, len(data)):
            distMat[i, j] = distMat[j, i] = np.sqrt((data[i] - data[j]).T @ matrix @ (data[i] - data[j]))

    return distMat


def run_tests(test_dataset, test_labels, method, EMD, method_arguments):
    x = method(**method_arguments)
    metric = x.metric
    time = x.time
    
    # use test data
    scores = []

    k_NN_mean_acc_across_data = []
    k_NN_std_acc_across_data = []
    error_rates_across_data = []

    if EMD:
        distances = calculate_wasserstein_distance_mat(test_dataset, metric)
    else:
        distances = calculate_final_distance_matrix(test_dataset, metric)

    scores.append(silhouette_score(distances, test_labels, metric='precomputed'))

    error_rates = [knn_test_error(distances, test_labels, i) for i in range(1, 21)]
    error_rates_across_data.append(error_rates)

    # run k-NN classification and calculate mean accuracy and standard deviation.
    k_NN_mean_acc = []
    k_NN_std_acc = []
    for k in range(1, 21):
        mean_acc, std_acc = evaluate_knn(distances, test_labels, K=k)
        k_NN_mean_acc.append(mean_acc)
        k_NN_std_acc.append(std_acc)

    dic = {'distances': metric, 
           'time': time,
           'ASW_scores': scores,
           'K-NN mean accuracies': k_NN_mean_acc,
           'K-NN std accuracies': k_NN_std_acc,
           'K-NN error rates': error_rates_across_data}

    return dic

def run_methods_on_dataset(partitioned_data, partitioned_labels, RBF_A, RBF_B):
    results = {}
    train_data, test_data = partitioned_data
    train_labels, test_labels = partitioned_labels

    results['Kernel'] = run_tests(test_dataset=test_data, test_labels=test_labels, method=my_setups.kernel_method, EMD=False, method_arguments={'data': train_data, 'RBF_A': RBF_A, 'RBF_B': RBF_B})
    results['Linear'] = run_tests(test_dataset=test_data, test_labels=test_labels, method=my_setups.linear_method, EMD=False, method_arguments={'data': train_data})
    results['LMNN'] = run_tests(test_dataset=test_data, test_labels=test_labels, method=my_setups.LMNN_method, EMD=False, method_arguments={'data': train_data, 'labels': train_labels})
    results['ITML'] = run_tests(test_dataset=test_data, test_labels=test_labels, method=my_setups.ITML_method, EMD=False, method_arguments={'data': train_data, 'labels': train_labels})
    results['L2'] = run_tests(test_dataset=test_data, test_labels=test_labels, method=my_setups.euclidean_method, EMD=False, method_arguments={'data': train_data, 'labels': train_labels})
    
    #results['u_GML'] = run_tests(test_datasets=partitioned_data, test_labels=partitioned_labels, EMD=True, method=my_setups.unsupervised_wasserstein_method, method_arguments={'data': train_data})
    #results['s_GML'] = run_tests(test_datasets=partitioned_data, test_labels=partitioned_labels, EMD=True, method=my_setups.supervised_wasserstein_method, method_arguments={'data': train_data, 'labels': train_labels})

    # Hellinger representation of histograms.
    hellinger_train_data, hellinger_test_data = [np.sqrt(data) for data in partitioned_data]
    results['Kernel-H'] = run_tests(test_dataset=hellinger_test_data, test_labels=test_labels, EMD=False, method=my_setups.kernel_method, method_arguments={'data': hellinger_train_data, 'RBF_A': RBF_A, 'RBF_B': RBF_B})
    results['Linear-H'] = run_tests(test_dataset=hellinger_test_data, test_labels=test_labels, EMD=False, method=my_setups.linear_method, method_arguments={'data': hellinger_train_data})
    results['LMNN-H'] = run_tests(test_dataset=hellinger_test_data, test_labels=test_labels, EMD=False, method=my_setups.LMNN_method, method_arguments={'data': hellinger_train_data, 'labels': train_labels})
    results['ITML-H'] = run_tests(test_dataset=hellinger_test_data, test_labels=test_labels, EMD=False, method=my_setups.ITML_method, method_arguments={'data': hellinger_train_data, 'labels': train_labels})
    results['L2-H'] = run_tests(test_dataset=hellinger_test_data, test_labels=test_labels, method=my_setups.euclidean_method, EMD=False, method_arguments={'data': hellinger_train_data, 'labels': train_labels})

    return results