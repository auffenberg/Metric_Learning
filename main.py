import time
from pathlib import Path
import my_testing
import my_datasets
import pickle


def classification(working_dir, n_classes, n_runs):
    RBF_A = 0.1
    RBF_B = 0.1
    results = {}

    for i in range(n_runs):
        partitioned_data, partitioned_labels = my_datasets.load_caltech_data(working_dir, n_classes)
        dataset_name = 'caltech256_' + str(i)

        start_time = time.time()
        print(f'Iteration count: {i + 1}')
        results[dataset_name] = my_testing.run_methods_on_dataset(partitioned_data=partitioned_data, partitioned_labels=partitioned_labels, RBF_A=RBF_A, RBF_B=RBF_B)
        print(f'Time: {time.time() - start_time}s')
    
    return results

working_dir = Path(__file__).parent / 'datasets/Caltech256/caltech256/256_ObjectCategories'
n_runs = 100

binary_results = classification(working_dir=working_dir, n_classes=2, n_runs=n_runs)
multiclass_results = classification(working_dir=working_dir, n_classes=4, n_runs=n_runs)

with open(f'{Path(__file__).parent}/binary_classification_results.pkl', 'wb') as file:
    pickle.dump(binary_results, file)

with open(f'{Path(__file__).parent}/multiclass_classification_results.pkl', 'wb') as file:
    pickle.dump(multiclass_results, file)