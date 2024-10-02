import time
from pathlib import Path
import my_testing
import my_datasets
import pickle
import concurrent.futures


def process_run(i, working_dir, n_classes, RBF_A, RBF_B):
    partitioned_data, partitioned_labels = my_datasets.load_caltech_data(working_dir, n_classes)
    dataset_name = 'caltech256_' + str(i)

    start_time = time.time()
    print(f'Iteration count: {i + 1}')
    result = my_testing.run_methods_on_dataset(partitioned_data=partitioned_data, partitioned_labels=partitioned_labels, RBF_A=RBF_A, RBF_B=RBF_B)
    print(f'Time for iteration {i + 1}: {time.time() - start_time}s')
    
    return dataset_name, result

def parallel_classification(working_dir, n_classes, n_runs, max_workers):
    RBF_A = 0.1
    RBF_B = 0.1
    results = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_run, i, working_dir, n_classes, RBF_A, RBF_B) for i in range(n_runs)]
        
        for future in concurrent.futures.as_completed(futures):
            dataset_name, result = future.result()
            results[dataset_name] = result
    
    return results

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

        # Save the partial results to the .pkl file after each iteration
        with open('binary_classification_results', 'wb') as file:
            pickle.dump(results, file)
    
    return results

if __name__ == '__main__':
    parallel = False
    max_workers = 4
    working_dir = Path(__file__).parent / 'datasets/Caltech256/caltech256/256_ObjectCategories'
    n_runs = 100

    binary_results = parallel_classification(working_dir=working_dir, n_classes=2, n_runs=n_runs, max_workers=max_workers) if parallel else classification(working_dir=working_dir, n_classes=2, n_runs=n_runs)
    with open(f'{Path(__file__).parent}/binary_classification_results.pkl', 'wb') as file:
        pickle.dump(binary_results, file)


    multiclass_results = parallel_classification(working_dir=working_dir, n_classes=4, n_runs=n_runs, max_workers=max_workers) if parallel else classification(working_dir=working_dir, n_classes=4, n_runs=n_runs)
    with open(f'{Path(__file__).parent}/multiclass_classification_results.pkl', 'wb') as file:
        pickle.dump(multiclass_results, file)