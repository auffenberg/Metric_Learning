import time
from pathlib import Path
import my_testing
import os
import my_datasets
import pickle
import concurrent.futures

def load_data(i, working_dir, n_classes):
    return my_datasets.load_caltech_data(working_dir, n_classes)

    
def process_run(i, working_dir, n_classes, RBF_A, RBF_B, binary_or_multi, max_workers):
    if i % max_workers ==0: 
        print(f'Starting processes {i + 1} to {i + max_workers} in {binary_or_multi}classification.')
    partitioned_data, partitioned_labels = load_data(i, working_dir, n_classes)

    dataset_name = 'caltech256_' + str(i)
    if i == 0:
        start_time = time.time()
    result = my_testing.run_methods_on_dataset(partitioned_data, partitioned_labels, RBF_A, RBF_B)
    if i == 0:
        print(f'First process in {binary_or_multi}classification done in {time.time() - start_time} seconds.')
    if i % max_workers == 0:
        print(f'Processes {i + 1} to {i + max_workers} in {binary_or_multi}classification completed.')
    return dataset_name, result


def save_results(results, output_file):
   with open(output_file, 'wb') as f:
       pickle.dump(results, f)
    
    
def parallel_classification(working_dir, n_classes, n_runs, max_workers, output_file, binary_or_multi):
    RBF_A = 0.1
    RBF_B = 0.1
    results = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_run, i, working_dir, n_classes, RBF_A, RBF_B, binary_or_multi, max_workers) for i in range(n_runs)]
        
        for future in concurrent.futures.as_completed(futures):
            dataset_name, result = future.result()
            results[dataset_name] = result

            save_results(results, output_file)

    return results


if __name__ == '__main__':

    #max_workers = get_number_of_cores()
    max_workers = os.cpu_count()
    data_dir = Path(__file__).parent / 'datasets/Caltech256/caltech256/256_ObjectCategories'
    n_runs = 500

    output_binary = f'{Path(__file__).parent}/binary_classification_results.pkl'
    parallel_classification(working_dir=data_dir, n_classes=2, n_runs=n_runs, max_workers=max_workers, output_file=output_binary, binary_or_multi='binary') 

    n_runs = 300
    output_multi = f'{Path(__file__).parent}/multi_classification_results.pkl'
    parallel_classification(working_dir=data_dir, n_classes=4, n_runs=n_runs, max_workers=max_workers, output_file=output_multi, binary_or_multi='multiclass') 

    print('All done')