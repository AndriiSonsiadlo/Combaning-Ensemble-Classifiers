from Algorithm import Algorithm
from Dataset import Dataset

if __name__ == '__main__':
    datasets = Dataset.get_datasets()
    alg = Algorithm(datasets)

    alg.iterate_validation_for_each_dataset()

    alg.display_figure()
    alg.display_table()
