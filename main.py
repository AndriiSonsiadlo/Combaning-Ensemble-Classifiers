from Algorithm import Algorithm
from Dataset import Dataset

if __name__ == '__main__':
   ens = Algorithm()
   datasets = Dataset.get_datasets()

   for dataset in datasets:

      ens.cross_val_5x2(dataset=dataset)


   ens.display_figure()
