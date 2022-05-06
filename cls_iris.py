from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetConfig:
    def __init__(self, dataset_name: str = '', names: list = None, global_path: str = None, local_path: str = None,
                 url: str = None):
        if names is None:
            self.names = []
        else:
            self.names = names

        self.dataset_name = dataset_name
        self.global_path = global_path
        self.local_path = local_path
        self.url = url

    def __get__(self):
        if self.local_path:
            path = self.local_path
        elif self.global_path:
            path = self.global_path
        elif self.url:
            path = self.url
        else:
            raise BufferError("Dataset path is empty. Set local path, global path or URL for Dataset.")
        return self.dataset_name, path, self.names


iris = load_iris()
print(iris["data"], iris["target"])


def load_dataset(dataset_config: DatasetConfig, targer_name:str):
    dataset = pd.read_csv(dataset_config.local_path, names=dataset_config.names)
    dataset_features = dataset.copy()
    dataset_labels = dataset_features.pop(targer_name)

    x_train, x_test, y_train, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.25)
    return x_train, x_test, y_train, y_test


abalone_dsc = DatasetConfig(dataset_name="Abalone",
                            names=["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                                   "Viscera weight", "Shell weight", "Rings"],
                            local_path="datasets/Abalone_DB/abalone.data")
abalone_splitted_dataset = load_dataset(abalone_dsc, "Sex")



