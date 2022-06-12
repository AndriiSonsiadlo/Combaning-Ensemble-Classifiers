import json

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class Dataset:
    file_json = 'datasets_config.json'
    dataset = None
    ordinal_dataset = None
    x_features = None
    y_labels = None
    x_train, x_test, y_train, y_test = None, None, None, None

    def __init__(self, name, feature_labels: list, target_label, path_or_url: str, record_count=-1,
                 test_size=0.33):
        self.name = name
        self.feature_labels = feature_labels
        self.target_label = target_label
        self.path_or_url = path_or_url
        self.test_size = test_size

        self.dataset = self.load_dataset(self.path_or_url, self.feature_labels).head(record_count)
        self.x_features, self.y_labels = self.process_dataset(self.dataset, self.target_label)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_dataset(self.x_features,
                                                                                  self.y_labels,
                                                                                  self.test_size)
        # self.__create_ordinal_dataset()

    def __call__(self, *args, **kwargs):
        return self.dataset

    @classmethod
    def get_datasets(cls, *args):
        datasets_lst = []

        with open(cls.file_json, 'r') as f:
            datasets = json.load(f)["datasets"]

        for dataset in datasets:
            name = datasets[dataset]["name"]
            path_or_url = datasets[dataset]["path_or_url"]
            feature_labels = datasets[dataset]["feature_labels"]
            target_label = datasets[dataset]["target_label"]
            record_count = datasets[dataset]["record_count"]

            dataset = Dataset(name=name, feature_labels=feature_labels, target_label=target_label,
                              path_or_url=path_or_url, record_count=record_count)

            if len(args) and (dataset.name.lower() in args or dataset.name in args):
                datasets_lst.append(dataset)
            elif not len(args):
                datasets_lst.append(dataset)

        return datasets_lst

    @classmethod
    def get_dataset(cls, name: str):
        lst = cls.get_datasets()

        for dataset in lst:
            if dataset.name.lower() == name.lower():
                return dataset

        raise NameError(f" {name.capitalize()} dataset not exist")

    def load_dataset(self, path_or_url, feature_names):
        if self.name.lower() in ['bank', 'student', 'winequalitywhite', 'winequalityred']:
            delimiter = ';'
        else:
            delimiter = None
        dataset = pd.read_csv(path_or_url, names=feature_names, delimiter=delimiter)

        return dataset

    @staticmethod
    def process_dataset(dataset, target_name):

        label_encoder = LabelEncoder()
        ordinal_encoder = OrdinalEncoder()

        dataset_features = dataset.copy()
        if isinstance(dataset, pd.DataFrame):
            if isinstance(target_name, str):
                dataset_labels = dataset_features.pop(target_name)

            else:
                raise NameError("Target name must be in string!")
        else:
            raise TypeError("Dataset must be in type pd.DataFrame. You can use static function ndarray_to_df()")

        dataset_features = ordinal_encoder.fit_transform(dataset_features)
        dataset_labels = label_encoder.fit_transform(dataset_labels)

        #    return dataset_features, dataset_labels

        return np.array(dataset_features), np.array(dataset_labels)

    @staticmethod
    def split_dataset(data, target, test_size):

        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size)

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    @staticmethod
    def ndarray_to_df(array, names: (list or tuple)):
        if isinstance(array, np.ndarray):
            print(names)
            df = pd.DataFrame(array, columns=names)
            return df
        else:
            raise TypeError("Given data is not ndarray type!")

    def plot(self):
        plt.close()
        sns.set_style("whitegrid")

        ds = self.ordinal_dataset if self.ordinal_dataset else self.dataset
        sns.pairplot(ds, hue=self.target_label, height=self._get_class_count())
        plt.show()

    def _get_class_count(self):
        return len(set(self.feature_labels)) - 1

    def _to_df(self):
        return self.dataset

    def _to_np(self):
        return np.array(self.dataset)

    def _get_dataset_x_and_y(self):
        return self.x_features, self.y_labels

    def _get_dataset_train_xy_and_test_xy(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def __create_ordinal_dataset(self):
        ordinal_encoder = OrdinalEncoder()

        ordinal_dataset_np = ordinal_encoder.fit_transform(self.x_features)
        self.ordinal_dataset = self.ndarray_to_df(ordinal_dataset_np, names=self.feature_labels[:-1])
        self.ordinal_dataset[self.target_label] = self.y_labels

# dataset_conf = Dataset(dataset_name="Iris",
#                        feature_labels=["sepal length in cm", "sepal width in cm",
#                                        "petal length in cm", "petal width in cm", "class"],
#                        path_or_url="datasets/Iris_DB/iris.data",
#                        target_label="class",
#                        record_count=-1,
#                        test_size=0.33)
#
# a = dataset_conf()
#
# dataset_conf.plot()
