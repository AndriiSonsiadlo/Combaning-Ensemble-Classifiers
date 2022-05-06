import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class DatasetConfig:
    dataset_features = None
    dataset_labels = None
    x_train, x_test, y_train, y_test = None, None, None, None

    def __init__(self, dataset_name, names: list, target_name, path_or_url: str, delete_names=None):
        self.dataset_name = dataset_name
        self.names = names
        self.delete_names = delete_names
        self.target_name = target_name
        self.path_or_url = path_or_url

    def processing_data(self):
        dataset = pd.read_csv(self.path_or_url, names=self.names)
        if self.delete_names:
            for name in self.delete_names:
                dataset.pop(name)
        self.dataset_features = dataset.copy()
        self.dataset_labels = self.dataset_features.pop(self.target_name)

        x_train, x_test, y_train, y_test = train_test_split(self.dataset_features, self.dataset_labels, test_size=0.25)
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    iris = load_iris()
    print(iris["data"], iris["target"])

    abalone_dataset = DatasetConfig(dataset_name="Abalone",
                                    names=["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                                           "Viscera weight", "Shell weight", "Rings"],
                                    path_or_url="datasets/Abalone_DB/abalone.data",
                                    target_name="Sex",
                                    delete_names=[]).processing_data()
    print(abalone_dataset)

    clfs = [RandomForestClassifier(n_estimators=1, n_jobs=10),
            DecisionTreeClassifier(),
            KNeighborsClassifier(n_neighbors=4)]
    for clf in clfs:
        clf.fit(abalone_dataset[0], abalone_dataset[2])
        y_pred_rf = clf.predict(abalone_dataset[1])
        print(accuracy_score(abalone_dataset[3], y_pred_rf))

    #rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    #rnd_clf.fit(abalone_dataset[0], abalone_dataset[2])
    #y_pred_rf = rnd_clf.predict(abalone_dataset[1])
    #print(accuracy_score(abalone_dataset[3], y_pred_rf))
