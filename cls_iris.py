from sklearn import datasets
from sklearn.ensemble import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
import sqlite3
import pandas as pd
import numpy as np

import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
import string, math, pickle
import warnings

warnings.filterwarnings("ignore")
from xgboost.sklearn import XGBModel, XGBRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class DatasetConfig:
    dataset = None
    ordinal_dataset = None
    dataset_features = None
    dataset_labels = None
    x_train, x_test, y_train, y_test = None, None, None, None

    def __init__(self, dataset_name, feature_labels: list, target_label, path_or_url: str, record_count=-1):
        self.dataset_name = dataset_name
        self.feature_labels = feature_labels
        self.target_label = target_label
        self.path_or_url = path_or_url

        self.dataset = self.load_dataset(self.path_or_url, self.feature_labels).head(record_count)
        print(self.dataset)
        self.dataset_features, self.dataset_labels = self.process_dataset(self.dataset, self.target_label)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_dataset(self.dataset_features,
                                                                                  self.dataset_labels)
        # self.__create_ordinal_dataset()

    @staticmethod
    def load_dataset(path_or_url, feature_names):
        dataset = pd.read_csv(path_or_url, names=feature_names)
        return dataset

    @staticmethod
    def process_dataset(dataset, target_name_or_index):
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        label_encoder = LabelEncoder()
        ordinal_encoder = OrdinalEncoder()

        dataset_features = dataset.copy()
        if isinstance(dataset, pd.DataFrame):
            if isinstance(target_name_or_index, str):
                dataset_labels = dataset_features.pop(target_name_or_index)
            else:
                raise NameError("Target name must be in string!")
        else:
            raise TypeError("Dataset must be in type pd.DataFrame. You can use static function ndarray_to_df()")

        dataset_features = ordinal_encoder.fit_transform(dataset_features)
        dataset_labels = label_encoder.fit_transform(dataset_labels)

        return dataset_features, dataset_labels

    @staticmethod
    def split_dataset(data, target):
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    @staticmethod
    def ndarray_to_df(array, names: (list or tuple)):
        if isinstance(array, np.ndarray):
            df = pd.DataFrame(array, columns=names)
            return df
        else:
            raise TypeError("Given data is not ndarray type!")

    def plot(self):
        plt.close()
        sns.set_style("whitegrid")
        print(self.dataset, self.target_label)
        ds = self.ordinal_dataset if self.ordinal_dataset else self.dataset
        sns.pairplot(ds, hue=self.target_label, size=self.__get_class_count())
        plt.show()

    def __get_class_count(self):
        return len(self.feature_labels) - 1

    def _get_dataset_in_dataframe(self):
        return self.dataset

    def _get_dataset_in_ndarray(self):
        return np.array(self.dataset)

    def _get_dataset_feature_and_label(self):
        return self.dataset_features, self.dataset_labels

    def _get_dataset_train_xy_and_test_xy(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def __create_ordinal_dataset(self):
        ordinal_encoder = OrdinalEncoder()

        ordinal_dataset_np = ordinal_encoder.fit_transform(self.dataset_features)
        self.ordinal_dataset = self.ndarray_to_df(ordinal_dataset_np, names=self.feature_labels[:-1])
        self.ordinal_dataset[self.target_label] = self.dataset_labels


def dataset_adult(records_count=-1):
    dataset_conf = DatasetConfig(dataset_name="Adult",
                                feature_labels=["age",
                                                "workclass",
                                                "fnlwgt",
                                                "education",
                                                "education-num",
                                                "marital-status",
                                                "occupation",
                                                "relationship",
                                                "race",
                                                "sex",
                                                "capital-gain",
                                                "capital-loss",
                                                "hours-per-week",
                                                "native-country",
                                                "class"],
                                path_or_url="datasets/Adult_DB/adult.data",
                                target_label="class",
                                record_count=records_count)
    return dataset_conf


def dataset_wine(records_count=-1):
    dataset_conf = DatasetConfig(dataset_name="Wine",
                                feature_labels=["Class",
                                                "Alcohol",
                                                "Malicacid",
                                                "Ash",
                                                "Alcalinity of ash",
                                                "Magnesium",
                                                "Total phenols",
                                                "Flavanoids",
                                                "Nonflavanoid phenols",
                                                "Proanthocyanins",
                                                "Color intensity",
                                                "Hue",
                                                "OD280 / OD315 of diluted wines",
                                                "Proline"],
                                path_or_url="datasets/Wine_DB/wine.data",
                                target_label="Class",
                                record_count=records_count)

    return dataset_conf


def dataset_car(records_count=-1):
    dataset_conf = DatasetConfig(dataset_name="Car",
                                feature_labels=["buying",
                                                "maint",
                                                "doors",
                                                "persons",
                                                "lug_boot",
                                                "safety",
                                                "class"],
                                path_or_url="datasets/Car_DB/Car.data",
                                target_label="class",
                                record_count=records_count)

    return dataset_conf


def dataset_iris(records_count=-1):
    dataset_conf = DatasetConfig(dataset_name="Iris",
                                feature_labels=["sepal length in cm", "sepal width in cm",
                                                "petal length in cm", "petal width in cm", "class"],
                                path_or_url="datasets/Iris_DB/iris.data",
                                target_label="class",
                                record_count=records_count)
    return dataset_conf


def dataset_abalone(records_count=-1):
    dataset_conf = DatasetConfig(dataset_name="Abalone",
                                feature_labels=["Sex", "Length", "Diameter", "Height", "Whole weight",
                                                "Shucked weight",
                                                "Viscera weight", "Shell weight", "Rings"],
                                path_or_url="datasets/Abalone_DB/abalone.data",
                                target_label="Rings",
                                record_count=records_count)
    return dataset_conf


def ML(x_train, x_test, y_train, y_test):
    clfs = {"RFC": RandomForestClassifier(n_estimators=500, n_jobs=-1),
            "DCT": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=4)}

    for name, clf in clfs.items():
        clf.fit(x_train, y_train)
        y_pred_rf = clf.predict(x_test)
        print(f"{name}: {accuracy_score(y_test, y_pred_rf)}")


def learn(x_train, x_test, y_train, y_test):
    iris_dataset = pd.read_csv("iris.csv")
    X, y = iris_dataset.iloc[:, 0:4], iris_dataset.iloc[:, 4]

    from sklearn.preprocessing import LabelEncoder
    encoder_object = LabelEncoder()
    y = encoder_object.fit_transform(y)

    RANDOM_SEED = 0

    # Base Learners
    rf_clf = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
    et_clf = ExtraTreesClassifier(n_estimators=5, random_state=RANDOM_SEED)
    knn_clf = KNeighborsClassifier(n_neighbors=2)
    svc_clf = SVC(C=10000.0, kernel='rbf', random_state=RANDOM_SEED)
    rg_clf = RidgeClassifier(alpha=0.1, random_state=RANDOM_SEED)
    lr_clf = LogisticRegression(C=20000, penalty='l2', random_state=RANDOM_SEED)
    dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED)
    adab_clf = AdaBoostClassifier(n_estimators=5, learning_rate=0.001)

    classifier_array = [rf_clf, et_clf, knn_clf, svc_clf, rg_clf, lr_clf, dt_clf, adab_clf]
    labels = [clf.__class__.__name__ for clf in classifier_array]

    normal_accuracy = []
    normal_std = []

    bagging_accuracy = []
    bagging_std = []

    for clf in classifier_array:
        cv_scores = cross_val_score(clf, X, y, cv=3, n_jobs=-1)
        bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=3, random_state=RANDOM_SEED)
        bagging_scores = cross_val_score(bagging_clf, X, y, cv=3, n_jobs=-1)

        normal_accuracy.append(np.round(cv_scores.mean(), 4))
        normal_std.append(np.round(cv_scores.std(), 4))

        bagging_accuracy.append(np.round(bagging_scores.mean(), 4))
        bagging_std.append(np.round(bagging_scores.std(), 4))

        print("Accuracy: %0.4f (+/- %0.4f) [Normal %s]" % (cv_scores.mean(), cv_scores.std(), clf.__class__.__name__))
        print("Accuracy: %0.4f (+/- %0.4f) [Bagging %s]\n" % (
            bagging_scores.mean(), bagging_scores.std(), clf.__class__.__name__))


if __name__ == '__main__':
    #  ML(*dataset_car()._get_dataset_train_xy_and_test_xy())
    #  ML(*dataset_wine()._get_dataset_train_xy_and_test_xy())

    iris = dataset_iris(500)
    ML(*iris._get_dataset_train_xy_and_test_xy())
    iris.plot()
