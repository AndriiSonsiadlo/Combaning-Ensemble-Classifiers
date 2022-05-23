from statistics import mean

from sklearn import datasets
from sklearn.ensemble import *
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
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

from Dataset import Dataset

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





def dataset_adult(records_count=-1):
    dataset_conf = Dataset(name="Adult",
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


def dataset_wine(records_count=-1, test_size=0.33):
    dataset_conf = Dataset(name="Wine",
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
                           record_count=records_count,
                           test_size=test_size)

    return dataset_conf


def dataset_car(records_count=-1):
    dataset_conf = Dataset(name="Car",
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


def dataset_iris(records_count=-1, test_size=0.33):
    dataset_conf = Dataset(name="Iris",
                           feature_labels=["sepal length in cm", "sepal width in cm",
                                                 "petal length in cm", "petal width in cm", "class"],
                           path_or_url="datasets/Iris_DB/iris.data",
                           target_label="class",
                           record_count=records_count,
                           test_size=test_size)
    return dataset_conf


def dataset_abalone(records_count=-1, test_size=0.33):
    dataset_conf = Dataset(name="Abalone",
                           feature_labels=["Sex", "Length", "Diameter", "Height", "Whole weight",
                                                 "Shucked weight",
                                                 "Viscera weight", "Shell weight", "Rings"],
                           path_or_url="datasets/Abalone_DB/abalone.data",
                           target_label="Rings",
                           record_count=records_count,
                           test_size=test_size)
    return dataset_conf


def ML(x_train, x_test, y_train, y_test):
    clfs = {"RFC": RandomForestClassifier(n_estimators=500, n_jobs=-1),
            "DCT": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=4)}

    for name, clf in clfs.items():
        clf.fit(x_train, y_train)
        y_pred_rf = clf.predict(x_test)
        print(f"{name}: {accuracy_score(y_test, y_pred_rf)}")


def process_clfs(RANDOM_SEED=1):
    clfs_list = [RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED),
                 ExtraTreesClassifier(n_estimators=5, random_state=RANDOM_SEED),
                 KNeighborsClassifier(n_neighbors=4),
                 SVC(C=10000.0, kernel='rbf', random_state=RANDOM_SEED, probability=True),
                 RidgeClassifier(alpha=0.1, random_state=RANDOM_SEED),
                 LogisticRegression(C=20000, penalty='l2', random_state=RANDOM_SEED, max_iter=3000),
                 DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED),
                 AdaBoostClassifier(n_estimators=5, learning_rate=0.001)]


    return clfs_list

    clfs = dict()

    for clf in clfs_list:
        clfs[clf.__class__.__name__] = clf
    del clfs_list

    return clfs


def bagging(dataset_config: Dataset):
    X, y = dataset_config._get_dataset_x_and_y()

    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    encoder_object = LabelEncoder()
    y = encoder_object.fit_transform(y)

    RANDOM_SEED = 99
    clfs = process_clfs(RANDOM_SEED)

    normal_accuracy = []
    normal_std = []

    bagging_accuracy = []
    bagging_std = []

    for name, clf in clfs.items():
        cv_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)

        bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=3, random_state=RANDOM_SEED)
        bagging_scores = cross_val_score(bagging_clf, X, y, cv=3, n_jobs=-1)

        normal_accuracy.append(np.round(cv_scores.mean(), 4))
        normal_std.append(np.round(cv_scores.std(), 4))

        bagging_accuracy.append(np.round(bagging_scores.mean(), 4))
        bagging_std.append(np.round(bagging_scores.std(), 4))

        print("Accuracy: %0.4f (+/- %0.4f) [Normal %s]" % (cv_scores.mean(), cv_scores.std(), name))
        print("Accuracy: %0.4f (+/- %0.4f) [Bagging %s]\n" % (
            bagging_scores.mean(), bagging_scores.std(), name))

    print(mean(normal_accuracy))
    print(mean(normal_std))
    print(mean(bagging_accuracy))
    print(mean(bagging_std))

    ### PLOT

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(20, 10))
    n_groups = len(clfs)
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = .7
    error_config = {'ecolor': '0.2'}

    normal_clf = ax.bar(index, normal_accuracy, bar_width, alpha=opacity, color='g', yerr=normal_std,
                        error_kw=error_config, label='Normal Classifier')
    bagging_clf = ax.bar(index + bar_width, bagging_accuracy, bar_width, alpha=opacity, color='y', yerr=bagging_std,
                         error_kw=error_config, label='Bagging Classifier')

    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Accuracy scores with variance')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels((clfs.keys()))
    ax.legend()

    # fig.tight_layout()
    plt.show()

    ### ##################################################################################### ###

    max_samples_dict = {"Max Samples": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    various_bagging_scores = {}

    for name, clf in clfs.items():
        cv_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
        print("\nAccuracy: %0.4f (+/- %0.4f) [Normal %s]" % (cv_scores.mean(), cv_scores.std(), name))

        mean_bagging_score = []
        for ratio in max_samples_dict["Max Samples"]:
            bagging_clf = BaggingClassifier(clf, max_samples=ratio, max_features=3, random_state=RANDOM_SEED)
            bagging_scores = cross_val_score(bagging_clf, X, y, cv=5, n_jobs=-1)
            mean_bagging_score.append(np.round(bagging_scores.mean(), 3))
            # print("Bagging accuracy: %0.4f [max_samples %0.2f]" % (bagging_scores.mean(), ratio))
        various_bagging_scores[name] = mean_bagging_score

    # Compare performance and display it in a pretty table.
    from prettytable import PrettyTable
    table = PrettyTable()

    # table.field_names = various_bagging_scores.keys()

    table.add_column("Max Samples", max_samples_dict["Max Samples"])
    for key, value in various_bagging_scores.items():
        table.add_column(key, value)

    print(table)

    ### PLOT

    x_axes = max_samples_dict["Max Samples"]

    color_map = ['blue', 'g', 'r', 'c', 'grey', 'y', 'black', 'm']
    plt.figure(figsize=(20, 10))
    for index, name in enumerate(clfs.keys()):
        plt.plot(x_axes, various_bagging_scores[name], color=color_map[index], label=name)
    plt.xlabel('Sub sampling Ratio')
    plt.ylabel('Accuracy')
    plt.title("Comparison b/w accuracy of different classifiers at various sub sampling ratio")
    plt.legend()
    plt.show()


def boosting(dataset_config: Dataset):
    X, y = dataset_config._get_dataset_x_and_y()
    train_x, test_x, train_y, test_y = dataset_config._get_dataset_train_xy_and_test_xy()

    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from mlxtend.classifier import EnsembleVoteClassifier
    from xgboost import XGBClassifier

    ada_boost = AdaBoostClassifier(n_estimators=5)
    grad_boost = GradientBoostingClassifier(n_estimators=10)
    xgb_boost = XGBClassifier(max_depth=5, learning_rate=0.001)

    ensemble_clf = EnsembleVoteClassifier(clfs=list(process_clfs().values()), voting='hard')
    boosting_labels = ['Ada Boost', 'Gradient Boost', 'XG Boost', 'Ensemble']

    for clf, label in zip([ada_boost, grad_boost, xgb_boost, ensemble_clf], boosting_labels):
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print("Accuracy: {0:.3f}, Variance: (+/-) {1:.3f} [{2}]".format(scores.mean(), scores.std(), label))

    for clf, label in zip([ada_boost, grad_boost, xgb_boost, ensemble_clf], boosting_labels):
        clf.fit(train_x, train_y)
        y_pred_rf = clf.predict(test_x)
        print(f"ACC {label}: {accuracy_score(test_y, y_pred_rf)}")

    ### PLOT

    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.gridspec as gridspec
    import itertools

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(20, 16))

    value = 1.5
    width = 0.75
    for clf, label, grd in zip([ada_boost, grad_boost, xgb_boost, ensemble_clf], boosting_labels,
                               itertools.product([0, 1], repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])

        #        fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2,
        #                                    filler_feature_values={2: 1, 3: 1},
        #                                    filler_feature_ranges={2: 1, 3: 1}
        #                                    )
        plt.title(label)

    plt.show()


if __name__ == '__main__':
    #  ML(*dataset_car()._get_dataset_train_xy_and_test_xy())
    #  ML(*dataset_wine()._get_dataset_train_xy_and_test_xy())

    iris = dataset_iris(records_count=-1, test_size=0.33)
    iris.plot()
    bagging(iris)
    boosting(iris)




    # x = dataset_iris().iloc[:, 0:4]
    # y = dataset_iris().iloc[:, 4]