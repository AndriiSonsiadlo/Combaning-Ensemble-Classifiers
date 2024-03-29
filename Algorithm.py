import random

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.classifier import EnsembleVoteClassifier
from tabulate import tabulate
from prettytable import PrettyTable
from scipy.stats import studentized_range, ttest_rel, ttest_ind
from sklearn import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from scipy.stats import studentized_range
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from scipy import stats
from tqdm import tqdm

from Dataset import Dataset


def print_dataset_name(func):
    def wrapper(*args, **kwargs):
        dataset = kwargs["dataset"]
        print("=" * 25, dataset.name, "=" * 25, end="\n")

        result = func(*args, **kwargs)

        print("=" * 60, end="\n\n\n")
        return result

    return wrapper


def static_test(func):
    def wrapper(*args, **kwargs):
        dataset = kwargs["dataset"]

        result = func(*args, **kwargs)

        print("=" * 60, end="\n\n\n")
        return result

    return wrapper


class ClassifierResult:
    def __init__(self, name):
        self.name = name
        self.clf = None

        # result
        self.scores = []
        self.mean = None
        self.std = None

    def calculate_mean_and_std(self):
        self.mean = round(np.mean(self.scores), 4)
        self.std = round(np.std(self.scores), 4)


class DatasetResult:

    def __init__(self, dataset):
        self.name = dataset.name
        self.num_records = len(dataset.y_labels),
        self.num_classes = len(set(dataset.y_labels))

        self.mod_clf = ClassifierResult("mod_clf")
        self.bag_clf = ClassifierResult("bag_clf")
        self.boost_clf = ClassifierResult("boost_clf")

        self.t_student_mod_bag = None
        self.t_student_mod_boost = None
        self.p_value_mod_bag = None
        self.p_value_mod_boost = None

    def calculate_mean_and_std(self):
        self.mod_clf.calculate_mean_and_std()
        self.bag_clf.calculate_mean_and_std()
        self.boost_clf.calculate_mean_and_std()


class Algorithm:
    seeds = [20, 45888, 10000, 89, 65487]
    k = 20
    n = 20
    max_depth = 10
    n_jobs = -1
    n_samples_max = 0.1
    alfa = .05

    def __init__(self, datasets: list):
        self.datasets = datasets
        self.result = {}
        for dataset in datasets:
            self.result[dataset.name] = DatasetResult(dataset)

    def iterate_validation_for_each_dataset(self):
        for dataset in self.datasets:
            self.cross_val_5x2(dataset=dataset)

    def __get_classifiers(self, trn_x, trn_y):
        n_samples = int(self.n_samples_max * len(trn_y))
        classifier_list = self.generate_boosting_list(trn_x, trn_y, n_samples)

        clf_modified_1 = VotingClassifier(estimators=classifier_list, voting='soft', n_jobs=self.n_jobs)
        clf_bagging_2 = BaggingClassifier(DecisionTreeClassifier(max_depth=self.max_depth), n_estimators=self.n,
                                          bootstrap=True,
                                          n_jobs=self.n_jobs)
        clf_boosting_3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.max_depth), n_estimators=self.k)

        return clf_modified_1, clf_bagging_2, clf_boosting_3

    def generate_boosting_list(self, trn_x, trn_y, n_samples):
        classifier_list = []
        for i in range(self.n):
            X_sample = resample(trn_x, n_samples=n_samples, random_state=i, replace=True)
            y_sample = resample(trn_y, n_samples=n_samples, random_state=i, replace=True)
            ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=self.k)
            ada_clf.fit(X_sample, y_sample)
            classifier_list.append(('ada' + str(i), ada_clf))
        return classifier_list

    @print_dataset_name
    def cross_val_5x2(self, dataset):
        mod_clf = self.result[dataset.name].mod_clf
        bag_clf = self.result[dataset.name].bag_clf
        boost_clf = self.result[dataset.name].boost_clf

        x, y = dataset.x_features, dataset.y_labels

        n_splits = 5
        n_repeats = 2

        # Split the dataset in 2 parts with the current seed
        rskfolds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.seeds[2])


        # Go through the current 2 fold
        for fold_id, (trn_idx, val_idx) in enumerate(rskfolds.split(x, y)):

            # Split the data - 50 / 50
            trn_x, trn_y = x[trn_idx], y[trn_idx]
            val_x, val_y = x[val_idx], y[val_idx]

            mod_clf.clf, bag_clf.clf, boost_clf.clf = self.__get_classifiers(trn_x, trn_y)

            # Train classifiers
            mod_clf.clf.fit(trn_x, trn_y)
            bag_clf.clf.fit(trn_x, trn_y)
            boost_clf.clf.fit(trn_x, trn_y)

            # Compute scores and keep score history for mean and stdev calculation
            try:
                preds_1 = mod_clf.clf.predict(val_x)[:, 1]
                preds_2 = bag_clf.clf.predict(val_x)[:, 1]
                preds_3 = boost_clf.clf.predict(val_x)[:, 1]
                mod_clf.scores.append(accuracy_score(val_y, preds_1))
                bag_clf.scores.append(accuracy_score(val_y, preds_2))
                boost_clf.scores.append(accuracy_score(val_y, preds_3))
            except BaseException as e:
                preds_1 = mod_clf.clf.predict(val_x)
                preds_2 = bag_clf.clf.predict(val_x)
                preds_3 = boost_clf.clf.predict(val_x)
                mod_clf.scores.append(accuracy_score(val_y, preds_1))
                bag_clf.scores.append(accuracy_score(val_y, preds_2))
                boost_clf.scores.append(accuracy_score(val_y, preds_3))


            # keep score history for mean and stdev calculation
            # diff_scores.append((score_1 - score_2, scores_1 - score_3))
            print("Fold %2d score difference between clf_modified and clf_bagging = %.6f" % (
                fold_id + 1, mod_clf.scores[-1] - bag_clf.scores[-1]))
            print("Fold %2d score difference between clf_modified and clf_boosting = %.6f" % (
                fold_id + 1, mod_clf.scores[-1] - boost_clf.scores[-1]))

        self.result[dataset.name].calculate_mean_and_std()

        t_statistic = np.zeros((3, 3))
        p_value = np.zeros((3, 3))


        scores = []
        scores.append(mod_clf.scores)
        scores.append(bag_clf.scores)
        scores.append(boost_clf.scores)
        for i in range(3):
            for j in range(3):
                t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
        print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
        self.result[dataset.name].t_student_mod_bag = t_statistic[0, 1]
        self.result[dataset.name].t_student_mod_boost = t_statistic[0, 2]
        self.result[dataset.name].p_value_mod_bag = p_value[0, 1]
        self.result[dataset.name].p_value_mod_boost = p_value[0, 2]


        print("Classifier Mod mean score and stdev: %.4f (+/- %.4f)" % (mod_clf.mean, mod_clf.std))
        print("Classifier Bag mean score and stdev: %.4f (+/- %.4f)" % (bag_clf.mean, bag_clf.std))
        print("Classifier Boost mean score and stdev: %.4f (+/- %.4f)" % (boost_clf.mean, boost_clf.std))
        # print("Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[0]), np.std(diff_scores[0])))


        headers = ["Mod", "Bag", "Boost"]
        names_column = np.array([["Mod"], ["Bag"], ["Boost"]])

        advantage = np.zeros((3, 3))
        advantage[t_statistic > 0] = 1
        advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
        print("\nAdvantage:\n", advantage_table)

        significance = np.zeros((3, 3))
        significance[p_value <= self.alfa] = 1
        significance_table = tabulate(np.concatenate(
            (names_column, significance), axis=1), headers)
        print("\nsignificance (alpha = 0.05):\n".capitalize(), significance_table)

        stat_better = significance * advantage
        stat_better_table = tabulate(np.concatenate(
            (names_column, stat_better), axis=1), headers)
        print("\nStatistically significantly better:\n", stat_better_table)





    def get_mean_std_lists(self):

        mean_mod, mean_bag, mean_boost = [], [], []
        std_mod, std_bag, std_boost = [], [], []

        for key, value in self.result.items():
            mean_mod.append(value.mod_clf.mean)
            mean_bag.append(value.bag_clf.mean)
            mean_boost.append(value.boost_clf.mean)
            std_mod.append(value.mod_clf.std)
            std_bag.append(value.bag_clf.std)
            std_boost.append(value.boost_clf.std)

        return mean_mod, mean_bag, mean_boost, std_mod, std_bag, std_boost

    def display_table(self):
        table = PrettyTable()
        table.field_names = ["Dataset", "t - bag", "p - bag",  "t - boost","p - boost", "Differ - mod and bag",
                             "Differ - mod and boost", "Num records", "Num classes", "Mod clf - mean",
                             "Bagging clf - mean", "Boosting clf - mean", "Mod clf - std", "Bagging clf - std",
                             "Boosting clf - std"]

        for key, data_res in self.result.items():
            t_bag = round(data_res.t_student_mod_bag, 2)
            t_boost = round(data_res.t_student_mod_boost, 2)
            p_bag = round(data_res.p_value_mod_bag, 3)
            p_boost = round(data_res.p_value_mod_boost, 3)

            if t_bag > 2.571:
                mod_or_bag = "Modified"
            elif t_bag < -2.571:
                mod_or_bag = "Bagging"
            else:
                mod_or_bag = "Similar"

            if t_boost > 2.571:
                mod_or_boost = "Modified"
            elif t_boost < -2.571:
                mod_or_boost = "Boosting"
            else:
                mod_or_boost = "Similar"

            table.add_row([key, t_bag, p_bag, t_boost, p_boost, mod_or_bag, mod_or_boost,
                           data_res.num_records[0], data_res.num_classes,
                           data_res.mod_clf.mean, data_res.bag_clf.mean, data_res.boost_clf.mean,
                           data_res.mod_clf.std, data_res.bag_clf.std, data_res.boost_clf.std])

            print(fr"{key} & {t_bag} & {p_bag}  & {t_boost} & {p_boost} \\")
            print(r"\hline")

        print(f"k = {self.k}")
        print(f"n = {self.n}")
        print(f"max_depth = {self.max_depth}")

        print(table)

    def display_figure(self):

        mean_mod, mean_bag, mean_boost, std_mod, std_bag, std_boost, *_ = self.get_mean_std_lists()

        fig, ax = plt.subplots(figsize=(20, 10))
        n_groups = len(self.result)
        index = np.arange(n_groups)
        bar_width = 0.25

        opacity = .7
        error_config = {'ecolor': '0.2'}

        modified_clf = ax.bar(index, mean_mod, bar_width, alpha=opacity, color='#71C375', yerr=std_mod,
                              error_kw=error_config, label='Boosting in Bagging')
        bagging_clf = ax.bar(index + bar_width, mean_bag, bar_width, alpha=opacity, color='#DF60AD', yerr=std_bag,
                             error_kw=error_config, label='Bagging')
        boosting_clf = ax.bar(index + (2 * bar_width), mean_boost, bar_width, alpha=opacity, color='#FC9426',
                              yerr=std_boost,
                              error_kw=error_config, label='Boosting')

        ax.set_xlabel('Classifiers')
        ax.set_ylabel('Accuracy scores with variance')
        ax.set_title('Scores by classifier groups and datasets')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels((self.result.keys()))
        ax.legend()

        fig.tight_layout()
        plt.show()
