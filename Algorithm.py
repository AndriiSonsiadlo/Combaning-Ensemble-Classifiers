import random

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.classifier import EnsembleVoteClassifier
from prettytable import PrettyTable
from scipy.stats import studentized_range, ttest_rel
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

    def calculate_mean_and_std(self):
        self.mod_clf.calculate_mean_and_std()
        self.bag_clf.calculate_mean_and_std()
        self.boost_clf.calculate_mean_and_std()


class Algorithm:
    seeds = [20, 45888, 10000, 89, 65487]
    n = 10
    k = 10
    max_depth = 1
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

        clf_modified_1 = VotingClassifier(estimators=classifier_list, voting='soft')
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

        # scores_1, scores_2, scores_3, diff_scores = [], [], [], []
        p_1_1_bag, p_1_1_boost = 0.0, 0.0         # Initialize the score difference for the 1st fold of the 1st iteration
        s_sqr_bag, s_sqr_boost = 0.0, 0.0         # Initialize a place holder for the variance estimate


        # Iterate through 5 2-fold CV
        for i_s, seed in enumerate(self.seeds):

            p_i_bag = np.zeros(2)    # Initialize score differences
            p_i_boost = np.zeros(2)    # Initialize score differences

            # Split the dataset in 2 parts with the current seed
            folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

            # Go through the current 2 fold
            for i_f, (trn_idx, val_idx) in enumerate(folds.split(x, y)):

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
                    preds_1 = mod_clf.clf.predict_proba(val_x)[:, 1]
                    preds_2 = bag_clf.clf.predict_proba(val_x)[:, 1]
                    preds_3 = boost_clf.clf.predict_proba(val_x)[:, 1]
                    mod_clf.scores.append(roc_auc_score(val_y, preds_1, multi_class='ovr'))
                    bag_clf.scores.append(roc_auc_score(val_y, preds_2, multi_class='ovr'))
                    boost_clf.scores.append(roc_auc_score(val_y, preds_3, multi_class='ovr'))
                except np.AxisError:
                    preds_1 = mod_clf.clf.predict_proba(val_x)
                    preds_2 = bag_clf.clf.predict_proba(val_x)
                    preds_3 = boost_clf.clf.predict_proba(val_x)
                    mod_clf.scores.append(roc_auc_score(val_y, preds_1, multi_class='ovr'))
                    bag_clf.scores.append(roc_auc_score(val_y, preds_2, multi_class='ovr'))
                    boost_clf.scores.append(roc_auc_score(val_y, preds_3, multi_class='ovr'))

                # keep score history for mean and stdev calculation
                # diff_scores.append((score_1 - score_2, scores_1 - score_3))
                print("Fold %2d score difference between clf_modified and clf_bagging = %.6f" % (
                    i_f + 1, mod_clf.scores[-1] - bag_clf.scores[-1]))
                print("Fold %2d score difference between clf_modified and clf_boosting = %.6f" % (
                    i_f + 1, mod_clf.scores[-1] - boost_clf.scores[-1]))

                p_i_bag[i_f] = mod_clf.scores[-1] - bag_clf.scores[-1]
                p_i_boost[i_f] = mod_clf.scores[-1] - boost_clf.scores[-1]
                if (i_s == 0) & (i_f == 0):
                    p_1_1_bag = p_i_bag[i_f]
                    p_1_1_boost = p_i_boost[i_f]

            # Compute mean of scores difference for the current 2-fold CV
            p_i_bar_bag = (p_i_bag[0] + p_i_bag[1]) / 2
            p_i_bar_boost = (p_i_boost[0] + p_i_boost[1]) / 2
            # Compute the variance estimate for the current 2-fold CV
            s_i_sqr_bag = (p_i_bag[0] - p_i_bar_bag) ** 2 + (p_i_bag[1] - p_i_bar_bag) ** 2
            s_i_sqr_boost = (p_i_boost[0] - p_i_bar_boost) ** 2 + (p_i_boost[1] - p_i_bar_boost) ** 2
            # Add up to the overall variance
            s_sqr_bag += s_i_sqr_bag
            s_sqr_boost += s_i_sqr_boost

        t_bar_bag = p_1_1_bag / ((s_sqr_bag / 5) ** .5)
        t_bar_boost = p_1_1_boost / ((s_sqr_boost / 5) ** .5)

        self.result[dataset.name].t_student_mod_bag = t_bar_bag
        self.result[dataset.name].t_student_mod_boost = t_bar_boost
        self.result[dataset.name].calculate_mean_and_std()

        print("bag", t_bar_bag)
        print("boost", t_bar_boost)
        print("Classifier Mod mean score and stdev: %.4f (+/- %.4f)" % (mod_clf.mean, mod_clf.std))
        print("Classifier Bag mean score and stdev: %.4f (+/- %.4f)" % (bag_clf.mean, bag_clf.std))
        print("Classifier Boost mean score and stdev: %.4f (+/- %.4f)" % (boost_clf.mean, boost_clf.std))
        # print("Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[0]), np.std(diff_scores[0])))

    def get_mean_result_2(self):
        table = PrettyTable()
        table.field_names = ["Dataset", "Num records", "Num classes", "Mod clf - mean", "Mod clf - std",
                             "Bagging clf - mean", "Bagging clf - std", "Boosting clf - mean", "Boosting clf - std"]

        mean_clf1, mean_clf2, mean_clf3 = [], [], []
        std_clf1, std_clf2, std_clf3 = [], [], []

        for key, value in self.result.items():
            num_records = value.num_records[0]
            num_classes = value.num_classes

            clf1_m = value.mod_clf.mean
            clf2_m = value.bag_clf.mean
            clf3_m = value.boost_clf.mean
            clf1_s = value.mod_clf.std
            clf2_s = value.bag_clf.std
            clf3_s = value.boost_clf.std

            mean_clf1.append(clf1_m)
            mean_clf2.append(clf2_m)
            mean_clf3.append(clf3_m)
            std_clf1.append(clf1_s)
            std_clf2.append(clf2_s)
            std_clf3.append(clf3_s)

            table.add_row([key, num_records, num_classes, clf1_m, clf1_s, clf2_m, clf2_s, clf3_m, clf3_s])

        print(table)

        return mean_clf1, std_clf1, mean_clf2, std_clf2, mean_clf3, std_clf3

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
        table.field_names = ["Dataset", "t - mod_bag", "t - mod_boost", "Differ - bag", "Differ - boost", "Num records", "Num classes", "Mod clf - mean", "Bagging clf - mean",
                             "Boosting clf - mean", "Mod clf - std", "Bagging clf - std", "Boosting clf - std"]

        for key, data_res in self.result.items():
            print(f"k = {self.k}")
            print(f"n = {self.n}")
            print(f"max_depth = {self.max_depth}")

            t_bag = round(data_res.t_student_mod_bag, 2)
            t_boost = round(data_res.t_student_mod_boost, 2)

            table.add_row([key, t_bag, t_boost, "+" if t_bag > 2.71 else "-", "+" if t_boost > 2.71 else "-",
                           data_res.num_records[0], data_res.num_classes,
                           data_res.mod_clf.mean, data_res.bag_clf.mean, data_res.boost_clf.mean,
                           data_res.mod_clf.std, data_res.bag_clf.std, data_res.boost_clf.std])
        print(table)




    def display_figure(self):

        mean_mod, mean_bag, mean_boost, std_mod, std_bag, std_boost, *_ = self.get_mean_std_lists()

        fig, ax = plt.subplots(figsize=(20, 10))
        n_groups = len(self.result)
        index = np.arange(n_groups)
        bar_width = 0.27

        opacity = .7
        error_config = {'ecolor': '0.2'}

        modified_clf = ax.bar(index, mean_mod, bar_width, alpha=opacity, color='g', yerr=std_mod,
                              error_kw=error_config, label='Boosting in Bagging')
        bagging_clf = ax.bar(index + bar_width, mean_bag, bar_width, alpha=opacity, color='r', yerr=std_bag,
                              error_kw=error_config, label='Bagging')
        boosting_clf = ax.bar(index + (2 * bar_width), mean_boost, bar_width, alpha=opacity, color='y', yerr=std_boost,
                             error_kw=error_config, label='Boosting')

        ax.set_xlabel('Classifiers')
        ax.set_ylabel('Accuracy scores with variance')
        ax.set_title('Scores by classifier groups and datasets')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels((self.result.keys()))
        ax.legend()

        fig.tight_layout()
        plt.show()
