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


class Algorithm:
    seeds = [random.randint(1, 50000) for i in range(5)]
    n = 20
    k = 20
    n_jobs = -1
    n_samples_max = 0.1
    alfa = .05

    result = {}

    @print_dataset_name
    def cross_val_5x2_backup(self, dataset):

        t_statistic = np.zeros((3, 3))
        p_value = np.zeros((3, 3))          # num classifiers

        X, y = dataset.x_features, dataset.y_labels

        n_samples = int(self.n_samples_max * len(y) * .5)

        # Initialize the score difference for the 1st fold of the 1st iteration
        p_1_1 = 0.0
        # Initialize a place holder for the variance estimate
        s_sqr = 0.0
        # Initialize scores list for both classifiers
        scores_1, scores_2, scores_3, diff_scores = [], [], [], []

        # Iterate through 5 2-fold CV
        for i_s, seed in enumerate(self.seeds):

            # Split the dataset in 2 parts with the current seed
            folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            # Initialize score differences
            p_i = np.zeros(2)

            # Go through the current 2 fold
            for i_f, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

                # Split the data - 50 / 50
                trn_x, trn_y = X[trn_idx], y[trn_idx]
                val_x, val_y = X[val_idx], y[val_idx]

                classifier_list = self.generate_boosting_list(trn_x, trn_y, n_samples)
                clf_modified_1 = VotingClassifier(estimators=classifier_list, voting='soft')

                clf_bagging_2 = BaggingClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.n,
                                                  bootstrap=True,
                                                  n_jobs=self.n_jobs)

                clf_boosting_3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.k)

                # Train classifiers
                clf_modified_1.fit(trn_x, trn_y)
                clf_bagging_2.fit(trn_x, trn_y)
                clf_boosting_3.fit(trn_x, trn_y)

                # Compute scores
                try:
                    preds_1 = clf_modified_1.predict_proba(val_x)[:, 1]
                    preds_2 = clf_bagging_2.predict_proba(val_x)[:, 1]
                    preds_3 = clf_boosting_3.predict_proba(val_x)[:, 1]
                    score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
                    score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')
                    score_3 = roc_auc_score(val_y, preds_3, multi_class='ovr')
                except np.AxisError:
                    preds_1 = clf_modified_1.predict_proba(val_x)
                    preds_2 = clf_bagging_2.predict_proba(val_x)
                    preds_3 = clf_boosting_3.predict_proba(val_x)
                    score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
                    score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')
                    score_3 = roc_auc_score(val_y, preds_3, multi_class='ovr')

                # keep score history for mean and stdev calculation
                scores_1.append(score_1)
                scores_2.append(score_2)
                scores_3.append(score_3)
                diff_scores.append((score_1 - score_2, scores_1 - score_3))
                print("Fold %2d score difference between clf_modified and clf_bagging = %.6f" % (
                    i_f + 1, score_1 - score_2))
                print("Fold %2d score difference between clf_modified and clf_boosting = %.6f" % (
                    i_f + 1, score_1 - score_3))

                # Compute score difference for current fold
                p_i[i_f] = score_1 - score_2
                # Keep the score difference of the 1st iteration and 1st fold
                if (i_s == 0) & (i_f == 0):
                    p_1_1 = p_i[i_f]

            # Compute mean of scores difference for the current 2-fold CV
            p_i_bar = (p_i[0] + p_i[1]) / 2
            # Compute the variance estimate for the current 2-fold CV
            s_i_sqr = (p_i[0] - p_i_bar) ** 2 + (p_i[1] - p_i_bar) ** 2
            # Add up to the overall variance
            s_sqr += s_i_sqr

        # Compute t value as the first difference divided by the square root of variance estimate
        t_bar = p_1_1 / ((s_sqr / 5) ** .5)



        self.result[dataset.name] = {
            "info": {
                "num_records": len(y),
                "num_classes": len(set(y))
            },
            "clf1": {
                "mean": round(np.mean(scores_1), 4),
                "std": round(np.std(scores_1), 4)
            },
            "clf2": {
                "mean": round(np.mean(scores_2), 4),
                "std": round(np.std(scores_2), 4)
            },
            "clf3": {
                "mean": round(np.mean(scores_3), 4),
                "std": round(np.std(scores_3), 4)
            }
        }
        print("ttest_rel:", ttest_rel(scores_1, scores_2))
        print("t value:", abs(t_bar))
        print("Classifier 1 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf1"]["mean"],
                                                                      self.result[dataset.name]["clf1"]["std"]))
        print("Classifier 2 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf2"]["mean"],
                                                                      self.result[dataset.name]["clf2"]["std"]))
        print("Classifier 3 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf3"]["mean"],
                                                                      self.result[dataset.name]["clf3"]["std"]))
        print("Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[0]), np.std(diff_scores[0])))
        # print("[Mod and Bag] Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[:][0]), np.std(diff_scores[:][0])))
        # print("[Mod and Boost] Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[:][1]), np.std(diff_scores[:][1])))

    @print_dataset_name
    def cross_val_5x2_3(self, dataset):
        t_statistic = np.zeros((3, 3))
        p_value = np.zeros((3, 3))  # num classifiers

        X, y = dataset.x_features, dataset.y_labels

        n_samples = int(self.n_samples_max * len(y) * .5)

        # Initialize the score difference for the 1st fold of the 1st iteration
        p_1_1_bag = 0.0
        p_1_1_boost = 0.0
        # Initialize a place holder for the variance estimate
        s_sqr_bag = 0.0
        s_sqr_boost = 0.0
        # Initialize scores list for both classifiers
        scores_1, scores_2, scores_3, diff_scores = [], [], [], []

        # Iterate through 5 2-fold CV
        for i_s, seed in enumerate(self.seeds):

            # Split the dataset in 2 parts with the current seed
            folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            # Initialize score differences
            p_i_bag = np.zeros(2)
            p_i_boost = np.zeros(2)

            # Go through the current 2 fold
            for i_f, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

                # Split the data - 50 / 50
                trn_x, trn_y = X[trn_idx], y[trn_idx]
                val_x, val_y = X[val_idx], y[val_idx]

                classifier_list = self.generate_boosting_list(trn_x, trn_y, n_samples)
                clf_modified_1 = VotingClassifier(estimators=classifier_list, voting='soft')

                clf_bagging_2 = BaggingClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.n,
                                                  bootstrap=True,
                                                  n_jobs=self.n_jobs)

                clf_boosting_3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.k)

                # Train classifiers
                clf_modified_1.fit(trn_x, trn_y)
                clf_bagging_2.fit(trn_x, trn_y)
                clf_boosting_3.fit(trn_x, trn_y)

                # Compute scores
                try:
                    preds_1 = clf_modified_1.predict_proba(val_x)[:, 1]
                    preds_2 = clf_bagging_2.predict_proba(val_x)[:, 1]
                    preds_3 = clf_boosting_3.predict_proba(val_x)[:, 1]
                    score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
                    score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')
                    score_3 = roc_auc_score(val_y, preds_3, multi_class='ovr')
                except np.AxisError:
                    preds_1 = clf_modified_1.predict_proba(val_x)
                    preds_2 = clf_bagging_2.predict_proba(val_x)
                    preds_3 = clf_boosting_3.predict_proba(val_x)
                    score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
                    score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')
                    score_3 = roc_auc_score(val_y, preds_3, multi_class='ovr')

                # keep score history for mean and stdev calculation
                scores_1.append(score_1)
                scores_2.append(score_2)
                scores_3.append(score_3)
                diff_scores.append((score_1 - score_2, scores_1 - score_3))
                print("Fold %2d score difference between clf_modified and clf_bagging = %.6f" % (
                    i_f + 1, score_1 - score_2))
                print("Fold %2d score difference between clf_modified and clf_boosting = %.6f" % (
                    i_f + 1, score_1 - score_3))

                # Compute score difference for current fold
                p_i_bag[i_f] = score_1 - score_2
                p_i_boost[i_f] = score_1 - score_3
                # Keep the score difference of the 1st iteration and 1st fold
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

        # Compute t value as the first difference divided by the square root of variance estimate
        t_bar_bag = p_1_1_bag / ((s_sqr_bag / 5) ** .5)
        t_bar_boost = p_1_1_boost / ((s_sqr_boost / 5) ** .5)

        self.result[dataset.name] = {
            "info": {
                "num_records": len(y),
                "num_classes": len(set(y))
            },
            "clf1": {
                "mean": round(np.mean(scores_1), 4),
                "std": round(np.std(scores_1), 4)
            },
            "clf2": {
                "mean": round(np.mean(scores_2), 4),
                "std": round(np.std(scores_2), 4)
            },
            "clf3": {
                "mean": round(np.mean(scores_3), 4),
                "std": round(np.std(scores_3), 4)
            }
        }
        t_statistic[0, 0], p_value[0,0] = ttest_rel(scores_1, scores_1)
        t_statistic[0, 1], p_value[0,1] = ttest_rel(scores_1, scores_2)
        t_statistic[0, 2], p_value[0,2] = ttest_rel(scores_1, scores_3)

        print(t_statistic)

        print("ttest_rel:", ttest_rel(scores_1, scores_3))
        print("ttest_rel:", ttest_rel(scores_1, scores_1))
        print("t-clf2 value:", t_bar_bag)
        print("t-clf3 value:", t_bar_boost)
        print("Classifier 1 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf1"]["mean"],
                                                                      self.result[dataset.name]["clf1"]["std"]))
        print("Classifier 2 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf2"]["mean"],
                                                                      self.result[dataset.name]["clf2"]["std"]))
        print("Classifier 3 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf3"]["mean"],
                                                                      self.result[dataset.name]["clf3"]["std"]))
        print("Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[0]), np.std(diff_scores[0])))
        # print("[Mod and Bag] Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[:][0]), np.std(diff_scores[:][0])))
        # print("[Mod and Boost] Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[:][1]), np.std(diff_scores[:][1])))

    @print_dataset_name
    def cross_val_5x2_3(self, dataset):
        X, y = dataset.x_features, dataset.y_labels

        n_samples = int(self.n_samples_max * len(y) * .5)

        scores_1, scores_2, scores_3, diff_scores = [], [], [], []

        # Iterate through 5 2-fold CV
        for i_s, seed in enumerate(self.seeds):

            # Split the dataset in 2 parts with the current seed
            folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

            # Go through the current 2 fold
            for i_f, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

                # Split the data - 50 / 50
                trn_x, trn_y = X[trn_idx], y[trn_idx]
                val_x, val_y = X[val_idx], y[val_idx]

                classifier_list = self.generate_boosting_list(trn_x, trn_y, n_samples)
                clf_modified_1 = VotingClassifier(estimators=classifier_list, voting='soft')

                clf_bagging_2 = BaggingClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.n,
                                                  bootstrap=True,
                                                  n_jobs=self.n_jobs)

                clf_boosting_3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.k)

                # Train classifiers
                clf_modified_1.fit(trn_x, trn_y)
                clf_bagging_2.fit(trn_x, trn_y)
                clf_boosting_3.fit(trn_x, trn_y)

                # Compute scores
                try:
                    preds_1 = clf_modified_1.predict_proba(val_x)[:, 1]
                    preds_2 = clf_bagging_2.predict_proba(val_x)[:, 1]
                    preds_3 = clf_boosting_3.predict_proba(val_x)[:, 1]
                    score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
                    score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')
                    score_3 = roc_auc_score(val_y, preds_3, multi_class='ovr')
                except np.AxisError:
                    preds_1 = clf_modified_1.predict_proba(val_x)
                    preds_2 = clf_bagging_2.predict_proba(val_x)
                    preds_3 = clf_boosting_3.predict_proba(val_x)
                    score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
                    score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')
                    score_3 = roc_auc_score(val_y, preds_3, multi_class='ovr')

                # keep score history for mean and stdev calculation
                scores_1.append(score_1)
                scores_2.append(score_2)
                scores_3.append(score_3)
                diff_scores.append((score_1 - score_2, scores_1 - score_3))
                print("Fold %2d score difference between clf_modified and clf_bagging = %.6f" % (
                    i_f + 1, score_1 - score_2))
                print("Fold %2d score difference between clf_modified and clf_boosting = %.6f" % (
                    i_f + 1, score_1 - score_3))


        self.result[dataset.name] = {
            "info": {
                "num_records": len(y),
                "num_classes": len(set(y))
            },
            "mod_clf": {
                "mean": round(np.mean(scores_1), 4),
                "std": round(np.std(scores_1), 4)
            },
            "bag_clf": {
                "mean": round(np.mean(scores_2), 4),
                "std": round(np.std(scores_2), 4)
            },
            "boost_clf": {
                "mean": round(np.mean(scores_3), 4),
                "std": round(np.std(scores_3), 4)
            }
        }


        print("Classifier 1 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf1"]["mean"],
                                                                      self.result[dataset.name]["clf1"]["std"]))
        print("Classifier 2 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf2"]["mean"],
                                                                      self.result[dataset.name]["clf2"]["std"]))
        print("Classifier 3 mean score and stdev: %.4f (+/- %.4f)" % (self.result[dataset.name]["clf3"]["mean"],
                                                                      self.result[dataset.name]["clf3"]["std"]))
        print("Score difference mean (+/- stdev): %.4f (+/- %.4f)" % (np.mean(diff_scores[0]), np.std(diff_scores[0])))


    def generate_boosting_list(self, trn_x, trn_y, n_samples):
        classifier_list = []
        for i in range(self.n):
            X_sample = resample(trn_x, n_samples=n_samples, random_state=i, replace=True)
            y_sample = resample(trn_y, n_samples=n_samples, random_state=i, replace=True)
            ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.k)
            ada_clf.fit(X_sample, y_sample)
            classifier_list.append(('ada' + str(i), ada_clf))
        return classifier_list

    def get_mean_result(self):
        table = PrettyTable()
        table.field_names = ["Dataset", "Num records", "Num classes", "Mod clf - mean", "Mod clf - std",
                             "Bagging clf - mean", "Bagging clf - std", "Boosting clf - mean", "Boosting clf - std"]

        mean_clf1, mean_clf2, mean_clf3 = [], [], []
        std_clf1, std_clf2, std_clf3 = [], [], []

        for key, value in self.result.items():
            num_records = value["info"]["num_records"]
            num_classes = value["info"]["num_classes"]

            clf1_m = value["clf1"]["mean"]
            clf2_m = value["clf2"]["mean"]
            clf3_m = value["clf3"]["mean"]
            clf1_s = value["clf1"]["std"]
            clf2_s = value["clf2"]["std"]
            clf3_s = value["clf3"]["std"]

            mean_clf1.append(clf1_m)
            mean_clf2.append(clf2_m)
            mean_clf3.append(clf3_m)
            std_clf1.append(clf1_s)
            std_clf2.append(clf2_s)
            std_clf3.append(clf3_s)

            table.add_row([key, num_records, num_classes, clf1_m, clf1_s, clf2_m, clf2_s, clf3_m, clf3_s])

        print(table)

        return mean_clf1, std_clf1, mean_clf2, std_clf2, mean_clf3, std_clf3

    def display_figure(self):

        mean_clf1, std_clf1, mean_clf2, std_clf2, mean_clf3, std_clf3 = self.get_mean_result()

        fig, ax = plt.subplots(figsize=(20, 10))
        n_groups = len(self.result)
        index = np.arange(n_groups)
        bar_width = 0.27

        opacity = .7
        error_config = {'ecolor': '0.2'}

        modified_clf = ax.bar(index, mean_clf1, bar_width, alpha=opacity, color='g', yerr=std_clf1,
                              error_kw=error_config, label='Boosting in Bagging')
        boosting_clf = ax.bar(index + bar_width, mean_clf3, bar_width, alpha=opacity, color='r', yerr=std_clf3,
                              error_kw=error_config, label='Bagging')
        bagging_clf = ax.bar(index + (2 * bar_width), mean_clf2, bar_width, alpha=opacity, color='y', yerr=std_clf2,
                             error_kw=error_config, label='Bagging')

        ax.set_xlabel('Classifiers')
        ax.set_ylabel('Accuracy scores with variance')
        ax.set_title('Scores by classifier groups and datasets')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels((self.result.keys()))
        ax.legend()

        fig.tight_layout()
        plt.show()
