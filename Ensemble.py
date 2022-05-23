import sklearn
import scipy
import numpy as np
import pandas as pd
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from Dataset import Dataset
from cls_iris import dataset_adult, dataset_iris, dataset_car, process_clfs


class Ensemble:
    seeds = [54, 50, 7897, 4589, 8897]
    n = 20
    k = 20
    n_jobs = -1
    n_samples_max = 0.1

    result = {}

    def cross_val_5x2(self, dataset):
        print("", end="\n\n")
        print("*" * 60, end="\n\n")
        print(dataset.name)

        X, y = dataset.x_features, dataset.y_labels
        nSamples = int(self.n_samples_max * len(y) * .5)

        # Initialize the score difference for the 1st fold of the 1st iteration
        p_1_1 = 0.0

        # Initialize a place holder for the variance estimate
        s_sqr = 0.0

        # Initialize scores list for both classifiers
        scores_1 = []
        scores_2 = []
        diff_scores = []

        # Iterate through 5 2-fold CV
        for i_s, seed in enumerate(self.seeds):

            # Split the dataset in 2 parts with the current seed
            folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

            # Initialize score differences
            p_i = np.zeros(2)

            # Go through the current 2 fold
            for i_f, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

                # Split the data
                trn_x, trn_y = X[trn_idx], y[trn_idx]
                val_x, val_y = X[val_idx], y[val_idx]

                classifier_list = []
                for i in range(self.n):
                    X_sample = resample(trn_x, n_samples=nSamples, random_state=i, replace=True)
                    y_sample = resample(trn_y, n_samples=nSamples, random_state=i, replace=True)
                    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.k)
                    ada_clf.fit(X_sample, y_sample)
                    classifier_list.append(('ada' + str(i), ada_clf))

                #clf1 = VotingClassifier(estimators=classifier_list, voting='soft')
                clf1 = BaggingClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.n, bootstrap=True,
                                         n_jobs=self.n_jobs)
                #clf2 = VotingClassifier(estimators=classifier_list, voting='soft', n_jobs=self.n_jobs)
                # clf2 = BaggingClassifier(AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=k), n_estimators=n, bootstrap=True)

                #clf1 = EnsembleVoteClassifier(clfs=process_clfs(seed), voting='soft')
                clf2 = EnsembleVoteClassifier(clfs=[clf[1] for clf in classifier_list], voting='soft')

                # Train classifiers
                clf1.fit(trn_x, trn_y)
                clf2.fit(trn_x, trn_y)

                # Compute scores
                try:
                    preds_1 = clf1.predict_proba(val_x)[:, 1]
                    preds_2 = clf2.predict_proba(val_x)[:, 1]
                    score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
                    score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')
                except np.AxisError:
                    preds_1 = clf1.predict_proba(val_x)
                    preds_2 = clf2.predict_proba(val_x)
                    score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
                    score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')

                # keep score history for mean and stdev calculation
                scores_1.append(score_1)
                scores_2.append(score_2)
                diff_scores.append(score_1 - score_2)
                print("Fold %2d score difference = %.6f" % (i_f + 1, score_1 - score_2))
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
        print("t value:", abs(t_bar))
        "Accuracy: %0.4f (+/- %0.4f) [Normal %s]"

        print("Classifier 1 mean score and stdev: %.4f (+/- %.4f)" % (np.mean(scores_1), np.std(scores_1)))
        print("Classifier 2 mean score and stdev: %.4f (+/- %.4f)" % (np.mean(scores_2), np.std(scores_2)))

        print("Score difference mean (+/- stdev): %.4f (+/- %.4f)"
              % (np.mean(diff_scores), np.std(diff_scores)))

        self.result[dataset.name] = [[round(np.mean(scores_1), 4), round(np.std(scores_1), 4)],
                                     [round(np.mean(scores_2), 4), round(np.std(scores_2), 4)]]


ens = Ensemble()
datasets = Dataset.get_datasets('WineQualityRed')

# print(datasets[0]._to_df())

for dataset in datasets:
    ens.cross_val_5x2(dataset)

print(ens.result)

mean_clf1 = []
mean_clf2 = []
std_clf1 = []
std_clf2 = []
print("\t\tbag clf, our clf")
for i in ens.result:
    print(f"{i}:", ens.result[i][0][0], ens.result[i][1][0])
    mean_clf1.append(ens.result[i][0][0])
    mean_clf2.append(ens.result[i][1][0])
    std_clf1.append(ens.result[i][0][1])
    std_clf2.append(ens.result[i][1][1])

print("\nbag clf, our clf")
print(np.mean(mean_clf1), np.mean(mean_clf2))

fig, ax = plt.subplots(figsize=(20, 10))
n_groups = len(ens.result)
index = np.arange(n_groups)
bar_width = 0.35

opacity = .7
error_config = {'ecolor': '0.2'}

normal_clf = ax.bar(index, mean_clf1, bar_width, alpha=opacity, color='g', yerr=std_clf1,
                    error_kw=error_config, label='Bagging')
bagging_clf = ax.bar(index + bar_width, mean_clf2, bar_width, alpha=opacity, color='y', yerr=std_clf2,
                     error_kw=error_config, label='Boosting in bagging')

ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy scores with variance')
ax.set_title('Scores by classifier groups and datasets')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels((ens.result.keys()))
ax.legend()

fig.tight_layout()
plt.show()

# # iris_dataset = dataset_adult()
# iris_dataset = Dataset.get_dataset("iris")
#
# X, y = iris_dataset.x_features, iris_dataset.y_labels
#
# nSamples = int(0.1 * len(y))
#
# n = 20
# k = 20
#
# # clf1 = LGBMClassifier(n_estimators=100, n_jobs=2)
# # clf2 = LGBMClassifier(n_estimators=100, reg_alpha=1, reg_lambda=1, min_split_gain=2, n_jobs=2)
#
# # Choose seeds for each 2-fold iterations
# seeds = [13, 51, 137, 24659, 347]
#
# # Initialize the score difference for the 1st fold of the 1st iteration
# p_1_1 = 0.0
#
# # Initialize a place holder for the variance estimate
# s_sqr = 0.0
#
# # Initialize scores list for both classifiers
# scores_1 = []
# scores_2 = []
# diff_scores = []
#
# # Iterate through 5 2-fold CV
# for i_s, seed in enumerate(seeds):
#
#     # Split the dataset in 2 parts with the current seed
#     folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
#
#     # Initialize score differences
#     p_i = np.zeros(2)
#
#     # Go through the current 2 fold
#     for i_f, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#
#         # Split the data
#         trn_x, trn_y = X[trn_idx], y[trn_idx]
#         val_x, val_y = X[val_idx], y[val_idx]
#
#         classifierList = []
#         for i in range(n):
#             X_sample = resample(trn_x, n_samples=nSamples, random_state=i, replace=True)
#             y_sample = resample(trn_y, n_samples=nSamples, random_state=i, replace=True)
#             ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=k)
#             ada_clf.fit(X_sample, y_sample)
#             classifierList.append((str('ada' + str(i)), ada_clf))
#
#         # clf1 = VotingClassifier(estimators=classifierList, voting='soft')
#         clf1 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n, bootstrap=True)
#         # clf2 = VotingClassifier(estimators=classifierList, voting='soft')
#         # clf2 = BaggingClassifier(AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=k), n_estimators=n, bootstrap=True)
#
#         # clf1 = EnsembleVoteClassifier(clfs=[clf[1] for clf in classifierList], voting='soft')
#         clf2 = EnsembleVoteClassifier(clfs=[clf[1] for clf in classifierList], voting='soft')
#
#         # Train classifiers
#         clf1.fit(trn_x, trn_y)
#         clf2.fit(trn_x, trn_y)
#
#         # Compute scores
#         preds_1 = clf1.predict_proba(val_x)[:, 1]
#         score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
#         preds_2 = clf2.predict_proba(val_x)[:, 1]
#         score_2 = roc_auc_score(val_y, preds_2, multi_class='ovr')
#
#         # keep score history for mean and stdev calculation
#         scores_1.append(score_1)
#         scores_2.append(score_2)
#         diff_scores.append(score_1 - score_2)
#         print("Fold %2d score difference = %.6f" % (i_f + 1, score_1 - score_2))
#         # Compute score difference for current fold
#         p_i[i_f] = score_1 - score_2
#         # Keep the score difference of the 1st iteration and 1st fold
#         if (i_s == 0) & (i_f == 0):
#             p_1_1 = p_i[i_f]
#     # Compute mean of scores difference for the current 2-fold CV
#     p_i_bar = (p_i[0] + p_i[1]) / 2
#     # Compute the variance estimate for the current 2-fold CV
#     s_i_sqr = (p_i[0] - p_i_bar) ** 2 + (p_i[1] - p_i_bar) ** 2
#     # Add up to the overall variance
#     s_sqr += s_i_sqr
#
# # Compute t value as the first difference divided by the square root of variance estimate
#
# t_bar = p_1_1 / ((s_sqr / 5) ** .5)
# print(abs(t_bar))
#
# print("Classifier 1 mean score and stdev : %.6f + %.6f" % (np.mean(scores_1), np.std(scores_1)))
# print("Classifier 2 mean score and stdev : %.6f + %.6f" % (np.mean(scores_2), np.std(scores_2)))
# print("Diff mean score and stdev : %.6f + %.6f" % (
#     np.mean([np.mean(scores_1), np.mean(scores_2)]), np.std([np.mean(scores_1), np.mean(scores_2)])))
#
# print("Score difference mean + stdev : %.6f + %.6f"
#       % (np.mean(diff_scores), np.std(diff_scores)))
