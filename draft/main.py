import sklearn
import numpy as np
import pandas as pd
import scipy
from scipy.constants import sigma
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
from Dataset import Dataset

if __name__ == "__main__":
    iris_dataset = Dataset(name="Iris",
                           feature_labels=["sepal length in cm", "sepal width in cm",
                                           "petal length in cm", "petal width in cm", "class"],
                           path_or_url="../datasets/Iris_DB/iris.data",
                           target_label="class",
                           record_count=-1,
                           test_size=0.33)

    X, y = iris_dataset.x_features, iris_dataset.y_labels

    # ======================== EXAMPLE ========================
    # https://datascience.eu/pl/uczenie-maszynowe/k-krotna-walidacja-krzyzowa/

    #
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X = scaler.fit_transform(X)
    #
    # scores = []
    # best_svr = RandomForestClassifier(n_estimators=10, random_state=42)
    # cv = KFold(n_splits=10, shuffle=True, random_state=42)
    # for train_index, test_index in cv.split(X):
    #     print('Train Index: ', train_index, '\n')
    #     print('Indeks testowy: ', test_index)
    #
    #     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    #     best_svr.fit(X_train, y_train)
    #     scores.append(int(round(best_svr.score(X_test, y_test), 2)*100))
    #
    # print(f"\nScores for every Fold [%]:\n{scores}")

    print(X, y)

    """""""""""""""""""""""""""""""""""""""CROSS VAL IS COMPLETED"""""""""""""""""""""""""""""""""""""""

    # # ======================== EXAMPLE - CROSS VAL IS COMPLETED  ========================
    #
    # clf1 = LGBMClassifier(n_estimators=100, n_jobs=2)
    # clf2 = LGBMClassifier(n_estimators=100, reg_alpha=1, reg_lambda=1, min_split_gain=2, n_jobs=2)
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
    #         print(trn_idx, val_idx)
    #
    #         # Split the data
    #         trn_x, trn_y = X.iloc[trn_idx], y.iloc[trn_idx]
    #         val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]
    #
    #         # Train classifiers
    #         clf1.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
    #         clf2.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
    #
    #         # Compute scores
    #         preds_1 = clf1.predict_proba(val_x, num_iteration=clf1.best_iteration_)
    #         score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
    #         preds_2 = clf2.predict_proba(val_x, num_iteration=clf2.best_iteration_)
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
    # print("Diff mean score and stdev : %.6f + %.6f" % (np.mean([np.mean(scores_1), np.mean(scores_2)]), np.std([np.mean(scores_1), np.mean(scores_2)])))
    #
    # print("Score difference mean + stdev : %.6f + %.6f"
    #       % (np.mean(diff_scores), np.std(diff_scores)))

    # ======================== EXAMPLE - CROSS VAL IS COMPLETED  ========================

    clf1 = LGBMClassifier(n_estimators=100, n_jobs=2)
    clf2 = LGBMClassifier(n_estimators=100, reg_alpha=1, reg_lambda=1, min_split_gain=2, n_jobs=2)

    # Choose seeds for each 2-fold iterations
    seeds = [13, 51, 137, 24659, 347]

    # Initialize the score difference for the 1st fold of the 1st iteration
    p_1_1 = 0.0

    # Initialize a place holder for the variance estimate
    s_sqr = 0.0

    # Initialize scores list for both classifiers
    scores_1 = []
    scores_2 = []
    diff_scores = []

    # Iterate through 5 2-fold CV
    for i_s, seed in enumerate(seeds):

        # Split the dataset in 2 parts with the current seed
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

        # Initialize score differences
        p_i = np.zeros(2)

        # Go through the current 2 fold
        for i_f, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

            print(trn_idx, val_idx)

            # Split the data
            trn_x, trn_y = X.iloc[trn_idx], y.iloc[trn_idx]
            val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]

            # Train classifiers
            clf1.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
            clf2.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)

            # Compute scores
            preds_1 = clf1.predict_proba(val_x, num_iteration=clf1.best_iteration_)
            score_1 = roc_auc_score(val_y, preds_1, multi_class='ovr')
            preds_2 = clf2.predict_proba(val_x, num_iteration=clf2.best_iteration_)
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
    if t_bar > 2.571:     # if greater than 2 sigma
        print("Classifiers aren't accepted")

    print("Classifier 1 mean score and stdev : %.6f + %.6f" % (np.mean(scores_1), np.std(scores_1)))
    print("Classifier 2 mean score and stdev : %.6f + %.6f" % (np.mean(scores_2), np.std(scores_2)))
    print("Diff mean score and stdev : %.6f + %.6f" % (
    np.mean([np.mean(scores_1), np.mean(scores_2)]), np.std([np.mean(scores_1), np.mean(scores_2)])))

    print("Score difference mean + stdev : %.6f + %.6f"
          % (np.mean(diff_scores), np.std(diff_scores)))











