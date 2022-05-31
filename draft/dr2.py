
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
