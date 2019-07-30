import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib

from sklearn import datasets
from sklearn import model_selection
from sklearn import tree
from sklearn import neural_network
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors

X_ufc = np.load('ufcData.npy')
y_ufc = np.load('ufcTarget.npy')
test_size = int(len(y_ufc) * .8)
X_train_ufc = X_ufc[:test_size, :]
X_test_ufc = X_ufc[test_size:, :]
y_train_ufc = y_ufc[:test_size, :]
y_test_ufc = y_ufc[test_size:, :]

X_train_ufc = np.array(X_train_ufc)
X_test_ufc = np.array(X_test_ufc)
y_train_ufc = np.array(y_train_ufc)
y_test_ufc = np.array(y_test_ufc)



def analysis_cancer(estimator_grid):
    print estimator_grid.best_params_
    print estimator_grid.best_score_
    print estimator_grid.score(X_test_cancer, y_test_cancer)
    print ""

def analysis_ufc(estimator_grid):
    print estimator_grid.best_params_
    print estimator_grid.best_score_
    print estimator_grid.score(X_test_ufc, y_test_ufc)
    print ""


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


cancer = datasets.load_breast_cancer()
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = \
    model_selection.train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

tree_cancer = pickle.load(open('../models/cancer/tree.pkl', 'rb'))
nn_cancer = pickle.load(open('../models/cancer/nn.pkl', 'rb'))
boosted_cancer = pickle.load(open('../models/cancer/boosted.pkl', 'rb'))
svm_cancer = pickle.load(open('../models/cancer/svm.pkl', 'rb'))
knn_cancer = pickle.load(open('../models/cancer/knn.pkl', 'rb'))

tree_ufc = pickle.load(open('tree.pkl', 'rb'))
nn_ufc = pickle.load(open('nn.pkl', 'rb'))
boosted_ufc = pickle.load(open('boosted.pkl', 'rb'))
knn_ufc = pickle.load(open('knn.pkl', 'rb'))


analysis_cancer(tree_cancer)
analysis_cancer(nn_cancer)
analysis_cancer(boosted_cancer)
analysis_cancer(svm_cancer)
analysis_cancer(knn_cancer)

analysis_ufc(tree_ufc)
analysis_ufc(nn_ufc)
analysis_ufc(boosted_ufc)
analysis_ufc(knn_ufc)


# tree_res = pd.DataFrame.from_dict(tree_cancer.cv_results_)
# nn_res = pd.DataFrame.from_dict(nn_cancer.cv_results_)
# boosted_res = pd.DataFrame.from_dict(boosted_cancer.cv_results_)
# svm_res = pd.DataFrame.from_dict(svm_cancer.cv_results_)
# knn_res = pd.DataFrame.from_dict(knn_cancer.cv_results_)

matplotlib.rcParams.update({'font.size': 22})

df = pd.DataFrame(data=X_train_ufc)
plt.matshow(df.corr())
plt.show()


# plot_learning_curve(tree_ufc, "UFC Learning Curves: Decision Tree", X_train_ufc, y_train_ufc)
# plt.show()
#
# plot_learning_curve(nn_ufc, "UFC Learning Curves: Neural Network", X_train_ufc, y_train_ufc)
# plt.show()

plot_learning_curve(boosted_ufc, "UFC Learning Curves: Boosted Ensemble", X_train_ufc, y_train_ufc)
plt.show()
