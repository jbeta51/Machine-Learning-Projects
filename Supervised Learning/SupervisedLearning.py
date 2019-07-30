import numpy as np
import pickle
import pandas as pd

from sklearn import model_selection

from sklearn import tree
from sklearn import neural_network
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors

from sklearn import datasets

cancer = datasets.load_breast_cancer()
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = \
    model_selection.train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

tree_params = {'max_depth': range(3, 20),
               'min_impurity_decrease': np.arange(0.0, 1., 0.05)}
tree_model_cancer = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), tree_params)
tree_model_cancer.fit(X_train_cancer, y_train_cancer)
pickle.dump(tree_model_cancer, open('tree.pkl', 'wb'))


nn_params = {'alpha': [.001, .0001, .00001],
             'activation': ["identity", "logistic", "tanh", "relu"],
             'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)]}
nn_model_cancer = model_selection.GridSearchCV(neural_network.MLPClassifier(), nn_params)
nn_model_cancer.fit(X_train_cancer, y_train_cancer)
pickle.dump(nn_model_cancer, open('nn.pkl', 'wb'))


boosted_params = {'max_depth': range(1, 4),
                  'min_impurity_decrease': np.arange(0.75, 1., 0.05),
                  'n_estimators': range(70, 110, 10)}
boosted_model_cancer = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), boosted_params)
boosted_model_cancer.fit(X_train_cancer, y_train_cancer)
pickle.dump(boosted_model_cancer, open('boosted.pkl', 'wb'))


svm_params = {'C': [.001, .01, .1, 1, 10],
              'gamma': [0.001, 0.01, 0.1, 1],
              'kernel': ['linear', 'poly', 'rbf']}
svm_model_cancer = model_selection.GridSearchCV(svm.SVC(max_iter=300), svm_params)
svm_model_cancer.fit(X_train_cancer, y_train_cancer)
pickle.dump(svm_model_cancer, open('svm.pkl', 'wb'))


knn_params = {'n_neighbors': [3, 5, 7, 15]}
knn_model_cancer = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), knn_params)
knn_model_cancer.fit(X_train_cancer, y_train_cancer)
pickle.dump(knn_model_cancer, open('knn.pkl', 'wb'))


best_tree = tree_model_cancer.best_estimator_
best_nn = nn_model_cancer.best_estimator_
best_boosted = boosted_model_cancer.best_estimator_
best_svm = svm_model_cancer.best_estimator_
best_knn = knn_model_cancer.best_estimator_
