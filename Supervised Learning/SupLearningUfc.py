import numpy as np
import pickle
import pandas as pd

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

# tree_params = {'max_depth': range(50, 100, 10),
#                'min_impurity_decrease': np.arange(0.0, 1., 0.05)}
# tree_model_ufc = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), tree_params)
# tree_model_ufc.fit(X_train_ufc, y_train_ufc)
# pickle.dump(tree_model_ufc, open('tree.pkl', 'wb'))

# nn_params = {'alpha': [.001, .0001, .00001],
#              'activation': ["identity", "logistic", "tanh", "relu"],
#              'hidden_layer_sizes': [(100, 100, 100, 100), (100, 100, 100, 100, 100), (100, 100, 100, 100, 100, 100)]}
# nn_model_ufc = model_selection.GridSearchCV(neural_network.MLPClassifier(), nn_params)
# nn_model_ufc.fit(X_train_ufc, y_train_ufc)
# pickle.dump(nn_model_ufc, open('nn.pkl', 'wb'))


# boosted_params = {'max_depth': range(2, 8, 2),
#                   'min_impurity_decrease': np.arange(0.0, 1., 0.05),
#                   'n_estimators': range(10, 50, 10)}
# boosted_model_ufc = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), boosted_params)
# boosted_model_ufc.fit(X_train_ufc, y_train_ufc)
# pickle.dump(boosted_model_ufc, open('boosted.pkl', 'wb'))
# print 'done'

# change target from multi class to single integer
y_train_ufc_single = list()


svm_params = {'C': [.001, .01, .1, 1, 10],
              'gamma': [0.001, 0.01, 0.1, 1],
              'kernel': ['linear', 'poly', 'rbf']}
svm_model_ufc = model_selection.GridSearchCV(svm.SVC(max_iter=300), svm_params)
svm_model_ufc.fit(X_train_ufc, y_train_ufc)
pickle.dump(svm_model_ufc, open('svm.pkl', 'wb'))

#
# knn_params = {'n_neighbors': [3, 5, 7, 15]}
# knn_model_ufc = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), knn_params)
# knn_model_ufc.fit(X_train_ufc, y_train_ufc)
# pickle.dump(knn_model_ufc, open('knn.pkl', 'wb'))

