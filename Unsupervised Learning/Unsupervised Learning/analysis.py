import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn import neural_network


def groundTruthIris():
    iris = datasets.load_iris()

    # Dataset Slicing
    x_axis = iris.data[:, 0]  # Sepal Length
    y_axis = iris.data[:, 2]  # Sepal Width

    # Plotting
    plt.scatter(x_axis, y_axis, c=iris.target)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")

    plt.show()

def groundTruthCancer():
    cancer = datasets.load_breast_cancer()

    # Dataset Slicing
    x_axis = cancer.data[:, 3]  # Area Mean
    y_axis = cancer.data[:, 5]  # Compactness Mean

    # Plotting
    print cancer.target
    plt.scatter(x_axis, y_axis, c=cancer.target)
    plt.xlabel("Area Mean")
    plt.ylabel("Compactness Mean")

    plt.show()
    plt.clf()


def iterationsKMeansIris():
    iris = datasets.load_iris()
    x_axis = iris.data[:, 0]  # Sepal Length
    y_axis = iris.data[:, 2]  # Sepal Width
    est = KMeans(init="random", n_clusters=3, max_iter=10, random_state=427, n_init=1, verbose=1)
    est.fit(iris.data)
    labels = est.labels_
    centers = est.cluster_centers_

    plt.scatter(x_axis, y_axis, c=labels.astype(np.float))
    for c in centers:
        plt.plot(c[0], c[2], "or", marker="o", markersize=12)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")

    plt.show()
    plt.clf()

def iterationsKMeansCancer():
    cancer = datasets.load_breast_cancer()
    x_axis = cancer.data[:, 3]  # Area Mean
    y_axis = cancer.data[:, 5]  # Compactness Mean
    est = KMeans(init="random", n_clusters=2, max_iter=10, random_state=427, n_init=1, verbose=1)
    est.fit(cancer.data)
    labels = est.labels_
    centers = est.cluster_centers_

    plt.scatter(x_axis, y_axis, c=labels.astype(np.float))
    for c in centers:
        plt.plot(c[3], c[5], "or", marker="o", markersize=12)
    plt.xlabel("Area Mean")
    plt.ylabel("Compactness Mean")

    plt.show()
    plt.clf()

def draw_ellipse(position, covariance, **kwargs):
    # Convert covariance to principal axes
    U, s, Vt = np.linalg.svd(covariance)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * np.sqrt(s)

    for nsig in range(1, 4):
        plt.gca().add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))



def iterationsGMMIris(k):
    iris = datasets.load_iris()
    x_axis = iris.data[:, 0]  # Sepal Length
    y_axis = iris.data[:, 2]  # Sepal Width
    est = GMM(n_components=3, n_iter=20, random_state=427, n_init=1, verbose=1, covariance_type=k)
    est.fit(iris.data)
    labels = est.predict(iris.data)
    centers = est.means_

    plt.scatter(x_axis, y_axis, c=labels.astype(np.float))
    for c in centers:
        plt.plot(c[0], c[2], "or", marker="o", markersize=12)

    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title(k)

    plt.show()
    plt.clf()

def iterationsGMMCancer(k):
    cancer = datasets.load_breast_cancer()
    x_axis = cancer.data[:, 3]  # Area Mean
    y_axis = cancer.data[:, 5]  # Compactness Mean
    est = GMM(n_components=2, n_iter=20, random_state=427, n_init=1, verbose=1, covariance_type=k)
    est.fit(cancer.data)
    labels = est.predict(cancer.data)
    centers = est.means_

    plt.scatter(x_axis, y_axis, c=labels.astype(np.float))
    for c in centers:
        plt.plot(c[3], c[5], "or", marker="o", markersize=12)
    plt.xlabel("Area Mean")
    plt.ylabel("Compactness Mean")
    plt.title(k)

    plt.show()
    plt.clf()


def PCAIris():
    iris = datasets.load_iris()
    train = iris.data[:int(.8 * len(iris.data))]
    pca = PCA(n_components=2)
    pca.fit(train)
    new_iris = pca.transform(iris.data)

    # Plotting
    plt.scatter(new_iris[:, 0], new_iris[:, 1], c=iris.target)
    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    plt.title("Iris PCA reduced")

    plt.show()
    plt.clf()


def PCACancer():
    cancer = datasets.load_breast_cancer()
    train = cancer.data[:int(.8 * len(cancer.data))]
    pca = PCA(n_components=2)
    pca.fit(train)
    new_cancer = pca.transform(cancer.data)

    # Plotting
    plt.scatter(new_cancer[:, 0], new_cancer[:, 1], c=cancer.target)
    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    plt.title("Cancer PCA reduced")

    plt.show()
    plt.clf()


def reducedClusteringIris():
    iris = datasets.load_iris()
    train = iris.data[:int(.8 * len(iris.data))]
    pca = PCA(n_components=2)
    pca.fit(train)
    new_iris = pca.transform(iris.data)

    est = KMeans(init="random", n_clusters=3, max_iter=10, random_state=427, n_init=1, verbose=1)
    est.fit(new_iris)
    labels = est.labels_
    centers = est.cluster_centers_

    plt.scatter(new_iris[:, 0], new_iris[:, 1], c=labels.astype(np.float))
    for c in centers:
        print c
        plt.plot(c[0], c[1], "or", marker="o", markersize=12)
    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    plt.title("Reduced K-Means Iris")

    plt.show()
    plt.clf()

    est = GMM(n_components=3, n_iter=20, random_state=427, n_init=1, verbose=1, covariance_type='diag')
    est.fit(new_iris)
    labels = est.predict(new_iris)
    centers = est.means_

    plt.scatter(new_iris[:, 0], new_iris[:, 1], c=labels.astype(np.float))
    for c in centers:
        print c
        plt.plot(c[0], c[1], "or", marker="o", markersize=12)
    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    plt.title("Reduced GMM Iris")

    plt.show()
    plt.clf()



def reducedClusteringCancer():
    cancer = datasets.load_breast_cancer()
    train = cancer.data[:int(.8 * len(cancer.data))]
    pca = PCA(n_components=2)
    pca.fit(train)
    new_cancer = pca.transform(cancer.data)

    est = KMeans(init="random", n_clusters=2, max_iter=10, random_state=427, n_init=1, verbose=1)
    est.fit(new_cancer)
    labels = est.labels_
    centers = est.cluster_centers_

    plt.scatter(new_cancer[:, 0], new_cancer[:, 1], c=labels.astype(np.float))
    for c in centers:
        plt.plot(c[0], c[1], "or", marker="o", markersize=12)
    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    plt.title("Reduced K-Means Cancer")

    plt.show()
    plt.clf()

    est = GMM(n_components=2, n_iter=20, random_state=427, n_init=1, verbose=1, covariance_type='full')
    est.fit(new_cancer)
    labels = est.predict(new_cancer)
    centers = est.means_

    plt.scatter(new_cancer[:, 0], new_cancer[:, 1], c=labels.astype(np.float))
    for c in centers:
        print c
        plt.plot(c[0], c[1], "or", marker="o", markersize=12)
    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    plt.title("Reduced GMM Cancer")

    plt.show()
    plt.clf()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

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


def reducedCancerNN():
    cancer = datasets.load_breast_cancer()
    train = cancer.data[:int(.8 * len(cancer.data))]
    pca = PCA(n_components=.9999)
    pca.fit(train)
    new_cancer = pca.transform(cancer.data)
    print new_cancer
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = \
        train_test_split(new_cancer, cancer.target, test_size=0.2, random_state=42)
    nn = neural_network.MLPClassifier(alpha=0.001, activation='logistic', hidden_layer_sizes=(100,))
    plot_learning_curve(nn, "Learning Curves: Neural Network", X_train_cancer, y_train_cancer, cv=3)
    plt.show()
    plt.clf()


def kMeansFeaturesCancer():
    cancer = datasets.load_breast_cancer()
    est = KMeans(init="random", n_clusters=4, max_iter=10, random_state=427, n_init=1, verbose=1)
    est.fit(cancer.data)
    labels = est.predict(cancer.data)
    centers = est.cluster_centers_

    pca = PCA(n_components=2)
    pca.fit(cancer.data)
    new_cancer = pca.transform(cancer.data)
    new_centers = pca.transform(centers)

    plt.scatter(new_cancer[:, 0], new_cancer[:, 1], c=labels.astype(np.float))
    for c in new_centers:
        plt.plot(c[0], c[1], "or", marker="o", markersize=12)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

    plt.show()
    plt.clf()

    # run nn on clusters
    nn = neural_network.MLPClassifier(alpha=0.001, activation='relu', hidden_layer_sizes=(100,))
    plot_learning_curve(nn, "NN on KMeans: Learning Curves", labels.reshape(-1, 1), cancer.target, cv=3)
    plt.show()


def gmmFeaturesCancer():
    cancer = datasets.load_breast_cancer()
    est = GMM(n_components=2, n_iter=20, random_state=427, n_init=1, verbose=1, covariance_type='diag')
    est.fit(cancer.data)
    labels = est.predict(cancer.data)
    centers = est.means_

    pca = PCA(n_components=2)
    pca.fit(cancer.data)
    new_cancer = pca.transform(cancer.data)
    new_centers = pca.transform(centers)

    plt.scatter(new_cancer[:, 0], new_cancer[:, 1], c=labels.astype(np.float))
    for c in new_centers:
        plt.plot(c[0], c[1], "or", marker="o", markersize=12)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

    plt.show()
    plt.clf()

    # run nn on clusters
    nn = neural_network.MLPClassifier(alpha=0.001, activation='relu', hidden_layer_sizes=(100,))
    plot_learning_curve(nn, "NN on GMM: Learning Curves", labels.reshape(-1, 1), cancer.target, cv=3)
    plt.show()

# ************** MAIN SCRIPT

# groundTruthIris()
# groundTruthCancer()

# iterationsKMeansIris()
# iterationsKMeansCancer()


# for i in ['spherical', 'tied', 'diag', 'full']:
#     iterationsGMMIris(i)
#
# for i in ['spherical', 'tied', 'diag', 'full']:
#     iterationsGMMCancer(i)
#
# PCAIris()
# PCACancer()
#
# reducedClusteringIris()
# reducedClusteringCancer()

# reducedCancerNN()
# kMeansFeaturesCancer()
# gmmFeaturesCancer()



