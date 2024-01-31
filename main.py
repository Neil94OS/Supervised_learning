#Neil O'Sullivan
#R00206266
#SDH4-C
import math
import datetime

import numpy as np
import pandas as pd
from sklearn import svm, metrics, model_selection, tree
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier

# Read in csv file
data = pd.read_csv("fashion-mnist_train.csv")
vary_n = [500, 1250, 2500, 5000, 10000]
k=5

def task1():
    data_to_read= [5,7,9]
    new_data = data.loc[data['label'].isin(data_to_read)]
    labels = new_data['label']
    features = new_data.loc[:, new_data.columns != 'label']

    titles = ['Sandal', 'Sneaker', 'Ankle Boot']

    fig, axis = plt.subplots(1, len(data_to_read))


    for i, label in enumerate(data_to_read):
        # Find an image with the specified label
        image = features[label == labels].iloc[0].to_numpy().reshape(28, 28)

        axis[i].imshow(image)
        axis[i].set_title(titles[i])
        axis[i].axis('off')

    plt.show()

    return labels, features

def task2(labels, features, clf, k, n, name):
    kf = KFold(n_splits=k)

    accuracies = []
    features_n_set= features[:n]
    labels_n_set = labels[:n]

    training_times = []
    prediction_times = []
    print("Classifier : ",name)
    print("\n")

    for train_index, test_index in kf.split(features_n_set, labels_n_set):
        X_train, y_train = features_n_set.iloc[train_index, :], labels_n_set.iloc[train_index]
        X_test, y_test = features_n_set.iloc[test_index, :], labels_n_set.iloc[test_index]

        time_started = datetime.datetime.now()
        clf.fit(X_train, y_train)
        clf.predict(X_train)

        train_time = datetime.datetime.now() - time_started
        training_times.append(train_time.total_seconds())

        time_started = datetime.datetime.now()
        prediction_test = clf.predict(X_test)

        prediction_time = datetime.datetime.now() - time_started
        prediction_times.append(prediction_time.total_seconds())

        confusion = metrics.confusion_matrix(y_test, prediction_test)
        print("Confusion matrix:\n", confusion)
        print("\n")

        accuracy = metrics.accuracy_score(y_test, prediction_test)
        print("Accuracy:",accuracy)
        print("\n")
        accuracies.append(accuracy)

    min_training_time = min(training_times)
    max_training_time = max(training_times)
    avg_training_time = np.mean(training_times)

    min_prediction_time = min(prediction_times)
    max_prediction_time = max(prediction_times)
    avg_prediction_time = np.mean(prediction_times)

    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    avg_accuracy = np.mean(accuracies)

    print("Minimum training time : ", min_training_time)
    print("Maximum training time : ", max_training_time)
    print("Average training time : ", avg_training_time)
    print("\n")

    print("Minimum prediction time : ", min_prediction_time)
    print("Maximum prediction time : ", max_prediction_time)
    print("Average prediction time : ", avg_prediction_time)
    print("\n")

    print("Minimum prediction accuracy : ", min_accuracy )
    print("Maximum prediction accuracy : ", max_accuracy)
    print("Average prediction accuracy : ", avg_accuracy)
    print("\n")

    return avg_accuracy, avg_training_time, avg_prediction_time


def task3and4(labels, features, clf, name):
    training_times =[]
    pred_times=[]
    
    for n in vary_n:
        avg_accuracy, training_time, prediction_time = task2(labels, features, clf, k, n, name)
        training_times.append(training_time)
        pred_times.append(prediction_time)

        print("Mean prediction accuracy  for n ", n, " : ", avg_accuracy)
        print("\n")

    plt.subplot(1,2,1)
    plt.plot(vary_n, training_times)
    plt.suptitle(name)
    plt.title("Training time v Data size")
    plt.xlabel("Data size")
    plt.ylabel("Training time")

    plt.subplot(1, 2, 2)
    plt.plot(vary_n, pred_times)
    plt.suptitle(name)
    plt.title("Prediction time v Data size")
    plt.xlabel("Data size")
    plt.ylabel("Prediction time")

    plt.show()



def task5(labels, features):
    training_times = []
    pred_times = []
    accuracies = []
    vary_knn=[2,3,4,5,6]
    best_knn = 0
    best_accuracy=0

    for i in vary_knn:
        clf = KNeighborsClassifier(n_neighbors=i)
        avg_accuracy, training_time, prediction_time = task2(labels, features, clf, k, 30000, "K-nearest neighbour ")
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_knn = i

    clf = KNeighborsClassifier(n_neighbors=best_knn)
    for n in vary_n:
        avg_accuracy, training_time, prediction_time = task2(labels, features, clf, k, n, "K-nearest neighbour ")
        training_times.append(training_time)
        pred_times.append(prediction_time)
        accuracies.append(avg_accuracy)

    best_accuracy = np.max(accuracies)
    print("Best mean prediction accuracy : ", best_accuracy, " for k :", best_knn, "with n =", vary_n[np.array(accuracies).argmax()])
    print("\n")

    plt.subplot(1, 2, 1)
    plt.plot(vary_n, training_times)
    plt.suptitle("K-nearest neighbour with k = " + str(best_knn))
    plt.title("Training time v Data size")
    plt.xlabel("Data size")
    plt.ylabel("Training time")

    plt.subplot(1, 2, 2)
    plt.plot(vary_n, pred_times)
    plt.suptitle("K-nearest neighbour with k = " + str(best_knn))
    plt.title("Prediction time v Data size")
    plt.xlabel("Data size")
    plt.ylabel("Prediction time")

    plt.show()


def task6(labels, features):
    training_times = []
    pred_times = []
    accuracies = []
    vary_gamma = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001]
    best_gamma = 0
    best_accuracy = 0

    for i in vary_gamma:
        clf = svm.SVC(kernel='rbf', gamma=i)
        avg_accuracy, training_time, prediction_time = task2(labels, features, clf, k, 1000, "Support Vector Machine")
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_gamma = i

    clf = svm.SVC(kernel='rbf', gamma=best_gamma)

    for n in vary_n:
        avg_accuracy, training_time, prediction_time = task2(labels, features, clf, k, n, "Support Vector Machine")
        training_times.append(training_time)
        pred_times.append(prediction_time)
        accuracies.append(avg_accuracy)

    best_accuracy = np.max(accuracies)
    print("Best mean prediction accuracy : ", best_accuracy, " for gamma :", best_gamma , "with n =", vary_n[np.array(accuracies).argmax()])
    print("\n")

    plt.subplot(1, 2, 1)
    plt.plot(vary_n, training_times)
    plt.suptitle("Support Vector Machine with gamma = " + str(best_gamma))
    plt.title("Training time v Data size")
    plt.xlabel("Data size")
    plt.ylabel("Training time")

    plt.subplot(1, 2, 2)
    plt.plot(vary_n, pred_times)
    plt.suptitle("Support Vector Machine with gamma = " + str(best_gamma))
    plt.title("Prediction time v Data size")
    plt.xlabel("Data size")
    plt.ylabel("Prediction time")

    plt.show()




def main():
    labels, features = task1()

    perceptron = Perceptron(tol=1e-3, random_state=0)
    task3and4(labels, features, perceptron, "Perceptron")

    dTree = tree.DecisionTreeClassifier(max_depth=5, random_state=0)
    task3and4(labels, features, dTree, "Decision Tree")

    task5(labels, features)
    task6(labels, features)
    #Task 7 answered at the end of the pdf.



main()