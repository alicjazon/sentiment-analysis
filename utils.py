#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
'''
Dummy classifier:
stratified - generates random predictions by respecting the training set class distribution.
most_frequent - always predicts the most frequent label in the training set.
prior -  always predicts the class that maximizes the class prior (like most_frequent) and predict_proba returns the class prior.
uniform -  generates predictions uniformly at random.
constant -  always predicts a constant label that is provided by the user.
'''
def dummy_classifier(train, test, strategy):
    y_tr = [] 
    y_test = [] 
    x_test = []
    for step, node in enumerate(train):
        _, l, _ = node
        y_tr.append(l)
    
    for step, node in enumerate(test):
        _, l, w = node
        y_test.append(l)
        x_test.append(w)
    clf = DummyClassifier(strategy=strategy, random_state=0)
    clf.fit(None, y_tr)
    print("Dummy classifier - strategy: " + strategy)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred)) 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Macierz konfuzji',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Etykiety rzeczywiste')
    plt.xlabel('Etykiety przewidziane')
    plt.tight_layout()

def subplot_metrics(epoch, metr1, metr2, name1, name2, metric):
    plt.plot(epoch, metr1, 'r--')
    plt.plot(epoch, metr2, 'b-')
    plt.legend([name1, name2])
    plt.xlabel('Epoka')
    plt.ylabel(metric)
    plt.show();
    
def plot(epoch, metric, name):
    plt.plot(epoch, metric, 'b-')
    plt.xlabel('Epoka')
    plt.ylabel(name)
    plt.show();
#saver = tf.train.Saver()
#with tf.Session() as sess:
#saver.restore(sess, "/models/model.ckpt")
    
