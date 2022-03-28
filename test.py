import argparse
from hashlib import new
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
import pickle

def test(testSet, modlePath):
    # Prepare the test set data
    samples = pickle.load(open(testSet, "rb"))

    featureData = []
    labels = []
    for feature, label in samples:
        featureData.append(feature)
        labels.append(label)

    # load the model and apply it to test set
    modle = pickle.load(open(modlePath, 'rb'))

    prediction = modle.predict(featureData)

    accuracy = metrics.accuracy_score(labels, prediction)
    precision = metrics.precision_score(labels, prediction, average='micro')
    recall = metrics.recall_score(labels, prediction, average='micro')
    f1Score = metrics.f1_score(labels, prediction, average='micro')

    print("Categorical NB Performance Metrics")
    print("=" * 34)
    print("Accuracy: \t", accuracy)
    print("Precision: \t", precision)
    print("recall: \t", recall)
    print("F1 score: \t", f1Score)


if __name__ == "__main__":

    '''
        This script take train set in the pickled format and return the trained model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The path of the test set in .p format.")
    parser.add_argument("modle", help="The path of the trained model in .sav format.")

    args = parser.parse_args()
    filePath = args.file
    modelPath = args.modle

    test(filePath, modelPath)