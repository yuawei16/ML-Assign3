import argparse
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn import svm
import pickle

def trainModel(path):
    samples = pickle.load(open(path, "rb"))

    # unpack and take feature vectors and labels
    featureData = []
    labels = []
    for feature, label in samples:
        featureData.append(feature)
        labels.append(label)

    # Train categoricalNB and SVC model and save model to pickle.
    model = CategoricalNB().fit(featureData, labels)

    pickle.dump(model, open("CNB.sav", 'wb'))

    svcModel = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(featureData, labels)
    pickle.dump(model, open("svc.sav", 'wb'))

    return print("CategoricalNB Model saved as CNB.sav. SVM Model saved as svc.sav.")



if __name__ == "__main__":

    '''
        This script take train set in the pickled format and return the trained model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The path of the train set in .p format.")

    args = parser.parse_args()
    filePath = args.file

    trainModel(filePath)