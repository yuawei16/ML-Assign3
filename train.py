import argparse
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
import pickle

def trainModel(path):
    samples = pickle.load(open(path, "rb"))

    # unpack and take feature vectors and labels
    featureData = []
    labels = []
    for feature, label in samples:
        featureData.append(feature)
        labels.append(label)

    # Train categoricalNB model
    model = CategoricalNB().fit(featureData, labels)

    # Save model to pickle.
    pickle.dump(model, open("CNB.sav", 'wb'))

    return print("Model saved as CNB.sav.")



if __name__ == "__main__":

    '''
        This script take train set in the pickled format and return the trained model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The path of the train set in .p format.")

    args = parser.parse_args()
    filePath = args.file

    trainModel(filePath)