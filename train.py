import argparse
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
import pickle

def trainModel(path):
    sampleList = pickle.load(open(path, "rb"))
    sent = []

    # Vectorize feature data
    fourLetters = [x for x, _ in sampleList]
    labels = [y for _, y in sampleList]
    features = list(set([y for x in fourLetters for y in x]))
    featureData = []
    
    for letterTuple in fourLetters:
        temp = []
        for sample in features:
            if sample in letterTuple:
                temp.append(1)
            else:
                temp.append(0)
        featureData.append(temp)
    
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