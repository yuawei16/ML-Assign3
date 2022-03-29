'''
    The concept of perplexity of the model is not very clear to me, so this script will consider each prediction like a unigram.
'''

import argparse
from hashlib import new
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
import pickle

def calculate_perplexity(testSet, modelPath):
    # Prepare the test set data
    samples = pickle.load(open(testSet, "rb"))

    featureData = []
    labels = []
    for feature, label in samples:
        featureData.append(feature)
        labels.append(label)

    # load the models depending on input and apply it to test set and report metrics
    model = pickle.load(open(modelPath, 'rb'))

    probs = model.predict(featureData)
        
    # Since the prediction is carried without considering dependencies, so each one is treated like a unigram.
    probList = []
    finalPP = 0
    for prob in probs:
        loga = 0.0
        for p in prob:
            loga += np.log2(p)
        pp = np.power(2, -(loga/len(prob)))
        finalPP += pp
        probList.append(pp)
    
    print("Reprot for Perplexity:")
    print("=" * 30)
    if 'cnb' in modelPath.lower():
        print("CategoricalNB PP value:", finalPP/len(probs))
    elif 'svc' in modelPath.lower():
        print("SVC PP value:", finalPP/len(probs))

    return finalPP/len(probs)   # Normalise the PP values by total number of the test set.



if __name__ == "__main__":

    '''
        This script take train set in the pickled format and return the trained model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The path of the test set in .p format.")
    parser.add_argument("model", help="The path of the trained model in .sav format.")

    args = parser.parse_args()
    filePath = args.file
    modelPath = args.model

    calculate_perplexity(filePath, modelPath)