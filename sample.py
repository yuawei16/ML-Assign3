import argparse
from cgi import print_arguments
import gzip
import pickle
from sklearn import preprocessing

def getSamples(path):
    newPath = path.split('.')
    allSents = []
    if newPath[-1] == 'gz':
        with gzip.open(path, 'rt') as file:
            for line in file:
                if '\x00' in line or '\n' == line:
                    continue
                line = line.strip("\n")
                allSents.append(line.lower())
    elif newPath[-1] == 'txt':
        with open(path, 'r') as file:
            for line in file:
                line = line.strip('\n')
                allSents.append(line.lower())
    else:
        print("File format not supported!")

    consonantLetters = 'bcdfghjklmnpqrstvwxyz'
    vowelLetters = 'aeiou'

    bigSent = " ".join(allSents)    # Combine all sentences in a long string.
    letterWithPosition = []
    span = 4

    for i in range(0, len(bigSent)):
        fourLetters = ()
        if len(bigSent[i: i+span]) == 4:
            fourLetters = bigSent[i]+"_1", bigSent[i+1]+"_2", bigSent[i+2]+"_3", bigSent[i+3]+"_4"
        label = ''
        for letter in bigSent[i+span: ]:
            if letter in consonantLetters:
                label = letter
                break
        if label != '':
            letterWithPosition.append((fourLetters, label))

    # Vectorize feature data
    fourLetters = [x for x, _ in letterWithPosition]
    labels = [y for _, y in letterWithPosition]
    features = list(set([y for x in fourLetters for y in x]))
    featureData = []
    
    # Encode labels and split train and test sets
    encod = preprocessing.LabelEncoder()
    encodedLabes = encod.fit_transform(labels)

    for i in range(len(letterWithPosition)):
        temp = []
        for sample in features:
            if sample in letterWithPosition[i]:
                temp.append(1)
            else:
                temp.append(0)
        featureData.append((temp, encodedLabes[i]))
    
    split = int(len(letterWithPosition)*0.8)

    trainSet = featureData[:split]
    testSet = featureData[split:]

    # pickle the 2 sets for furture use.
    pickle.dump(letterWithPosition, open("allSamples.p", "wb"))
    pickle.dump(trainSet, open("trainSet.p", "wb"))
    pickle.dump(testSet, open("testSet.p", "wb"))

    
    # The returns are for test purpose.
    return print("Training set and test set are save in pickle files.")



if __name__ == "__main__":

    '''
        This script take file path and return 2 sets. The split of the training and test set 
        is 0.8 which is quite common.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The path of the file in gz or txt format")

    args = parser.parse_args()
    filePath = args.file

    getSamples(filePath)