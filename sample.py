import argparse
import gzip
import pickle

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
                allSents.append(line)
    else:
        print("File format not supported!")

    consonantLetters = 'bcdfghjklmnpqrstvwxyz'
    vowelLetters = 'aeiou'

    letterWithPosition = []
    span = 4
    for s in allSents:
        for i in range(0, len(s)):
            fourLetters = ()
            if len(s[i: i+span]) == 4:
                fourLetters = s[i]+"_1", s[i+1]+"_2", s[i+2]+"_3", s[i+3]+"_4"
            label = ''
            for letter in s[i+span: ]:
                if letter in consonantLetters:
                    label = letter
                    break
            letterWithPosition.append((fourLetters, label))
            
    split = int(len(letterWithPosition)*0.8)

    trainSet = letterWithPosition[:split]
    testSet = letterWithPosition[split:]

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