import numpy as np


def extractData():
    f = open("trainingData/feature103_Train.txt")

    data = []
    floatData = []
    label = []
    for lineNumber, line in enumerate(f):
        if lineNumber != 0:
            entries = line.split('\t')
            data.append(list(map(float, entries[2:])))
            label.append(float(entries[1]))

    
    classLabel = np.array(label)
    trainData = np.array(data)

    f.close()

    # print(classLabel)
    # print(classLabel.shape)
    # print(trainData.shape)

if __name__ == "__main__":
    extractData()