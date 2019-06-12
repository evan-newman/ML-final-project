import sys
import operator
import numpy as np
import math

TOTAL_FEATURES = 104

def checkCommandLine():
    if len(sys.argv) != 4:
        print("Incorrect amount of arguments given. Exiting...")
        sys.exit()

def extractData(dataFile):
    f = open(dataFile, 'r')

    data = []
    floatData = []
    #label = []
    for lineNumber, line in enumerate(f):
        if lineNumber != 0:
            entries = line.split('\t')
            data.append(list(map(float, entries[1:])))
            #label.append(float(entries[1]))
    #         data.append((list(map(float, entries[2:])), int(entries[1])))
    
    # print(data[0])

    
    #classLabel = np.array(label)
    #trainData = np.array(data)

    f.close()

    return data 

def countRatio(data):

    pos = 0.0
    neg = 0.0

    for k in data:
        if k == 1.0:
            pos += 1
        elif k == 0.0:
            neg += 1
        else:
            print("Error incorrect class given")
            sys.exit()

    return pos, neg

def pickStump(trainData, depth):

    totalEx = float(len(trainData))
    diagnosis = [x[0] for x in trainData]

    rootPos, rootNeg = countRatio(diagnosis)
    totalEntropy = entropy(totalEx, rootPos, rootNeg)
    
    #print("Stump: [" +str(rootPos) + "+," + str(rootNeg) + "] " + str(totalEntropy))

    bestSplit = 0.0
    splitLoc = 0
    splitNumber = 0.0
    stump = np.array([])
    bestFeat = 0

    for i in range(1, len(trainData)):
        print("********")
        print(i)
        feat = [x[i] for x in trainData]

        combine = np.array(list(zip(diagnosis, feat)))
        
        stump = combine[np.argsort(combine[:, 1])]

        for j in range(len(stump) - 1):
            current = stump[j]
            nxt     = stump[j+1]

            if current[0] != nxt[0]:
                benefitOfSplit = uncertainty(totalEntropy, totalEx, stump[:j+1], stump[j+1:])

                if bestSplit < benefitOfSplit:
                    bestSplit = benefitOfSplit
                    splitLoc = j
                    splitNumber = (abs((current[1] - nxt[1]) / 2)) + current[1]
                    bestFeat = i

    if depth == 1:
        return ("end", bestFeat, splitNumber)
    
    ldata, rdata = splitTheData(trainData, bestFeat, splitNumber)
    lfirst = [x[0] for x in ldata]
    rfirst = [x[0] for x in rdata]
    lerror = error(lfirst)
    rerror = error(rfirst)
    
    if lerror == 0 and rerror == 0:
        return ("pure", bestFeat, splitNumber)
    elif lerror == 0:
        split = pickStump(rdata, depth-1)
        return ("pure left", bestFeat, splitNumber, split) 
    elif rerror == 0:
        split = pickStump(ldata, depth-1)
        return ("pure right", bestFeat, splitNumber, split)
    else:
        splitLeft  = pickStump(ldata, depth-1)
        splitRight = pickStump(rdata, depth-1)
        return ("tainted", bestFeat, splitNumber, splitLeft, splitRight)
        
def uncertainty(totalEntropy, totalEx, left, right):
    
    ltotal = float(len(left))
    rtotal = float(len(right))

    lfirst = [x[0] for x in left]
    rfirst = [x[0] for x in right]
    lpos, lneg = countRatio(lfirst)
    rpos, rneg = countRatio(rfirst)

    lentropy = entropy(ltotal, lpos, lneg)
    rentropy = entropy(rtotal, rpos, rneg)

    return totalEntropy - ( (ltotal/totalEx)*lentropy + (rtotal/totalEx)*rentropy )

def entropy(total, pos, neg):
    if pos == 0.0:
        return -( (neg/total) * math.log(neg/total, 2.0) )
    elif neg == 0.0:
        return -( (pos/total) * math.log(pos/total, 2.0) ) 
    else:
        return -( (pos/total) * math.log(pos/total, 2.0) + (neg/total) * math.log(neg/total, 2.0) )

def main():
    checkCommandLine()
    trainData = extractData(sys.argv[1])
    testData = extractData(sys.argv[2])

    depth = int(sys.argv[3])

    chaos = pickStump(trainData, depth)
    
    trainError = recursiveError(chaos, trainData)
    testError = recursiveError(chaos, testData)
    print("Training Error: " + str(trainError / len(trainData) * 100))
    print("Testing Error: " + str(testError / len(testData) * 100))
    
def recursiveError(chaos, data):

    hint = chaos[0]
    bestFeat = chaos[1]
    splitNum = chaos[2]

    ldata, rdata = splitTheData(data, bestFeat, splitNum)
    
    if hint == "end" or hint == "pure":
        lfirst = [x[0] for x in ldata]
        rfirst = [x[0] for x in rdata]
        lerror = error(lfirst)
        rerror = error(rfirst)
       
    elif hint == "tainted":
        left = chaos[3]
        right = chaos[4]
        lerror = recursiveError(left, ldata)
        rerror = recursiveError(right, rdata)
    
    elif hint == "pure right":
        rfirst = [x[0] for x in rdata]
        rerror = error(rfirst)
        left = chaos[3]
        lerror = recursiveError(left, ldata)

    elif hint == "pure left":
        lfirst = [x[0] for x in ldata]
        lerror = error(lfirst)
        right = chaos[3]
        rerror = recursiveError(right, rdata)

    return lerror + rerror

def splitTheData(data, bestFeat, splitNumber):

    left = []
    right = []
    for i in range(len(data)):
        if data[i][bestFeat] <= splitNumber:
            left.append(data[i])
        elif data[i][bestFeat] > splitNumber:
            right.append(data[i])

    return left, right

def error(data):

    pos, neg = countRatio(data)
    if (pos > neg):
        err = neg
    else:
        err = pos
    
    return err

if __name__ == "__main__":
    main()