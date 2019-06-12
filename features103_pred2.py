import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

def main():
    f = open("trainingData/feature103_Train.txt")

    data = []
    label = []
    for lineNumber, line in enumerate(f):
        if lineNumber != 0:
            newLine = line.rstrip()
            entries = newLine.split('\t')
            data.append(list(map(float, entries[2:])))
            label.append(int(entries[1]))

    f.close()

    f2 = open("testingData/features103_test.txt")

    extractTest = []
    extractIds = []
    for lineNumber, line in enumerate(f2):
        if lineNumber != 0:
            newLine = line.rstrip()
            entries = newLine.split('\t')
            extractTest.append(list(map(float, entries[1:])))
            extractIds.append(entries[0])

    f2.close()


    trainLabel = np.array(label)
    trainData = np.array(data)

    testData = np.asarray(extractTest)

    ###########################################################

    X_train, X_test, y_train, y_test = model_selection.train_test_split(trainData, trainLabel, test_size = 0.33)

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    clf_train = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf_train = clf_train.fit(X_train, y_train)

    y_dt_pred_train = clf_train.predict_proba(X_test)

    onlyPKpredictions_train = y_dt_pred_train[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, onlyPKpredictions_train)
    print("accuracy DT AUC:",metrics.auc(fpr,tpr))

    ###########################################################

    X_train, y_train = SMOTE().fit_resample(trainData, trainLabel)
 
    clf_test = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf_test = clf_test.fit(X_train, y_train)

    y_dt_pred_test = clf_test.predict_proba(testData)

    onlyPKpredictions_test = y_dt_pred_test[:,1]

    ###########################################################

    o = open('features103_pred2.txt', 'w')
    for i in range(len(extractIds)):
        entry = extractIds[i] + "\t" + str(onlyPKpredictions_test[i])
        o.write(entry)
        o.write('\n')

    o.close()

if __name__ == "__main__":
    main()