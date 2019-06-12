import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn import svm
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier 

# 2995  - pk
# 24362 - pkf
# 1:8 ratio

def main():
    f = open("trainingData/featuresall_train.txt")

    data = []
    floatData = []
    label = []
    for lineNumber, line in enumerate(f):
        if lineNumber != 0:
            entries = line.split('\t')
            data.append(list(map(float, entries[2:])))
            label.append(int(entries[1]))
            #data.append((list(map(float, entries[2:])), int(entries[1])))

    f.close()

    # f2 = open("testingData/features103_test.txt")

    # extractTest = []
    # extractTestLabels = []
    # for lineNumber, line in enumerate(f2):
    #     if lineNumber != 0:
    #         entries = line.split('\t')
    #         extractTest.append(list(map(float, entries[1:])))
    #         # extractTestLabels.append(float(entries[1]))


    # testData = np.asarray(extractTest)
    # # testLabels = np.asarray(extractTestLabels)
    
    classLabel = np.array(label)
    trainData = np.array(data)

    # ros = RandomOverSampler(random_state=0)

    # X_resampled, y_resampled = SMOTE().fit_resample(trainData, classLabel)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(trainData, classLabel, test_size = 0.33)
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X_resampled, y_resampled, test_size = 0.33)

    # X_train, y_train = ros.fit_resample(X_train, y_train)
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    # Gaussian Naive Bayes SUCKS don't use it
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_nb_pred = nb.predict_proba(X_test)
    # print(y_nb_pred)
    onlyPKpredictions = y_nb_pred[:,1]
    # print(onlyPKpredictions)
    # print("accuracy KAPPA:",metrics.cohen_kappa_score(y_test, y_nb_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, onlyPKpredictions)
    print("accuracy NB AUC:",metrics.auc(fpr,tpr))
    # print("confusion:\n",metrics.confusion_matrix(y_test, y_nb_pred))

    # We are blessed with DT
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)#, max_depth=20)
    # clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X_train, y_train)

    y_dt_pred = clf.predict_proba(X_test)
    # print(y_dt_pred[:,0])

    onlyPKpredictions = y_dt_pred[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, onlyPKpredictions)
    print("accuracy DT AUC:",metrics.auc(fpr,tpr))
    # print(y_dt_pred)

    # print("accuracy KAPPA:",metrics.cohen_kappa_score(y_test, y_dt_pred))
    # print("accuracy AUC:",metrics.roc_auc_score(y_test, y_dt_pred))
    # print("DT:\n",metrics.confusion_matrix(y_test, y_dt_pred))

    model = BalancedRandomForestClassifier(n_estimators=100, max_depth=5)
    model = model.fit(X_train, y_train)
    y_rfc_pred = model.predict_proba(X_test)
    onlyPKpredictions = y_rfc_pred[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, onlyPKpredictions)
    print("accuracy RFC AUC:",metrics.auc(fpr,tpr))
    # print(y_rfc_pred)
    # print("accuracy AUC:",metrics.roc_auc_score(y_test, y_rfc_pred))

    f3 = open('results_train103.txt')
    grab = []
    for line in f3:
        spl = line.split('\t')
        grab.append(float(spl[1]))

    nn = np.array(grab) 

    fpr, tpr, thresholds = metrics.roc_curve(classLabel, nn)
    print("accuracy NN AUC:",metrics.auc(fpr,tpr))


    f3.close()

    # SVM take way too long...why? Too large? DT went very fast though
    # model = svm.SVC(gamma='auto')
    # y_svc_pred = model.fit(X_train, y_train)
    # print("accuracy KAPPA:",metrics.cohen_kappa_score(y_test, y_svc_pred))


if __name__ == "__main__":
    main()