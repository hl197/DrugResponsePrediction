import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from dataset import divide_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


LR = []
RF = []
SVM = []
for i in range(20):
    train, val, test = divide_data('../count_processed_rnaseq.csv', numpy=True, seed=i)

    X, y = train[:, 1:-1], train[:, -1].astype(int)
    X_test, y_test = test[:, 1:-1], test[:, -1].astype(int)

    clf = LinearRegression().fit(X, y)
    y_pred = clf.predict(X_test)

    LR.append(accuracy_score(y_test, (y_pred > 0.5).astype(int)))

    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=7)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    RF.append(accuracy_score(y_test, (y_pred > 0.5).astype(int)))

    clf = SVC(gamma='auto')
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    SVM.append(accuracy_score(y_test, (y_pred > 0.5).astype(int)))

print("LR average {}, standard dev {}".format(np.mean(LR), np.std(LR)))
print("RF average {}, standard dev {}".format(np.mean(RF), np.std(RF)))
print("SVM average {}, standard dev {}".format(np.mean(SVM), np.std(SVM)))
