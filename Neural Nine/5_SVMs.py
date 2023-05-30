from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

breast_data = load_breast_cancer()

X = breast_data.data
Y = breast_data.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = SVC(kernel= 'linear', C=3)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))