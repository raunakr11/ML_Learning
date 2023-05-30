from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

breast_data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(np.array(breast_data.data), np.array(breast_data.target), test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))


