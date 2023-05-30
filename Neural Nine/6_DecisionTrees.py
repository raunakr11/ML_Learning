from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

breast_data = load_breast_cancer()

X = breast_data.data
Y = breast_data.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)

print(f'Decision Tree Classifier: {clf1.score(X_test, y_test)}')

clf2 = RandomForestClassifier()
clf2.fit(X_train, y_train)

print(f'Random Forest Classifier: {clf2.score(X_test, y_test)}')
