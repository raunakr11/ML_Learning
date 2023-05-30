from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading dataset
iris = datasets.load_iris()

#print(iris.DESCR)

#defining
features = iris.data
labels = iris.target

#training classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

predict = clf.predict([[9.1, 9.5, 6.4, 0.4]])

print(predict)