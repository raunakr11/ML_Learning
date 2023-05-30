from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

#print(list(iris.keys()))
#print(iris['data'])
#print(iris['target'])

x = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(np.int)

clf = LogisticRegression()
clf.fit(x,y)

example = clf.predict(([[2.5]]))
#print(example)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)
plt.plot(X_new, y_prob[:,1], "g-", label = 'virginica')
plt.show()