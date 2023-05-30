from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits

digit_data = load_digits()

data = scale(digit_data.data)

model = KMeans(n_clusters=10, init= 'random', n_init=10)

model.fit(data)