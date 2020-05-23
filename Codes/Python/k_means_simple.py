import pandas as pd
from sklearn.cluster import KMeans

X = pd.read_csv('X_train.txt', delim_whitespace = True, header = None)
print('\n..................................Data loaded')

km = KMeans(n_clusters=6)
y_km = km.fit_predict(X)
print("\n\nClustering completed")