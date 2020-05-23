import pandas as pd
from sklearn.cluster import KMeans

X = pd.read_csv('X_train.txt', delim_whitespace = True, header = None)
y = pd.read_csv('y_train.txt', delim_whitespace = True, header = None)
print('\n..................................Data loaded')

km = KMeans(n_clusters=6)
y_km = km.fit_predict(X)
print("\n\nClustering completed")

y_km_dic = dict()
y_dic = dict()
for i in y_km:
    y_km_dic[i] = y_km_dic.get(i, 0) + 1
for j in y.loc[ : , 0]:
    j = int(j)
    y_dic[j] = y_dic.get(j, 0) + 1
y_km_dic = sorted([(v,k) for k,v in y_km_dic.items()], reverse=True)
y_dic = sorted([(v,k) for k,v in y_dic.items()], reverse=True)

print("\nClustered labels:", y_km_dic)
print("\nActual labels:", y_dic)