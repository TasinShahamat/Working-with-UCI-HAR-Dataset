import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

X = pd.read_csv('X_train.txt', delim_whitespace = True)
y = pd.read_csv('y_train.txt', delim_whitespace = True)
Xtest = pd.read_csv('X_test.txt', delim_whitespace = True)
ytest = pd.read_csv('y_test.txt', delim_whitespace = True)

print('\n\n............Dataset Loaded\n')

svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X, y)

print('..................Testing training accuracy on training set')
y_pred = svclassifier.predict(X)
print(confusion_matrix(y,y_pred))
print(classification_report(y,y_pred))

print('\n\n..................Testing training accuracy on test set')
y_pred = svclassifier.predict(Xtest)
print(confusion_matrix(ytest,y_pred))
print(classification_report(ytest,y_pred))

# Using linear kernel, we got 99% training set accuracy 
#                             and 96% test set accuracy

# Using gaussian(rbf) kernel, we got 98% training set accuracy 
#                                    and 95% test set accuracy

# Using polynomial kernel (8th degree), we got 100% training set accuracy 
#                                               and 96% test set accuracy

# Using sigmoid kernel, we got 89% training set accuracy 
#                              and 86% test set accuracy