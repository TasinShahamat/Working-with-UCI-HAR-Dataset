clc
clear

X = load('X_train.txt');
y = load('y_train.txt');
X_test = load('X_test.txt');
y_test = load('y_test.txt');

model = svmtrain(y, X, '-t 0 -c 0.1');
[p, accuracy, values] = svmpredict(y, X, model);
fprintf('\nTrain accuracy: %f\n', accuracy(1));

[p, accuracy, values] = svmpredict(y_test, X_test, model);
fprintf('\nTest accuracy: %f\n', accuracy(1));


% Using SVM, we found train accuracy = 98.993471% and test accuracy = 96.063794%