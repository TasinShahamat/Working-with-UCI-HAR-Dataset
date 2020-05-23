clc
clear

X = load('X_train.txt');
y = load('y_train.txt');
X_test = load('X_test.txt');
y_test = load('y_test.txt');

lambda = 0.3;
[all_theta] = oneVsAll(X, y, num_labels=6, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
pause;

pred = predictOneVsAll(all_theta, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

%Using a simple one vs all classifier (logistic regression), we found training set accuracy = 98.000544% and test set accuracy = 95.690533%