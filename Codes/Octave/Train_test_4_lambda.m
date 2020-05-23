clc
clear

X = load('X_train.txt');
y = load('y_train.txt');
X_test = load('X_test.txt');
y_test = load('y_test.txt');

train_accuracy_vec = zeros(10, 1);
test_accuracy_vec = zeros(10, 1);
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	[all_theta] = oneVsAll(X, y, num_labels=6, lambda);
	pred_train = predictOneVsAll(all_theta, X);
	pred_test = predictOneVsAll(all_theta, X_test);
	train_accuracy_vec(i) =  mean(double(pred_train == y)) * 100;
	test_accuracy_vec(i) = mean(double(pred_test == y_test)) * 100;
endfor

plot(lambda_vec, train_accuracy_vec, lambda_vec, test_accuracy_vec);
legend('Train', 'Test');
xlabel('lambda');
ylabel('Error');