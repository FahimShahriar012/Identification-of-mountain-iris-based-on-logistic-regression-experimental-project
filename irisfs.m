clear
clc
close all

%% %% 数据准备 (Data Preparation)
% 1. Load feature data
load('Iris_Data.mat');
data = IrisData;

% 2. Load binary label data
load('Iris_Data_binary.mat');
label = IrisDatabinary;

% 3. Convert tables to numeric matrices
data_matrix = table2array(data(:, 1:4));

% IMPORTANT:
% IrisDatabinary has 3 columns:
% column 1 = sepal length
% column 2 = sepal width
% column 3 = binary label
label_matrix = table2array(label(:, 3));

% 4. Match feature rows with label rows
num_samples = size(label_matrix, 1);
data_matrix = data_matrix(1:num_samples, :);

% 5. Use only first two features
data_features = data_matrix(:, 1:2);

%% %% Min-Max Normalization
data_norm = (data_features - min(data_features)) ./ ...
            (max(data_features) - min(data_features));

%% %% Add Bias Term
data_ready = [data_norm, ones(size(data_norm, 1), 1)];

%% %% Shuffle Data
randIndex = randperm(num_samples);
data_new = data_ready(randIndex, :);
label_new = label_matrix(randIndex, :);

%% %% 80% Training 20% Testing
num_samples = size(data_new, 1);
k = round(0.8 * num_samples);

% Training Data
X1 = data_new(1:k, :);
Y1 = label_new(1:k, :);

% Testing Data
X2 = data_new(k+1:end, :);
Y2 = label_new(k+1:end, :);

[m1, n1] = size(X1);
[m2, n2] = size(X2);
Features = size(data_ready, 2);

%% %% 开始训练 (Start Training)
delta = 0.2;      % Learning Rate
lamda = 0;        % Regularization Coefficient

% Initial beta = 0
theta1 = zeros(1, Features);

%% %% Gradient Descent Training
num = 10000;
L = zeros(1, num);

for iter = 1:num
    dt = zeros(1, Features);
    loss = 0;

    for i = 1:m1
        xx = X1(i, 1:Features);
        yy = Y1(i, 1);

        % Sigmoid Function
        h = 1 / (1 + exp(-(theta1 * xx')));

        % Avoid log(0)
        h = max(min(h, 1 - 1e-15), 1e-15);

        % Gradient
        dt = dt + (h - yy) * xx;

        % Loss Function
        loss = loss + yy * log(h) + (1 - yy) * log(1 - h);
    end

    % Average Loss
    loss = -loss / m1;
    L(iter) = loss;

    % Parameter Update
    theta1 = theta1 - delta * dt / m1 - lamda * theta1 / m1;
end

%% %% Draw Figures
figure

%% %% Cost Function Curve
subplot(1, 2, 1)
plot(L, 'LineWidth', 2)
xlabel('Iteration')
ylabel('Loss')
title('Cost Function')
grid on

%% %% Logistic Regression Separation Curve
subplot(1, 2, 2)
hold on

% Draw Class 1 points
idx1 = find(label_matrix == 1);
plot(data_norm(idx1, 1), data_norm(idx1, 2), 'ro', ...
     'MarkerFaceColor', 'r', 'MarkerSize', 5)

% Draw Class 0 points
idx0 = find(label_matrix == 0);
plot(data_norm(idx0, 1), data_norm(idx0, 2), 'go', ...
     'MarkerFaceColor', 'g', 'MarkerSize', 5)

% Separation boundary:
% w1 * SepalLength + w2 * SepalWidth + b = 0
x = 0:0.01:1;
y = (-theta1(1) * x - theta1(3)) / theta1(2);
plot(x, y, 'k', 'LineWidth', 2)

xlabel('Normalized Sepal Length')
ylabel('Normalized Sepal Width')
title('Logistic Regression Boundary: Sepal Length vs Sepal Width')

legend('Setosa / Mountain Iris (1)', ...
       'Versicolor / Colorful Iris (0)', ...
       'Separation Boundary', ...
       'Location', 'best')

axis([-0.05 1.05 -0.05 1.05])
grid on
hold off

%% %% Testing Accuracy
acc = 0;

for i = 1:m2
    xx = X2(i, 1:Features)';
    yy = Y2(i);

    final = 1 / (1 + exp(-theta1 * xx));

    if final > 0.5 && yy == 1
        acc = acc + 1;
    end

    if final <= 0.5 && yy == 0
        acc = acc + 1;
    end
end

accuracy = acc / m2 * 100;
fprintf('Accuracy = %.2f%%\n', accuracy);

%% %% Print Parameters
fprintf('w1 = %.8f\n', theta1(1));
fprintf('w2 = %.8f\n', theta1(2));
fprintf('b  = %.8f\n', theta1(3));
fprintf('Final loss = %.8f\n', L(end));