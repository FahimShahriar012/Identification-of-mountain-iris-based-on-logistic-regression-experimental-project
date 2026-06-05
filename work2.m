%% car test data
weight = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300]';
tested = [48 42 31 34 31 21 23 23 21 16 17 21]';
failed = [1 2 0 3 8 8 14 17 19 15 17 21]';

proportion = failed ./ tested;

figure;
plot(weight, proportion, 'rp');
title(['Linear Fitting'])
xlabel('Weight (kg)');
ylabel('Failure proportion');

b = glmfit(weight, [failed tested], 'binomial', 'link', 'logit');

x = (2100:10:4300)';
y = 1 ./ (1 + exp(-(b(1) + b(2) * x)));

hold on;
plot(x, y, 'b');

fprintf('Logistic regression model:\n');
fprintf('y = 1 / (1 + exp(-(%f + %f*x)))\n', b(1), b(2));