%% car test data
weight = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300];
tested = [48 42 31 34 31 21 23 23 21 16 17 21];
failed = [1 2 0 3 8 8 14 17 19 15 17 21];

proportion = failed ./ tested;

figure;
plot(weight, proportion, 'rp');
title(['Linear Fitting'])
xlabel('Weight (kg)');
ylabel('Failure proportion');

p = polyfit(weight, proportion, 1);
w = p(1);
b = p(2);

x = 2100:10:4300;
y = w * x + b;

hold on;
plot(x, y, 'b');

fprintf('y = %.6fx + %.6f\n', w, b);