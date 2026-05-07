% ============================================================
% ЛАБОРАТОРНА РОБОТА 4 (9) — Варіант 5
% Моделювання об'єктів управління на основі нейронних мереж
% ============================================================
% Функція 1 (двовимірна): z1(x,y) = x * sin(x + y)
%   Входи x, y в [0, pi/2]
% Функція 2 (одновимірна): y(x) = sin(x) + cos(3*x^2)
%   Вхід x в [0, pi]
% ============================================================

%% --- Конфігурації мереж (спільні для обох функцій) ---
configs = {
    'feedforwardnet',   [10],     'Feed-forward, 1 шар, 10 нейронів';
    'feedforwardnet',   [20],     'Feed-forward, 1 шар, 20 нейронів';
    'cascadeforwardnet',[20],     'Cascade-forward, 1 шар, 20 нейронів';
    'cascadeforwardnet',[10 10],  'Cascade-forward, 2 шари по 10 нейронів';
    'elmannet',         [15],     'Elman, 1 шар, 15 нейронів';
    'elmannet',         [5 5 5],  'Elman, 3 шари по 5 нейронів';
};

short_labels = {'FF 1x10','FF 1x20','CF 1x20','CF 2x10','EL 1x15','EL 3x5'};

%% ============================================================
%  ЧАСТИНА 1: z1(x,y) = x * sin(x + y)
%% ============================================================
fprintf('\n');
fprintf('==========================================================\n');
fprintf('   ЧАСТИНА 1:  z1(x,y) = x * sin(x + y)                 \n');
fprintf('==========================================================\n');

N = 500;
x1 = linspace(0, pi/2, N);
y1 = linspace(0, pi/2, N);
Input1  = [x1; y1];
Output1 = x1 .* sin(x1 + y1);

results1 = zeros(1,6);
nets1    = cell(1,6);

for i = 1:6
    fprintf('\n=== Конфігурація %d: %s ===\n', i, configs{i,3});
    net = create_net(configs{i,1}, configs{i,2});
    [net, tr] = train(net, Input1, Output1);
    Out_nn = net(Input1);
    err = mean(abs(Output1 - Out_nn) ./ (abs(Output1) + 1e-10)) * 100;
    results1(i) = err;
    nets1{i}    = net;
    fprintf('Середня відносна похибка: %.6f %%\n', err);
    fprintf('MSE на тренуванні:        %.2e\n', tr.best_perf);
end

print_table(configs, results1, 'z1(x,y) = x*sin(x+y)');

%% ============================================================
%  ЧАСТИНА 2: y(x) = sin(x) + cos(3*x^2)
%% ============================================================
fprintf('\n');
fprintf('==========================================================\n');
fprintf('   ЧАСТИНА 2:  y(x) = sin(x) + cos(3*x^2)               \n');
fprintf('==========================================================\n');

x2 = linspace(0, pi, N);
y2 = zeros(1, N);          % фіктивний другий вхід
Input2  = [x2; y2];
Output2 = sin(x2) + cos(3*x2.^2);

results2 = zeros(1,6);
nets2    = cell(1,6);

for i = 1:6
    fprintf('\n=== Конфігурація %d: %s ===\n', i, configs{i,3});
    net = create_net(configs{i,1}, configs{i,2});
    [net, tr] = train(net, Input2, Output2);
    Out_nn = net(Input2);
    err = mean(abs(Output2 - Out_nn) ./ (abs(Output2) + 1e-10)) * 100;
    results2(i) = err;
    nets2{i}    = net;
    fprintf('Середня відносна похибка: %.6f %%\n', err);
    fprintf('MSE на тренуванні:        %.2e\n', tr.best_perf);
end

print_table(configs, results2, 'y(x) = sin(x)+cos(3x^2)');

%% ============================================================
%  ГРАФІКИ
%% ============================================================

%% --- Графік 1: Порівняння похибок (обидві функції) ---
figure('Name','Порівняння похибок — Варіант 5','Color','w','Position',[50 50 800 420]);
xb = 1:6; wb = 0.35;
b1 = bar(xb-wb/2, results1, wb, 'FaceColor','flat');
b1.CData = repmat([0.18 0.55 0.80], 6, 1);
hold on;
b2 = bar(xb+wb/2, results2, wb, 'FaceColor','flat', 'FaceAlpha', 0.7);
b2.CData = repmat([0.85 0.33 0.10], 6, 1);
set(gca,'XTickLabel', short_labels, 'FontSize', 10);
legend({'z1(x,y) = x*sin(x+y)', 'y(x) = sin(x)+cos(3x^2)'}, 'Location','NorthWest');
ylabel('Середня відносна похибка (%)');
title('Варіант 5 — Порівняння похибок конфігурацій НМ');
grid on;

%% --- Графік 2: Апроксимація z1 (краща конфігурація) ---
[~, best1] = min(results1);
Out_best1  = nets1{best1}(Input1);

figure('Name','Апроксимація z1 — Варіант 5','Color','w','Position',[50 520 700 350]);
plot(x1, Output1,   'b-',  'LineWidth', 2,   'DisplayName','Еталон z1');
hold on;
plot(x1, Out_best1, 'r--', 'LineWidth', 1.5, 'DisplayName',['НМ: ',configs{best1,3}]);
xlabel('x  (y = x)'); ylabel('z1'); grid on;
title(sprintf('z1(x,y)=x*sin(x+y) — краща НМ: %s (%.4f%%)', ...
    configs{best1,3}, results1(best1)));
legend('Location','best');

%% --- Графік 3: 3D-поверхня z1 (еталон vs НМ) ---
Ng = 30;
xg = linspace(0, pi/2, Ng);
yg = linspace(0, pi/2, Ng);
[Xg,Yg] = meshgrid(xg, yg);
Zg_true = Xg .* sin(Xg + Yg);
Zg_nn   = reshape(nets1{best1}([Xg(:)'; Yg(:)']), Ng, Ng);

figure('Name','3D z1 — Варіант 5','Color','w','Position',[800 50 1000 420]);
subplot(1,2,1);
surf(Xg,Yg,Zg_true,'EdgeColor','none'); colormap(gca,'parula'); colorbar;
xlabel('x'); ylabel('y'); zlabel('z1'); grid on;
title('Еталон  z1 = x*sin(x+y)');
subplot(1,2,2);
surf(Xg,Yg,Zg_nn,'EdgeColor','none'); colormap(gca,'parula'); colorbar;
xlabel('x'); ylabel('y'); zlabel('z1 НМ'); grid on;
title(sprintf('НМ: %s', short_labels{best1}));
sgtitle('Варіант 5 — z1(x,y) = x*sin(x+y)', 'FontSize',13);

%% --- Графік 4: Апроксимація y(x) (краща конфігурація) ---
[~, best2] = min(results2);
Out_best2  = nets2{best2}(Input2);

figure('Name','Апроксимація y(x) — Варіант 5','Color','w','Position',[800 520 700 350]);
plot(x2, Output2,   'b-',  'LineWidth', 2,   'DisplayName','Еталон y(x)');
hold on;
plot(x2, Out_best2, 'r--', 'LineWidth', 1.5, 'DisplayName',['НМ: ',configs{best2,3}]);
xlabel('x'); ylabel('y'); grid on;
title(sprintf('y(x)=sin(x)+cos(3x^2) — краща НМ: %s (%.4f%%)', ...
    configs{best2,3}, results2(best2)));
legend('Location','best');

%% ============================================================
%  ДОПОМІЖНІ ФУНКЦІЇ
%% ============================================================

function net = create_net(net_type, layers)
    switch net_type
        case 'feedforwardnet',    net = feedforwardnet(layers);
        case 'cascadeforwardnet', net = cascadeforwardnet(layers);
        case 'elmannet',          net = elmannet(layers);
    end
    net.trainParam.epochs     = 1000;
    net.trainParam.goal       = 1e-6;
    net.trainParam.show       = 25;
    net.trainParam.showWindow = false;
end

function print_table(configs, results, func_name)
    fprintf('\n--- Зведена таблиця: %s ---\n', func_name);
    fprintf('%-45s | Похибка (%%)\n', 'Конфігурація мережі');
    fprintf('%s\n', repmat('-',1,58));
    for i = 1:6
        marker = '';
        if results(i) == min(results), marker = ' <- найкраща'; end
        if results(i) == max(results), marker = ' <- найгірша'; end
        fprintf('%-45s | %.6f%s\n', configs{i,3}, results(i), marker);
    end
    fprintf('%s\n', repmat('=',1,58));
end
