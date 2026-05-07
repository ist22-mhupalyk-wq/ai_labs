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

% Назви типів мережі для заголовків (як у звіті)
net_type_titles = {
    '1. Тип мережі: feed forward backprop';
    '1. Тип мережі: feed forward backprop';
    '2. Тип мережі: cascade - forward backprop';
    '2. Тип мережі: cascade - forward backprop';
    '3. Тип мережі: elman backprop';
    '3. Тип мережі: elman backprop';
};

sub_titles = {
    'a) 1 внутрішній шар з 10 нейронами;';
    'b) 1 внутрішній шар з 20 нейронами;';
    'a) 1 внутрішній шар з 20 нейронами;';
    'b) 2 внутрішніх шари по 10 нейронів у кожному;';
    'a) 1 внутрішній шар з 15 нейронами;';
    'b) 3 внутрішніх шари по 5 нейронів у кожному;';
};

approx_labels = {'FF 1x10','FF 1x20','CF 1x20','CF 2x10','EL 1x15','EL 3x5'};

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

x2      = linspace(0, pi, N);
Input2  = x2;           % одновимірний вхід
Output2 = sin(x2) + cos(3*x2.^2);

results2 = zeros(1,6);
nets2    = cell(1,6);
trs2     = cell(1,6);   % зберігаємо tr для графіків навчання

for i = 1:6
    fprintf('\n=== Конфігурація %d: %s ===\n', i, configs{i,3});
    net = create_net(configs{i,1}, configs{i,2});
    [net, tr] = train(net, Input2, Output2);
    Out_nn = net(Input2);
    err = mean(abs(Output2 - Out_nn) ./ (abs(Output2) + 1e-10)) * 100;
    results2(i) = err;
    nets2{i}    = net;
    trs2{i}     = tr;
    fprintf('Середня відносна похибка: %.6f %%\n', err);
    fprintf('MSE на тренуванні:        %.2e\n', tr.best_perf);
end

print_table(configs, results2, 'y(x) = sin(x)+cos(3x^2)');

%% ============================================================
%  ГРАФІКИ — ЧАСТИНА 2: по одному рисунку на конфігурацію
%  Кожен рисунок: ліво — апроксимація, право — крива MSE
%% ============================================================

colors_approx = lines(6);   % різні кольори для кожної конф.

for i = 1:6
    Out_nn_i = nets2{i}(Input2);
    tr_i     = trs2{i};
    err_i    = results2(i);

    fig = figure('Color','w','Position',[100 100 900 340]);
    sgtitle(sprintf('%s\n%s', net_type_titles{i}, sub_titles{i}), ...
            'FontSize', 11, 'FontWeight','bold');

    % --- Ліва частина: апроксимація ---
    subplot(1,2,1);
    plot(x2, Output2, 'b-',  'LineWidth', 2,   'DisplayName','Еталонна функція');
    hold on;
    plot(x2, Out_nn_i,'r--', 'LineWidth', 1.5, 'DisplayName','Вихід НМ');
    xlabel('x  (y = x)'); ylabel('y');
    title(sprintf('Апроксимація: %s', approx_labels{i}));
    legend('Location','best','FontSize',8);
    grid on;
    % підпис похибки в лівому нижньому куті
    xl = xlim; yl = ylim;
    text(xl(1)+0.02*(xl(2)-xl(1)), yl(1)+0.07*(yl(2)-yl(1)), ...
         sprintf('Похибка: %.4f%%', err_i), ...
         'FontSize',9,'Color',[0.1 0.5 0.1],'BackgroundColor','w', ...
         'EdgeColor',[0.7 0.7 0.7]);

    % --- Права частина: крива навчання MSE ---
    subplot(1,2,2);
    semilogy(tr_i.epoch, tr_i.perf, 'Color',[1.0 0.6 0.1], 'LineWidth', 1.5);
    xlabel('Епоха'); ylabel('MSE (log)');
    title('Графік навчання (MSE)');
    grid on;
end

%% ============================================================
%  ЗВЕДЕНІ ГРАФІКИ (як на фото — 4. Зробити висновки)
%% ============================================================

%% --- Графік A: Стовпчаста діаграма похибок y(x) ---
fig_bar = figure('Color','w','Position',[100 100 750 400]);
bar_colors = [
    0.18 0.55 0.80;   % FF 1x10  — синій
    0.18 0.55 0.80;   % FF 1x20  — синій
    0.30 0.70 0.30;   % CF 1x20  — зелений
    0.30 0.70 0.30;   % CF 2x10  — зелений
    1.00 0.55 0.00;   % EL 1x15  — оранжевий
    0.85 0.20 0.20;   % EL 3x5   — червоний
];
bh = bar(1:6, results2, 'FaceColor','flat');
bh.CData = bar_colors;
set(gca,'XTick',1:6,'XTickLabel', short_labels,'FontSize',10);
ylabel('Середня відносна похибка (%)');
title('Порівняння похибок різних конфігурацій нейронних мереж');
grid on; hold on;
% числа над стовпцями
for i = 1:6
    text(i, results2(i)+0.3, sprintf('%.3f%%', results2(i)), ...
         'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end
xlabel('Конфігурація мережі');

%% --- Графік B: Криві навчання всіх конфігурацій на одному графіку ---
fig_curves = figure('Color','w','Position',[100 550 750 380]);
cmap = lines(6);
hold on;
for i = 1:6
    semilogy(trs2{i}.epoch, trs2{i}.perf, ...
             'Color', cmap(i,:), 'LineWidth', 1.5, ...
             'DisplayName', short_labels{i});
end
xlabel('Епоха'); ylabel('MSE (log scale)');
title('Криві навчання всіх конфігурацій нейронних мереж');
legend('Location','NorthEast','FontSize',9);
grid on;

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
