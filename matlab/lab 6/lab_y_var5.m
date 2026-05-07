%% ============================================================
%  Лабораторна робота №1 — Варіант 5
%  Функція: y(x) = sin(x) + cos(3*x^2)
%  (моделюється як функція двох змінних: y-вхід фіктивний)
%  Запуск: >> lab_y_var5
%% ============================================================
clear; clc; close all;

%% ── Параметри ───────────────────────────────────────────────
f   = @(x,~) sin(x) + cos(3*x.^2);   % y-аргумент не використовується
N_IN  = 6;
N_OUT = 9;
X_MIN = 0; X_MAX = pi;
Y_MIN = 0; Y_MAX = pi;
Z_MIN = -2.05; Z_MAX = 2.05;

x_centers = linspace(X_MIN, X_MAX, N_IN);
y_centers = linspace(Y_MIN, Y_MAX, N_IN);
z_centers = linspace(Z_MIN, Z_MAX, N_OUT);

%% ── Таблиця значень функції ─────────────────────────────────
fprintf('\n=== y(x)=sin(x)+cos(3*x^2)  [ значення у точках МФ ] ===\n');
fprintf('%10s','');
for j=1:N_IN, fprintf('%9.3f', y_centers(j)); end; fprintf('\n');
Z_tbl = zeros(N_IN);
for i=1:N_IN
    fprintf('%10.3f', x_centers(i));
    for j=1:N_IN
        Z_tbl(i,j) = f(x_centers(i), y_centers(j));
        fprintf('%9.3f', Z_tbl(i,j));
    end; fprintf('\n');
end

%% ── Таблиця правил ──────────────────────────────────────────
Rule_tbl = zeros(N_IN);
fprintf('\n=== Таблиця правил (індекс вихідної МФ 1..%d) ===\n', N_OUT);
fprintf('%10s','');
for j=1:N_IN, fprintf('%6s',['my',num2str(j)]); end; fprintf('\n');
for i=1:N_IN
    fprintf('%10s',['mx',num2str(i)]);
    for j=1:N_IN
        [~,idx] = min(abs(z_centers - Z_tbl(i,j)));
        Rule_tbl(i,j) = idx;
        fprintf('%6s',['mf',num2str(idx)]);
    end; fprintf('\n');
end

%% ── Функція побудови FIS ────────────────────────────────────
function fis = make_fis(mf_type, rule_tbl, ...
        x_min,x_max, y_min,y_max, z_min,z_max, n_in, n_out)

    fis = mamfis('Name', mf_type);

    % --- вхід x ---
    fis = addInput(fis,[x_min x_max],'Name','x');
    step_x = (x_max-x_min)/(n_in-1);
    for k=1:n_in
        c = x_min + (k-1)*step_x;
        switch mf_type
          case 'gaussmf'
            fis = addMF(fis,'x','gaussmf',[step_x*0.45 c],'Name',['mx',num2str(k)]);
          case 'trimf'
            fis = addMF(fis,'x','trimf',[max(x_min,c-step_x) c min(x_max,c+step_x)],...
                        'Name',['mx',num2str(k)]);
          case 'trapmf'
            fis = addMF(fis,'x','trapmf',...
                [max(x_min,c-step_x*.95) max(x_min,c-step_x*.35) ...
                 min(x_max,c+step_x*.35) min(x_max,c+step_x*.95)],...
                'Name',['mx',num2str(k)]);
        end
    end

    % --- вхід y (фіктивний для y(x)) ---
    fis = addInput(fis,[y_min y_max],'Name','y');
    step_y = (y_max-y_min)/(n_in-1);
    for k=1:n_in
        c = y_min + (k-1)*step_y;
        switch mf_type
          case 'gaussmf'
            fis = addMF(fis,'y','gaussmf',[step_y*0.45 c],'Name',['my',num2str(k)]);
          case 'trimf'
            fis = addMF(fis,'y','trimf',[max(y_min,c-step_y) c min(y_max,c+step_y)],...
                        'Name',['my',num2str(k)]);
          case 'trapmf'
            fis = addMF(fis,'y','trapmf',...
                [max(y_min,c-step_y*.95) max(y_min,c-step_y*.35) ...
                 min(y_max,c+step_y*.35) min(y_max,c+step_y*.95)],...
                'Name',['my',num2str(k)]);
        end
    end

    % --- вихід z ---
    fis = addOutput(fis,[z_min z_max],'Name','z');
    step_z = (z_max-z_min)/(n_out-1);
    for k=1:n_out
        c = z_min + (k-1)*step_z;
        switch mf_type
          case 'gaussmf'
            fis = addMF(fis,'z','gaussmf',[step_z*0.4 c],'Name',['mf',num2str(k)]);
          case 'trimf'
            fis = addMF(fis,'z','trimf',[max(z_min,c-step_z) c min(z_max,c+step_z)],...
                        'Name',['mf',num2str(k)]);
          case 'trapmf'
            fis = addMF(fis,'z','trapmf',...
                [max(z_min,c-step_z*.95) max(z_min,c-step_z*.35) ...
                 min(z_max,c+step_z*.35) min(z_max,c+step_z*.95)],...
                'Name',['mf',num2str(k)]);
        end
    end

    % --- правила ---
    rules = [];
    for i=1:n_in
        for j=1:n_in
            rules(end+1,:) = [i j rule_tbl(i,j) 1 1]; %#ok<AGROW>
        end
    end
    fis = addRule(fis, rules);
end

%% ── Сітка для оцінки похибки ────────────────────────────────
N_eval  = 10;
xs_eval = linspace(X_MIN+0.1, X_MAX-0.1, N_eval);
ys_eval = linspace(Y_MIN+0.1, Y_MAX-0.1, N_eval);
[Xe,Ye] = meshgrid(xs_eval, ys_eval);
Ze_true = f(Xe, Ye);

mf_types  = {'gaussmf','trimf','trapmf'};
mf_labels = {'Гаусова','Трикутна','Трапецієподібна'};
err_full  = zeros(1,3);
err_diag  = zeros(1,3);

%% ── Цикл по типах МФ ────────────────────────────────────────
for m = 1:3
    mft = mf_types{m};

    % ---- Повна база (36 правил) ----
    fis36 = make_fis(mft, Rule_tbl, ...
        X_MIN,X_MAX, Y_MIN,Y_MAX, Z_MIN,Z_MAX, N_IN, N_OUT);

    Ze_pred = zeros(size(Xe));
    for ii=1:numel(Xe)
        Ze_pred(ii) = evalfis(fis36,[Xe(ii) Ye(ii)]);
    end
    eps36 = abs(Ze_true-Ze_pred)./(abs(Ze_true)+1)*100;
    err_full(m) = mean(eps36(:));

    % ---- Діагональна база (6 правил) ----
    Rule_diag = ones(N_IN)*Rule_tbl(1,1);
    for k=1:N_IN, Rule_diag(k,k) = Rule_tbl(k,k); end
    for i=1:N_IN, for j=1:N_IN
        if i~=j
            [~,ni] = min(abs((1:N_IN)-mean([i j])));
            Rule_diag(i,j) = Rule_tbl(ni,ni);
        end
    end; end

    fis6 = make_fis(mft, Rule_diag, ...
        X_MIN,X_MAX, Y_MIN,Y_MAX, Z_MIN,Z_MAX, N_IN, N_OUT);

    Ze_diag = zeros(size(Xe));
    for ii=1:numel(Xe)
        Ze_diag(ii) = evalfis(fis6,[Xe(ii) Ye(ii)]);
    end
    epsd = abs(Ze_true-Ze_diag)./(abs(Ze_true)+1)*100;
    err_diag(m) = mean(epsd(:));

    fprintf('\n[%s]  Повна база: %.2f%%   Діагональ: %.2f%%\n',...
        mf_labels{m}, err_full(m), err_diag(m));

    if m==1, assignin('base','fis_y_gauss',fis36); end
end

%% ── Графік 1: еталонна функція ──────────────────────────────
figure('Name','y(x) — Еталон','Color','w','Position',[50 50 600 420]);
xp = linspace(X_MIN, X_MAX, 300);
plot(xp, sin(xp) + cos(3*xp.^2), 'b-', 'LineWidth',2);
xlabel('x'); ylabel('y'); grid on;
title('Еталонна функція y(x) = sin(x) + cos(3x^2)','FontSize',12);

%% ── Графік 2: МФ входу ──────────────────────────────────────
figure('Name','МФ входу','Color','w','Position',[50 520 900 300]);
univ = linspace(X_MIN,X_MAX,500);
subplot(1,3,1); hold on;
for k=1:N_IN
    c = x_centers(k);
    plot(univ, gaussmf(univ,[0.45*(X_MAX-X_MIN)/(N_IN-1) c]),'LineWidth',1.6);
end
title('Гаусова'); xlabel('x'); ylabel('\mu'); grid on; ylim([0 1.1]);

subplot(1,3,2); hold on;
step = (X_MAX-X_MIN)/(N_IN-1);
for k=1:N_IN
    c = x_centers(k);
    plot(univ, trimf(univ,[max(X_MIN,c-step) c min(X_MAX,c+step)]),'LineWidth',1.6);
end
title('Трикутна'); xlabel('x'); ylabel('\mu'); grid on; ylim([0 1.1]);

subplot(1,3,3); hold on;
for k=1:N_IN
    c = x_centers(k);
    plot(univ, trapmf(univ,...
        [max(X_MIN,c-step*.95) max(X_MIN,c-step*.35) ...
         min(X_MAX,c+step*.35) min(X_MAX,c+step*.95)]),'LineWidth',1.6);
end
title('Трапецієподібна'); xlabel('x'); ylabel('\mu'); grid on; ylim([0 1.1]);
sgtitle('Функції приналежності входу x  (y(x) = sin(x)+cos(3x^2))');

%% ── Графік 3: порівняння похибок ────────────────────────────
figure('Name','Порівняння похибок','Color','w','Position',[700 50 640 400]);
xb = 1:3; wb = 0.35;
b1 = bar(xb-wb/2, err_full, wb, 'FaceColor','flat');
b1.CData = [0.18 0.63 0.35; 0.85 0.25 0.25; 0.22 0.48 0.78];
hold on;
b2 = bar(xb+wb/2, err_diag, wb, 'FaceColor','flat','FaceAlpha',0.5);
b2.CData = b1.CData;
set(gca,'XTickLabel',mf_labels,'FontSize',11);
legend({'36 правил (повна)','6 правил (діагональ)'},'Location','NorthWest');
ylabel('Середня похибка \epsilon, %');
title('y(x) = sin(x)+cos(3x^2)  — Порівняння похибок');
grid on;
for k=1:3
    text(k-wb/2, err_full(k)+0.2, sprintf('%.1f%%',err_full(k)),...
         'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
    text(k+wb/2, err_diag(k)+0.2, sprintf('%.1f%%',err_diag(k)),...
         'HorizontalAlignment','center','FontSize',9,'Color',[0.5 0 0]);
end

%% ── Висновки ────────────────────────────────────────────────
[~,bi]=min(err_full); [~,wi]=max(err_full);
fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
fprintf('║   ВИСНОВКИ  y(x) = sin(x)+cos(3*x^2)                   ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║ П.1  FIS Mamdani: 2 входи × %d МФ, 1 вихід × %d МФ       ║\n',N_IN,N_OUT);
fprintf('║      База знань: %d×%d = %d правил.                       ║\n',N_IN,N_IN,N_IN^2);
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║ П.2  Найкраща МФ  : %-16s (%.2f%%)         ║\n',mf_labels{bi},err_full(bi));
fprintf('║      Найгірша МФ  : %-16s (%.2f%%)         ║\n',mf_labels{wi},err_full(wi));
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║ П.3  Діагональ (6 правил): похибка зростає в %.1fx      ║\n',mean(err_diag./err_full));
fprintf('║      Повна база правил є необхідною.                    ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║ П.4  Оптимум: %s МФ + 36 правил.          ║\n',mf_labels{bi});
fprintf('╚══════════════════════════════════════════════════════════╝\n');
