load("data/ad_data.mat")

root=cd;
addpath(genpath([root '/SLEP']));

opts.rFlag = 1;  % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4;  % termination options.
opts.maxIter = 5000; % maximum iterations.

pars = 0:0.001:1; % regularization values to try
auc_vec = zeros(1, length(pars));
for i = 1:numel(pars)
     [w, c] = LogisticR(X_train, y_train, pars(i), opts);
     yhat = X_test * w + c;
     [X,Y,T,AUC] = perfcurve(y_test,yhat, 1);
       auc_vec(i) = AUC;
end

[AUC, i] = max(auc_vec);
[w, c] = LogisticR(X_train, y_train, pars(i), opts) % recaluclate using best value
AUC
pars(i)
sum(w == 0)

figure;
plot(pars, auc_vec, '-r');
title('Logistic Reg Exeriment');
xlabel('penalty ');
ylabel('AUC');
saveas(gcf,'reg_experiment.png');
