load("data/ad_data.mat")

% Specify the options
opts.rFlag = 1;  % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4;  % termination options.
opts.maxIter = 5000; % maximum iterations.

addpath(genpath([root '/SLEP']));

pars = 0.001:0.0001:0.1; % regularization values to try
auc_vec = zeros(1, length(pars));
for i = 1:numel(pars)
    [w, c] = LogisticR(X_train, y_train, pars(i), opts);
     phat = (1 + exp(- (X_test * w + c))); % convert estimates to probs
     [X,Y,T,AUC] = perfcurve(y_test,phat,1);
    auc_vec(i) = AUC;
end
[AUC, i] = max(auc_vec);
[w, c] = LogisticR(X_train, y_train, pars(i), opts); % recaluclate using best value
AUC
pars(i)
sum(w == 0);
