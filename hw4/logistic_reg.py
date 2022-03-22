import numpy as np
import pandas as pd


def logistic_train(data, labels, epsilon=1e-5, max_iter=1000):
    X, t = data, labels
    w = np.zeros(data.shape[1])
    # start with a vector of 0s with the goal of improving
    # with each iteratioon
    for _ in range(max_iter):
        w_old = w
        yhat = 1 / (1 + np.exp(-(X @ w)))
        R = np.diag(yhat * (1 - yhat))  # Eq 4.98
        z = yhat - np.linalg.inv(R) @ (yhat - t)  # Eq 4.100
        w = np.linalg.inv(X.T @ R @ X) @ X.T @ R @ z  # Eq 4.99
        converged = epsilon > np.sum(np.abs(w - w_old))
        # if the updated weights are no longer different than the algorythm
        # converged to a solution. This does not imply a good fit
        if converged:
            break

    if not converged:
        from warnings import warn

        warn(f"Newton-Rapson Algorthm did not converge to error < {epsilon}")

    return w  # coeffecients use X @ w to get a prediction


if __name__ == "__main__":
    # lets take this for a spin!
    data = "https://raw.githubusercontent.com/jiayuzhou/CSE847/master/data/spam_email/data.txt"
    labels = "https://raw.githubusercontent.com/jiayuzhou/CSE847/master/data/spam_email/labels.txt"

    X = pd.read_csv(data, sep="  ", header=None, engine="python").to_numpy()
    y = pd.read_csv(labels, header=None).to_numpy().ravel()
    X_train, X_test, y_train, y_test = X[0:2000, :], X[2000:, :], y[0:2000], y[2000:]

    # lets see if the accuacy improves as n increases
    for n in [200, 500, 1000, 1500, 2000]:
        X_temp = X_train[0:n, :]
        y_temp = y_train[0:n]
        w = logistic_train(X_temp, y_temp)  # use default iter and epsilon settings
        phat = 1 / (1 + np.exp(-X_test @ w))
        yhat = np.where(phat > 0.5, 1, 0)  # adjust depending on application
        acc = np.sum(yhat == y_test) / y_test.shape[0]
        print(f"sample of n = {n} predicted an accuracy of {round(acc*100, 2)}%")
