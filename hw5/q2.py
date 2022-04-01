import numpy as np


def my_decomp(p, matrix):
    U, lam, Vt = np.linalg.svd(matrix, full_matrices=False)
    # regular svd, then do elementwise multiplication of eigenvales by 
    # (1,1,...p, 0, 0, ... k) where k is the rank of the matrix
    # diagnalize and multiply the matricies and get a lower representation
    ones_vec = np.concatenate((np.ones(p), np.zeros(lam.shape[0]-p)))
    return U @ np.diag(np.multiply(lam, ones_vec)) @ Vt 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    data = loadmat('data/USPS.mat')
    A = data["A"]
    
    fig, axs = plt.subplots(4,2, figsize=(6, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 2.1)


    axs = axs.ravel()
    for i, p in enumerate([10, 50, 100, 200]):

        A_decomposed = my_decomp(p, A)
        a0 = np.reshape(A_decomposed[0, :], (16, 16))
        a0_error = np.sum(np.abs(A_decomposed[0, :] - A[0, :]))
        a1 = np.reshape(A_decomposed[1, :], (16, 16))
        a1_error = np.sum(np.abs(A_decomposed[1, :] - A[1, :]))
        axs[i].imshow(a0,  cmap='gray')
        axs[i].set_title(f"number of components\n p = {p}\n error = {round(a0_error, 2)}")
        axs[i+4].imshow(a1,  cmap='gray')
        axs[i+4].set_title(f"number of components\n p = {p}\n error = {round(a0_error,2)}")
    plt.savefig('question2_fig.png')