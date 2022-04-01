import numpy as np

def my_knn(X, k, tolerance=1e-5, max_iter=1e6):

    centroids = X[np.random.randint(X.shape[0], size=k), :] # grab k rows at random
    dist = np.empty((X.shape[0], k))

    for _ in range(int(max_iter)): 
        
        old_centroids = centroids.copy()
        
        for j in range(k):
            dist[:, j] = np.sqrt(np.sum((centroids[j, :] - X)**2, axis=1))
            # need all distances
        
        assigned_centroid = np.argmin(dist, axis=1)
        
        for j in range(k):
            centroids[j, :] = X[assigned_centroid == j, :].mean(axis=0)  
        
        
        # if the centroids are no longer different than the algorythm
        # converged to a solution. 
        converged = np.sum(np.abs(centroids - old_centroids)) < tolerance
        
        if converged:
            break

    if not converged:
        from warnings import warn
        warn(f"Did not converge to a solution. This is bad given it's gurenteed to converge")
        
    return centroids

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    n = 100
    rng = np.random.RandomState(0)
    groups = np.random.binomial(2, 0.5, n) * 5
    # we need to recorver within some error
    # (x, y) = {(0, 0), (5, 5), (10, 10))
    x = rng.randn(n) + groups 
    y = rng.randn(n) + groups
    
    X = np.vstack((x, y)).T
    centroids = my_knn(X, k=3)
    
    # compare with spectral clustering
    from sklearn.cluster import SpectralClustering
    spectral_clustering = SpectralClustering(n_clusters=3).fit(X)
    centroids_spectral = np.empty((3, X.shape[1]))
    
    for j in range(3):
        centroids_spectral[j, :] = X[spectral_clustering.labels_ == j, :].mean(axis=0)  
    
    plt.plot(x, y, 'o', color="black")
    plt.plot(centroids_spectral[:, 0], centroids_spectral[:, 1], "o", color='blue', markersize=10)
    plt.plot(centroids[:, 0], centroids[:, 1], "x", color='red', markersize=10)
    plt.title("K means parameters recovery")
    plt.xlabel("X")
    plt.ylabel("Y") 
    plt.savefig('question1_fig.png')
