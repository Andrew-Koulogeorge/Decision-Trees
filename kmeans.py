"""
K-Means Clustering Implementation
-> "If you are doing loops in numpy or torch, you are probably doing something slowly"
"""
import numpy as np
import matplotlib.pyplot as plt

class Kmeans: 
    def __init__(self, k:int, tol=0.01) -> None:
        self.k = k
        self.been_fit = False
        self.tol = tol

    def fit(self, X: np.ndarray, iterations:int) -> np.ndarray:
        """
        K-means iterates over 2 major steps:
        1) Assign each datapoint to the lable of its nearest cluster
        2) Update each cluster to be the mean of all points with its label

        Input: 
        - X: (n,d) feature matrix 
        - iterations: number of loops before we terminate
        """
        self.been_fit = True
        self.X = X
        self.n, self.d = self.X.shape
        

        # init each cluster to be random point in X
        self.clusters = X[np.random.choice(self.n, self.k, replace=False)] # (k,d)
        self.labels = np.full(self.n, -1)
        for iters in range(iterations):
            # could also imporve the stopping criterion by comparing norms of the clusters before and after the loop
            prior_clusters = self.clusters.copy()

            # assign each datapoint to a cluster based on nearest l2 distance
            for i, x in enumerate(self.X):
                self.labels[i] = self._nearest_cluster(x)
            
            # update the centroid of each cluster            
            for cluster_label in range(self.k):
                pointer_in_cluster = self.X[self.labels == cluster_label]
                self.clusters[cluster_label] = pointer_in_cluster.mean(axis=0)
            
            # early stopping criterion
            cluster_diff = np.linalg.norm(self.clusters - prior_clusters)
            if cluster_diff < self.tol:
                break
        self.total_iters = iters
    
    def get_labels(self):
        return self.labels if self.been_fit else None

    def _nearest_cluster(self, x):
        min_dist = float("inf")
        label = -1
        for curr_label, curr_cluster in enumerate(self.clusters):
            dist = np.linalg.norm(x - curr_cluster)
            if dist < min_dist:
                label = curr_label
                min_dist = dist
        return label

    

if __name__ == "__main__":
    np.random.seed(2)

    mean1 = [1, 2]        
    cov1 = [[1, 2],       
            [2, 1]]       
    
    mean2 = [7, 4]
    cov2 = [[1, 0],
            [0, 1]]

    n_samples = 100
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples)    
    X = np.vstack((X1, X2))

    kmeans = Kmeans(k=2)
    kmeans.fit(X, iterations=100)

    data_labels = kmeans.get_labels()
    print(f"Number of iterations to terminate: {kmeans.total_iters}")

    plt.scatter(X[:, 0], X[:, 1], c=data_labels, cmap='viridis', s=50)
    plt.scatter(kmeans.clusters[:, 0], kmeans.clusters[:, 1], s=200, c='red', marker='X', label='Cluster Centers')
    plt.title('Clusters Identified by Your Algorithm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()
    
