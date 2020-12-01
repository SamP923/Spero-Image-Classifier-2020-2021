
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import numpy as np

class BIC:
    def __init__(self, cluster_range):
        self.cluster_range = cluster_range
    def bicMethod(self, data):
        bics = []
        for n_clusters in tqdm(self.cluster_range):
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
            gmm.fit(data)
            bics.append(gmm.bic(data))
        return self.cluster_range[bics.index(min(bics))]
    def customBIC(self, data):
        bics = []
        for n_clusters in self.cluster_range:
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
            gmm.fit(data)
            clusters = []
            for i in range(n_clusters):
                clusters.append([])
            clusterSizes = np.zeros(n_clusters)
            labels = gmm.predict(data)
            for i in range(len(labels)):
                clusters[labels[i]].append(data[i])
                clusterSizes[labels[i]] += 1
            log_sigma = np.zeros(n_clusters)
            for i in range(n_clusters):
                log_sigma[i] = np.log(abs(np.linalg.det(np.cov(np.transpose(np.array(clusters[i]))))))
            bic = 0
            bic += np.dot(clusterSizes, np.log(clusterSizes))
            bic -= np.dot(clusterSizes, log_sigma)/3
            q = n_clusters*(n_clusters+3)/2
            bic -= q*np.sum(np.log(clusterSizes))/2
            print(bic)
            bics.append(bic)
        return self.cluster_range[bics.index(max(bics))]
