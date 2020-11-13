
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import numpy as np

class BIC:
    def __init__(self, cluster_range):
        self.cluster_range = cluster_range
    def bicMethod(self, data):
        bics = []
        scores = []
        for n_clusters in tqdm(self.cluster_range):
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
            gmm.fit(data)
            scores.append(gmm.score(data))
            bics.append(n_clusters*np.log(data.shape[0])-2*np.sqrt(data.shape[0])*gmm.score(data))

        print(scores)
        print(bics)
        return self.cluster_range[bics.index(min(bics))]