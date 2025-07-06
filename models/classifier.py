import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import torch
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from joblib import dump, load
from numba import jit
# from multiprocessing import Pool
from tqdm import tqdm


class Classifier:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()

    def train(self, data):
        pass

    def _normalize(self, data):
        return self.scaler.transform(data)

    def cluster(self, data):
        pass


class KMeansClassifier(Classifier):
    def __init__(self, n_clusters):
        super(KMeansClassifier, self).__init__(n_clusters)
        self.model = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')

    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.fit(data)
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'kmeans_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'kmeans_scaler.joblib'))

    def cluster(self, data):
        data = self._normalize(data)
        return self.model.predict(data)
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'kmeans_classifier.joblib'))
        self.scaler = load(os.path.join(path, 'kmeans_scaler.joblib'))


class MeanShiftClassifier(Classifier):
    def __init__(self):
        super(MeanShiftClassifier, self).__init__(n_clusters=None)
        self.model = MeanShift(cluster_all=True, n_jobs=-1)

    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.fit(data)
        self.n_clusters = len(np.unique(self.model.labels_))
        print(f'Mean shift identified {self.n_clusters} clusters')
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'mean_shift_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'mean_shift_scaler.joblib'))

    def cluster(self, data):
        data = self._normalize(data)
        return self.model.predict(data)
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'mean_shift_classifier.joblib'))
        self.scaler = load(os.path.join(path, 'mean_shift_scaler.joblib'))


class GaussianMixtureClassifier(Classifier):
    def __init__(self, n_clusters):
        super(GaussianMixtureClassifier, self).__init__(n_clusters)
        self.model = GaussianMixture(n_components=n_clusters, random_state=0)

    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.fit(data)
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'gmm_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'gmm_scaler.joblib'))

    def cluster(self, data):
        data = self._normalize(data)
        return self.model.predict(data)
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'gmm_classifier.joblib'))
        self.scaler = load(os.path.join(path, 'gmm_scaler.joblib'))


class WassersteinKMeansClassifier(KMeansClassifier):
    def __init__(self, n_clusters):
        super(WassersteinKMeansClassifier, self).__init__(n_clusters)
        self.model = KMeansWasserstein(n_clusters=n_clusters, random_state=0)

    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.fit(data)
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'wasserstein_kmeans_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'wasserstein_kmeans_scaler.joblib'))
             
    def load_model(self, path):
        self.model = load(os.path.join(path, 'wasserstein_kmeans_classifier.joblib'))
        self.scaler = load(os.path.join(path, 'wasserstein_kmeans_scaler.joblib'))
                           
    def cluster(self, data):
        data = self._normalize(data)
        return self.model.predict(data)
    

class KMeansWasserstein(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None, distance_metric="wasserstein", init='k-means++', n_jobs=16):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance_metric = distance_metric
        self.init = init
        self.n_jobs = n_jobs
        self.centers = None

    def _initialize_centers(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        self.centers = [X[np.random.randint(n_samples)]]
        for _ in range(1, self.n_clusters):
            # Compute distances from each point to the nearest center
            if self.distance_metric == 'euclidean':
                distances = np.array([min(np.linalg.norm(x - center)**2 for center in self.centers) for x in X])
            elif self.distance_metric == 'wasserstein':
                distances = np.array([min(wasserstein_distance(x, center) for center in self.centers) for x in X])

            # Choose the next center
            probabilities = distances / distances.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            next_center_idx = np.searchsorted(cumulative_probabilities, r)
            self.centers.append(X[next_center_idx])
        self.centers = np.array(self.centers)

    def _compute_distances(self, X):
        if self.n_jobs is None or self.n_jobs == 1:
            return self._compute_distances_single(X)
        else:
            return self._compute_distances_parallel(X)
        
    def _compute_distances_single(self, X):
        if self.distance_metric == 'euclidean':
            return cdist(X, self.centers, 'euclidean')
        elif self.distance_metric == 'wasserstein':
            return np.array([[wasserstein_distance(x, center) for center in self.centers] for x in X])
        else:
            raise ValueError("Unsupported distance metric")

    def _compute_distances_parallel(self, X):
        split_size = X.shape[0] // self.n_jobs
        X_split = np.array_split(X, self.n_jobs)
    
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if self.distance_metric == 'euclidean':
                # Use the standalone function for euclidean distance
                results = list(executor.map(euclidean_parallel_helper, [(x, self.centers) for x in X_split]))
            elif self.distance_metric == 'wasserstein':
                # Use the standalone function for wasserstein distance
                results = list(executor.map(wasserstein_parallel_helper, [(x, self.centers) for x in X_split]))
            return np.vstack(results)

    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def _update_centers(self, X, labels):
        new_centers = []
        for i in range(self.n_clusters):
            if np.any(labels == i):
                new_center = X[labels == i].mean(axis=0)
            else:
                # Reinitialize empty cluster to a random data point
                new_center = X[np.random.randint(0, X.shape[0])]
            new_centers.append(new_center)
        new_centers = np.array(new_centers)
        shift = np.linalg.norm(self.centers - new_centers)
        self.centers = new_centers
        return shift

    def fit(self, X):
        self._initialize_centers(X)
        for i in range(self.max_iter):
            distances = self._compute_distances(X)
            labels = self._assign_clusters(distances)
            shift = self._update_centers(X, labels)
            if shift < self.tol:
                break

    def predict(self, X):
        distances = self._compute_distances(X)
        return self._assign_clusters(distances)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
    
def compute_euclidean_distance(x, centers):
    return cdist([x], centers, 'euclidean').flatten()

def compute_wasserstein_distance(x, centers):
    all_dist = []
    for sample in x:
        dist = [wasserstein_distance(sample, center) for center in centers]
        all_dist.append(dist)
    return np.array(all_dist)

def euclidean_parallel_helper(args):
    return compute_euclidean_distance(*args)

def wasserstein_parallel_helper(args):
    return compute_wasserstein_distance(*args)