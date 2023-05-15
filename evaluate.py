from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN 

from fcmeans import FCM
import skfda
import skfuzzy as fuzz



def AGC(emb, label, nb_clusters = None):


    if nb_clusters is None:
        nb_clusters = len(np.unique(label))
    
    sc_pred = AgglomerativeClustering(n_clusters=nb_clusters).fit(emb).labels_
    nmi = normalized_mutual_info_score(label, sc_pred)
    return nmi

def GMM(emb, label, nb_clusters = None):
    if nb_clusters is None:
        nb_clusters = len(np.unique(label))

    gmm = GaussianMixture(n_components=nb_clusters)
    gmm.fit(emb)
    gmm_pred = gmm.predict(emb)
    nmi = normalized_mutual_info_score(label, gmm_pred)
    return nmi

def kmeans(emb, label, nb_clusters = None):

    """
    Community detection: computes AMI and ARI scores
    from a K-Means clustering of nodes in an embedding space
    :param emb: n*d matrix of embedding vectors for all nodes
    :param label: ground-truth node labels
    :param nb_clusters: int number of ground-truth communities in the graph
    :return: Adjusted Mutual Information (AMI) and Adjusted Rand Index (ARI)
    """

    if nb_clusters is None:
        nb_clusters = len(np.unique(label))

    kmeans_pred = KMeans(n_clusters = nb_clusters, init = 'k-means++',
                             n_init = 200, max_iter = 500).fit(emb).labels_
    nmi = normalized_mutual_info_score(label, kmeans_pred)
    return nmi


def dbscan(emb, label,  nb_clusters = None):

    if nb_clusters is None:
        nb_clusters = len(np.unique(label))

    dbpred = DBSCAN().fit(emb).labels_ # eps=3, min_samples=2
    nmi = normalized_mutual_info_score(label, dbpred)
    return nmi

def fuzzy(emb, label,  nb_clusters = None):

    if nb_clusters is None:
        nb_clusters = len(np.unique(label))


    # Apply fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            emb.T, nb_clusters, 2, error=0.005, maxiter=1000, init=None)
 
    # Predict cluster membership for each data point
    cluster_membership = np.argmax(u, axis=0)
    nmi = normalized_mutual_info_score(label, cluster_membership)
    return nmi

    

#https://www.geeksforgeeks.org/ml-fuzzy-clustering/

# https://github.com/scikit-fuzzy/scikit-fuzzy
