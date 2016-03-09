import cv2
import sklearn
import numpy as np
import os
from Images import Images
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics

path1 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/data/'

img = Images(path1)
img.cnt
img.applyCanny()
img.createFeatures()

# Compute DBSCAN
db1 = DBSCAN(eps=10, min_samples=10).fit(img.getEdgesFeature(0))
core_samples_mask = np.zeros_like(db1.labels_, dtype=bool)
core_samples_mask[db1.core_sample_indices_] = True
labels = db1.labels_
print(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#Calculating Maximum and Minimum of each cluster and storing 1 for max and -1 for min value in corresponding index position of maxminarray

maxminarray = np.zeros((len(labels),1))

for i in range(1,n_clusters_):
      b = [item for item in range(len(labels)) if labels[item] == i]
      temp = img.getEdgesFeature(0)[b][:,1]
      maxvalue = max(temp)
      minvalue = min(temp)
      for j in range(0,len(b)):
          if temp[j] == maxvalue:
              maxminarray[b[j]] = 1
          if temp[j] == minvalue:
              maxminarray[b[j]] = -1

      
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

def findMaxInCluster(cluster):
    m1 = max(cluster)
    b = [item for item in range(len(cluster)) if cluster[item] == m1]
    return b
    
def findMinInCluster(cluster):
    m1 = min(cluster)
    b = [item for item in range(len(cluster)) if cluster[item] == m1]
    return b

    




