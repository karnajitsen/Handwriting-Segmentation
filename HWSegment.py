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
db1 = DBSCAN(eps=5, min_samples=12).fit(img.getEdgesFeature(0))
core_samples_mask = np.zeros_like(db1.labels_, dtype=bool)
core_samples_mask[db1.core_sample_indices_] = True
labels = db1.labels_
print(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#Calculating Maximum and Minimum of each cluster and storing 1 for max and -1 for min value in corresponding index position of maxminarray .. . Hell sloww!!!!

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



#Classifying the maxima and minima in 4 classes .. Hell sloww!!!!

for i in range(2,n_clusters_-2):
     
      maxvalcur = findMaxInCluster(i,img.getEdgesFeature(0),maxminarray,labels)
      minvalcur = findMinInCluster(i,img.getEdgesFeature(0),maxminarray,labels)
      
      maxvalnext = findMaxInCluster(i+1,img.getEdgesFeature(0),maxminarray,labels)
      minvalnext = findMinInCluster(i+1,img.getEdgesFeature(0),maxminarray,labels)
      
      maxvalprev = findMaxInCluster(i-1,img.getEdgesFeature(0),maxminarray,labels)
      minvalprev = findMinInCluster(i-1,img.getEdgesFeature(0),maxminarray,labels)
      
      diffmaxnxt = abs(maxvalnext - maxvalcur)
      diffmaxprv = abs(maxvalprev - maxvalcur)
      diffminnxt = abs(minvalnext - minvalcur)
      diffminprv = abs(minvalprev - minvalcur)
      print(diffmaxnxt)
      print(diffmaxprv)
      diffcurr = abs(maxvalcur - minvalcur)
      p1 = p2 = p3 = 0
      if diffcurr > 3 and diffcurr <= 10:
          p1 = 1 #lower and upper baseline
      else:
          if (diffmaxnxt > 2 and diffmaxnxt <=6) or (diffmaxprv > 2 and diffmaxprv <=6):
              p2 = 1 
          if (diffminnxt > 2 and diffminnxt <=6) or (diffminprv > 2 and diffminprv <=6):
              p3 = 1
       
      b = [item for item in range(len(labels)) if labels[item] == i]
               
      maxindex = [b[j] for j in range(len(b)) if img.getEdgesFeature(0)[b[j]][1] == maxvalcur]
      minindex = [b[j] for j in range(len(b)) if img.getEdgesFeature(0)[b[j]][1] == minvalcur]
           
      if p1==1:
          img.getEdgesFeature(0)[maxindex][:,2]=0
          img.getEdgesFeature(0)[minindex][:,2]=1
      elif p2==1 and p3 ==1:
          img.getEdgesFeature(0)[maxindex][:,2]=2
          img.getEdgesFeature(0)[minindex][:,2]=2
      elif p2==1 and p3==0:
          img.getEdgesFeature(0)[maxindex][:,2]=2
          img.getEdgesFeature(0)[minindex][:,2]=1
      elif p3==1 and p2==0:
          img.getEdgesFeature(0)[maxindex][:,2]=0
          img.getEdgesFeature(0)[minindex][:,2]=2
      else: 
          img.getEdgesFeature(0)[maxindex][:,2]=3
          img.getEdgesFeature(0)[minindex][:,2]=3



t = invertImage(img.getEdgeImage(0))
cv2.imwrite( "edge.jpg", t )
t = markImage(t,img.getEdgesFeature(0))
backtorgb = cv2.cvtColor(t,cv2.COLOR_GRAY2RGB )
cv2.imwrite( "edgemarked.jpg",backtorgb )
# print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))

# Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

def findMaxInCluster(i,image,maxminarray,labels):
      b = [item for item in range(len(labels)) if labels[item] == i]
      maxval = 0
      for j in range(0,len(b)):
          if maxminarray[b[j]] == 1:
             maxval = image[b[j]][1]
             break
        
      return maxval
    
def findMinInCluster(i,image,maxminarray,labels):
      b = [item for item in range(len(labels)) if labels[item] == i]
      minval = 0
      for j in range(0,len(b)):
          if maxminarray[b[j]] == -1:
             minval = image[b[j]][1]
             break
        
      return minval
      
def invertImage(img):
    dim = np.shape(img)
    for i in range(dim[0]):
        for j in range(dim[1]):
            if img[i][j]==0:
                img[i][j] = 255
            else:
                img[i][j] = 0
                
    return img
            
def markImage(img, featureImg):
    a = np.shape(featureImg)
    for i in range(a[0]):
        if featureImg[i][2] == 0:
            img[featureImg[i][0]][featureImg[i][1]] = 50
            #print(i)
        if featureImg[i][2] == 1:
            img[featureImg[i][0]][featureImg[i][1]] = 100    
        if featureImg[i][2] == 2:
            img[featureImg[i][0]][featureImg[i][1]] = 150
        if featureImg[i][2] == 3:
            img[featureImg[i][0]][featureImg[i][1]] = 200
    return img    




