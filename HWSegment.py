import cv2
import cv
import sklearn
import numpy as np
import os
from Images import Images
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

path1 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/data/'

img = Images(path1)
print(img.cnt)
img.applyCanny()
img.createFeatures()
fimgi = img.getEdgesFeature(1)
imgi = img.getImage(0)

#dividing the image into two pages and removing the marginal white part from 4 sides.
sh = np.shape(imgi)
limgi = imgi[:,0:sh[1]/2-1]
rimgi = imgi[:,sh[1]/2:sh[1]-1]
plt.imshow(rimgi,cmap='gray')
plt.show()
sh=np.shape(limgi)
up = 120
down=150
left = 400
right = 90
lpimgi = limgi[up:sh[0]-down,left:sh[1]-right]
plt.imshow(lpimgi,cmap='gray')
plt.show()

sh=np.shape(rimgi)
up = 120
down=150
left = 300
right = 250
rpimgi = rimgi[up:sh[0]-down,left:sh[1]-right]
plt.imshow(rpimgi,cmap='gray')
plt.show()

# blpimgi=binarize(lpimgi)
# plt.imshow(blpimgi,cmap='gray')
# plt.show()
# brpimgi=binarize(rpimgi)
# plt.imshow(brpimgi,cmap='gray')
# plt.show()
# cv2.cvtColor(brpimgi,cv2.COLOR_B)

precan=rpimgi
# precan[brpimgi==255]=255
# plt.imshow(precan,cmap='gray')
# plt.show()
  
# Canny
imgcan = cv2.Canny(precan,100,200)
imgcan[imgcan==255]=5
imgcan[imgcan==0]=255
imgcan[imgcan==5]=0
plt.imshow(imgcan,cmap='gray')
plt.show()
sh=np.shape(imgcan)
imgcutcan=imgcan[10:300,20:sh[1]-200]
plt.imshow(imgcutcan,cmap='gray')
plt.show()


a = imgcutcan[0:10,100:150]
plt.imshow(a,cmap='gray')
plt.show()

fimgb = createFeatures(imgcutcan)

#fourier transform to remove noise .. Not Required!!

# f = np.fft.fft2(imgcutcan)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# plt.subplot(121),plt.imshow(imgcutcan, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
# 
# fg = fshift;
# 
# fg[np.abs(fg < 500)]=0;
# 
# p=300
# sh = np.shape(fshift)
# fshift[0:sh[0]/4+p,:] = 0 
# fshift[:,0:sh[1]/4+p] = 0
# fshift[3*sh[0]/4-p:sh[0]-1,:] = 0
# fshift[:,3*sh[1]/4-p:sh[1]-1] = 0
# 
# f_ishift = np.fft.ifftshift(fg)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)
# 
# plt.subplot(131),plt.imshow(rpimgi, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after LPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.imshow(img_back, cmap = 'gray')

#plt.figure()
#plt.scatter(fimgb[:,0],fimgb[:,1])
# plt.show()
# #Clustering ..
# fimgi = fimgb
# pca = PCA(n_components=2).fit(fimgb)
# # Compute DBSCAN .. little slow!!
# #db1 = DBSCAN(eps=2, min_samples=5).fit(fimgb)
# af1 = AffinityPropagation(preference=None).fit(fimgb)
# #core_samples_mask = np.zeros_like(db1.labels_, dtype=bool)
# #core_samples_mask[db1.core_sample_indices_] = True
# labels = db1.labels_
# #print(labels)
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#Calculating Maximum and Minimum of each cluster and storing 1 for max and -1 for min value in corresponding index position of maxminarray ...

maxminarray = np.zeros((len(labels),1)) 

for i in range(1,n_clusters_):
      b = np.where(labels==i)
      #temp = fimgi[b][:,1]
      maxvalue = max(fimgi[b][:,1])
      minvalue = min(fimgi[b][:,1])
      maxminarray[np.where(fimgi[b][:,1] == maxvalue)] = 1
      maxminarray[np.where(fimgi[b][:,1] == minvalue)] = 2
          

fimgi[:,3] = maxminarray[:,0]
    
markAndSave(imgi,fimgi)
visualizeCluster(imgi,fimgi,labels,n_clusters_)
#Classifying the maxima and minima in 4 classes Preparing label data for training.

for i in range(2,n_clusters_-2):
      timg = 
      maxvalcur = findMaxInCluster(i,fimgi,maxminarray,labels)
      minvalcur = findMinInCluster(i,fimgi,maxminarray,labels)
      
      maxvalnext = findMaxInCluster(i+1,fimgi,maxminarray,labels)
      minvalnext = findMinInCluster(i+1,fimgi,maxminarray,labels)
      
      maxvalprev = findMaxInCluster(i-1,fimgi,maxminarray,labels)
      minvalprev = findMinInCluster(i-1,fimgi,maxminarray,labels)
      
       diffmaxnxt = maxvalnext - maxvalcur
       diffmaxprv = maxvalprev - maxvalcur
       diffminnxt = minvalnext - minvalcur
       diffminprv = minvalprev - minvalcur
      # #print(diffmaxnxt)
      # #print(diffmaxprv)
      diffcurr = abs(maxvalcur - minvalcur)
      p1 = p2 = p3 = 0
      if diffcurr >= 30 and diffcurr <= 90:
          p1 = 1 #lower and upper baseline
      if diffcurr < 5:
          p2 = 1 # both are middle line
      else:
          if (diffmaxnxt < -10 and diffmaxnxt > -55) or (diffmaxprv < -10 and diffmaxprv > -55):
              p2 = 1 
          if (diffminnxt > 10 and diffminnxt <=55) or (diffminprv > 10 and diffminprv <=55):
              p3 = 1 
          if p2 == 0 and p3 ==0:
              
          if diffcurr < 30 and diffmaxnxt < 10 and diffmaxprv < 10 and diffminnxt < 10 and diffminprv < 10:
              p2 = 1
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

      print(diffcurr,diffmaxnxt, diffmaxprv , diffminnxt, diffminprv, p1,p2,p3)

t = invertImage(img.getImage(0))
cv2.imwrite( "edge.jpg", t )
t = markImage(img.getImage(0),img.getEdgesFeature(0))
backtorgb = cv2.cvtColor(t,cv2.COLOR_GRAY2RGB )
cv2.imwrite( "edgemarked.jpg",img.getImage(0) )
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
      b = np.where(labels==i)
      maxval = image[np.where(maxminarray[b] == 1)][1]
      if np.shape(maxval) > 0:
          return maxval[0]
      else:
          return -1
    
def findMinInCluster(i,image,maxminarray,labels):
      b = np.where(labels==i)
      minval = image[np.where(maxminarray[b] == 2)][1]
      if np.shape(minval) > 0:
          return minval[0]
      else:
          return -1
      
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

def markAndSave(img, fimg):
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    r = np.where(fimg[:,3] == 1)
    s = r[0][:]
    t = fimg[s,1].astype(int)
    u = fimg[s,0].astype(int)
    for i in range(len(s)):
        cimg[t[i]:t[i]+20,u[i]:u[i]+20,:] = [255,1,0]
        
    r = np.where(fimg[:,3] == 2)
    s = r[0][:]
    t = fimg[s,1].astype(int)
    u = fimg[s,0].astype(int)
    for i in range(len(s)):
        cimg[t[i]:t[i]+20,u[i]:u[i]+20,:] = [0,1,255]
    cv2.imwrite('colormarkedmaxmin.jpg',cimg)
    
  
def visualizeCluster(imgi,fimgi,labels,n_clusters_):
    R = 255
    G = 1
    B = 0
    cimg = cv2.cvtColor(imgi,cv2.COLOR_GRAY2RGB)
    for i in range(1,n_clusters_):
        b = np.where(labels==i)
        cimg[fimgi[b,1].astype(int),fimgi[b,0].astype(int),:] = random_color()
        
        # if R == min(R,G,B):
        #     R = R+10
        # if G == min(R,G,B):
        #     G = G + 10
        # if B == min(R,G,B):
        #     B = B + 10
    cv2.imwrite('colormarkedcluster.jpg',cimg)
   # with open('test.txt','wb') as f:
   #     np.savetxt(f,cimg.astype(int), delimiter=" ", fmt = '%d')
 
def random_color():
    rgbl=[255,0,0]
    np.random.shuffle(rgbl)
    return tuple(rgbl)
    
def createFeatures(img):
    #edgesFeatures = [[]]
    feature = np.zeros((np.count_nonzero(img),2))
    c = np.nonzero(img)
    feature[:,0] = c[1]
    feature[:,1] = c[0]
    return feature

def binarize(img):
    imgb = np.zeros((np.shape(img)))
    imgb[img > 160] = 255
    imgb[img <= 160] = 0
    return imgb
    
