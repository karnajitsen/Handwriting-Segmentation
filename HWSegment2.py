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
import lasagne
import theano
import theano.tensor as T
theano.config.exception_verbosity='high'
theano.config.traceback.limit=100
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


precan=rpimgi

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

a=imgcutcan[110:130,225:245]
cv2.imwrite('/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/testdata2/t6.png',a)


a = imgcutcan[0:10,100:150]
plt.imshow(a,cmap='gray')
plt.show()



path2 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/testdata2/'

testimg = Images(path2)

timgi = testimg.getImage(0)
sh=timgi.shape
sh2=[1,1,sh[0],sh[1]]
var_in = T.tensor4('var_in')
var_t = T.tensor4('var_t')

network1 = lasagne.layers.InputLayer(shape=sh2,input_var=var_in)

network2 = lasagne.layers.Conv2DLayer(
        network1, num_filters=1, filter_size=(19,19),pad='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Constant(0.0),b=lasagne.init.Constant(0.))

network3 = lasagne.layers.Pool2DLayer(network2, pool_size=(2, 2),mode='average_exc_pad')

network4 = lasagne.layers.Conv2DLayer(network3, num_filters=1,pad='same', filter_size=(5,5),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.Constant(0.0),b=lasagne.init.Constant(0.))

network5 = lasagne.layers.Pool2DLayer(network4, pool_size=(2, 2),mode='average_exc_pad')

network6 = lasagne.layers.DenseLayer(network5,num_units = 1,nonlinearity=lasagne.nonlinearities.softmax,W=lasagne.init.Constant(0.0), b=lasagne.init.Constant(0.))

prediction = lasagne.layers.get_output(network6)
prediction=T.clip(prediction,1e-2, 1.0 - 1e-2)
loss = lasagne.objectives.binary_crossentropy(prediction, var_t).mean()
all_params = lasagne.layers.get_all_params(network6)
updates=lasagne.updates.adagrad(loss,all_params)
    
train = theano.function([var_in, var_t], loss, updates=updates)

theano.config.optimizer_verbose = 1
theano.config.compute_test_value = 'warn'
theano.config.optimizer='fast_compile'

varin= timgi[None,None,:,:,0]

# varin = np.transpose(varin, (0,3,1,2))
varout=[[[(100,)]]]
for i in range(100):
    [l u] = train(varin,varout)
    print "{}, {}".format(network2, network2.output_shape)
    

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
    
