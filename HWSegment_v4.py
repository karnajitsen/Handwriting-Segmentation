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

path2 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/testdata2/uppercontour/'

uppercontourimg = Images(path2)
#testimg.applyCanny()
#testimg.invertImages()
#testimg.normalize()
#plt.imshow(testimg.getImage(2),cmap='gray')
#plt.show()
#testimg.createImageFeatures()
uppercontourno = uppercontourimg.cnt

path2 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/testdata2/lowercontour/'

imglowercont = Images(path2)
#testimg.applyCanny()
#imglowercont.invertImages()
#testimg.normalize()
#plt.imshow(imglowercont.getImage(2),cmap='gray')
#plt.show()
#testimg.createImageFeatures()
lowercontourno = imglowercont.cnt


batchsize = lowercontourno + uppercontourno
#batchsize = uppercontourno
#batchsize = lowercontourno

#CNN Configuration

var_in = T.tensor4('var_in')
var_t = T.imatrix('var_t')
#batchsize = T.scalar('batchsize')
network = lasagne.layers.InputLayer(shape=(batchsize,1,150,150),input_var=var_in)

network1 = lasagne.layers.Conv2DLayer(
        network, num_filters=30, filter_size=(10,10),pad='full',
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.))

network = lasagne.layers.Pool2DLayer(network1, pool_size=(2, 2),mode='max')

network = lasagne.layers.Conv2DLayer(network, num_filters=40,pad='full', filter_size=(5,5),nonlinearity=lasagne.nonlinearities.sigmoid,W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.))

network = lasagne.layers.Pool2DLayer(network, pool_size=(2, 2),mode='max')

#network = lasagne.layers.Conv2DLayer(network, num_filters=batchsize*3,pad='same', filter_size=(3,3),nonlinearity=lasagne.nonlinearities.sigmoid,W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.))

#network = lasagne.layers.Pool2DLayer(network, pool_size=(2, 2),mode='max')

network = lasagne.layers.DenseLayer(network,num_units = 2,nonlinearity=lasagne.nonlinearities.softmax,W=lasagne.init.GlorotNormal(), b=lasagne.init.Constant(0.))
#weights = network2.W.get_value()

prediction = lasagne.layers.get_output(network)

loss = lasagne.objectives.squared_error(prediction, var_t).mean()
all_params = lasagne.layers.get_all_params(network, trainable=True)
updates=lasagne.updates.sgd(loss,all_params,0.1)
    
train = theano.function([var_in, var_t], loss, updates=updates )


#theano.config.optimizer_verbose = 1
#theano.config.compute_test_value = 'warn'
#theano.config.optimizer='fast_compile'


#Training Data preparation


varout=[]
varin=()
#output = np.eye(imglowercont.cnt,batchsize)
#output=output.astype(int)
size=150


for i in range(0,uppercontourno):
    timgi = uppercontourimg.getImage(i)/255.0
    sh = np.shape(timgi[:,:,0])
    dest = np.ones((size,size,1))
    dest[0:sh[0],0:sh[1],:]=timgi[:,:,0,None]
    lst = list(varin)
    lst.append(dest)
    varin=tuple(lst)
    varout.append(tuple([1.0,0.0]))
    
for i in range(0,lowercontourno):
    timgi = imglowercont.getImage(i)/255.0
    sh = np.shape(timgi[:,:,0])
    dest = np.ones((size,size,1))
    dest[0:sh[0],0:sh[1],:]=timgi[:,:,0,None]
    lst = list(varin)
    lst.append(dest)
    varin=tuple(lst)
    varout.append(tuple([0.0,1.0]))
    
varin=np.transpose(varin,(0,3,1,2))


#training

maxepoch=10
for epoch in range(maxepoch):
    loss = 0
    loss += train(varin, varout)
    print("Epoch %d: Loss %g" % (epoch + 1, loss))

#test data preparation

path1 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/data/'
img = Images(path1)
imgcan = img.getImage(0)
sh=np.shape(imgcan)
imgcan[imgcan>180]=255
sh=np.shape(imgcan)


a=imgcan[sh[0]/2+250:sh[0]/2 + 400,sh[1]/2+500:sh[1]/2 + 650]
plt.imshow(a,cmap='gray')
plt.show()

imgcutcan=imgcan[sh[0]/2+250:sh[0]/2 + 400,sh[1]/2+500:sh[1]/2 + 650]/255.0
#imgsh=imgcan[sh[0]/2+250:sh[0]/2 + 400,sh[1]/2+500:sh[1]/2 + 650]


#testing
varin = ()
varout=[]
for i in range(batchsize):
    timgi = imgcutcan[None,95:120,:,0]
    sh = np.shape(timgi[:,:,:])
    dest = np.ones((1,size,size))
    dest[:,0:sh[1],0:sh[2]]=timgi[None,:,:]
    lst = list(varin)
    lst.append(dest)
    varin=tuple(lst)

#Testing
test_prediction = lasagne.layers.get_output(network, deterministic=False)
#predict_fn = theano.function([var_in], T.argmax(test_prediction, axis=1))
predict_fn = theano.function([var_in], test_prediction)
print("Predicted class for first test input: %r" % predict_fn(varin))


