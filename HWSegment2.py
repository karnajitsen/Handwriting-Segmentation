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

#Data preparation

path2 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/testdata2/'

testimg = Images(path2)

plt.imshow(testimg.getImage(1),cmap='gray')
plt.show()

batchsize = testimg.cnt
varout=[]
varin=()
output = np.eye(batchsize,batchsize)
output=output.astype(int)
size=150
for i in range(batchsize):
    timgi = testimg.getImage(i)/255
    sh = np.shape(timgi[:,:,0])
    dest = np.zeros((size,size,1))
    dest[0:sh[0],0:sh[1],:]=timgi[:,:,0,None]
    lst = list(varin)
    lst.append(dest)
    varin=tuple(lst)
    varout.append(tuple(output[i]))
    
varin=np.transpose(varin,(0,3,1,2))



#CNN Configuration

var_in = T.tensor4('var_in')
var_t = T.imatrix('var_t')

network = lasagne.layers.InputLayer(shape=(batchsize,1,size,size),input_var=var_in)

network = lasagne.layers.Conv2DLayer(
        network, num_filters=batchsize, filter_size=(19,19),pad='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.))

network = lasagne.layers.Pool2DLayer(network, pool_size=(2, 2),mode='average_exc_pad')

network = lasagne.layers.Conv2DLayer(network, num_filters=batchsize*2,pad='same', filter_size=(5,5),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(),b=lasagne.init.Constant(0.))

network = lasagne.layers.Pool2DLayer(network, pool_size=(2, 2),mode='average_exc_pad')

network = lasagne.layers.DenseLayer(network,num_units = batchsize,nonlinearity=lasagne.nonlinearities.softmax,W=lasagne.init.GlorotNormal(), b=lasagne.init.Constant(0.))

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.binary_crossentropy(prediction, var_t).mean()
all_params = lasagne.layers.get_all_params(network, trainable=True)
updates=lasagne.updates.sgd(loss,all_params,0.1)
    
train = theano.function([var_in, var_t], loss, updates=updates)


#theano.config.optimizer_verbose = 1
#theano.config.compute_test_value = 'warn'
#theano.config.optimizer='fast_compile'


#Training
for epoch in range(40):
    loss = 0
    loss += train(varin, varout)
    print("Epoch %d: Loss %g" % (epoch + 1, loss))


path1 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/data/'

img = Images(path1)
img.applyCanny()
imgcan = img.getEdgeImage(0)
sh=np.shape(imgcan)
imgcan[imgcan==255]=5
imgcan[imgcan==0]=255
imgcan[imgcan==5]=0
sh=np.shape(imgcan)
imgcutcan=imgcan[None,sh[0]/2+250:sh[0]/2 + 400,sh[1]/2+500:sh[1]/2 + 650]


#a=imgcutcan[125:150,70:100]
#cv2.imwrite('/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/testdata2/t8.png',a)

plt.imshow(imgcutcan,cmap='gray')
plt.show()

plt.imshow(imgcan,cmap='gray')
plt.show()

#varin=imgcutcan[None,]
varin = ()
varout=[]
for i in range(batchsize):
    timgi = imgcutcan/255
    lst = list(varin)
    lst.append(timgi)
    varin=tuple(lst)
    varout.append(tuple(output[i]))
   
#Testing
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([var_in], T.argmax(test_prediction, axis=1))
print("Predicted class for first test input: %r" % predict_fn(varin))


path1 = '/home/karna/Karna_Work/Handwriting_Project/Handwriting-Segmentation/data/'

img = Images(path1)


