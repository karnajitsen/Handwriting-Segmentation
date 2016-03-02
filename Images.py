import numpy as np
import os
import cv2

class Images:
    def __init__(self,path):
        self.path = path
        listing = os.listdir(path)
        self.img = []
        self.cnt = 0
        for file in listing:
            self.cnt = self.cnt + 1
            fl = cv2.imread(path + file,0)
            self.img.append(fl)

        
    def createFeatures(self):
        self.imgFeatures = []
        print(self.cnt)
        for n in range(0,self.cnt-1):
            feature = np.zeros((np.count_nonzero(self.img[n]),2))
            k = 0
            timg = self.img[n]
            print(n)
            print(np.shape(timg))
            for i in range(0,np.shape(timg)[0]):
                for j in range(0,np.shape(timg)[1]):
                        if timg[i][j] <> 0:
                            feature[k] = [i,j]
                            k = k + 1
        self.imgFeatures.append(feature)
        self.edgesFeatures = []
        for n in range(0,self.cnt-1):
            feature = np.zeros((np.count_nonzero(self.edges[n]),2))
            k = 0
            tedge = self.edges[n]
            for i in range(0,np.shape(tedge)[0]):
                for j in range(0,np.shape(tedge)[1]):
                        if tedge[i][j] <> 0:
                            feature[k] = [i,j]
                            k = k + 1
        self.edgesFeatures.append(feature)
        
     
    def applyCanny(self):
        self.edges = []
        for i in range(0,self.cnt-1):
           self.edges.append(cv2.Canny(self.img[i],100,200))
    
    
    def getImage(self,i):
        return self.img[i]
       
    def getEdgeImage(self,i):
        return self.edges[i]
    
    def getImgFeature(self,i):
        return self.imgFeatures[i]
        
    def getEdgesFeature(self,i):
        return self.edgesFeatures[i]
      
    
    def getCount(self):
        return self.cnt
        
