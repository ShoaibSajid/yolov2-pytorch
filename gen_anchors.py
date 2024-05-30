'''
Created on Feb 20, 2017

@author: jumabek
'''
from os import listdir
from os.path import isfile, join
import argparse
#import cv2
import numpy as np
import sys
import os
import shutil
import random 
import math
from pathlib import Path

# width_in_cfg_file = 416.
# height_in_cfg_file = 416.

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

def write_anchors_to_file(centroids, X, anchor_file, args):
    f = open(anchor_file,'w')
    
    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0]*=args.img_width/32.
        anchors[i][1]*=args.img_height/32.
         

    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])
        
    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    #there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))
    
    f.write('%f\n'%(avg_IOU(X,centroids)))
    print()

def kmeans(X, centroids, eps, anchor_file, args):
    
    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        
        if (assignments == prev_assignments).all() :
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids, X, anchor_file, args)
            return

        #calculate new centroids
        #centroid_sums=np.zeros((k,dim),np.float)
        # **** np.float is deprecated in NumPy 1.20 and later ****
        centroid_sums=np.zeros((k,dim), float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', 
                        default = '\\path\\to\\voc\\filelist\\train.txt', type = str,
                        help='path to filelist\n' )
    
    parser.add_argument('-output_dir', 
                        default = 'generated_anchors/anchors', type = str, 
                        help='Output anchor directory\n' )  
    
    parser.add_argument('-num_clusters', 
                        default = 0, type = int, 
                        help='number of clusters\n' )  
    
    parser.add_argument('-img_width', 
                        default = 416, type = int, 
                        help='Image width to which the image is resized\n' )  
    
    parser.add_argument('-img_height', 
                        default = 416, type = int, 
                        help='Image height to which the image is resized\n' )  

    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    except:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

    f = open(args.filelist)
  
    lines = [line.rstrip('\n') for line in f.readlines()]
    
    annotation_dims = []
    labels_used = 0

    size = np.zeros((1,1,3))
    for line in lines:
                    
        #line = line.replace('images','labels')
        #line = line.replace('img1','labels')
        line = line.replace('JPEGImages','labels')        
        

        line = line.replace('.jpg','.txt')
        line = line.replace('.png','.txt')
        print(line)
        try:
            f2 = open(line)
        except:continue
        labels_used+=1
        for line in f2.readlines():
            line = line.rstrip('\n')
            # w,h = line.split(' ')[3:]   
            
            #**** changing w,h = line.split(' ')[3:]  --> line.split(' ')[3:5] ****
            
            w,h = line.split(' ')[3:5]           
            #print(w,h)
            annotation_dims.append(tuple(map(float,(w,h))))
    annotation_dims = np.array(annotation_dims)
  
    eps = 0.005
    
    if args.num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = join( args.output_dir,'anchors%d.txt'%(num_clusters))

            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims,centroids,eps,anchor_file,args)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join( args.output_dir,'anchors%d.txt'%(args.num_clusters))
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims,centroids,eps,anchor_file,args)
        print('centroids.shape', centroids.shape)
    print(f"\nTotal no. of Label files used: {labels_used}")

if __name__=="__main__":
    main(sys.argv)
