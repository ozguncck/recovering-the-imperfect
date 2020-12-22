import numpy as np
import h5py
import tifffile
import glob
import os
import argparse
import cv2
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True,
                    metavar="/path/to/dataset/",
                    help='Root directory of the dataset')
args = parser.parse_args()
path = args.path 
base_path='/'.join(path.split('/')[:-1])+'/'
file_name=path.split('/')[-1][:-4]

file=np.array(tifffile.imread(path))
os.mkdir(base_path+file_name+'/')
os.rename(path, base_path+file_name+'/'+file_name+'.tif')
tifffile.imsave(base_path+file_name+'/ch0.tif',file[:,0,:,:])
tifffile.imsave(base_path+file_name+'/ch1.tif',file[:,1,:,:])

for j in range(file.shape[0]):
    f=h5py.File(base_path+file_name+'/im'+str(j)+'.h5','w')
    f['data']=file[j,0,:,:].reshape((file[j,0,:,:].shape[0],file[j,0,:,:].shape[1],1))/(2**16-1)
    f.close()
    tifffile.imsave(base_path+file_name+'/'+str(j)+'.tif',file[j,1,:,:])
    
     
    
