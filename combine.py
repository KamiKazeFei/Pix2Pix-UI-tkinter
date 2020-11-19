import os
import math
import random
import numpy as np
import cv2
from PIL import Image
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

input_path = ""        #set your own input images folder 
target_path  = ""      #set your own target images folder 
save_dir = ""          #set your own save images folder

input_img = make_dataset(input_path)    #create a list of images
target_img  = make_dataset(target_path) #same

for inp, tar in zip(input_img, target_img):

    filename = inp.split("\\")[-1].split(".")[0] #split the filename from path
    print(filename)
    inp_img  = cv2.imread(inp) #input image
    tar_img = cv2.imread(tar)  #target image
    h, w, _ = inp_img.shape
    
    #make sure two images has same size
    tar_img = cv2.resize(tar_img,(w,h),interpolation=cv2.INTER_AREA)

    #combine two images as one
    tmp = np.hstack((tar_img,inp_img))

    #save the combine image to save images folder
    cv2.imwrite(os.path.join(save_dir + clr[31:33] + ".png"),tmp)
    print(filename + " Save...") #comfirm the process
