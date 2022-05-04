
import os
import sys
import glob
import numpy as np
import nibabel as nib
from itertools import permutations
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
from scipy.ndimage import rotate
from skimage import measure
import cv2
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

def fixed_nobed_clahe(fixed):
        #ㅡㅡㅡㅡㅡfixed lung bed remove startㅡㅡㅡㅡㅡ#
    dataarray = np.array(fixed)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(1,1))

    i = 0
    maskimg=dataarray.copy()
    maskimg[maskimg>-500]=1
    maskimg[maskimg!=1]=0

    fixed_zeros = np.zeros((dataarray.shape))

    for i in range(maskimg.shape[2]):

        cutimg = maskimg[:, :, i]
        mm = np.zeros((600, 600))

        mm[44:556, 44:556] = cutimg[:,:]

        bestidx = 0
        mm = cv2.medianBlur(np.array(mm, dtype='uint8'), 5)
        kernel = np.ones((3, 3))
        mm = cv2.erode(mm, kernel=kernel, iterations=1)
        mm = cv2.dilate(mm, kernel=kernel, iterations=1)

        bestvolu = 0
        lines = measure.find_contours(mm, 0.95)
        area=[]
        contours=[]
        for iddx, contour in enumerate(lines):
            hull = ConvexHull(contour)
            contours.append([contour,hull.volume])

        img=np.zeros( mm.shape)
        contours = sorted(contours, key=lambda s: s[1])
        contours.reverse()
        if len(contours)>=5:
            contours=contours[:5]

        for cont in contours:
            imgcopy = Image.new('L', mm.shape, 0)
            contour = cont[0]
            x = contour[:, 0]
            y = contour[:, 1]
            polygon_tuple = list(zip(x, y))
            ImageDraw.Draw(imgcopy).polygon(polygon_tuple, outline=0, fill=1)
            imgcopy = np.array(imgcopy)

            img=img+imgcopy

        mask = np.array(img)
        mask[mask>0]=1
        kernel=np.ones((5,5))
        mask=cv2.erode(mask,kernel=kernel,iterations=10)
        mask = cv2.dilate(mask, kernel=kernel, iterations=10)
        mask = mask[44: 556,44:556]
        mask[mask>0]=1
        dataarray[:,:,i] = (dataarray[:,:,i]-np.min(dataarray[:,:,i]))/(np.max(dataarray[:,:,i]-np.min(dataarray[:,:,i])))
        dataarray[:,:,i] = dataarray[:,:,i]*255
        dataarray[:, :, i] = dataarray[:,:,i]* mask.T
        #ㅡㅡㅡㅡㅡfixed lung bed remove endㅡㅡㅡㅡㅡ#
         
        #ㅡㅡㅡㅡㅡfixed lung clahe startㅡㅡㅡㅡㅡ#
        for i_2 in range(37):
            if i_2 ==0 :
                fixed_zeros[:,:,i] = clahe.apply(np.array(dataarray[:,:,i],dtype='uint8')) #clahe
            elif i_2>=1:
                fixed_zeros[:,:,i] = clahe.apply(np.array(fixed_zeros[:,:,i],dtype='uint8'))        
  
        fixed_zeros[:,:,i][fixed_zeros[:,:,i]<35]=0
        fixed_zeros[:,:,i]=fixed_zeros[:,:,i]/255
        print(i)
    fixed_clahe =resize(fixed_zeros,(256,256,256))

    #ㅡㅡㅡㅡㅡfixed lung clahe endㅡㅡㅡㅡㅡ#
    return fixed_clahe

def moving_data(moving,fixed_clahe,moving_label) :
    
    #ㅡㅡㅡㅡㅡmoving lung bed remove startㅡㅡㅡㅡㅡ#
    dataarray = np.array(moving) #(512,512,147)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(1,1))

    i = 0
    maskimg=dataarray.copy() #(512,512,147)
    maskimg[maskimg>-500]=1
    maskimg[maskimg!=1]=0

    moving_zeros = np.zeros((dataarray.shape))#(512,512,147)

    for i in range(maskimg.shape[2]):

        cutimg = maskimg[:, :, i]
        mm = np.zeros((600, 600))

        mm[44:556, 44:556] = cutimg[:,:]

        mm = cv2.medianBlur(np.array(mm, dtype='uint8'), 5)
        kernel = np.ones((3, 3))
        mm = cv2.erode(mm, kernel=kernel, iterations=1)
        mm = cv2.dilate(mm, kernel=kernel, iterations=1)
        lines = measure.find_contours(mm, 0.95)
        contours=[]
        for iddx, contour in enumerate(lines):
            hull = ConvexHull(contour)
            contours.append([contour,hull.volume])

        img=np.zeros( mm.shape)
        contours = sorted(contours, key=lambda s: s[1])
        contours.reverse()
        if len(contours)>=5:
            contours=contours[:5]

        for cont in contours:
            imgcopy = Image.new('L', mm.shape, 0)
            contour = cont[0]
            x = contour[:, 0]
            y = contour[:, 1]
            polygon_tuple = list(zip(x, y))
            ImageDraw.Draw(imgcopy).polygon(polygon_tuple, outline=0, fill=1)
            imgcopy = np.array(imgcopy)

            img=img+imgcopy

        mask = np.array(img)
        mask[mask>0]=1
        kernel=np.ones((5,5))
        mask=cv2.erode(mask,kernel=kernel,iterations=10)
        mask = cv2.dilate(mask, kernel=kernel, iterations=10)
        mask = mask[44: 556,44:556]
        mask[mask>0]=1
        # mask = mask[0:512, 0:512]
        dataarray[:,:,i] = (dataarray[:,:,i]-np.min(dataarray[:,:,i]))/(np.max(dataarray[:,:,i]-np.min(dataarray[:,:,i])))
        dataarray[:,:,i] = dataarray[:,:,i]*255
        # dataarray[:, :, i][mask.T != 1] =0
        dataarray[:, :, i] = dataarray[:,:,i]* mask.T
        #ㅡㅡㅡㅡㅡmoving lung bed removed endㅡㅡㅡㅡㅡㅡ#
         
        #ㅡㅡㅡㅡㅡmoving lung clahe startㅡㅡㅡㅡㅡ#
        for i_2 in range(37):
            if i_2 ==0 :
                moving_zeros[:,:,i] = clahe.apply(np.array(dataarray[:,:,i],dtype='uint8')) #clahe 
            elif i_2>=1:
                moving_zeros[:,:,i] = clahe.apply(np.array(moving_zeros[:,:,i],dtype='uint8'))
              
        moving_zeros[:,:,i][moving_zeros[:,:,i]<35]=0
        moving_zeros[:,:,i]=moving_zeros[:,:,i]/255
        print(i)       
    moving_clahe =resize(moving_zeros,(256,256,256))
    #ㅡㅡㅡㅡㅡmoving lung clahe endㅡㅡㅡㅡㅡ#



    #ㅡㅡㅡㅡㅡmoving label color startㅡㅡㅡㅡㅡ#
    moving_label = nib.load(moving_label).get_fdata()
    for i in range(moving_label.shape[2]):
        moving_label_s = moving_label[:,:,i] 
        A = np.zeros((512,512,3)) # color_shape 
        val = np.unique(moving_label_s)
        print(val)
        A= np.array(A,dtype='uint8')


        if 1 in np.unique(moving_label_s):
            A[moving_label_s==1] = np.array((0,200,0))
        if 2 in np.unique(moving_label_s):
            A[moving_label_s==2] = np.array((0,0,200))
        if 3 in np.unique(moving_label_s):
            A[moving_label_s==3] = np.array((250,0,0))
        if 4 in np.unique(moving_label_s):
            A[moving_label_s==4] = np.array((0,250,0))
        if 5 in np.unique(moving_label_s):
            A[moving_label_s==5] = np.array((0,0,250))
        if 6 in np.unique(moving_label_s):
            A[moving_label_s==6] = np.array((200,200,0))
        if 7 in np.unique(moving_label_s):
            A[moving_label_s==7] = np.array((0,200,200))
        if 8 in np.unique(moving_label_s):
            A[moving_label_s==8] = np.array((200,0,200))
        if 9 in np.unique(moving_label_s):
            A[moving_label_s==9] = np.array((150,100,250))
        if 10 in np.unique(moving_label_s):
            A[moving_label_s==10] = np.array((50,50,100))
        if 11 in np.unique(moving_label_s):
            A[moving_label_s==11] = np.array((25,75,255))
        if 12 in np.unique(moving_label_s):
            A[moving_label_s==12] = np.array((100,100,0))    
        if 13 in np.unique(moving_label_s):
            A[moving_label_s==13] = np.array((255,255,255))
        
        if i==0 :
            moving_label_s_acc = A[:,:,np.newaxis,:] # (512,512,1,3)
        elif i>=1 :
            anotehr_s = A[:,:,np.newaxis,:] # (512,512,1,3)
            moving_label_s_acc = np.concatenate((moving_label_s_acc,anotehr_s),axis=2) # (512,512,147,3)

    moving_label_color =resize(moving_label_s_acc,(256,256,256)) # (256,256,256,3)
    moving_label_base = resize(moving_label,(256,256,256)) 
    #ㅡㅡㅡㅡㅡmoving label color endㅡㅡㅡㅡㅡ#

    #ㅡㅡㅡㅡㅡmoving_clahe , label crop(3d) startㅡㅡㅡㅡㅡ#
    for i in range(moving_clahe.shape[2]):
        moving_crop_1 = moving_clahe[:,:,i]
        if np.sum(moving_crop_1) !=0:
            break
    for i_2 in range(moving_clahe.shape[2]-1,0,-1):
        moving_crop_2 = moving_clahe[:,:,i_2]
        if np.sum(moving_crop_2) !=0:
            break
    for i_3 in range(moving_clahe.shape[2]):
        moving_crop_3 = moving_clahe[:,i_3,:]
        if np.sum(moving_crop_3) !=0:
            break  
    for i_4 in range(moving_clahe.shape[2]-1,0,-1):
        moving_crop_4 = moving_clahe[:,i_4,:]
        if np.sum(moving_crop_4) !=0:
            break
    for i_5 in range(moving_clahe.shape[2]):
        moving_crop_5 = moving_clahe[i_5,:,:]
        if np.sum(moving_crop_5) !=0:
            break
    for i_6 in range(moving_clahe.shape[2]-1,0,-1):
        moving_crop_6 = moving_clahe[i_6,:,:]
        if np.sum(moving_crop_6) !=0:
            break
    moving_clahe_crop = moving_clahe[i_5:i_6, i_3:i_4, i:i_2]
    moving_label_crop = moving_label_base[i_5:i_6, i_3:i_4, i:i_2]
    moving_label_color_crop = moving_label_color[i_5:i_6, i_3:i_4, i:i_2]
    #ㅡㅡㅡㅡㅡmoving_clahe crop(3d) endㅡㅡㅡㅡㅡ#

    #ㅡㅡㅡㅡㅡfixed_clahe crop(3d) startㅡㅡㅡㅡㅡ#
    for k in range(fixed_clahe.shape[2]):
        moving_crop_1 = fixed_clahe[:,:,k]
        if np.sum(moving_crop_1) !=0:
            break
    for k_2 in range(fixed_clahe.shape[2]-1,0,-1):
        moving_crop_2 = fixed_clahe[:,:,k_2]
        if np.sum(moving_crop_2) !=0:
            break
    for k_3 in range(fixed_clahe.shape[2]):
        moving_crop_3 = fixed_clahe[:,k_3,:]
        if np.sum(moving_crop_3) !=0:
            break  
    for k_4 in range(fixed_clahe.shape[2]-1,0,-1):
        moving_crop_4 = fixed_clahe[:,k_4,:]
        if np.sum(moving_crop_4) !=0:
            break
    for k_5 in range(fixed_clahe.shape[2]):
        moving_crop_5 = fixed_clahe[k_5,:,:]
        if np.sum(moving_crop_5) !=0:
            break
    for k_6 in range(fixed_clahe.shape[2]-1,0,-1):
        moving_crop_6 = fixed_clahe[k_6,:,:]
        if np.sum(moving_crop_6) !=0:
            break
    fixed_clahe_crop = fixed_clahe[k_5:k_6, k_3:k_4, k:k_2]
    
    #ㅡㅡㅡㅡㅡfixed_clahe crop(3d) endㅡㅡㅡㅡㅡ#    
        

    #ㅡㅡㅡㅡㅡmoving_fixed resize startㅡㅡㅡㅡㅡ#
    moving_crop_reisze =resize(moving_clahe_crop,(fixed_clahe_crop.shape[0],fixed_clahe_crop.shape[1],fixed_clahe_crop.shape[2]))
    moving_label_color_resize =resize(moving_label_color_crop,(fixed_clahe_crop.shape[0],fixed_clahe_crop.shape[1],fixed_clahe_crop.shape[2]))
    moving_label_crop_resize =resize(moving_label_crop,(fixed_clahe_crop.shape[0],fixed_clahe_crop.shape[1],fixed_clahe_crop.shape[2]))

    #ㅡㅡㅡㅡㅡmoving_fixed resize endㅡㅡㅡㅡㅡ#

    #ㅡㅡㅡㅡㅡmoving_fixed padding startㅡㅡㅡㅡㅡ#
    moving_crop_reisze_padding = np.pad(moving_crop_reisze,((k_5,256-k_6),(k_3,256-k_4),(k,256-k_2))) #256,256,256
    moving_label_color_resize_padding = np.pad(moving_label_color_resize,((k_5,256-k_6),(k_3,256-k_4),(k,256-k_2),(0,0)))#256,256,256
    moving_label_crop_resize_padding = np.pad(moving_label_crop_resize,((k_5,256-k_6),(k_3,256-k_4),(k,256-k_2)))#256,256,256
    
    #ㅡㅡㅡㅡㅡmoving_fixed padding endㅡㅡㅡㅡㅡ#

    return moving_crop_reisze_padding, fixed_clahe,moving_label_crop_resize_padding,moving_label_color_resize_padding


def load_file_vol(x,fixed,seg):
    moving = nib.load(x).get_fdata()
    moving,_,seg_base,seg_color = moving_data(moving,fixed,seg)
    moving = moving[np.newaxis,np.newaxis,...]
    seg_base = seg_base[np.newaxis,np.newaxis,...]
    seg_color = seg_color[np.newaxis,np.newaxis,...]
    
    return  moving,seg_base,seg_color


#ㅡㅡㅡㅡㅡ data generator ㅡㅡㅡㅡㅡ#

def scan_to_scan(vol_names, fixed_names,seg_names, batch_size=2, return_segs=False):
    fixed_img = fixed_nobed_clahe(fixed_names)

    while True:
        mvs_list=[]
        fixed_list=[]
        seg_base_list=[]
        seg_color_list=[]
        vol = os.path.join(vol_names, '*')
        seg = os.path.join(seg_names, '*')
        all_vol = glob.glob(vol)
        all_seg = glob.glob(seg)
        indices = np.random.randint(len(all_vol), size=batch_size)
        for i in indices:
            moving ,seg_base_m ,seg_color_m  = load_file_vol(all_vol[i],fixed_img,all_seg[i]) 
            mvs_list.append(moving)
            fixed_img_new=fixed_img[np.newaxis,np.newaxis,...]
            fixed_list.append(fixed_img_new)
            seg_base_list.append(seg_base_m)
            seg_color_list.append(seg_color_m)
    
        moving = np.concatenate(mvs_list,axis=0)
        fixed = np.concatenate(fixed_list,axis=0)
        seg_base = np.concatenate(seg_base_list,axis=0)
        seg_color = np.concatenate(seg_color_list,axis=0)
        # vols_seg = np.concatenate(imgs_seg, axis=0)

        invols  = [moving, fixed,seg_base,seg_color]
        

        yield (invols)

