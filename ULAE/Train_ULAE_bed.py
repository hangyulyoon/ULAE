import os, glob, sys
from argparse import ArgumentParser
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import rotate
from skimage import measure
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

# Network # 
from Loss import *
from ULAE_bgr import ULAE_bgr, SpatialTransformer 

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-3, help="learning rate") # leanring rate
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=300, # epoch
                    help="number of total iterations")
parser.add_argument("--smooth", type=float, #Grad loss 
                    dest="smooth", default=10.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--bs_ch", type=int,
                    dest="bs_ch", default=16,
                    help="number of basic channels")
parser.add_argument("--modelname", type=str,
                    dest="model_name",
                    default='ULAE',
                    help="Name for saving")
parser.add_argument("--gpu", type=str,
                    dest="gpu",
                    default='6',
                    help="gpus")
parser.add_argument("--gamma", type=float, #Ncc loss
                    dest="gamma",
                    default='9.5', #9.5
                    help="hype-param for mutiwindow loss")               
opt = parser.parse_args()

lr = opt.lr
bs_ch = opt.bs_ch
smooth = opt.smooth
model_name = opt.model_name
iteration = opt.iteration
gamma = opt.gamma
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

# data load, processing    
moving_image = nib.load("/data/tu_data/img/img0001.nii.gz").get_fdata()
fixed_image = nib.load("/data/tu_data/img/img0024.nii.gz").get_fdata()
moving_label_base = nib.load("/data/tu_data/label/label0001.nii.gz").get_fdata()
fixed_label_base = nib.load("/data/tu_data/label/label0024.nii.gz").get_fdata()
moving_image[moving_image<-1024] = -1024
moving_image = moving_image-moving_image.min()
fixed_image[fixed_image<-1024] = -1024
fixed_image = fixed_image-fixed_image.min()

#??????????????? make label color start???????????????#
for i in range(moving_label_base.shape[2]):
        moving_label_s = moving_label_base[:,:,i] # slice  
        A = np.zeros((512,512,3)) # A = color shape 
        val = np.unique(moving_label_s)
        print(val)
        A= np.array(A,dtype='uint8') # A value 0~255 

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
            moving_label_s_acc = A[:,:,np.newaxis,:] # first slice 
        elif i>=1 :
            anotehr_s = A[:,:,np.newaxis,:] # another slice 
            moving_label_s_acc = np.concatenate((moving_label_s_acc,anotehr_s),axis=2) # (512,512,147,3)

for i in range(fixed_label_base.shape[2]):
        fixed_label_s = fixed_label_base[:,:,i] # slice  
        A = np.zeros((512,512,3)) # A = color shape 
        val = np.unique(fixed_label_s)
        print(val)
        A= np.array(A,dtype='uint8') # A value 0~255 

        if 1 in np.unique(fixed_label_s):
            A[fixed_label_s==1] = np.array((0,200,0))
        if 2 in np.unique(fixed_label_s):
            A[fixed_label_s==2] = np.array((0,0,200))
        if 3 in np.unique(fixed_label_s):
            A[fixed_label_s==3] = np.array((250,0,0))
        if 4 in np.unique(fixed_label_s):
            A[fixed_label_s==4] = np.array((0,250,0))
        if 5 in np.unique(fixed_label_s):
            A[fixed_label_s==5] = np.array((0,0,250))
        if 6 in np.unique(fixed_label_s):
            A[fixed_label_s==6] = np.array((200,200,0))
        if 7 in np.unique(fixed_label_s):
            A[fixed_label_s==7] = np.array((0,200,200))
        if 8 in np.unique(fixed_label_s):
            A[fixed_label_s==8] = np.array((200,0,200))
        if 9 in np.unique(fixed_label_s):
            A[fixed_label_s==9] = np.array((150,100,250))
        if 10 in np.unique(fixed_label_s):
            A[fixed_label_s==10] = np.array((50,50,100))
        if 11 in np.unique(fixed_label_s):
            A[fixed_label_s==11] = np.array((25,75,255))
        if 12 in np.unique(fixed_label_s):
            A[fixed_label_s==12] = np.array((100,100,0))    
        if 13 in np.unique(fixed_label_s):
            A[fixed_label_s==13] = np.array((255,255,255))
        
        if i==0 :
            fixed_label_acc = A[:,:,np.newaxis,:] # first slice 
        elif i>=1 :
            anotehr_fs = A[:,:,np.newaxis,:] # another slice 
            fixed_label_acc = np.concatenate((fixed_label_acc,anotehr_fs),axis=2) # (512,512,147,3)
#??????????????? make label color ???????????????#

# clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(1,1))

# for i in range(moving_image.shape[2]):
#     for i_2 in range(37):
#         if i_2 ==0 :
#             moving_image[:,:,i] = clahe.apply(np.array(moving_image[:,:,i],dtype='uint8'))
#         elif i_2>=1:
#             moving_image[:,:,i] = clahe.apply(np.array(moving_image[:,:,i],dtype='uint8'))    

# for i_3 in range(fixed_image.shape[2]):
#     for i_4 in range(37):
#         if i_4 ==0 :
#             fixed_image[:,:,i_3] = clahe.apply(np.array(fixed_image[:,:,i_3],dtype='uint8'))
#         elif i_4>=1:
#             fixed_image[:,:,i_3] = clahe.apply(np.array(fixed_image[:,:,i_3],dtype='uint8'))    

moving_resize = resize(moving_image,(256,256,256))
fixed_resize = resize(fixed_image,(256,256,256))



moving_label_color =resize(moving_label_s_acc,(256,256,256)) # (256,256,256,3)
fixed_label_color =resize(fixed_label_acc,(256,256,256)) # (256,256,256,3)
moving_label_base = resize(moving_label_base,(256,256,256))

#??????????????? data save start ???????????????#

# AA  =nib.Nifti1Image(moving_resize,None)
# nib.save(AA,'/data/tu_data/img_processing/moving_resize.nii.gz')
# AA  =nib.Nifti1Image(fixed_resize,None)
# nib.save(AA,'/data/tu_data/img_processing/fixed_resize.nii.gz')
# AA  =nib.Nifti1Image(moving_label_base,None)
# nib.save(AA,'/data/tu_data/img_processing/moving_label_base.nii.gz')
AA  =nib.Nifti1Image(fixed_label_color,None)
nib.save(AA,'/data/tu_data/img_processing/fixed_label_color.nii.gz')
#??????????????? data save end  ???????????????#


moving_resize= moving_resize[np.newaxis,np.newaxis,...] # batch, channel, (shape)

fixed_resize= fixed_resize[np.newaxis,np.newaxis,...]

moving_label_base = moving_label_base[np.newaxis,np.newaxis,...] 
moving_label_base = torch.from_numpy(moving_label_base).cuda().float()

moving_label_color = moving_label_color[np.newaxis,np.newaxis,...] 
moving_label_color = torch.from_numpy(moving_label_color).cuda().float()

fixed_label_base = nib.load("/data/tu_data/label/label0024.nii.gz").get_fdata()
fixed_label_base = resize(fixed_label_base,(256,256,256))
fixed_label_base = fixed_label_base[np.newaxis,np.newaxis,...]
fixed_label_base = torch.from_numpy(fixed_label_base).cuda().float()

imgshape = moving_resize.shape[2:]






def train_ULAE():

    model = ULAE_bgr(2, 3, bs_ch, True, imgshape).cuda() # Network 

    sim_loss_fn = multi_window_loss(win=[11,9,7], gamma=gamma) # Ncc loss 
    smo_loss_fn = smoothloss # Grad loss 
    
    def mse(y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    mse_loss_fn = mse

    transform = SpatialTransformer(size=imgshape).cuda()
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_dir = '/data/tu_data/model/'
    os.makedirs('/data/tu_data/model/',exist_ok=True)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    loss_all = np.zeros((5, iteration + 1))

    step = 0
    
    load_model = False
    if load_model is True:
        model_path = '/home/tukim/Wrokspace/ULAE-net-master/ULAE/model/ULAE165_0200.pth'
        step = 199
        model.load_state_dict(torch.load(model_path))
        loss_load = np.load("/home/tukim/Wrokspace/ULAE-net-master/ULAE/model/lossULAE165_0200.npy")
        loss_all[:, :step] = loss_load[:, :step]

    while step <= iteration:
            print(__file__) # ?????? ?????? 
            X = torch.from_numpy(moving_resize).cuda().float()
            Y = torch.from_numpy(fixed_resize).cuda().float()

            warps, flows, _,input_image_warp,input_image_warp_two = model(X, Y,moving_label_base,moving_label_color)  #return [warp1, warp2, warp3], [flowa, flowb, flowc], [flow1, flow2, flow3]

            sim_loss_1, sim_loss_2, sim_loss_3 = sim_loss_fn(warps[2], Y) 
            smo_loss = smo_loss_fn(flows[-1])

            mse_loss_label = mse_loss_fn(fixed_label_base,input_image_warp) # ????????? label mse loss

            # loss = sim_loss_1 + sim_loss_2 + sim_loss_3 + (smooth * smo_loss) 
            loss = sim_loss_1 + sim_loss_2 + sim_loss_3 + (smooth * smo_loss)  + mse_loss_label * 0.5 
            optimizer.zero_grad() 
            
            loss.backward()
            optimizer.step()

            loss_all[:, step] = np.array(
                [loss.item(), sim_loss_1.item(), sim_loss_2.item(), sim_loss_3.item(), smo_loss.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_muti_window_NCC "{2:4f} {3:4f} {4:4f}" - smo "{5:.4f}"'.format(
                    step, loss.item(), sim_loss_1.item(), sim_loss_2.item(), sim_loss_3.item(), smo_loss.item()))
            sys.stdout.flush()

            if (step % 10 == 0):
                modelname = model_dir + '/' + model_name + str(bs_ch) + str(smooth).replace('.', '_') + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + str(bs_ch) + str(smooth).replace('.', '_') + str(step) + '.npy', loss_all)
                O=nib.Nifti1Image(warps[2][0,0,:,:,:].cpu().detach().numpy(),None)
                nib.save(O,'/data/tu_data/result_bed/'+str(step)+'result.nii.gz')            
                O=nib.Nifti1Image(input_image_warp[0,0,:,:,:].cpu().detach().numpy(),None)
                nib.save(O,'/data/tu_data/result_bed/'+str(step)+'result_new.nii.gz')
                O=nib.Nifti1Image(input_image_warp_two[0,0,:,:,:,:].cpu().detach().numpy(),None)
                nib.save(O,'/data/tu_data/result_bed/'+str(step)+'result_new_two.nii.gz')
                # O=nib.Nifti1Image(flows[-1][0,0,:,:,:].cpu().detach().numpy(),None)
                # nib.save(O,'/data/tu_data/result/'+str(step)+'flow.nii.gz')                
            print(step)
            step += 1

            if step > iteration:
                break
        # np.save(model_dir + '/loss' + model_name + str(bs_ch) + str(smooth).replace('.', '_') +  str(step) + '.npy', loss_all)


if __name__ == '__main__':
    train_ULAE()
