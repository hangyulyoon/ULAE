import os, glob, sys
from smtpd import DebuggingServer
from argparse import ArgumentParser
import numpy as np
import torch
import cv2
# from Transforms import *
from datagen_batch import *
from Loss import *
from ULAE_bgr_batch import ULAE_bgr_batch, SpatialTransformer
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import rotate
from skimage.transform import resize
from skimage import measure
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
import torch.nn 


parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-3, help="learning rate")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=300,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=10.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000,
                    help="frequency of saving models")
parser.add_argument("--bs_ch", type=int,
                    dest="bs_ch", default=16,
                    help="number of basic channels")
parser.add_argument("--modelname", type=str,
                    dest="model_name",
                    default='ULAE',
                    help="Name for saving")
# parser.add_argument("--gpu", type=str,
#                     dest="gpu",
#                     default='1',
#                     help="gpus")
parser.add_argument("--gamma", type=float,
                    dest="gamma",
                    default='9.5', #9.5
                    help="hype-param for mutiwindow loss")               
opt = parser.parse_args()

lr = opt.lr
bs_ch = opt.bs_ch
local_ori = opt.local_ori
n_checkpoint = opt.checkpoint
smooth = opt.smooth
model_name = opt.model_name
iteration = opt.iteration
gamma = opt.gamma
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_vol_names = nib.load('/home/tukim/Wrokspace/voxelmorph-dev_4/scripts/torch/adomen/3d_py/a_here/no_bed/img_clage/img_clahe_256.nii.gz').get_fdata()
moving_image = '/data/tu_data/img/'
moving_image_label = '/data/tu_data/label/'
fixed_image  = nib.load("/data/tu_data/img/img0024.nii.gz").get_fdata()

# dg = scan_to_scan(moving_image,fixed_image,moving_image_label,  batch_size=2)
gd = scan_to_scan(moving_image,fixed_image,moving_image_label,  batch_size=2)
imgshape = (256,256,256)

fixed_image = resize(fixed_image, (256,256,256))
fixed_image = fixed_image[np.newaxis,np.newaxis,...]
fixed_image = torch.from_numpy(fixed_image).cuda().float()



def train_ULAE():

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ULAE_bgr_batch(2, 3, bs_ch, True, imgshape)
    # model = model.to('cuda:0')
    model = torch.nn.DataParallel(model).cuda()
  
    sim_loss_fn = multi_window_loss(win=[11,9,7], gamma=gamma) # 11, 9 ,7
    smo_loss_fn = smoothloss
    
    def mse(y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)



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
    
    for Data in gd:

        X = Data #moving, fixed_img,seg_base
        print(__file__)
        moving = torch.from_numpy(X[0]).cuda().float()
        fixed = torch.from_numpy(X[1]).cuda().float()
        moving_seg = torch.from_numpy(X[2]).cuda().float()
        moving_seg_color = torch.from_numpy(X[3]).cuda().float()

        for step in range(iteration):
        

            warps, flows, _,input_image_warp,input_image_warp_two = model(moving,fixed,moving_seg,moving_seg_color) #return [warp1, warp2, warp3], [flowa, flowb, flowc], [flow1, flow2, flow3]

            sim_loss_1, sim_loss_2, sim_loss_3 = sim_loss_fn(warps[2], fixed)
            smo_loss = smo_loss_fn(flows[-1])
            # mse_loss_label = mse_loss_fn(fixed_image,input_image_warp) 

            loss = sim_loss_1 + sim_loss_2 + sim_loss_3 + (smooth * smo_loss)  
            # loss = sim_loss_1 + sim_loss_2 + sim_loss_3 + (smooth * smo_loss) + mse_loss_label * 0.5 
            # loss =  (smooth * smo_loss)  + mse_loss_label * 0.5
            # loss = (smooth * smo_loss)  + (20 * mse_loss) 
         
            optimizer.zero_grad() 
            # loss=loss.cpu()
            loss.backward()
            optimizer.step()

            loss_all[:, step] = np.array(
                [loss.item(), sim_loss_1.item(), sim_loss_2.item(), sim_loss_3.item(), smo_loss.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_muti_window_NCC "{2:4f} {3:4f} {4:4f}" - smo "{5:.4f}"'.format(
                    step, loss.item(), sim_loss_1.item(), sim_loss_2.item(), sim_loss_3.item(), smo_loss.item()))
            sys.stdout.flush()
            torch.cuda.empty_cache() 
            
            if (step % 10 == 0):
                modelname = model_dir + '/' + model_name + str(bs_ch) + str(smooth).replace('.', '_') + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + str(bs_ch) + str(smooth).replace('.', '_') + str(step) + '.npy', loss_all)
                O=nib.Nifti1Image(warps[2][0,0,:,:,:].cpu().detach().numpy(),None)
                nib.save(O,'/data/tu_data/result_batch/'+str(step)+'result.nii.gz')            
                O=nib.Nifti1Image(input_image_warp[0,0,:,:,:].cpu().detach().numpy(),None)
                nib.save(O,'/data/tu_data/result_batch/'+str(step)+'result_new.nii.gz')
                O=nib.Nifti1Image(input_image_warp_two[0,0,:,:,:,:].cpu().detach().numpy(),None)
                nib.save(O,'/data/tu_data/result_batch/'+str(step)+'result_new_two.nii.gz')
                O=nib.Nifti1Image(flows[-1][0,0,:,:,:].cpu().detach().numpy(),None)
                nib.save(O,'/data/tu_data/result_batch/'+str(step)+'flow.nii.gz')                
                # S = flows[-1].permute(0,2,3,4,1)
                # O=nib.Nifti1Image(S[0,:,:,:,:].cpu().detach().numpy(),None)
                # nib.save(O,'/home/tukim/Wrokspace/ULAE-net-master/ULAE/result_clahe/'+str(step)+'flow.nii.gz')     
                # O=nib.Nifti1Image(input_image_warp_two_2[0,0,:,:,:,:],None)
                # nib.save(O,'/home/tukim/Wrokspace/ULAE-net-master/ULAE/'+str(step)+'result_new_two_2.nii.gz')                

            print(step)
            step += 1


        np.save(model_dir + '/loss' + model_name + str(bs_ch) + str(smooth).replace('.', '_') +  str(step) + '.npy', loss_all)


if __name__ == '__main__':
    train_ULAE()
