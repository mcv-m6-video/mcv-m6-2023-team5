import os

mydir = os.getcwd() # would be the MAIN folder
mydir_tmp = mydir + "/RAFT/core/" # add the testA folder name
mydir_new = os.chdir(mydir_tmp) # change the current working directory
mydir = os.getcwd() # set the main directory again, now it calls testA

from raft import RAFT
import numpy as np
import torch
import argparse
from utils.utils import InputPadder, forward_interpolate


class RaftFlow:
    def __init__(self):
        
        # Set initial arguments
        model = "../../raft-kitti.pth"
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint", default = model)
        parser.add_argument('--dataset', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()
        
        # Init model
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))
        self.model.cuda()
        
    def inference(self, im1, im2, grayscale = True):
        
        # Eval mode
        self.model.eval()
        
        with torch.no_grad():
            
            # Init images
            if grayscale:
                im1 = np.tile(im1, (1, 1, 3))
                im2 = np.tile(im2, (1, 1, 3))
            img1 = torch.from_numpy(im1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(im2).permute(2, 0, 1).float()
            image1 = img1[None].cuda()
            image2 = img2[None].cuda()
            
            # Images pad
            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)
            
            # Inference
            flow_low, flow_pr = self.model(image1, image2, iters=24, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()
        
            flow = flow.permute(1, 2, 0).numpy()
        
        
        return flow
