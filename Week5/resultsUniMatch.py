import os

mydir = os.getcwd() # would be the MAIN folder
mydir_tmp = mydir + "/unimatch/" # add the testA folder name
mydir_new = os.chdir(mydir_tmp) # change the current working directory
mydir = os.getcwd() # set the main directory again, now it calls testA

from unimatch.unimatch import UniMatch
import torch
import numpy as np
import torch.nn.functional as F

class UniMatchOptFlow():
    def __init__(self):
        self.device = "cuda"
        
        self.feature_channels = 128
        self.num_scales = 2
        self.upsample_factor = 4
        self.num_head = 1
        self.ffn_dim_expansion = 4
        self.num_transformer_layers = 6
        self.reg_refine = True 
        self.resume = "../gmflow-scale2-regrefine6-kitti15-25b554d7.pth"
        self.padding_factor = 32
        self.attn_splits_list = [2, 8]
        self.corr_radius_list = [-1, 4]
        self.prop_radius_list = [-1, 1]
        self.num_reg_refine = 6
        self.task = "flow"
        self.pred_bwd_flow = False
        self.pred_bidir_flow = False
        self.attn_type='swin'
        
        
        # Init model
        self.model = UniMatch(feature_channels=self.feature_channels,
                             num_scales=self.num_scales,
                             upsample_factor=self.upsample_factor,
                             num_head=self.num_head,
                             ffn_dim_expansion=self.ffn_dim_expansion,
                             num_transformer_layers=self.num_transformer_layers,
                             reg_refine=self.reg_refine,
                             task=self.task).to(self.device)
        
        # Load weights
        checkpoint = torch.load(self.resume, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        
        
    
    def inference(self, im1, im2, grayscale = True, fixed_inference_size = None):
        
        # Eval
        self.model.eval()
        
        # Inference
        with torch.no_grad():
            transpose_img = False
            
            # Preprocess
            if grayscale:
                im1 = np.tile(im1, (1, 1, 3))
                im2 = np.tile(im2, (1, 1, 3))
            
            image1 = torch.from_numpy(im1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            image2 = torch.from_numpy(im2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            
            # the model is trained with size: width > height
            if image1.size(-2) > image1.size(-1):
                image1 = torch.transpose(image1, -2, -1)
                image2 = torch.transpose(image2, -2, -1)
                transpose_img = True
            
            nearest_size = [int(np.ceil(image1.size(-2) / self.padding_factor)) * self.padding_factor,
                                int(np.ceil(image1.size(-1) / self.padding_factor)) * self.padding_factor]
        
            # resize to nearest size or specified size
            inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size
        
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]
        
            # resize before inference
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                       align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                       align_corners=True)
        
            if self.pred_bwd_flow:
                image1, image2 = image2, image1
        
            results_dict = self.model(image1, image2,
                                 attn_type=self.attn_type,
                                 attn_splits_list=self.attn_splits_list,
                                 corr_radius_list=self.corr_radius_list,
                                 prop_radius_list=self.prop_radius_list,
                                 num_reg_refine=self.num_reg_refine,
                                 task=self.task,
                                 pred_bidir_flow=self.pred_bidir_flow,
                                 )
        
            flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
        
            # resize back
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
        
            if transpose_img:
                flow_pr = torch.transpose(flow_pr, -2, -1)
        
            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        
        return flow
    
    
    
    
