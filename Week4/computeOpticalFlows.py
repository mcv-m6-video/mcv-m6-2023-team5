import numpy as np
from resultsUniMatch import UniMatchOptFlow
# from resultsRAFT import RaftFlow
import cv2
import os
import torch
import torch.nn.functional as F
import flow_vis
import imageio

def computeOpticalFlows(videoPath, outputFolder, visualize = False):
    """
    This function generates optical flows for every frame.

    Parameters
    ----------
    videoPath : str
        Video path.
    outputFolder : str
        Output folder path.
    visualize: bool
        True to create a GIF with optical flows.

    Returns
    -------
    None.

    """
    
    # Init
    predictor = UniMatchOptFlow()
    # predictor = RaftFlow()
    # Load video
    cap = cv2.VideoCapture(videoPath)
    
    if visualize:
        ofPlots = []
        orPlots = []
    
    # Loop video    
    while True:
        
        # Read frame
        ret, frame = cap.read()
        
        # Check if frame was successfully read
        if not ret:
            break
        
        # Get frame
        imageId = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        print(imageId)
        
        if imageId == "1":
            previousFrame = frame
            continue
        
        # Resize due to memory limitations
        fixed_inference_size = [544, 960]
        # original_size = frame.shape[:-1]
        # previousFrame_res = torch.from_numpy(previousFrame).permute(2, 0, 1).unsqueeze(0).float()
        # previousFrame_res = F.interpolate(previousFrame_res, size=fixed_inference_size, mode='bilinear',
        #                         align_corners=True)
        # previousFrame_res = previousFrame_res.squeeze(0).permute(1, 2, 0).numpy()
        # frame_res = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        # frame_res = F.interpolate(frame_res, size=fixed_inference_size, mode='bilinear',
        #                         align_corners=True)
        # frame_res = frame_res.squeeze(0).permute(1, 2, 0).numpy()
        
        # Obtain optical flow
        flow = predictor.inference(previousFrame,frame, grayscale = False, 
                                  fixed_inference_size= fixed_inference_size)
        # flow = predictor.inference(previousFrame_res,frame_res, grayscale = False)
        # flow_res = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
        # flow_res = F.interpolate(flow_res, size=original_size, mode='bilinear',
        #                         align_corners=True)
        # flow_res[:, 0] = flow_res[:, 0] * original_size[-1] / fixed_inference_size[-1]
        # flow_res[:, 1] = flow_res[:, 1] * original_size[-2] / fixed_inference_size[-2]
        
        # flow = flow_res[0].permute(1, 2, 0).cpu().numpy()
        
        if visualize:
            if int(imageId) % 10 == 0:
                # Flow
                flowVis = flow_vis.flow_to_color(flow, convert_to_bgr=False)
                flowVis = cv2.resize(flowVis,(250, 120))
                # Save frame
                ofPlots.append(flowVis)
                # Orig
                orVis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orVis = cv2.resize(orVis,(250, 120))
                orPlots.append(orVis)
        # Save previous frame
        previousFrame = frame
        
        # Store optical flow
        np.save(outputFolder + imageId + ".npy", flow)
        
    if visualize:
        imageio.mimsave('../predictedOpticalFlow.gif', ofPlots, fps=5)
        imageio.mimsave('../orig.gif', orPlots, fps=5)

if __name__ == "__main__":
    
    # Path
    videoPath = "../../AICity_data/train/S03/c010/vdo.avi"
    outputFolder = "../opticalFlows_new/"
    
    # Create folder of outputs
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        
    computeOpticalFlows(videoPath, outputFolder, visualize = True)