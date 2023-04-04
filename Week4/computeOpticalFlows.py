import numpy as np
#from resultsUniMatch import UniMatchOptFlow
from resultsPyFlow import PyFlowOptFlow
import cv2
import os

def computeOpticalFlows(videoPath, outputFolder):
    """
    This function generates optical flows for every frame.

    Parameters
    ----------
    videoPath : str
        Video path.
    outputFolder : str
        Output folder path.

    Returns
    -------
    None.

    """
    
    # Init
    #predictor = UniMatchOptFlow()
    predictor = PyFlowOptFlow(colType = 0)
    # Load video
    cap = cv2.VideoCapture(videoPath)
    
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
        #fixed_inference_size = [544, 960]
        # Obtain optical flow
        #flow = predictor.inference(previousFrame,frame, grayscale = False, 
        #                           fixed_inference_size= fixed_inference_size)
        flow = predictor.inference(previousFrame,frame)
        
        # Save previous frame
        previousFrame = frame
        
        # Store optical flow
        np.save(outputFolder + imageId + ".npy", flow)

if __name__ == "__main__":
    
    # Path
    videoPath = "../../AICity_data/train/S03/c010/vdo.avi"
    outputFolder = "../opticalFlows_pyflow/"
    
    # Create folder of outputs
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        
    computeOpticalFlows(videoPath, outputFolder)