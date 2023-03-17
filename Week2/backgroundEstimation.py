import cv2
import numpy as np

def gaussianModel(videoPath):
    """
    This function model the background by taking first 25% of images, by calculating mean and
    std.

    Parameters
    ----------
    videoPath : str
        Video path.

    Returns
    -------
    backgroundMean : numpy array
        Background modeled mean.
    backgroundStd : numpy array
        Background modeled std.
    cap : video instance

    """
    # Read video
    cap = cv2.VideoCapture(videoPath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get 25%
    backgroundFrames = int(num_frames * 0.1)
    
    # Background model
    backgroundMean = np.zeros((height, width))
    backgroundM2 = np.zeros((height, width))
    n = 0

    
    # Generate background with 25% first images using the Welford's method
    for i in range(backgroundFrames):
        
        # Read frame
        ret, frame = cap.read()
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        # Check if frame was successfully read
        if not ret:
            break
        
        # Update
        n += 1
        delta = frameGray - backgroundMean
        backgroundMean = backgroundMean + delta/n
        backgroundM2 = backgroundM2 + delta*(frameGray - backgroundMean)

    
    # Compute std
    backgroundStd = np.sqrt(backgroundM2/n)
    
    return backgroundMean, backgroundStd, cap


def estimateForeground(image, backgroundMean, backgroundStd, alpha):
    """
    This function estimates if each pixel of the image is foreground or not by using the 
    alpha, background mean and std values.

    Parameters
    ----------
    image : numpy array
        Image to estimate.
    backgroundMean : numpy array
        Modeled background mean.
    backgroundStd : numpy array
        Modeled background std.
    alpha : float
        Segmentation parameter.

    Returns
    -------
    foreground : numpy array
        Array with bool values, true if pixel is considered foreground.

    """
    
    # Compute foreground
    foreground = np.abs(image - backgroundMean) >= alpha*(backgroundStd + 2)
    
    return foreground