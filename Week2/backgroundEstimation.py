import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio

def gaussianModel(videoPath, plot=False):
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
    backgroundFrames = int(num_frames * 0.25)
    
    # Background model
    backgroundMean = np.zeros((height, width))
    backgroundM2 = np.zeros((height, width))
    n = 0
    
    # Save mean and std video
    if plot:
        frames = []
    
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
        
        # Store frame info
        if plot:
            # Store one frame per 10
            if i % 10 == 0:
                print(i)
                # Compute std
                backgroundStd = np.sqrt(backgroundM2/n)
                
                # Plot std
                fig = plt.figure()
                plt.imshow(backgroundStd, cmap = "Blues", vmin=0, vmax=100)
                plt.colorbar()
                plt.axis('off')
                fig.savefig("test.png", dpi=fig.dpi)
                plt.close()
                plotStd = cv2.imread("test.png")
                plotStd = cv2.cvtColor(plotStd, cv2.COLOR_BGR2RGB)
                
                # Resize and convert to rgb
                frameGray = cv2.resize(frameGray, (500,250))
                frameGray = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2RGB)
                plotMean = backgroundMean.astype(np.uint8)
                plotMean = cv2.resize(plotMean, (500,250))
                plotMean = cv2.cvtColor(plotMean, cv2.COLOR_GRAY2RGB)
                plotStd = cv2.resize(plotStd, (500,250))
                
                # Stack images
                newFrame = np.hstack((frameGray, plotMean, plotStd))
                frames.append(newFrame)
    
    # Compute std
    backgroundStd = np.sqrt(backgroundM2/n)
    
    # Plot
    if plot:
        # Save the video frames as a GIF
        imageio.mimsave('evolution.gif', frames, fps=2)
        print("Final values:")
        # Save last background mean
        plotMean = backgroundMean.astype(np.uint8)
        cv2.imwrite("finalMean.png", plotMean)
        fig = plt.figure()
        plt.imshow(backgroundStd, cmap = "Blues", vmin=0, vmax=100)
        plt.colorbar()
        plt.axis('off')
        fig.savefig("finalStd.png", dpi=fig.dpi)
        plt.close()
    
    
    return backgroundMean, backgroundStd, cap

def objDet(foreground, imageId, n, nn, c_size):
    """
    Function to detect bounding boxes from foreground detection.

    Parameters
    ----------
    foreground : numpy array
        Image with foreground estimation.
    imageId : str
        Image id.
    n:		: size of the erotion kernel
    nn:		: size of dilation kernel
    c_size	: threshold value for contours 

    Returns
    -------
    BB : numpy array
        Array with the detected bboxes.
    imageIds : list
        Image ids.

    """
    # Filter noise
    kernel = np.ones((n,n))
    kernel1 = np.ones((nn,nn))
    # Remove noise
    foregroundFiltered = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    
    # Agroup results
    foregroundFiltered = cv2.morphologyEx(foregroundFiltered, cv2.MORPH_CLOSE, kernel1)
    


    # Get size of image
    h = foreground.shape[0]
    w = foreground.shape[1]

    # Detect obj
    contours, _ = cv2.findContours(foregroundFiltered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    BB = np.zeros((0,4))
    imageIds = []
    # Each contour one obj
    for contour in contours:
        area = cv2.contourArea(contour)         
        if area > c_size: 
            contour = contour[:, 0, :]
            xmin = np.min(contour[:,0])
            ymin = np.min(contour[:,1])
            xmax = np.max(contour[:,0])
            ymax = np.max(contour[:,1])
        
            if (xmax - xmin) < w * 0.4 and (ymax - ymin) < h*0.4:
                # Stack BBoxes
                BB = np.vstack((BB, np.array([xmin,ymin,xmax,ymax])))
                imageIds.append(imageId)
        
    return BB, imageIds, foregroundFiltered

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