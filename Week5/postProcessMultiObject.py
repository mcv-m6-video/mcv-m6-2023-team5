import numpy as np
import os
import cv2

def allowedDet(roiImage, x, y, w, h):
    
    if np.sum(roiImage[y:y+w, x:x+w] == False) > 0:
        return False
    else:
        return True

def postProcessTracks(roiImagePath, trackFile, outputFile):
    """
    This function removes the tracks from static objects and the tracklets where the object
    is in a not allowed position (no ROI).

    Parameters
    ----------
    roiImagePath: str
        ROI image path.
    trackFile : str
        Tracking results file with static objects.
    outputFile : str
        Path to store the output with tracks without the static and out of ROI objects.

    Returns
    -------
    None.

    """
    
    tracks = {}
    
    # Load detections
    f = open(trackFile, "r")
    lines = f.readlines()
    f.close()
    
    # Load ROI image
    roiImage = cv2.imread(roiImagePath)[:, :, 0] == 255
    
    # Load tracks
    i = 0
    while i < len(lines):
        line = lines[i]
        line_split = line.split(",")
        l_id = int(line_split[1])
        
        area = float(line_split[4])*float(line_split[5])
        if allowedDet(roiImage, int(float(line_split[2])), 
                      int(float(line_split[3])), 
                      int(float(line_split[4])), 
                      int(float(line_split[5]))) and area > 100:
            
            centerX = float(line_split[2]) + float(line_split[4])/2
            centerY = float(line_split[3]) + float(line_split[5])/2
            
            center = [centerX, centerY]
            
            if l_id in tracks.keys():
                tracks[l_id].append(center)
            else:
                tracks[l_id] = [center]
            
            i += 1
        
        else:
            
            del lines[i]
    
    # Detect static objects
    staticTracks = []
    for trackId in tracks.keys():
        
        centers = np.array(tracks[trackId])
        # maxX, minX = centers[:,0].max(), centers[:,0].min()
        # maxY, minY = centers[:,1].max(), centers[:,1].min()
        Xstd = centers[:,0].std()
        Ystd = centers[:,1].std()
        # if (maxX - minX) < 150 and (maxY - minY) < 150:
        if (Xstd + Ystd)/2 < 100:
            staticTracks.append(trackId)
    
    # Save not static tracks
    # Open file
    f = open(outputFile, "w")
    
    for line in lines:
        line_id = int(line.split(",")[1])
        
        if not(line_id in staticTracks):
            line_store = line
    
            f.write(line_store)
        
    # Close txt
    f.close()
            
        
    
if __name__ == "__main__":
    
    # Folder with all tracks
    folderTracks = "./SEQ3_tracks/"
    folderOutput = "./SEQ3_tracks_pp/"
    if not os.path.exists(folderOutput):
       os.makedirs(folderOutput)
    seqFolder = "../seqs/train/S03/"
    
    for trackingFile in os.listdir(folderTracks):
        # Get roi path
        roiPath = seqFolder + trackingFile[:-4].split("_")[-1] + "/roi.jpg"
        # Set output
        outputFile = folderOutput + "pp_"+ trackingFile[:-4] + ".txt"
        postProcessTracks(roiPath, folderTracks + trackingFile, outputFile)