import numpy as np
import os

def removeStaticTracks(trackFile, outputFile):
    """
    This function removes the tracks from static objects.

    Parameters
    ----------
    trackFile : str
        Tracking results file with static objects.
    outputFile : str
        Path to store the output with tracks without the static objects.

    Returns
    -------
    None.

    """
    
    tracks = {}
    
    # Load detections
    f = open(trackFile, "r")
    lines = f.readlines()
    f.close()
    
    # Load tracks
    for line in lines:
        line_split = line.split(",")
        l_id = int(line_split[1])
        
        centerX = float(line_split[2]) + float(line_split[4])/2
        centerY = float(line_split[3]) + float(line_split[5])/2
        
        center = [centerX, centerY]
        
        if l_id in tracks.keys():
            tracks[l_id].append(center)
        else:
            tracks[l_id] = [center]
    
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
    folderTracks = "./challengeData3_res_no/"
    folderOutput = "./challengeData3_res_no/"
    
    for trackingFile in os.listdir(folderTracks):
        # Set output
        outputFile = folderOutput + trackingFile[:-4] + "_remStatic.txt"
        removeStaticTracks(folderTracks + trackingFile, outputFile)