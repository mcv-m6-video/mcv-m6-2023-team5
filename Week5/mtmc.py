import numpy as np
from reId import ReIdByCentroids, ReIdByVoting
import os

if __name__ == "__main__":
        
    distances = [-1, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 4000]
    types = ["centroids", "votes"]
    for typeMatch in types:
        for distance in distances:
            folderEmbeddings = "./S03embeddings/"  
            folderTracks = "./SEQ3_tracks_pp/"
            folderSeqs = "../seqs/train/S03/"
            
            outputMultiCameraTrack = "./SO3mtmc/"
            if not os.path.exists(outputMultiCameraTrack):
               os.makedirs(outputMultiCameraTrack)
    
            initial = True
            if typeMatch == "centroids":
                centroids = True
            else: 
                centroids = False
                
            camerasList = []
            matchesList = [None]
            
            # Compute MTMC
            frameDiff = 10000
            for file in os.listdir(folderEmbeddings):
                
                if file[-14:-4] == "embeddings":
                    camerasList.append(file.split("_")[-2])
                    embeddings = np.load(folderEmbeddings + file)
                    ids =  np.load(folderEmbeddings + file[:-14] + "ids.npy")
                    tracks = {}
                    tracks["embeddings"] = embeddings
                    tracks["ids"] = ids
                    
                    if initial:
                        mainTracks = tracks
                        initial = False
                    else:
                        if centroids:
                            matches, mainTracks = ReIdByCentroids(mainTracks, tracks, distance)
                            matchesList.append(matches)
                        else:
                            matches, mainTracks = ReIdByVoting(mainTracks, tracks, distance)
                            matchesList.append(matches)
            
            # Save result MTMC
            resultTracks = []
            for i, camera in enumerate(camerasList):
                frameInit = frameDiff * i
                match = matchesList[i]
                
                for trackFile in os.listdir(folderTracks):
                    if trackFile[:-4].split("_")[-1] == camera:
                        break
                
                f = open(folderTracks + trackFile)
                lines = f.readlines()
                f.close()
                videoTrack = [line.split(",") for line in lines]
    
                for track in videoTrack:
                    
                    track[0] = str(int(track[0]) + frameInit)
                    
                    if not (match is None):
                        track[1] = str(match[int(track[1])])
                    
                    resultTracks.append(track)
            
            f = open(outputMultiCameraTrack + "mtmc"+ typeMatch + str(distance) + ".txt", "w")
            for track in resultTracks:
                line = ",".join(track)
                f.write(line)
            
            f.close()
            
            # Save GT MTMC
            resultTracks = []
            for i, camera in enumerate(camerasList):
                frameInit = frameDiff * i
                
                f = open(folderSeqs + camera + "/gt/gt.txt")
                lines = f.readlines()
                f.close()
                videoTrack = [line.split(",") for line in lines]
                
                for track in videoTrack:
                    
                    track[0] = str(int(track[0]) + frameInit)
                    
                    resultTracks.append(track)
            
            f = open(outputMultiCameraTrack + "gt" + typeMatch + str(distance) + ".txt", "w")
            for track in resultTracks:
                line = ",".join(track)
                f.write(line)
            
            f.close()
        
            
        