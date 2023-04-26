import numpy as np
import cv2
import imageio
import os
from PIL import Image

if __name__ == "__main__":
    
    # Path
    seq = "../seqs/train/S03/"
    cameras = os.listdir(seq)
    trackFile = "./SO3mtmc/gtvotes0.6.txt"
    #trackFile = "./data/gt/mot_challenge/MOT15-all/OFtracking_c022_remStatic/gt/gt.txt"
    f = open(trackFile, "r")
    lines = f.readlines()
    f.close()
    results = [line.split(",") for line in lines]
    
    cameraFrames = [[] for i in range(len(cameras))]
    
    # colours = np.random.rand(32, 3) #used only for display
    colours = np.random.randint(0, 256, size=(32, 3))
    currentCamera = -1
    
    plotedDetections = [False]*len(results)
    for index, result in enumerate(results):
        if currentCamera != int(result[0])//10000:
            currentCamera = int(result[0])//10000
            videoPath = seq + cameras[currentCamera] + "/vdo.avi"
            # Load video
            cap = cv2.VideoCapture(videoPath)
            current_frame = 1
        
            # Read frame
            ret, frame = cap.read()
        if plotedDetections[index]:
            continue
        while int(result[0]) - currentCamera * 10000 > current_frame:
            # Read frame
            ret, frame = cap.read()
            current_frame += 1
        
        print(current_frame)
        # if current_frame == 50:
        #     break
        if current_frame % 2 == 0:
            if current_frame > 500: 
                continue
            if current_frame > 100:
                for res_i, detection in enumerate(results):
                    detection = [float(e) for e in detection]
                    detection[4] += detection[2]
                    detection[5] += detection[3]
                    if detection[0] - currentCamera * 10000 == current_frame:
                        # draw the boxes on screen
                        plotedDetections[res_i] = True
                        id = int(detection[1])
                        positions = detection[2:6]# [x1, y1, x2, y2]
                        color = np.uint8(colours[id%32, :])
                        c = tuple(map(int, color))
                        #print(id, c)
        
                        cv2.rectangle(frame, (int(positions[0]), int(positions[1])), (int(positions[2]), int(positions[3])), color=c, thickness=10)
                        cv2.putText(frame, str(id), (int(positions[0]), int(positions[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 4, color=c, thickness=10)
                        
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(500, 300))
                # Select specific part
                #frame = frame[10:200, 500:1060]
                cameraFrames[currentCamera].append(frame)
    
    # Same color palette every frame:
    for j, oneCamFrames in enumerate(cameraFrames):
        for i, frame in enumerate(oneCamFrames):
            frame = Image.fromarray(frame)
            if j == 0 and i == 0:
                cameraFrames[j][i] = frame.quantize(colors=254, method=Image.Quantize.MAXCOVERAGE)
            else:
                cameraFrames[j][i] = frame.quantize(colors=254, palette=cameraFrames[0][0], method=Image.Quantize.MAXCOVERAGE)
                cameraFrames[j][i] = cameraFrames[j][i].convert('RGB')
                cameraFrames[j][i] = np.array(cameraFrames[j][i])
    cameraFrames[0][0] = cameraFrames[0][0].convert('RGB')      
    cameraFrames[0][0] = np.array(cameraFrames[0][0])
    
    for i, oneCamFrames in enumerate(cameraFrames):
        path = "results_" + cameras[i] + trackFile[:-4] + ".gif"
        imageio.mimsave(path, oneCamFrames, format  = 'GIF', fps=10)
    print("DONE!")