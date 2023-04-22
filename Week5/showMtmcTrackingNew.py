import numpy as np
import cv2
import imageio
import os
from PIL import Image

if __name__ == "__main__":
    
    # Path
    seq = "../seqs/train/S03/"
    cameras = os.listdir(seq)
    trackFile = "./SO3mtmc/mtmcvotes0.6.txt"
    #trackFile = "./data/gt/mot_challenge/MOT15-all/OFtracking_c022_remStatic/gt/gt.txt"
    f = open(trackFile, "r")
    lines = f.readlines()
    f.close()
    results = [line.split(",") for line in lines]
    
    cameraFrames = [[] for i in range(len(cameras))]
    
    # colours = np.random.rand(32, 3) #used only for display
    colours = np.random.randint(0, 256, size=(32, 3))
    
    frameWidth = 300
    frameHeight = 200
    
    for indexCam, camera in enumerate(cameras):
        videoPath = seq + cameras[indexCam] + "/vdo.avi"
        # Load video
        cap = cv2.VideoCapture(videoPath)
        current_frame = 1
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            print(current_frame)
            # if current_frame == 50:
            #     break
            if current_frame % 2 == 0:
                if current_frame > 800: 
                    break
                if current_frame > 100:
                    for res_i, detection in enumerate(results):
                        detection = [float(e) for e in detection]
                        detection[4] += detection[2]
                        detection[5] += detection[3]
                        if detection[0] - indexCam * 10000 == current_frame:
                            # draw the boxes on screen
                            id = int(detection[1])
                            positions = detection[2:6]# [x1, y1, x2, y2]
                            color = np.uint8(colours[id%32, :])
                            c = tuple(map(int, color))
                            #print(id, c)
            
                            cv2.rectangle(frame, (int(positions[0]), int(positions[1])), (int(positions[2]), int(positions[3])), color=c, thickness=10)
                            cv2.putText(frame, str(id), (int(positions[0]), int(positions[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 4, color=c, thickness=10)
                            
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame,(frameWidth, frameHeight))
                    # Select specific part
                    #frame = frame[10:200, 500:1060]
                    cameraFrames[indexCam].append(frame)
            current_frame += 1
    
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
        path = "results_s03_" + cameras[i] + trackFile[:-4].split("/")[-1] + ".gif"
        imageio.mimsave(path, oneCamFrames, format  = 'GIF', fps=10)
        
    # Create big gif
    rows = int(np.floor(np.sqrt(len(cameraFrames))))
    columns = int(np.ceil(len(cameraFrames) / rows))
    row = 0
    column = 0
    concatFrames = [np.zeros((rows*frameHeight, columns* frameWidth, 3), dtype = np.uint8) for i in range(len(cameraFrames[0]))]
    for oneCamFrames in cameraFrames:
        if column >= columns:
            row += 1
            column = 0
        
        for i, frame in enumerate(oneCamFrames):
            concatFrames[i][row*frameHeight: (row + 1)*frameHeight, column*frameWidth: (column + 1)*frameWidth] = frame
        column += 1
    path = "results_all_s03_" + trackFile[:-4].split("/")[-1] + ".gif"
    imageio.mimsave(path, concatFrames, format  = 'GIF', fps=10)
    
    print("DONE!")