import numpy as np
import cv2
import imageio
import os
from PIL import Image

if __name__ == "__main__":
    
    # Path
    videoPath = "./seqs/train/S03/c010/vdo.avi"
    trackFile = "./data/trackers/mot_challenge/MOT15-all/MPNTrack/data/c010.txt"
    #trackFile = "./data/gt/mot_challenge/MOT15-all/OFtracking_c022_remStatic/gt/gt.txt"
    f = open(trackFile, "r")
    lines = f.readlines()
    f.close()
    results = [line.split(",") for line in lines]
    
    # colours = np.random.rand(32, 3) #used only for display
    colours = np.random.randint(0, 256, size=(32, 3))
    
    frames = []

    # Load video
    cap = cv2.VideoCapture(videoPath)

    # current frame number
    current_frame = 1

    while True:
         # Read frame
        ret, frame = cap.read()
        
        # Check if frame was successfully read
        if not ret:
            break
        print(current_frame)
        # if current_frame == 50:
        #     break
        if current_frame % 2 == 0:
            if current_frame > 500: 
                break
            if current_frame > 100:
                for detection in results:
                    detection = [float(e) for e in detection]
                    detection[4] += detection[2]
                    detection[5] += detection[3]
                    if detection[0] == current_frame:
                        # draw the boxes on screen
                        
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
                frames.append(frame)
        
        current_frame += 1
    
    # Same color palette every frame:
    for i, frame in enumerate(frames):
        frame = Image.fromarray(frame)
        if i == 0:
            frames[i] = frame.quantize(colors=254, method=Image.Quantize.MAXCOVERAGE)
        else:
            frames[i] = frame.quantize(colors=254, palette=frames[0], method=Image.Quantize.MAXCOVERAGE)
            frames[i] = frames[i].convert('RGB')
            frames[i] = np.array(frames[i])
    frames[0] = frames[0].convert('RGB')      
    frames[0] = np.array(frames[0])
    imageio.mimsave('results_c010_our_no_rem.gif', frames, format  = 'GIF', fps=10)
    print("DONE!")