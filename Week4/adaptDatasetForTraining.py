import os
import cv2
from storeCOCOformat import storeCOCOformatJson

if __name__ == "__main__":
    
    imagesFolder = "./seqs/train/"
    
    trainGroupSeqs = [["S01", "S04"], ["S03", "S04"], ["S01", "S03"]]
    valGroupSeqs = [["S03"], ["S01"], ["S04"]]
    folders = ["./challengeData1/", "./challengeData2/", "./challengeData3/"]
    
    for i, outputFolder in enumerate(folders):
    
        trainSeqs = trainGroupSeqs[i]
        valSeqs = valGroupSeqs[i]
    
        trainFolder = outputFolder + "train2017/"
        valFolder = outputFolder + "val2017/"
        annotFolder = outputFolder + "annotations/"
        
        # Create folder of new split
        if not os.path.exists(trainFolder):
            os.makedirs(trainFolder)
            
        # Create folder of new split
        if not os.path.exists(valFolder):
            os.makedirs(valFolder) 
        
        # Create folder of new split
        if not os.path.exists(annotFolder):
            os.makedirs(annotFolder) 
        
        classIds = {"car": 3}
        
        
        
        annotsTrain = {}
        print("TRAIN:")
        # Train sequences
        for trainSeq in trainSeqs:
            print(trainSeq)
            for subSeq in os.listdir(imagesFolder + trainSeq + "/"):
                print(subSeq)
                annotsPath = imagesFolder + trainSeq + "/" + subSeq + "/" + "gt/gt.txt"
                f = open(annotsPath, "r")
                lines = f.readlines()
                f.close()
                results = [line.split(",") for line in lines]
                
                for line in results:
                    annotation = {}
                    xmin = float(line[2])
                    ymin = float(line[3])
                    w = float(line[4])
                    h = float(line[5])
                    annotation["bbox"] = [xmin, ymin, xmin + w, ymin + h]
                    annotation["name"] = "car"
                    
                    imageId = trainSeq + "_" + subSeq + "_" + line[0]
                    
                    if imageId in annotsTrain.keys():
                        annotsTrain[imageId].append(annotation)
                    else:
                        annotsTrain[imageId] = [annotation]
                
                frames = annotsTrain.keys()
                
                videoPath = imagesFolder + trainSeq + "/" + subSeq + "/" + "vdo.avi"
                
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
        
                    imageId = trainSeq + "_" + subSeq + "_" + str(current_frame)
                    
                    if imageId in frames:
                        cv2.imwrite(trainFolder + imageId + ".jpg", frame)
                    
                    current_frame += 1
    
        frames = annotsTrain.keys()
        # Save train annotations
        json_path = annotFolder + "instances_train2017.json"
        storeCOCOformatJson(annotsTrain, frames, classIds, trainFolder, json_path)
        
        
        
        annotsVal = {}
        print("VAL:")
        # Validation sequences
        for valSeq in valSeqs:
            print(valSeq)
            for subSeq in os.listdir(imagesFolder + valSeq + "/"):
                print(subSeq)
                annotsPath = imagesFolder + valSeq + "/" + subSeq + "/" + "gt/gt.txt"
                f = open(annotsPath, "r")
                lines = f.readlines()
                f.close()
                results = [line.split(",") for line in lines]
                
                
                
                
                for line in results:
                    annotation = {}
                    xmin = float(line[2])
                    ymin = float(line[3])
                    w = float(line[4])
                    h = float(line[5])
                    annotation["bbox"] = [xmin, ymin, xmin + w, ymin + h]
                    annotation["name"] = "car"
                    
                    imageId = valSeq + "_" + subSeq + "_" + line[0]
                    
                    if imageId in annotsVal.keys():
                        annotsVal[imageId].append(annotation)
                    else:
                        annotsVal[imageId] = [annotation]
                
                frames = annotsVal.keys()
                
                videoPath = imagesFolder + valSeq + "/" + subSeq + "/" + "vdo.avi"
                
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
        
                    imageId = valSeq + "_" + subSeq + "_" + str(current_frame)
                    
                    if imageId in frames:
                        cv2.imwrite(valFolder + imageId + ".jpg", frame)
                    
                    current_frame += 1
                break
            break
        
        frames = annotsVal.keys()
        # Save val annotations
        json_path = annotFolder + "instances_val2017.json"
        storeCOCOformatJson(annotsVal, frames, classIds, valFolder, json_path)
    
    
    
    