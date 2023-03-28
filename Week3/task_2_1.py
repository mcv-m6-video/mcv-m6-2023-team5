from sort import iou_batch, convert_bbox_to_z, convert_x_to_bbox
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

class Detection:
    def __init__(self, frame, ID, coords):
        self.frame = frame
        self.coords = coords
        self.ID = ID

    def __str__(self):
        return f"{self.frame}, {self.ID}, {self.coords}"

def computeIoU(detection1, detection2):
    return bb_intersection_over_union(detection1[2:6], detection2[2:6])

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def computeOverlapMaximization(dict, current_id, nr_frames, threshold = 0.8):

    result = []
    [result.append(x) for x in dict[0]] # attach the first frames

    # check last 5 previous frames for a bounding box (in case of occlusions)
    BACK_IN_TIME_FRAMES = 5
    
    # start from second frame, as the first one has the IDs
    for current_frame in range(1, nr_frames+1):
        for current_frame_detection in dict[current_frame]: # take all bboxes from current frame
            IoUValue = float('-inf')
            assignedID = -1

            previous_frames = [*range(max(current_frame - BACK_IN_TIME_FRAMES, 0), current_frame)]
    
            previous_detections = [det for key in previous_frames for det in dict.get(key)]
            
            for previous_detection in previous_detections:
                IoU = computeIoU(current_frame_detection, previous_detection)
                
                if(IoU >= threshold and IoU > IoUValue): # get the maximum IoU above the threshold
                    IoUValue = IoU
                    assignedID = previous_detection[1]
                    
            if assignedID != -1:
                current_frame_detection[1] = assignedID
            elif assignedID == -1: # didn't find anything that has an IoU big enought
                current_frame_detection[1] = current_id # new detection
                current_id += 1
            
            result.append(current_frame_detection)

    return result

if __name__ == '__main__':
    colours = np.random.rand(32, 3) #used only for display

    input_file = "det_detr.txt"
    seq_dets = np.loadtxt(input_file, delimiter=',') # array with every line in the file

    current_ID = 0
    # create dictionary to store the detections from each frame (faster than list) in paris <FRAME_NR, DETECTIONS_ARRAY>
    dict = {}
    nr_frames = 0

    for detection in seq_dets:
        frame = int(detection[0])
        nr_frames = max(nr_frames, frame)
        id = detection[1]
        rest = detection[2:7] 
        rest[2:4] += rest[0:2] # convert from [x, y, bb_width, bb_height] to [x1, y1, x2, y2]

        if(frame == 0):
            id = current_ID # set the first bboxes IDs in the first frame
            current_ID += 1
        
        if frame in dict:
            det = [frame, id, rest[0], rest[1], rest[2], rest[3]]
            dict[frame] = dict[frame] + [det]
        else:
            bb_nr_in_frame = 0
            det = [frame, id, rest[0], rest[1], rest[2], rest[3]]
            dict[frame] = [det]

    result = computeOverlapMaximization(dict, current_ID, nr_frames)

    print(result[0])


    # TO BE COMPLETED
    output_file = 'task21_output.txt'
    with open(output_file, 'w') as f:
        for detection in result:
            f.write(f"{detection}\n")

    f.close()






