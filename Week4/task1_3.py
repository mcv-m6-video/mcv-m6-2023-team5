import numpy as np
import cv2
import imageio

class Track:
    def __init__(self, center, idVal, bbox, conf):
        self.bbox = bbox
        self.actualCenter = center
        self.flowsX = []
        self.flowsY = []
        self.patience = 5
        self.id = idVal
        self.conf = conf
        
    def updateFlow(self, flow, size):
        """
        This functions updates the results with the optical flow predicted

        Parameters
        ----------
        flow : numpy array
            mean/median optical flow related to this track bbox.
        size : numpy array
            size of the bbox.

        Returns
        -------
        None.

        """
        # If 5 flows were previously added, get 4 newest and add new one, otherwise add it
        if len(self.flowsX) == 5:
            self.flowsX = self.flowsX[1:] + [flow[0]]
            self.flowsY = self.flowsY[1:] + [flow[1]]
        else:
            self.flowsX.append(flow[0])
            self.flowsY.append(flow[1])
            
        # Get mean
        flowXmean = np.array(self.flowsX).mean()
        flowYmean = np.array(self.flowsY).mean()
        
        # Predict new bbox based on optical flow
        self.pred_new_bbox = [min(size[1] - 1, max(0, self.bbox[0] + flowXmean)), 
                              min(size[0] - 1, max(0, self.bbox[1] + flowYmean)),
                              min(size[1] - 1, max(0, self.bbox[2] + flowXmean)), 
                              min(size[0] - 1, max(0, self.bbox[3] + flowYmean))]
        
        # Predict new bbox center
        self.pred_new_center = [flowXmean + (self.bbox[0]+self.bbox[2])/2, 
                                flowYmean + (self.bbox[1]+self.bbox[3])/2]
        
    def updateValidated(self, bbox, conf):
        """
        This functions updates the track by updating the results with a detected bbox

        Parameters
        ----------
        bbox : list
            Detected bbox and assigned to this track.
        conf : float
            Detection confidence value.

        Returns
        -------
        None.

        """
        # Compute new center
        newCenter = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        # Compute real flow
        realFlowX = newCenter[0] - self.actualCenter[0]
        realFlowY = newCenter[1] - self.actualCenter[1]
        # Update with real flow
        self.flowsX[-1] = realFlowX
        self.flowsY[-1] = realFlowY
        
        # Update values
        self.actualCenter = newCenter
        self.bbox = bbox
        self.patience = 5
        self.conf = conf
    
    def updateNotValidated(self):
        """
        This functions updates the values, when no detections were assigned to this track.
        It uses the ones predicted with optical flow. Patience is reduced.

        Returns
        -------
        None.

        """
        self.actualCenter = self.pred_new_center
        self.bbox = self.pred_new_bbox
        self.patience -= 1
        # If bbox is empty remove track (patience 0)
        if self.bbox[0] == self.bbox[2] or self.bbox[1] == self.bbox[2]:
            self.patience = 0

def detectionsFlow(tracks, flow, size, opType = "mean"):
    """
    This function updates the tracks by using the optical flow

    Parameters
    ----------
    tracks : list
        Actual track list.
    flow : numpy array
        Optical flow estimated for next frame.
    size : numpy array
        Images size.
    opType : str, optional
        How the bbox flow is calculated "mean" or "median". The default is "mean".

    Returns
    -------
    None.

    """
    
    # Update every track
    for track in tracks:
        # Get bbox flow
        bbox = track.bbox
        bbox_rounded = [int(np.around(e)) for e in bbox]
        bbox_flow = flow[bbox_rounded[1]:bbox_rounded[3],bbox_rounded[0]:bbox_rounded[2], :]
        
        # Compute mean or median
        if opType == "mean":
            flow_box_f = np.mean(bbox_flow, axis=(0,1))
        elif opType == "median":
            flow_box_f = np.median(bbox_flow, axis=(0,1))
        
        # Update track with optical flow
        track.updateFlow(flow_box_f, flow.shape)
        

def computeDis(center, detection):
    """
    This function computes the distacen between the detection bbox center and the center given.
    It returns inf value if the center is not inside the detection bbox.

    Parameters
    ----------
    center : list
        Center point.
    detection : numpy array
        Detection bbox.

    Returns
    -------
    dis : float
        Distance value between two centers (inf if the center is not inside the detection bbox).

    """
    
    # Init as inf
    dis = np.float32("inf")
    bbox = detection[2:7]
    
    # Center point inside the bbox
    if (center[0] >= bbox[0] and center[0] <= bbox[2]) and (center[1] >= bbox[1] and center[1] <= bbox[3]):
        # Compute detection center
        det_center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
        # Compute distance between centers
        dis = np.mean(np.square(det_center - center))
    
    return dis

def trackObjects(objDetectionFile, opticalFlows):
    """
    This function generate the tracks using the optical flow and detections.

    Parameters
    ----------
    objDetectionFile : str
        Detections file path.
    opticalFlows : str
        Optical flow folder.

    Returns
    -------
    results : list
        Result track values.

    """
    
    seq_dets = np.loadtxt(objDetectionFile, delimiter=',') # array with every line in the file

    current_ID = 1
    # create dictionary to store the detections from each frame (faster than list) in paris <FRAME_NR, DETECTIONS_ARRAY>
    results = {}
    nr_frames = 0
    
    tracks = []
    
    # Read detections
    for detection in seq_dets:
        frame = int(detection[0])
        nr_frames = max(nr_frames, frame)
        id = detection[1]
        rest = detection[2:7] 
        rest[2:4] += rest[0:2] # convert from [x, y, bb_width, bb_height] to [x1, y1, x2, y2]

        # Assign first frame object IDs
        if(frame == 1):
            id = current_ID # set the first bboxes IDs in the first frame
            current_ID += 1
        
        if frame in results:
            det = [frame, id, rest[0], rest[1], rest[2], rest[3], rest[4]]
            results[frame] = results[frame] + [det]
        else:
            det = [frame, id, rest[0], rest[1], rest[2], rest[3], rest[4]]
            results[frame] = [det]
        if (frame == 1):
            tracks.append(Track([(rest[0]+rest[2])/2, (rest[1]+rest[3])/2], id, rest[:4], rest[4]))

    # Assign new ids
    results = trackWithOpticalFlow(results, current_ID, nr_frames, opticalFlows, tracks)
    
    return results
    
def trackWithOpticalFlow(results, current_id, nr_frames, opticalFlows, tracks):
    """
    This function generates the tracks from frame 2 to the end.

    Parameters
    ----------
    results : dict
        Detections values.
    current_id : int
        Last id assigned.
    nr_frames : int
        Number of frames.
    opticalFlows : str
        Folder of files with predicted optical flow.
    tracks : list
        Initial tracks (frame 1).

    Returns
    -------
    result : list
        Computed tracks.

    """

    # Get first frames tracks
    result = [x for x in results[1]] # attach the first frames
    
    # start from second frame, as the first one has the IDs assigned
    for current_frame in range(2, nr_frames+1):
        
        # Read optical flow
        flow = np.load(opticalFlows + str(current_frame) + ".npy")
        
        # Get flow
        detectionsFlow(tracks, flow, flow.shape, "median")
        
        # Do not repeat current detection assigments
        alreadyAssig = [False]*len(results[current_frame])
        for i, track in enumerate(tracks): # take all tracks from previous
            # Init assignment
            minDis = np.float32("inf")
            det = None
            det_index = -1
            
            # Get current detection
            for index, current_detection in enumerate(results[current_frame]):
                
                if not alreadyAssig[index]:
                    dis = computeDis(track.pred_new_center, current_detection)
                    
                    if dis < np.float32("inf") - 1 and dis < minDis:
                        det = current_detection
                        det_index = index
                        minDis = dis
            
            # Assigned to a detection
            if not(det is None):
                det[1] = track.id
                # det = [current_frame, assignedID, newBBox[0], newBBox[1], 
                #                 newBBox[2], newBBox[3], previous_detection[6]]
                # results[current_frame][det_index] = det
                alreadyAssig[det_index] = True
                track.updateValidated(det[2:6], det[6])
                
            elif det is None: # didn't find anything create new bbox with same id
                track.updateNotValidated()
                if track.patience == 0:
                    continue
                det = [current_frame, track.id, track.bbox[0], track.bbox[1], 
                                track.bbox[2], track.bbox[3], track.conf]
                results[current_frame].append(det)
                alreadyAssig.append(True)
            result.append(det)
        
        # New objects found for tracking
        for i, assigned in enumerate(alreadyAssig):
            
            if not assigned:
                # Create new track
                newTrackBBox = results[current_frame][i][2:7]
                newTrack = Track([(newTrackBBox[0]+newTrackBBox[2])/2, 
                                  (newTrackBBox[1]+newTrackBBox[3])/2], 
                                 current_id, newTrackBBox[:4], newTrackBBox[4])
                tracks.append(newTrack)
                det = results[current_frame][i]
                det[1] = current_id
                current_id += 1
            
                result.append(det)
        
        # Remove tracks without patience
        ind = 0
        while ind < len(tracks):
            if tracks[ind].patience == 0:
                del tracks[ind]
            else:
                ind += 1
            
    return result

if __name__ == "__main__":
    objDets = "./det_detr_pretrained.txt"
    flows = "./opticalFlows/"
    results = trackObjects(objDets, flows)
    
    
    output_file = 'optical_based_tracking_pretrained.txt'
    # Open file
    f = open(output_file, "w")
    
    initial = True
    
    for result in results:
        imageId = str(result[0])
        objId = str(result[1])
        x = result[2]
        y = result[3]
        w = result[4] - result[2]
        h = result[5] - result[3]
        conf = result[6]
        if initial:
            line = imageId + "," + objId + ",{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},-1,-1,-1".format(x, y, w, h, conf)
            initial = False
        else:
            line = "\n" + imageId + "," + objId + ",{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},-1,-1,-1".format(x, y, w, h, conf)

        f.write(line)
        
    # Close txt
    f.close()
    
    
    display = True
    # colours = np.random.rand(32, 3) #used only for display
    colours = np.random.randint(0, 256, size=(32, 3))
    
    if(display):
        
        frames = []
        # Path
        videoPath = "../AICity_data/train/S03/c010/vdo.avi"

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
            for detection in results:
                if detection[0] == current_frame:
                    # draw the boxes on screen
                    
                    id = detection[1]
                    positions = detection[2:6]# [x1, y1, x2, y2]
                    color = np.uint8(colours[id%32, :])
                    c = tuple(map(int, color))

                    cv2.rectangle(frame, (int(positions[0]), int(positions[1])), (int(positions[2]), int(positions[3])), color=c, thickness=2)
                    cv2.putText(frame, str(id), (int(positions[0]), int(positions[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color=c, thickness=2)
                    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(250, 120))
            frames.append(frame)
            
            current_frame += 1
        imageio.mimsave('results_optical_flow.gif', frames, fps=10)