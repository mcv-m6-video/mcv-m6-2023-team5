import numpy as np

def add_noise(annot, imageIds, mean = 0, std = 1.0, delete = 0.5, gener = 0.5):
    """
    This function corrupts to a list of bboxes in three ways:
    1. Adds Gaussian Noise to the size and position
    2. Drops bboxes from the list
    3. Generates new random bboxes

    Args: 
        BBs:- is a list of dictionaries of bboxes. Structure of the dict elements: {'name': 'car', 'bbox': [558.36, 94.45, 663.49, 169.31], 'confidence': 1.0}
        mean:- mean of the Gaussian Noise
        std:- standard deviation of the Gaussian Noise
        dropout:- the probability of a bbox to be dropped from the list
        generate:- for each bbox, the probability to create an extra random bbox
    
    Return:
        newBBs:- list of coordinates of the corrupted bboxes. Each item is of the form [xmin, ymin, xmax, ymax]
    """

    new_bboxes = {}
    for ind in imageIds:
        new_bboxes[ind] = []
        for obj in annot[ind]:
            
            dropped = np.random.choice([True,False],1,p = [delete, 1-delete])
            #print(dropped)

            if not dropped: # Add Gaussian Noise to the bbox if not dropped
                [xmin, ymin, xmax, ymax] = obj['bbox']
    
                w, h = [xmax-xmin,ymax-ymin]
    
                xmin = xmin + np.random.normal(mean,std)
                ymin = ymin + np.random.normal(mean,std)
    
                xmax = xmin + (w + np.random.normal(mean,std))
                ymax = ymin + (h + np.random.normal(mean,std))
    
                BBnew = obj.copy()
                BBnew['bbox'] = [xmin, ymin, xmax, ymax]
                new_bboxes[ind].append(BBnew)
        
            generation = np.random.choice([True,False],1,p = [gener, 1-gener])
        
            if generation: # Generate new bbox
                w_max = 1920 # Width of the frame
                h_max = 1080 # Height of the frame
        
                xmin = np.random.uniform(0,1000)
                ymin = np.random.uniform(0,1000)
        
                xmax = xmin + np.random.uniform(0,w_max - xmin)
                ymax = ymin + np.random.uniform(0,h_max - ymin)
            
                rand_bboxes = {'name': obj["name"], 'bbox': [xmin, ymin, xmax, ymax]}
                new_bboxes[ind].append(rand_bboxes)

    
    return new_bboxes
