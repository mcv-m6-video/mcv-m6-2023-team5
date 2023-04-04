from utils2 import calculateMetrics
import numpy as np
from PIL import Image
import time
import cv2

# Paths
img1Path = "./flow_dataset/training/image_0/000045_10.png"
img2Path = "./flow_dataset/training/image_0/000045_11.png"
opt_gt = "./flow_dataset/training/flow_noc/000045_10.png"

# Read GT
flow_gt = cv2.cvtColor(cv2.imread(opt_gt, cv2.IMREAD_UNCHANGED).astype(np.uint16), cv2.COLOR_BGR2RGB)

# Read images
im1 = np.array(Image.open(img1Path))
im1 = im1[:,:,np.newaxis]
im2 = np.array(Image.open(img2Path))
im2 = im2[:,:,np.newaxis]

#from resultsPyFlow import PyFlowOptFlow
from resultsUniMatch import UniMatchOptFlow
#from resultsPerceiverIO import PercIOOptFlow
#from resultsRAFT import RaftFlow

# Init
predictor = UniMatchOptFlow()

# Obtain optical flow
times = []
for i in range(20):
    s = time.time()
    flow = predictor.inference(im1,im2)
    e = time.time()
    times.append(e-s)
print('Time Taken: %.2f seconds' % (
    np.median(times)))

# Measure results
msen, pepn = calculateMetrics(flow, flow_gt)
print("MSEN: ", msen)
print("PEPN: ", pepn)