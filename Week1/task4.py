import cv2
import numpy as np
import matplotlib.pyplot as plt


# select file
path = r"data/results_opticalflow_kitti/results/LKflow_000045_10.png"
# path = r"data/results_opticalflow_kitti/results/LKflow_000157_10.png"


def get_offset(img):
    ### Decode 16-bit .png OF representation
    img -= 2**15
    img /= 128
    return img

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)#[:, :, ::-1]
img = img.astype(np.float32)
img = get_offset(img)


def calc_magnitude_and_angle(img):
    ### Compute magnitude (l2-norm) and angle (atan2) of OF
    # compute magnitude
    squares = np.square(img[:, :, 1:])
    sum_of_squares = np.sum(squares, axis=2)
    magnitude = np.sqrt(sum_of_squares)
    # compute angle
    angle = np.arctan2(img[:, :, 2], img[:, :, 1])  # NOTE: y-first
    return magnitude, angle

def clip_magnitude(magnitude, bound=1):
    ### Limit maximum magnitude and normalize it. 
    ### Good bound values are typically around 3. 
    ### The less the bound -- the more compressed the OF is. The bigger the bound -- more information, but darker image.
    magnitude = np.clip(magnitude, 0, bound)
    magnitude /= bound
    return magnitude

def flow2hsv(img):
    ### Create an HSV image, assigning:
    ### Hue value to the direction of OF
    ### Setting maximum saturation to all the pixels
    ### Value (brightness) as the OF magnitude RESULT 
    ### Different colors represent different directions of OF. The brighter the pixels -- the stronger OF was at this point.
    magnitude, angle = calc_magnitude_and_angle(img)
    magnitude = clip_magnitude(magnitude)
    deg = np.rad2deg(angle)
    deg += 180  # map [-180;180] to [0;360]
    hsv = np.stack([
        deg,
        np.ones_like(deg),  # maximum saturation for all pixels
        magnitude
    ])
    hsv = np.moveaxis(hsv, 0, -1)
    return hsv

hsv = flow2hsv(img)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)



plt.imshow(rgb)
plt.show()


