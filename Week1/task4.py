import cv2
import numpy as np
import matplotlib.pyplot as plt


# select file
path = r"data/results_opticalflow_kitti/results/LKflow_000045_10.png"
# path = r"data/results_opticalflow_kitti/results/LKflow_000157_10.png"


def get_offset(img):
    img -= 2**15
    img /= 128
    return img

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)#[:, :, ::-1]
img = img.astype(np.float32)
img = get_offset(img)


def calc_magnitude_and_angle(img):
    # compute magnitude
    squares = np.square(img[:, :, 1:])
    sum_of_squares = np.sum(squares, axis=2)
    magnitude = np.sqrt(sum_of_squares)
    # compute angle
    angle = np.arctan2(img[:, :, 2], img[:, :, 1])  # NOTE: y-first
    return magnitude, angle

def clip_magnitude(magnitude, bound=1):
    magnitude = np.clip(magnitude, 0, bound)
    magnitude /= bound
    return magnitude

def flow2hsv(img):
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


