import optuna
from block_matching import block_match, block_match_log, l2
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

def decodeOpticalFlow(img):
    """
    Determine the optical flow in x and y direction
    
    Args: The numpy ndarrray of the image
    Return:
        Optical flow estimation in X and Y direction.
    """
    flow_x = (img[:,:,0].astype(float) - 2. ** 15) / 128
    flow_y = (img[:,:,1].astype(float) - 2. ** 15) / 128
    valid = img[:,:,2].astype(bool)
    return flow_x, flow_y, valid


def calculate_metrics(x_img, y_img, x_gt, y_gt, valid_gt):
    """
    Calculating The Mean Square Error and Percentage of Erroneous Pixels in non-ocluded areas
    
    Args:
        img: the predicted numpy.ndarray
        img_gt: the groundtruth of the prediction. It is also a numpy.ndarray
        ouputFile: Where you want to save the result.
    Return:
        msen: Mean Square Error in non-ocluded areas.
        pepn: Percentage of Erroneous Pixels in non-ocluded areas.
    """
    # is calculated for every pixel
    motion_vectors = np.sqrt( np.square(x_img - x_gt) + np.square(y_img - y_gt) )


    # erroneous pixels are the ones where motion_vector > 3 and are valid pixels 
    err_pixels = (motion_vectors[valid_gt == 1] > 3).sum()

    # calculate metrics
    msen = np.mean((motion_vectors)[valid_gt == 1])
    pepn = (err_pixels / (valid_gt == 1).sum()) * 100 # erroneous pixels / total valid pixels from the ground truth

    return msen, pepn



gt_path = r"/home/user/optical_flow/of_gt/000045_10.png"
curr_path = r"/home/user/optical_flow/data_stereo_flow/training/colored_0/000045_10.png"
ref_path = r"/home/user/optical_flow/data_stereo_flow/training/colored_0/000045_11.png"
curr = np.asarray(Image.open(curr_path))
ref = np.asarray(Image.open(ref_path))
gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)\
    [:, :, ::-1]  # X-Y-Valid
x_gt, y_gt, valid_gt = decodeOpticalFlow(gt_img)

def objective(trial: optuna.Trial):
    block_side = trial.suggest_int("block_side", 3, 40)
    win_scale = trial.suggest_float("win_scale", 1.1, 3.0)
    win_side = np.ceil(block_side * win_scale).astype(int) + 1
    block_shape = (block_side, block_side)
    win_shape = (win_side, win_side)
    
    flow = block_match(ref, curr, block_shape, win_shape, l2)
    
    # max_level = trial.suggest_int("max_level", 1, 5)
    # flow = block_match_log(ref, curr, block_shape, win_shape, l2, max_level)
    
    flow = flow.astype(np.float32)
    interpolation = trial.suggest_categorical("interpolation", [
        "nearest",
        "cubic"
    ])
    if interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    if interpolation == "cubic":
        interpolation = cv2.INTER_CUBIC
    flow_reshaped = cv2.resize(flow, (x_gt.shape[1], x_gt.shape[0]),
                               interpolation=interpolation)
    x_pred = flow_reshaped[..., 1]
    y_pred = flow_reshaped[..., 0]
    msen, pepn = calculate_metrics(x_pred, y_pred,
                                   x_gt, y_gt, valid_gt)
    return msen


study = optuna.create_study(study_name="bm8",
                            storage="sqlite:///block_matching.db",
                            load_if_exists=True)
study.optimize(objective, n_trials=100)