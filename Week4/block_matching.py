import numpy as np


def get_blocks(img, block_shape):
    """Divide the input image into blocks of specified shape.
    
    Take into account cases when img can not be evenly divided into
    blocks of specified shapes. In such cases create blocks with unequal
    shapes.
    
    Args:
        img: array of shape H, W, C, represents an image.
        block_shape: tuple that represents (H, W) of a block.
    
    Retrurns:
        blocks: list, where each element is a block.
        position: list, where each element is a left-upper positions
            of blocks in img coordinates.
        split_shape: tuple, (n_vertical_blocks, n_horizontal_blocks).
    """
    
    split_shape = np.ceil((
        img.shape[0]/block_shape[0],
        img.shape[1]/block_shape[1]
    )).astype(np.int32)
    
    blocks = []
    positions = []
    pos_u = 0
    for i in range(split_shape[0]):
        if i != split_shape[0]:
            stripe = img[i*block_shape[0]:i*block_shape[0]+block_shape[0]]
        else:
            stripe = img[i*block_shape[0]:]  # for uneven block/img shapes
        pos_v = 0
        for j in range(split_shape[1]):
            if j != split_shape[1]:
                block = stripe[:, j*block_shape[1]:\
                    j*block_shape[1]+block_shape[1]]
            else:
                block = stripe[:, j*block_shape[1]:]      
            blocks.append(block)
            positions.append((pos_u, pos_v))
            pos_v += block.shape[1]
        pos_u += block.shape[0]
    return blocks, positions, split_shape


def get_window(img, win_shape, block_shape, pos):
    """Return a window, centered in a block specified by its shape and
    and pos.
    
    Args:
        img: NumPy array, image which is been sliced.
        win_shape: tuple (win_height, win_width).
        block_shape: tuple (block_height, block_width).
        pos: tuple, top left position of block in img coords.

    Returns:
        window: slice of the img.
        win_pos: coordinates of top left corner of the window
            in img coords.
            
    """
    center = [pos[0]+block_shape[0]//2, pos[1]+block_shape[1]//2]
    top = np.max([0, center[0] - win_shape[0]//2])
    bot = np.min([img.shape[0], center[0] - win_shape[0]//2 + win_shape[0]])
    left = np.max([0, center[1] - win_shape[1]//2])
    right = np.min([img.shape[1], center[1] - win_shape[1]//2 + win_shape[1]])
    window = img[top:bot, left:right]
    win_pos = (top, left)

    return window, win_pos


def block_match(ref_frame, curr_frame, block_shape, win_shape, metric_func):
    """Perform block matching.
    
    Calculate apparent movement between blocks of two frames, using
    exhaustive search.

    Args:
        ref_frame: NumPy array, reference frame, image where
            correspondences should by found.
        curr_frame: NumPy array, current frame, image, from which blocks
            are obtained.
        block_shape: tuple, shape of a block (height, width).
        win_shape: tuple, shape of a search window (height, width).
        metric_func: callable, metric to compare current block with
            candidates withing search window. Outputs scores
            for n_candidates, when input is img_shape,
            and (n_candidates, *img_shape).
    
    Returns:
        flow: NumPy array, representing u and v movement between frames
        
    """
    flow = []
    blocks, positions, split_shape = get_blocks(curr_frame, block_shape)
    for block, pos in zip(blocks, positions):
        # NOTE: block_shape is the desired shape,
        # while block.shape is the real one.
        window, win_pos = get_window(ref_frame, win_shape, block.shape, pos)
        sliding = np.lib.stride_tricks.sliding_window_view(window, block.shape)
        scores = metric_func(sliding, block)
        scores = np.squeeze(scores)  # suppress redundant 3rd dim
        best_idx_flat = np.argmin(scores)
        best_idx_2d = np.unravel_index(best_idx_flat, scores.shape)
        best_idx_2d_img = (best_idx_2d[0] + win_pos[0],
                           best_idx_2d[1] + win_pos[1])
        displacement = (best_idx_2d_img[0] - pos[0],
                        best_idx_2d_img[1] - pos[1])
        # BUG: flow sometimes exceedes possible values
        flow.append(displacement)
    flow = np.array(flow).reshape((*split_shape, 2))
    
    return flow


def block_match_log(ref_frame, curr_frame, block_shape, win_shape, metric_func,
                max_level=3):
    """Perform logarithmic block matching.
    
    Calculate apparent movement between blocks of two frames, using
    recoursive (logarithmic) search.

    Args:
        ref_frame: NumPy array, reference frame, image where
            correspondences should by found.
        curr_frame: NumPy array, current frame, image, from which blocks
            are obtained.
        block_shape: tuple, shape of a block (height, width).
        win_shape: tuple, shape of a search window (height, width).
        metric_func: callable, metric to compare current block with
            candidates withing search window. Outputs scores
            for n_candidates, when input is img_shape,
            and (n_candidates, *img_shape).
        max_level: int, maximum number of iteration in log search

    Returns:
        flow: NumPy array, representing u and v movement between frames
        
    """
    flow = []
    blocks, positions, split_shape = get_blocks(curr_frame, block_shape)
    for block, pos in zip(blocks, positions):
        # NOTE: block_shape is the desired shape,
        # while block.shape is the real one.
        window, win_pos = get_window(ref_frame, win_shape, block.shape, pos)
        sliding = np.lib.stride_tricks.sliding_window_view(window, block.shape)
        sliding_center = np.array(sliding.shape)[:2] // 2
        for level in range(max_level):
            shift = np.array(sliding.shape)[:2] // 2**(level+1) - 1
            indices = np.array([
                sliding_center - shift,
                sliding_center,
                sliding_center + shift
            ]).T
            indices = np.array([
                [indices[0][0], indices[1][1]],
                
                [indices[0][1], indices[1][0]],
                [indices[0][1], indices[1][1]],
                [indices[0][1], indices[1][2]],
                
                [indices[0][2], indices[1][1]],
            ])
            indices = np.clip(indices, 0, np.array(sliding.shape)[:2]-1)            
            candidates = np.array([sliding[w[0]][w[1]] for w in indices])
            scores = metric_func(candidates, block)
            current_center = indices[np.argmin(scores)]
            if np.all(current_center == sliding_center):
                break
            sliding_center = current_center
            pass
        best_idx_2d = sliding_center
        best_idx_2d_img = (best_idx_2d[0] + win_pos[0],
                           best_idx_2d[1] + win_pos[1])
        displacement = (best_idx_2d_img[0] - pos[0],
                        best_idx_2d_img[1] - pos[1])
        # BUG: flow sometimes exceedes possible values
        flow.append(displacement)
    flow = np.array(flow).reshape((*split_shape, 2))
    
    return flow

    
# create a metric function
def l2(a, b):
    diff = a - b
    square = np.power(diff, 2)
    sq_sum = np.sum(square, axis=(-1, -2, -3))
    return sq_sum    


if __name__ == "__main__":
    # USAGE EXAMPLE
    from PIL import Image
    import matplotlib.pyplot as plt
    from time import time
    curr_path = r"/home/user/optical_flow/data_stereo_flow/training/colored_0/000045_10.png"
    ref_path = r"/home/user/optical_flow/data_stereo_flow/training/colored_0/000045_11.png"
    curr = np.asarray(Image.open(curr_path))
    ref = np.asarray(Image.open(ref_path))
    # flow = block_match(ref, curr, (20, 20), (39, 39), l2)
    flow = block_match_log(ref, curr, (20, 20), (39, 39), l2, 3)
    ace = np.sum(np.square(flow), axis=-1)
    plt.imshow(ace)
    plt.quiver(flow[:, :, 1], flow[:, :, 0], angles="xy")
    plt.show()
