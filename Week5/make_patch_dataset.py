import numpy as np
from pathlib import Path
import cv2


seq_dir = Path(r"../seqs/train/S03")
out_dir = Path(r"./val_dataset_3")
out_dir.mkdir(parents=True, exist_ok=True)


for cam_dir in seq_dir.iterdir():
    print(cam_dir)
    gt_path = cam_dir / r"gt/gt.txt"
    vid_path = cam_dir / r"vdo.avi"

    gt = np.genfromtxt(gt_path, delimiter=',', dtype=np.int32)
    
    cap = cv2.VideoCapture(str(vid_path))
    # TODO: check how the first frame is denoted in gt.txt: 0 or 1
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        anns = gt[frame_idx==gt[:, 0]]
        if frame_idx % 10 != 0:
            continue
        if anns.size == 0:
            continue
        for ann in anns:
            patch = frame[ann[3]:ann[3]+ann[5], ann[2]:ann[2]+ann[4]]
            patch_out_dir = out_dir / f"{ann[1]}"
            patch_out_dir.mkdir(parents=True, exist_ok=True)
            patch_out_path = patch_out_dir /\
                f"{cam_dir.parts[-2]+cam_dir.parts[-1]}f{frame_idx}.png"
            cv2.imwrite(str(patch_out_path), patch)
    cap.release()


