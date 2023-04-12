# Week 4

During the fourth week of the project, we tackled two main tasks: optical flow estimation and tracking. To achieve this goal, we initially implemented a block-matching method for optical flow estimation. We also tried different off-the-shelf methods such as PyFlow, RAFT, Perceiver IO, and GMFlow. Then, we implemented a tracking method based on the optical flow. Finally, we tested our tracking methods in the CVPR 2022 AI City Challenge.

* **Task 1.1**: Optical flow by block-,matching:
	* ``block_matching.py`` : functions for exhaustive and log -search block matching. There is also an example of usage in the ``main`` method.
    * ``finetune_block_matching.py`` : script for block matching hyperparameter optimization (optuna).
* **Task 1.2**: Off-the-shelf:
	* ``task1_2.py`` : script that can be used to create and evaluate the optical flow predictions of PyFlow, RAFT, Perceiver IO, and GMFlow methods (the model-based ones use the KITTI pretrained weights). The method can be defined in the script. The script provides MSEN, PEPN, mean inference time (s/frame_pair), and predicted optical flow visualization.

* **Task 1.3**: Tracking by optical flow:
	* ``task1_3.py`` : script that can be used to create a MOT format tracking file, given a detections file in the same MOT format. This script uses a folder where all the optical flows are stored in numpy files. The optical flows were predicted using the `computeOpticalFlows.py` script, where it uses the GMFlow pretrained model.


* **Task 2.1, 2.2 and 3**: Evaluate in CVPR 2022 AI City Challenge:
Several step were followed, in each task different sequences were defined in the same scripts:
	* `adaptDatasetForTraining.py`: script that was used to take different sequence videos and create a dataset in the COCO format.
	*  ``trainDetr.py`` : script that can be used to fine-tune the DETR model using weights from the COCO dataset. In this script, the dataset folder (in COCO 2017 format) can be defined. This script uses the official training script, so it is important to clone the repository before executing it:
``git clone https://github.com/facebookresearch/detr.git``
	* ``createModelDetections.py``: script that was used to generate the MOT format detections of the fine-tuned models in the test sequence videos.
	* ``computeOFtracking.py``: script that was used to obtain the tracking from the detections using the optical flow-based method. It uses the GMFlow pretrained model to predict the optical flows.
	* Evaluate tracking:
		* Clone the [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository by running `git clone https://github.com/JonathonLuiten/TrackEval.git`.
		* Create the `data/` folder with the same format as the MOT15-all challenge to evaluate.
		* Evaluate using the following command:
``python run_mot_challenge.py --GT_FOLDER /path/to/ground_truth/folder/ --TRACKERS_FOLDER /path/to/tracking/folder/ --BENCHMARK 'MOT15' --SPLIT_TO_EVAL 'all' --DO_PREPROC False``

		Make sure the paths of the folders are correct.

	* ``showTracking.py`` : script that can be used to visualize the tracking generated and create GIFs.


