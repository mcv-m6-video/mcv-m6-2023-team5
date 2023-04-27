# Week 5

During the fifth and final week of the project, our focus was on Multi-Target Multi-Camera (MTMC) tracking. To accomplish this, we initially used the finetuned DETR model for object detection, as we did in weeks 3 and 4. We then performed Multi-Target Single-Camera (MTSC) tracking and post-processed the results by removing static cars, as we did in previous weeks. Finally, we employed metric learning to combine the different camera tracks using various ReId methods such as Clusters or Voting to obtain our task results. Our methods and performance were evaluated in the [CVPR 2022 AI City Challenge](https://www.aicitychallenge.org/2022-data-and-evaluation/).

All the scripts have an example of usage in their respective `main` function.

Here's a brief description of the two main tasks we worked on during this week, along with the relevant scripts:

* **Task 1**: Multi-Target Single-Camera (MTSC tracking): In this task, we tracked multiple objects across frames within a single camera's view. The task involved detecting objects, tracking them, and removing static cars from the results. The relevant scripts are:
	* Detection:
		* ``createModelDetections.py`` : -   script that was used to generate the MOT format detections of the fine-tuned DETR model in the sequence videos.
	* MTSC:
		* ``sort.py``:  script (original implementation: [https://github.com/abewley/sort](https://github.com/abewley/sort)) to track and obtain the results using SORT method.
		* ``overlap_tracking.py``: script to assign an ID to each bounding box detection by using the highest overlapping of bounding boxes through frames method.
		* ``computeOFtracking.py``: script that was used to obtain the tracking from the detections using the optical flow-based method. It uses the GMFlow pretrained model to predict the optical flows.
		* ``deep_sort_app.py``: script used to obtain the tracking from deepsort project [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort))
    * Post-processing:
	    * ``postProcessMultiObject.py``:  this script is used to remove static car tracks and detections with a small area (less than 50 $pixel^2$) or those outside of the region of interest (ROI)

* **Task 2**: Multi-Target Multi-Camera (MTMC tracking): In this task, we tracked multiple objects across multiple cameras by combining the results of multiple MTSC trackers. The task involved training a model to extract feature vectors for each object and using metric learning to match objects across cameras. The relevant scripts are:
	* Metric Learning to get car feature vectors:
		* ``make_patch_dataset.py``: script to generate database from video detections.
		* ``siamese_train.py``: script to train a custom simple CNN model in a siamese way.
		*  ``train_resnet_triplet.py``:  script to Resnet18 model a using triplets.
		* ``visualizeResults.py``: visualize feature vector space learnt by the network.
	* MTMC:
		* ``computeEmbeddings.py``: script to compute the embeddings of the detections of videos using the trained network.
		* ``mtmc.py``: script to compute the MTMC using the embeddings and MTSC results.
		* ``showMtmcTracking.py``: script to visualize the results obtained in MTMC.
	* Evaluation: For evaluating our performance, we used the [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository and followed the MOT15-all challenge format. The relevant evaluation script is:
		* Clone the [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository by running `git clone https://github.com/JonathonLuiten/TrackEval.git`.
		* Create the `data/` folder with the same format as the MOT15-all challenge to evaluate.
		* Evaluate using the following command:
``python run_mot_challenge.py --GT_FOLDER /path/to/ground_truth/folder/ --TRACKERS_FOLDER /path/to/tracking/folder/ --BENCHMARK 'MOT15' --SPLIT_TO_EVAL 'all' --DO_PREPROC False``

		Make sure the paths of the folders are correct.
