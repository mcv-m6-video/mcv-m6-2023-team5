
# Week 2

In the second week of the project, our primary focus was on estimating the background of a sequence of images to detect moving objects. To achieve this goal, we initially used the Single Gaussian model to estimate the foreground, and later we improved our implementation by making the estimation adaptive. We also evaluated our results with different state-of-the-art methods. Additionally, we attempted to enhance the background estimation by using different color spaces and assessing their effectiveness.
## Available tasks

* **Task 1 (1.1, 1.2)**: Gaussian modelling:
	* ``task1.py`` : This script can be used to evaluate the single static Gaussian model. Parameters such as kernel sizes for morphological post-processing, minimum component area and the alpha parameter can be changed.
	* ``optuna_search_task1.py`` : This script can be used to do a parameter search and estimate the best values for post-processing and alpha. The number of trials can be adjusted.
* **Task 2 (2.1, 2.2)**: Adaptive Gaussian modelling
	* ``task2.py`` : This script can be used to evaluate the single adaptive Gaussian model. Parameters such as kernel sizes for morphological post-processing, minimum component area, alpha, and rho can be changed.
	* ``optuna_search_task2.py`` : This script can be used to do a parameter search and estimate the best values for alpha and rho. The number of trials can be adjusted.

* **Task 3**: Evaluate state-of-the-art
	
* **Task 4**: Colour sequences
	* ``task4.py``: This script models the background with respect to the HSV, LAB, and RGB color models. It estimates the background using all-channel or any-channel voting.
