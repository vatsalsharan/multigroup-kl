# multigroup-kl

This is preliminary code accompanying the paper KL Divergence Estimation with Multi-group Attribution. The code will be updated over time.

The code for the two algorithms we implement can be found as follows:

1. Node.py is the code for the multicalibration algorithm (MC).
2. multiA.py is the code for the MaxEnt/LL-KLIEP algorithm (LLK).

The code for the KLIEP and uLSIF algorithm is in MATLAB, and is directly taken from the website: http://www.ms.k.u-tokyo.ac.jp/sugi/software.html All rights to the MATLAB code for these two algorithms belong to the original developer (Masashi Sugiyama). We copy the code here for ease of reproducibility of the results using our setup.

The Python notebooks for running the experiments are the following:

1. exploration_mixture_gaussians.ipynb: We recommend users to begin with this notebook to understand their way around the algorithms and the data. This is a mixture of Gaussians experiments in 2 dimensions. Code can be found for visualizing the data, running the MC and LLK algorithm on this data, computing the overall KL-divergence, computing the KL-divergence for the subgroups, plotting the results, and visualizing the contours.

2. results_mixture_gaussians.ipynb: This is the notebook for running the mixture of Gaussians results for repeated trials, and getting the overall results.

3. mnist.ipynb: This is the notebook for running the MNIST experiment. The notebook downloads the MNIST data from scikit-learn.

The code for running the MATLAB experiments can be found in the matlab_algo/experiment_code folder. For the MNIST experiment, the data has to be generated and saved in Python, which is done in the mnist.ipynb notebook.
