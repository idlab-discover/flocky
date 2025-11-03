from keras.datasets import mnist
import numpy as np
from sklearn.utils import shuffle
import os

"""
This is being used to load in datasets and split them up in different files
With this method different nodes can use data from the same dataset but divided over the nodes
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# num_subsets is the number of subsets to create
# usually determined by the number of nodes in the cluster
# e.g. if there are 75 nodes, num_subsets = 75
num_subsets = 50
subset_size = len(x_train) // num_subsets

x_train, y_train = shuffle(x_train, y_train, random_state=42)
subsets = []
for i in range(num_subsets):
    start = i * subset_size
    end = start + subset_size
    x_subset = x_train[start:end]
    y_subset = y_train[start:end]
    subsets.append((x_subset, y_subset))
remaining_start = num_subsets * subset_size
if remaining_start < len(x_train):
    x_remaining = x_train[remaining_start:]
    y_remaining = y_train[remaining_start:]
    # Optionally, distribute remaining data among subsets or handle separately


for idx, (x_subset, y_subset) in enumerate(subsets):
    subset_dir = f'subsets_f{num_subsets}-e1/subset_f{idx}'
    os.makedirs(subset_dir, exist_ok=True)
    np.save(os.path.join(subset_dir, f'x_subset.npy'), x_subset)
    np.save(os.path.join(subset_dir, f'y_subset.npy'), y_subset)
