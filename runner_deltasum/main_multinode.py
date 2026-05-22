 # filepath: runner/main.py
import copy
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
from multiprocessing import shared_memory
import random
import re
import sys
import traceback
print(sys.path)
import numpy as np
import signal
from typing import List
import posix_ipc
from posix_ipc import Semaphore, O_CREAT

import numpy
from threading import Thread
from queue import Queue, Empty
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn.metrics.pairwise as pairwise
import torch.nn.functional as F
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)  # if using GPU
np.random.seed(1234)  # numpy consistency
import math
import json
import platform
import warnings
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger.info("[Python] Starting execution...")
sys.stdout.reconfigure(line_buffering=True)
if platform.system() == "Windows":
    raise RuntimeError("This code requires a POSIX-compatible system")

# Attempt to configure PyTorch to not use GPUs
try:
    device = torch.device('cuda')
    logger.info(f"[DEBUG] Using device: {device}")
except RuntimeError as e:
    logger.error(f"[ERROR] Could not configure device: {e}")
    pass # Hopes CUDA_VISIBLE_DEVICES worked

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"posix_ipc version: {posix_ipc.__version__}") # Or however its version is exposed

# ——————————————————————————————————————————————————————————————
#  Semaphore & Shared‑Memory Setup
# ——————————————————————————————————————————————————————————————
def load_config(file="config.json"):
    """Load configuration from a JSON file."""
    with open(file, "r") as f:
        content = f.read()
        # Replace JavaScript-style booleans and null with Python-compatible values
    content = (
        content.replace("true", "True")
        .replace("false", "False")
        .replace("null", "None")
    )
    try:
        config = eval(content)
    except SyntaxError as e:
        raise ValueError(f"Failed to parse the JSON file: {file}. Error: {e}")
    #logger.info(f"Loaded config: {config}")
    return config

#gossip_config = load_config("config.json")
#config_file = sys.argv[1] if len(sys.argv) > 1 else "defaultConfig.json"
#logger.info("loading arg %s", config_file)
#config = load_config(config_file)

# Example
NODE_COUNT = 75
MSG_DROPOUT = 0.7
DIV_STATIC = 0.30 # default 0.15
DIV_ROUNDS = 570.0 # default 1000
GOSSIP_BATCH_SIZE: int = 10 # Number of batches after which to perform gossip averaging
RESULTS_PREFIX = "CT_CONN_DEFAULT_N"
SUBSETS = "subsets_default"
TOPOS = "ct_conn"

logger.info("NODE_COUNT: %d", NODE_COUNT)
incoming_counter = 0
#node_id = config["nodeID"]

# def get_weight_info(model):
#     """Get weight information from PyTorch model"""
#     weights = []
#     shapes = []
#     dtypes = []
#     sizes = []
    
#     for param in model.parameters():
#         weights.append(param.data.cpu().numpy())  # Move to CPU and convert to numpy
#         shapes.append(param.data.shape)
#         dtypes.append(str(param.data.dtype))
#         sizes.append(param.data.cpu().numpy().nbytes)
    
#     total_size = sum(sizes)
#     return shapes, dtypes, sizes, total_size

def average_weights(weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    # assert this
    if len(weights) == 0:
        return []
    elif len(weights) == 1:
        return weights[0]
    else:
        ret_weights = copy.deepcopy(weights[0])
        for layer_idx in range(len(ret_weights)):
            layer_stack = np.stack([m[layer_idx] for m in weights], axis=0)
            # Compute the mean along the first axis (the one that stacks the models)
            avg_layer = np.mean(layer_stack, axis=0)
            ret_weights[layer_idx] = avg_layer
        return ret_weights

def average_weights_alpha(current_weights: List[np.ndarray], loaded_weights: List[np.ndarray]) -> List[np.ndarray]:
    alpha = 0.5
    # Debug: Verify shapes match
    assert len(current_weights) == len(
        loaded_weights
    ), "Weight lists have different lengths"
    for i, (w1, w2) in enumerate(zip(current_weights, loaded_weights)):
        assert (
            w1.shape == w2.shape
        ), f"Shape mismatch at layer {i}: {w1.shape} vs {w2.shape}"
    # Efficient vectorized averaging using numpy
    f1 = 1.0 - alpha
    return [((w1 * f1) + (w2 * alpha)) for w1, w2 in zip(current_weights, loaded_weights)]

def sum_weights(loaded_weights: List[np.ndarray], current_weights: List[np.ndarray]) -> List[np.ndarray]:
    # Debug: Verify shapes match
    assert len(current_weights) == len(
        loaded_weights
    ), "Weight lists have different lengths"
    for i, (w1, w2) in enumerate(zip(current_weights, loaded_weights)):
        assert (
            w1.shape == w2.shape
        ), f"Shape mismatch at layer {i}: {w1.shape} vs {w2.shape}"
    # Efficient vectorized averaging using numpy
    return [(w1 + w2) for w1, w2 in zip(current_weights, loaded_weights)]

def weight_diff(matrix1: List[np.ndarray], matrix2: List[np.ndarray]) -> List[np.ndarray]:
    # Debug: Verify shapes match
    assert len(matrix2) == len(
        matrix1
    ), "Weight lists have different lengths"
    for i, (w1, w2) in enumerate(zip(matrix1, matrix2)):
        assert (
            w1.shape == w2.shape
        ), f"Shape mismatch at layer {i}: {w1.shape} vs {w2.shape}"
    # Efficient vectorized averaging using numpy
    return [(w1 - w2) for w1, w2 in zip(matrix1, matrix2)]

def average_weights_n(loaded_weights, n: int):
    divider = float(n)
    # Efficient vectorized averaging using numpy
    return [(w1/divider) for w1 in loaded_weights]

def mult_weights(matrix: List[np.ndarray], mult: float) -> List[np.ndarray]:
    return [(np.multiply(w1,mult)) for w1 in matrix]

def update_corrected_model_averaging(
        models: dict[str,List[List[np.ndarray]]]
) -> List[np.ndarray]:
    return []

def variance_corrected_model_averaging(
    model_buffer: List[List[np.ndarray]], 
    eps: float = 1e-6
) -> List[np.ndarray]:
    """
    Implements Algorithm 3 (Variance-Corrected Model Averaging).
    Args:
        model_buffer: List of N models, where each model is itself a list of L 
                      numpy arrays (the layer weight tensors).
        eps:          Small constant to avoid division by zero in variance ratios.
    Returns:
        A new list of L numpy arrays corresponding to the variance-corrected
        averaged model.
    """
    N = len(model_buffer)
    if N == 0:
        raise ValueError("Model buffer must contain at least one model.")
    
    # Number of layers (L)
    # Assuming all models in the buffer have the same architecture (same number of layers)
    L = len(model_buffer[0])
    
    # Stack all models into arrays for efficient computation
    avg_model_layers: List[np.ndarray] = []
    for layer_idx in range(L):
        # Stack the current layer from all models in the buffer
        # This creates an array of shape (N, layer_shape_dims...)
        layer_stack = np.stack([m[layer_idx] for m in model_buffer], axis=0)
        # Compute the mean along the first axis (the one that stacks the models)
        avg_layer = np.mean(layer_stack, axis=0)
        avg_model_layers.append(avg_layer)
    
    corrected_model_layers: List[np.ndarray] = []
    
    # Iterate through each layer (l_avg) in the averaged model
    for layer_idx in range(L):
        l_avg = avg_model_layers[layer_idx] # Current layer from Model_avg
        
        # Calculate sigma_li^2: Variance of layer l in model i
        # This will be a list of N variances, one for each model in the buffer for the current layer
        variances_of_layer_across_models: List[float] = []
        for model_i in model_buffer:
            l_i = model_i[layer_idx] # Layer l from model i
            variances_of_layer_across_models.append(np.var(l_i))
        
        # Calculate sigma_l^2: Target variance for the current layer
        # This is the mean of the variances calculated above
        target_variance_for_layer_l = np.mean(variances_of_layer_across_models)
        
        # Calculate sigma_lavg^2: Variance of the current averaged layer l_avg
        variance_of_avg_layer = np.var(l_avg)
        
        # Calculate mean of the current averaged layer l_avg (referred to as l_avg_mean)
        mean_of_avg_layer = np.mean(l_avg)
        
        # Compute scaling factor: sigma_l / sigma_lavg
        # (Using square root of variances to get standard deviations)
        # Add eps to avoid division by zero or issues with zero variance
        scaling_factor = np.sqrt(target_variance_for_layer_l / (variance_of_avg_layer + eps))
        
        # Rescale each weight v in l_avg:
        # v_corrected = (v - mean_of_avg_layer) * scaling_factor + mean_of_avg_layer
        # This can be done element-wise on the entire layer array using NumPy
        corrected_layer = (l_avg - mean_of_avg_layer) * scaling_factor + mean_of_avg_layer
        corrected_model_layers.append(corrected_layer)
        
    return corrected_model_layers

def load_subset(path):
    x_path = os.path.join(path, "x_subset.npy")
    y_path = os.path.join(path, "y_subset.npy")
    if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
        raise FileNotFoundError(f"Could not find subset files in {path}")
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y

def to_categorical(y, num_classes=10):
    """Convert class labels to one-hot encoded vectors"""
    y = y.astype(int)
    y_categorical = np.zeros((len(y), num_classes))
    y_categorical[np.arange(len(y)), y] = 1
    return y_categorical

def log_metrics(
    id: str, timestamp, acc, prec, rec, f1, epoch, iter, batch, val_acc, val_loss, test_acc, test_loss, lr=0.001, path="metrics.csv"
):
    path = RESULTS_PREFIX + str(NODE_COUNT) + "_"  + str(MSG_DROPOUT) + "_" + str(GOSSIP_BATCH_SIZE) + "_" + path
    row = [id, epoch, batch, iter, acc, prec, rec, f1, timestamp, lr, val_acc, val_loss, test_acc, test_loss]
    line = ",".join(map(str, row))
    # This write() is atomic on Linux as long as len(line) < PIPE_BUF (≈4096)
    with open(path, "a", newline="") as f:
        f.write(line + "\n")

def init_metrics(path="metrics.csv"):
    path = RESULTS_PREFIX + str(NODE_COUNT) + "_"  + str(MSG_DROPOUT) + "_" + str(GOSSIP_BATCH_SIZE) + "_" + path
    with open(path, "a", newline="") as f:
        f.write("id,epoch,batch,iteration,accuracy,precision,recall,f1,timestamp,lr,val_acc,val_loss,test_acc,test_loss\n")


# ———————————————————————————————————————————————————————————————————
# Batch update processing functions
#  
# ———————————————————————————————————————————————————————————————————
class WeightUpdate: 
    def __init__(self, nodeId: str, baseWeights: List[np.ndarray], update: List[np.ndarray]):
        self.nodeid = nodeId
        self.baseweights = baseWeights
        self.update = update

class Metrics: 
    def __init__(self, accu, prec, rec, f1, test_loss, test_acc, val_loss, val_acc):
        self.accu = accu
        self.prec = prec
        self.rec = rec
        self.f1 = f1
        self.test_loss = test_loss
        self.test_acc = test_acc
        self.val_loss = val_loss
        self.val_acc = val_acc

def calc_metrics(model, x_test, y_test, x_val, y_val) -> Metrics:
    """Calculate metrics for PyTorch model"""
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Test predictions
        #x_test_tensor = torch.FloatTensor(x_test).to(device)
        #y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        test_outputs = model(x_test)
        y_test_pred = test_outputs.argmax(axis=1)
        y_test_true = y_test.argmax(axis=1)
        
        # Validation predictions
        x_val_tensor = torch.FloatTensor(x_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        val_outputs = model(x_val_tensor)
        y_val_pred = val_outputs.argmax(axis=1)
        y_val_true = y_val_tensor.argmax(axis=1)
        
        # Convert to numpy for sklearn metrics
        y_test_pred_np = y_test_pred.cpu().numpy()
        y_test_true_np = y_test_true.cpu().numpy()
        y_val_pred_np = y_val_pred.cpu().numpy()
        y_val_true_np = y_val_true.cpu().numpy()
        
        # Calculate metrics
        acc = accuracy_score(y_test_true_np, y_test_pred_np)
        prec = precision_score(
            y_test_true_np, y_test_pred_np, average="macro", zero_division=0
        )
        rec = recall_score(
            y_test_true_np, y_test_pred_np, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test_true_np, y_test_pred_np, average="macro", zero_division=0
        )
        
        # Loss calculation
        test_loss = -F.cross_entropy(test_outputs, y_test_tensor).item()
        val_loss = -F.cross_entropy(val_outputs, y_val_tensor).item()
        test_acc = accuracy_score(y_test_true_np, y_test_pred_np)
        val_acc = accuracy_score(y_val_true_np, y_val_pred_np)
        
    return Metrics(acc, prec, rec, f1, test_loss, test_acc, val_loss, val_acc)

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)
  def exit_gracefully(self, signum, frame):
    logger.error(f"Signal detected: {signum}")
    self.kill_now = True

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv1.weight = torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1.bias = nn.init.zeros_(self.conv1.bias)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        #self.dropout1.manual_seed(1234)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2.weight  = torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2.bias = nn.init.zeros_(self.conv2.bias)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)
        #self.dropout1.manual_seed(1234)
        self.dropout3 = nn.Dropout(0.2)
        #self.dropout1.manual_seed(1234)
        self.fc1 = nn.Linear(32 * 7 * 7, 32)  # After flattening
        self.fc1.weight  = torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias = nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(32, 10)
        self.fc2.weight  = torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias = nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# Helper functions for PyTorch model
def get_model_weights_numpy(model):
    """Get model weights as numpy arrays"""
    weights = []
    for param in model.parameters():
        weights.append(copy.deepcopy(param.data.cpu().numpy()))
    return weights

def set_model_weights_numpy(model, weights):
    """Set model weights from numpy arrays"""
    for i, param in enumerate(model.parameters()):
        param.data = torch.FloatTensor(weights[i]).to(device)

def compute_weight_diff_numpy(new_weights, old_weights):
    """Compute difference between new and old weights"""
    return [(nw - ow) for nw, ow in zip(new_weights, old_weights)]

# Initialize model and move to device
#model = CNNModel()
#model.to(device)

def l2_norm(update: List[np.ndarray]) -> float:
    s = 0.0
    for w in update:
        s += np.sum(w * w)
    return np.sqrt(s)

def ema_update(old, new, beta):
    if old is None:
        return [w.copy() for w in new]
    return [(beta * o + (1 - beta) * n) for o, n in zip(old, new)]

def clip_update(update: List[np.ndarray], tau: float) -> List[np.ndarray]:
    # Compute global L2 norm
    norm_sq = 0.0
    for w in update:
        norm_sq += np.sum(w * w)
    norm = np.sqrt(norm_sq)

    # Clip if needed
    if norm > tau:
        scale = tau / (norm + 1e-12)
        return [w * scale for w in update]
    return update

def l2_distance(update_a, update_b):
    dist = 0.0
    for a, b in zip(update_a, update_b):
        diff = a - b
        dist += np.sum(diff * diff)
    return np.sqrt(dist)

def compute_update_stats(gossip_list):
    norms = []

    # Collect norms of all "effective updates"
    for node_name in gossip_list:
        first = gossip_list[node_name][0]
        last = gossip_list[node_name][-1]

        # same one as in delta update
        delta = sum_weights(
            weight_diff(last.baseweights, first.baseweights),
            last.update
        )

        norms.append(l2_norm(delta))

    if len(norms) == 0:
        return 1.0, 1.0  # fallback

    norms = np.array(norms)

    # stats
    median = np.median(norms)
    p75 = np.percentile(norms, 75)

    return median, p75

class NodeSimulator:
    def __init__(self, number, config_file, basemodel):
        """
        Initialize a node simulator with its configuration
        
        Args:
            node_config: Dict containing node configuration
        """
        node_config = load_config(config_file)
        self.node_id = node_config["nodeID"]
        self.max_ping = node_config["maxPing"]
        self.cheaty_map = node_config["cheatyMinimalPingMap"]
        self.weight_queue = Queue()
        self.current_weights = []
        self.previous_weights = []
        self.delta_momentum = None
        self.node_number = number
        self.model = copy.deepcopy(basemodel) #CNNModel()
        self.x_train = torch.FloatTensor()
        self.x_test = torch.FloatTensor()
        self.y_train = []
        self.y_test = []
        self.cosine_sim = []
        self.global_histo = []
        self.local_histo = []
        self.num_batches = 0
        self.optimizer = [] #optim.Adam()
        self.previous_weights = []
        self.x_train_split = torch.Tensor()
        self.x_val_split = torch.Tensor()
        self.y_train_split = torch.Tensor()
        self.y_val_split = torch.Tensor()
        self.gossip_list: dict[str, List[WeightUpdate]] = {}
    
    def load_model_and_data(self, prefix, batch_size):
        self.model.to(device)
        # Load data
        x_test, y_test = load_subset(f"{SUBSETS}/subsets_f{prefix}-e1/mnist")  # MNIST dataset
        x_train, y_train = load_subset(f"{SUBSETS}/subsets_f{prefix}-e1/subset_{self.node_id}")
        
        # Convert to PyTorch tensors
        self.x_train = torch.FloatTensor(x_train).reshape(-1, 28, 28) / 255.0
        self.x_test = torch.FloatTensor(x_test).reshape(-1, 28, 28) / 255.0
        
        # Convert labels to one-hot encoding
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)
        logger.info("Node: %s", self.node_id)
        #logger.info(y_test)
        self.global_histo, bin_edges = numpy.histogram(y_test, bins=10, range=(0.,10.))
        #logger.info(self.global_histo)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.previous_weights = get_model_weights_numpy(self.model)
        # Print model architecture
        #logger.info(self.model)

    def split_test_dataset(self):
        train_size = int(0.8 * len(self.x_train))
        val_size = len(self.x_train) - train_size
        self.x_train_split, self.x_val_split = torch.split(self.x_train, [train_size, val_size])
        self.y_train_split, self.y_val_split = torch.split(torch.FloatTensor(self.y_train), [train_size, val_size])
        self.num_batches = math.ceil(self.x_train_split.shape[0] / batch_size)
        
        y_val_arr = torch.argmax(self.y_val_split, dim=1).float()
        #logger.info(y_val_arr)
        self.local_histo, bin_edges = torch.histogram(y_val_arr, bins=10, range=(0.,10.))
        #logger.info(self.local_histo)

        global_arr = self.global_histo #/ NODE_COUNT
        local_arr = self.local_histo.numpy()
        self.cosine_sim = pairwise.cosine_similarity([global_arr], [local_arr])[0][0]
        logger.info(self.cosine_sim)

    def get_nodes_in_range(self):
        """Determine which other nodes should communicate with this node"""
        in_range_nodes = []
        
        for neighbor_id, ping_time in self.cheaty_map.items():
            if ping_time <= self.max_ping:
                in_range_nodes.append(neighbor_id)
        
        return in_range_nodes
    
    def receive_weight_update(self, weight_update):
        """Receive weight update and enqueue it"""
        self.weight_queue.put(weight_update)
    
    def get_next_weight_update(self):
        """Get next weight update from queue"""
        if not self.weight_queue.empty():
            try:
                return self.weight_queue.get_nowait()
            except:
                return None
        return None

try:
    #killer = GracefulKiller()
    init_metrics()
    # ——————————————————————————————————————————————————————————————
    #  Build and Train Model
    # ——————————————————————————————————————————————————————————————
    # Load data
    #x_test, y_test = load_subset(f"subsets_{prefix}/mnist")  # MNIST dataset
    #x_train, y_train = load_subset(f"subsets_{prefix}/subset_{node_id}")
    
    # Convert to PyTorch tensors
    #x_train = torch.FloatTensor(x_train).reshape(-1, 28, 28) / 255.0
    #x_test = torch.FloatTensor(x_test).reshape(-1, 28, 28) / 255.0
    
    # Convert labels to one-hot encoding
    #y_train = to_categorical(y_train, 10)
    #y_test = to_categorical(y_test, 10)
    batch_size = 64 
    nodes: dict[str, NodeSimulator] = {}
    basemodel = CNNModel()
    for i in range(NODE_COUNT):
        config = f"{TOPOS}/f{NODE_COUNT}-e1/fog{i}.json"
        node_sim = NodeSimulator(i, config, basemodel)
        node_sim.load_model_and_data(NODE_COUNT, batch_size)
        node_sim.split_test_dataset()
        nodes[node_sim.node_id] = node_sim 

    for i in nodes:
        print(nodes[i].cosine_sim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    
    # ——————————————————————————————————————————————————————————————
    #  Get weights and metadata on the weights
    #  (and write to shared memory)
    #  This is once per model, not per epoch
    #  (unless you change the model architecture)
    # ——————————————————————————————————————————————————————————————
    # Prepare shared memory once, based on initial weights
    #current_weights = get_model_weights_numpy(model)
    
    # Print the model summary to see its architecture
    #logger.info("PyTorch CNN Model:")
    #logger.info(model)
    
    # ——————————————————————————————————————————————————————————————
    #  Create a thread to read and send weights from Go
    #  Create queue to communicate between threads
    # ——————————————————————————————————————————————————————————————
    #send_queue = Queue()
    #receive_queue: Queue[WeightUpdate] = Queue()
    # Threads = []
    # t_read_weights = Thread(
    #     target=read_weights,
    #     args=(receive_queue,),
    #     name="read_weights_thread",
    #     daemon=False,
    # )
    # Threads.append(t_read_weights)
    # t_read_weights.start()
    # t_write_weights = Thread(
    #     target=exchange_weights_with_go,
    #     args=(send_queue, model),  # Pass model to the thread
    #     name="write_weights_thread",
    #     daemon=False,
    # )
    # Threads.append(t_write_weights)
    # t_write_weights.start()
    

    ###################################################################
    #  Train the model
    ###################################################################
    # Split the data into training and validation sets
    #train_size = int(0.8 * len(x_train))
    #val_size = len(x_train) - train_size
    #x_train_split, x_val_split = torch.split(x_train, [train_size, val_size])
    #y_train_split, y_val_split = torch.split(torch.IntTensor(y_train), [train_size, val_size])
    
    num_epochs = 200
    num_gossips = 50

    gossip_round = 0
    batching_counter = 0
    epoch_loss = 0
    epoch_acc = 0
    #gossip_list: dict[str, List[WeightUpdate]] = {}
    #model_list: List[List[np.ndarray]] = [] # List to hold models for averaging

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        for idx in nodes:
            node = nodes[idx]
            logger.info(f"Train node {node.node_id}")
            epoch_loss = 0
            epoch_acc = 0
            
            # Create data loader
            train_dataset = TensorDataset(node.x_train_split, node.y_train_split)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            
            node.model.train()  # Set model to training mode
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.unsqueeze(1).to(device)  # Add channel dimension and move to device
                y_batch = y_batch.to(device)  # Convert one-hot to class labels
                
                # Forward pass
                outputs = node.model(x_batch)
                #print(outputs)
                #print(y_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                node.optimizer.zero_grad()
                loss.backward()
                node.optimizer.step()
                
                epoch_loss += loss.item()
                # Calculate accuracy for this batch
                predicted = outputs
                actual = y_batch
                #print(predicted[0])
                #print(actual[0])
                epoch_acc += (predicted == actual).sum().item()
            
            avg_loss = epoch_loss / node.num_batches
            avg_acc = epoch_acc / len(train_dataset)  # Overall accuracy
            
            # Calculate metrics for test set
            node.model.eval()
            with torch.no_grad():
                x_test_tensor = torch.FloatTensor(node.x_test).reshape(-1, 28, 28).unsqueeze(1).to(device)
                y_test_tensor = torch.argmax(torch.FloatTensor(node.y_test), dim=1).to(device)
                
                test_outputs = node.model(x_test_tensor)
                y_test_pred = test_outputs.argmax(axis=1)
                y_test_true = y_test_tensor
                
                # Convert to numpy for sklearn metrics
                y_test_pred_np = y_test_pred.cpu().numpy()
                y_test_true_np = y_test_true.cpu().numpy()
                
                test_loss = criterion(test_outputs, y_test_tensor).item()
                acc = accuracy_score(y_test_true_np, y_test_pred_np)
                prec = precision_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                rec = recall_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                f1 = f1_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                
                ts = time.time()
                
                # Validation metrics
                x_val_tensor = torch.FloatTensor(node.x_val_split).unsqueeze(1).to(device)
                y_val_tensor = torch.argmax(node.y_val_split, dim=1).to(device)
                
                val_outputs = node.model(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                y_val_pred = val_outputs.argmax(axis=1)
                
                val_acc = (y_val_pred == y_val_tensor).sum().item() / len(x_val_tensor)
                #str, timestamp, acc, prec, rec, f1, epoch, iter, batch, val_acc, val_loss, test_acc, test_loss, lr=0.001
                log_metrics(node.node_id, ts, acc, prec, rec, f1, epoch, -1, batching_counter, val_acc, val_loss, acc, test_loss, 0.001)
                logger.info(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
                logger.info(f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")
                logger.info(f"Epoch {epoch+1} - Test loss: {test_loss:.4f}, Test accuracy: {acc:.4f}")
                
                # Extract weights at the end of epoch
                current_weights = get_model_weights_numpy(node.model)
                
                # 2. Hand them off to Go (and wait for Go to read)
                # exchange_weights_with_go(current_weights)
                neighbours = node.get_nodes_in_range()
                diff = compute_weight_diff_numpy(current_weights, node.previous_weights)
                
                for neigh_node in neighbours:
                    if neigh_node != "e0" and random.random() > MSG_DROPOUT:
                        neighbour = nodes[neigh_node]
                        neighbour.weight_queue.put(
                            WeightUpdate(node.node_id, node.previous_weights, diff)
                        )  # Send weights to the thread for writing to shared memory
            

        ##################################################################
        # Delta corrected averaging normal
        ##################################################################
        incoming_counter += 1
        # if len(incoming_weights) > 0:
        gossip_round += 1
        
        for idx in nodes:
            node = nodes[idx]
            logger.info(f"Process weight updates node {node.node_id}")
            while not node.weight_queue.empty():
                try:
                    wupdate = node.weight_queue.get_nowait()
                    #incoming_weights.append(incoming_weight)
                    logger.info(f"Weight update from {wupdate.nodeid}")
                    if wupdate.nodeid in node.gossip_list:
                        node.gossip_list[wupdate.nodeid].append(wupdate)
                    else:
                        node.gossip_list[wupdate.nodeid] = [wupdate]
                except Empty:
                    pass
            
            local_diff = compute_weight_diff_numpy(get_model_weights_numpy(node.model), node.previous_weights)
            localupdate = WeightUpdate("local", node.previous_weights, local_diff)
            if "local" in node.gossip_list:
                logger.info(f"Weight update from local")
                node.gossip_list["local"].append(localupdate)
            else:
                logger.info(f"Weight update from local")
                node.gossip_list["local"] = [localupdate]
        
        # success = batch_avg_normal(gossip_list, incoming_counter)
        if incoming_counter == 200 or (incoming_counter % GOSSIP_BATCH_SIZE == 0 and incoming_counter > 0):
            
            for idx in nodes:
                node = nodes[idx]
                logger.info(f"Average weight updates updates node {node.node_id}")

                # If we have enough models, perform averaging
                if len(node.gossip_list) >= 1:  # pretty useless otherwise
                    locallist = node.gossip_list["local"]
                    local_delta = sum_weights(
                            weight_diff(locallist[-1].baseweights, locallist[0].baseweights),
                            locallist[-1].update
                        )

                    # --- compute stats once per integration ---
                    median_norm, p75_norm = compute_update_stats(node.gossip_list)
                    iqr = p75_norm - median_norm
                    TAU_CLIP = median_norm + 0.9 * iqr
                    ADAPT_C = 1.0 / (median_norm + iqr + 1e-12)
                    TRUST_WEIGHT = 1.0
                    EMA_BETA = 0.5

                    logger.info(f"TAU_CLIP={TAU_CLIP:.6f}, ADAPT_C={ADAPT_C:.6f}, MEDIAN_NORM={median_norm:.6f}, IQR={iqr:.6f}")

                    #DELTA SUM
                    logger.info("Performing delta averaging of weights...")
                    model_list: List[List[np.ndarray]] = []

                    for node_name in node.gossip_list:
                        logger.info(f"NODE_NAME: {node_name} LEN {len(node.gossip_list[node_name])}")
                        wupdate = node.gossip_list[node_name][0]
                        model_list.append(wupdate.baseweights)

                    new_weights = average_weights(model_list)

                    for node_name in node.gossip_list:
                        first = node.gossip_list[node_name][0]
                        last = node.gossip_list[node_name][-1]

                        delta = sum_weights(
                            weight_diff(last.baseweights, first.baseweights),
                            last.update
                        )

                        # Clip
                        delta = clip_update(delta, TAU_CLIP)

                        # Adaptive scaling
                        disagree = l2_distance(delta, local_delta)
                        adapt = 1.0 / (1.0 + TRUST_WEIGHT * ADAPT_C * disagree)

                        divider = DIV_STATIC + float(incoming_counter) / DIV_ROUNDS
                        delta = mult_weights(delta, divider * adapt)
                        logger.info(f"disagree={disagree:.6f},divider={divider:.6f}, adapt param={adapt:.6f}, effective delta={divider * adapt:.6f}")

                        # EMA
                        node.delta_momentum = ema_update(node.delta_momentum, delta, EMA_BETA)
                        #print(node.delta_momentum)
                        #print(delta)
                        new_weights = sum_weights(new_weights, node.delta_momentum)
                    
                    #DELTA SUM
                    # logger.info("Performing delta averaging of weights...")
                    # model_list: List[List[np.ndarray]] = []
                    # for node_name in node.gossip_list:
                    #     wupdate = node.gossip_list[node_name][0]
                    #     model_list.append(wupdate.baseweights)
                    
                    # new_weights = average_weights(model_list)
                    
                    # for node_name in node.gossip_list:
                    #     first = node.gossip_list[node_name][0]
                    #     last = node.gossip_list[node_name].pop()
                    #     delta = sum_weights(weight_diff(last.baseweights, first.baseweights), last.update)
                    #     divider = DIV_STATIC + float(incoming_counter) / DIV_ROUNDS
                    #     delta = mult_weights(delta, divider)
                    #     new_weights = sum_weights(new_weights, delta)
                    
                    #AVERAGING
                    # logger.info("Performing batch averaging of weights...")
                    # model_list: List[List[np.ndarray]] = []
                    # for node_name in node.gossip_list:
                    #     wupdate = node.gossip_list[node_name].pop()
                    #     model_list.append(sum_weights(wupdate.baseweights, wupdate.update))
                    
                    # new_weights = average_weights(model_list)

                    # Apply averaged weights to model
                    set_model_weights_numpy(node.model, new_weights)
                    logger.info(f"Applied {len(node.gossip_list)} new weights to model.")
                    model_list.clear()
                    node.gossip_list.clear()
                    
                    # Calculate and log metrics
                    with torch.no_grad():
                        x_test_tensor = torch.FloatTensor(node.x_test).reshape(-1, 28, 28).unsqueeze(1).to(device)
                        y_test_tensor = torch.argmax(torch.FloatTensor(node.y_test), dim=1).to(device)
                        
                        test_outputs = node.model(x_test_tensor)
                        y_test_pred = test_outputs.argmax(axis=1)
                        y_test_true = y_test_tensor
                        
                        # Convert to numpy for sklearn metrics
                        y_test_pred_np = y_test_pred.cpu().numpy()
                        y_test_true_np = y_test_true.cpu().numpy()
                        
                        test_loss = criterion(test_outputs, y_test_tensor).item()
                        acc = accuracy_score(y_test_true_np, y_test_pred_np)
                        prec = precision_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                        rec = recall_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                        f1 = f1_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                        
                        ts = time.time()
                        
                        # Validation metrics
                        x_val_tensor = torch.FloatTensor(node.x_val_split).unsqueeze(1).to(device)
                        y_val_tensor = torch.argmax(node.y_val_split, dim=1).to(device)
                        
                        val_outputs = node.model(x_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                        y_val_pred = val_outputs.argmax(axis=1)
                        
                        val_acc = (y_val_pred == y_val_tensor).sum().item() / len(x_val_tensor)
                        #str, timestamp, acc, prec, rec, f1, epoch, iter, batch, val_acc, val_loss, test_acc, test_loss, lr=0.001
                        log_metrics(node.node_id, ts, acc, prec, rec, f1, epoch, gossip_round, batching_counter, val_acc, val_loss, acc, test_loss, 0.001)

                    logger.info(f"Batch {batching_counter} accuracy: {acc:.4f}")
                
            batching_counter += 1
        
        for idx in nodes:
            node = nodes[idx]
            node.previous_weights = get_model_weights_numpy(node.model)
        
        #time.sleep(1)  # Sleep for 1 second
    
    ##################################################################
    #  Final evaluation
    ##################################################################
    for idx in nodes:
        node = nodes[idx]
        logger.info(f"Final eval for {node.node_id}")
        node.model.eval()
        with torch.no_grad():
            x_test_tensor = torch.FloatTensor(node.x_test).reshape(-1, 28, 28).unsqueeze(1).to(device)
            y_test_tensor = torch.argmax(torch.FloatTensor(node.y_test), dim=1).to(device)
            
            test_outputs = node.model(x_test_tensor)
            y_test_pred = test_outputs.argmax(axis=1)
            y_test_true = y_test_tensor
            
            # Convert to numpy for sklearn metrics
            y_test_pred_np = y_test_pred.cpu().numpy()
            y_test_true_np = y_test_true.cpu().numpy()
            
            acc = accuracy_score(y_test_true_np, y_test_pred_np)
            prec = precision_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
            rec = recall_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
            f1 = f1_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
            
            ts = time.time()
            
            # Validation metrics
            x_val_tensor = torch.FloatTensor(node.x_val_split).unsqueeze(1).to(device)
            y_val_tensor = torch.argmax(node.y_val_split, dim=1).to(device)
            
            val_outputs = node.model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            y_val_pred = val_outputs.argmax(axis=1)
            
            val_acc = (y_val_pred == y_val_tensor).sum().item() / len(x_val_tensor)
            
            # Test metrics
            test_loss = criterion(test_outputs, y_test_tensor).item()
            test_acc = (y_test_pred == y_test_true).sum().item() / len(x_test_tensor)
            
            log_metrics(node.node_id, ts, acc, prec, rec, f1, -1, 0, batching_counter, val_acc, val_loss, test_acc, test_loss, 0.001)
            
            logger.info("Test loss: %f", test_loss)
            logger.info("Test accuracy: %f", test_acc)
            logger.info("[PROCESSING] Weights received from Go, metrics:")
    
    # --- BEGIN CONTINUOUS GOSSIP AND AVERAGING PHASE ---
    logger.info("Entering continuous gossip and averaging phase...")
    try:
        for epoch in range(num_gossips):
            # weights_to_send_to_go = None
            # incoming_weights: List[WeightUpdate] = []
            # while not receive_queue.empty():
            #     try:
            #         incoming_weight = receive_queue.get_nowait()
            #         incoming_weights.append(incoming_weight)
            #     except Empty:
            #         pass
            for idx in nodes:
                node = nodes[idx]
                logger.info(f"Process weight updates node {node.node_id}")
                while not node.weight_queue.empty():
                    try:
                        wupdate = node.weight_queue.get_nowait()
                        #incoming_weights.append(incoming_weight)
                        logger.info(f"Weight update from {wupdate.nodeid}")
                        if wupdate.nodeid in node.gossip_list:
                            node.gossip_list[wupdate.nodeid].append(wupdate)
                        else:
                            node.gossip_list[wupdate.nodeid] = [wupdate]
                    except Empty:
                        pass
                

            for idx in nodes:
                node = nodes[idx]
                logger.info(f"Gossip only eval for {node.node_id}")
                
                #if len(incoming_weights) > 0:
                #    logger.info("GOSSIP PHASE: Received weights from Go.")
                #    for wupdate in incoming_weights: 
                #        gossip_list[wupdate.nodeid] = [wupdate]

                model_list = []  
                for node_name in node.gossip_list:
                    wupdate = node.gossip_list[node_name].pop()
                    total = wupdate.baseweights #sum_weights(wupdate.baseweights, wupdate.update)
                    model_list.append(total)
                
                model_list.append(get_model_weights_numpy(node.model))  # Add the current model weights to the list
                
                new_averaged_weights = average_weights(model_list)
                    
                # Apply averaged weights to model
                set_model_weights_numpy(node.model, new_averaged_weights)
                logger.info("GOSSIP PHASE: Applied averaged weights to model.")
                #weights_to_send_to_go = [new_averaged_weights, get_previous_weights_numpy()]
                model_list.clear()
                node.gossip_list.clear()
            
                with torch.no_grad():
                    x_test_tensor = torch.FloatTensor(node.x_test).reshape(-1, 28, 28).unsqueeze(1).to(device)
                    y_test_tensor = torch.argmax(torch.FloatTensor(node.y_test), dim=1).to(device)
                    
                    test_outputs = node.model(x_test_tensor)
                    y_test_pred = test_outputs.argmax(axis=1)
                    y_test_true = y_test_tensor
                    
                    # Convert to numpy for sklearn metrics
                    y_test_pred_np = y_test_pred.cpu().numpy()
                    y_test_true_np = y_test_true.cpu().numpy()
                    
                    test_loss = criterion(test_outputs, y_test_tensor).item()
                    acc = accuracy_score(y_test_true_np, y_test_pred_np)
                    prec = precision_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                    rec = recall_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                    f1 = f1_score(y_test_true_np, y_test_pred_np, average="macro", zero_division=0)
                    
                    ts = time.time()
                    
                    # Validation metrics
                    x_val_tensor = torch.FloatTensor(node.x_val_split).unsqueeze(1).to(device)
                    y_val_tensor = torch.argmax(node.y_val_split, dim=1).to(device)
                    
                    val_outputs = node.model(x_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    y_val_pred = val_outputs.argmax(axis=1)
                    
                    val_acc = (y_val_pred == y_val_tensor).sum().item() / len(x_val_tensor)
                    #str, timestamp, acc, prec, rec, f1, epoch, iter, batch, val_acc, val_loss, test_acc, test_loss, lr=0.001
                    log_metrics(node.node_id, ts, acc, prec, rec, f1, -2, gossip_round, -1, val_acc, val_loss, acc, test_loss, 0.001)
                
                logger.info(f"GOSSIP PHASE: Validation after averaging - Loss: {val_loss:.4f}, "
                        f"Accuracy: {val_acc:.4f}")
                logger.info(f"GOSSIP PHASE: Validation after averaging - Test Loss: {test_loss:.4f}, "
                        f"Accuracy: {acc:.4f}")
                
            #else:
                # No new weights from Go, so we'll send our current model weights
            #    weights_to_send_to_go = [get_model_weights_numpy(node.model), get_model_weights_numpy(node.model)]
            #    logger.info("GOSSIP PHASE: No new weights from Go. Will send current model weights.")
            
                # 5. Push weights to Go
                weights_to_send_to_go = [new_averaged_weights, new_averaged_weights]
                #send_queue.put(weights_to_send_to_go)
                neighbours = node.get_nodes_in_range()
                for neigh_node in neighbours:
                    if neigh_node != "e0":
                        neighbour = nodes[neigh_node]
                        neighbour.weight_queue.put(
                            WeightUpdate(node.node_id, new_averaged_weights, new_averaged_weights)
                        )  # Send weights to the thread for writing to shared memory
                logger.info("GOSSIP PHASE: Sent new averaged weights to Go.")
            
                #time.sleep(5)  # Adjust sleep time as needed (e.g., 2-5 seconds)
                node.previous_weights = get_model_weights_numpy(node.model)
            gossip_round += 1
    
    except KeyboardInterrupt:
        logger.warning("GOSSIP PHASE: KeyboardInterrupt. Exiting continuous gossip phase...")

except KeyboardInterrupt:
    # User pressed Ctrl+C
    logger.warning("KeyboardInterrupt: Exiting...")
    for thread in Threads:
        thread.join(2)
        time.sleep(1)
        if thread.is_alive():
            logger.warning(f"Thread {thread.name} is still alive after 1 second.")
        else:
            logger.warning(f"Thread {thread.name} has finished.")
except EOFError:
    logger.error("EOFError: The end of the file was reached unexpectedly.")
except MemoryError:
    logger.error("MemoryError: Not enough memory to allocate the shared memory block.")
except Exception as e:
    logger.error(f"An error occurred: {e}")
    logger.error(traceback.format_exc())
finally:
    logger.info("[Python] Script completed.")

