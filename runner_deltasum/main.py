# filepath: runner/main.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
from multiprocessing import shared_memory
import re
import sys

print(sys.path)
import numpy as np
import signal
from typing import List
import posix_ipc
from posix_ipc import Semaphore, O_CREAT


from threading import Thread
from queue import Queue, Empty

import time
import tensorflow as tf
tf.random.set_seed(1234)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.datasets import mnist

# cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.initializers import GlorotUniform
initializer = GlorotUniform(seed=1234)

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


# Attempt to configure TensorFlow to not use GPUs
# try:
#     tf.config.set_visible_devices([], 'GPU')
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     logger.info(f"[DEBUG] Logical GPUs after setting visible devices: {len(logical_gpus)}")
# except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     logger.error(f"[ERROR] Could not set visible devices: {e}")
#     pass # Hopes CUDA_VISIBLE_DEVICES worked


logger.info(f"TensorFlow version: {tf.__version__}")
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
    logger.info(f"Loaded config: {config}")
    return config


gossip_config = load_config("config.json")
config_file = sys.argv[1] if len(sys.argv) > 1 else "defaultConfig.json"
logger.info("loading arg %s", config_file)

config = load_config(config_file)

def extract_after_f(prefix: str) -> int:
    m = re.match(r"^f(\d+)", prefix)
    if not m:
        raise ValueError(f"No number after 'f' in {prefix!r}")
    return int(m.group(1))

prefix = sys.argv[2] if len(sys.argv) > 2 else "f1"
# Example
NODE_COUNT = extract_after_f(prefix)  # → 75
logger.info("NODE_COUNT: %d", NODE_COUNT)

incoming_counter = 0

node_id = config["nodeID"]
# Names for the IPC objects
SHM_NAME = gossip_config["shared_memory"]["weights_name"] + node_id
SHM_META_NAME = gossip_config["shared_memory"]["metadata_name"] + node_id
SEM_PY2GO = gossip_config["semaphores"]["python_to_go"] + node_id  # Python → Go signal
SEM_GO2PY = gossip_config["semaphores"]["go_to_python"] + node_id  # Go → Python signal
SEM_META = gossip_config["semaphores"]["metadata"] + node_id

SHM_NAME_GO = gossip_config["shared_memory_go2py"]["weights_name"] + node_id
SEM_GO_PY2GO = (
    gossip_config["semaphores_go2py"]["python_to_go"] + node_id
)  # Python → Go signal
SEM_GO_GO2PY = (
    gossip_config["semaphores_go2py"]["go_to_python"] + node_id
)  # Go → Python signal

# ——————————————————————————————————————————————————————————————
# Helper Functions
# ——————————————————————————————————————————————————————————————


def get_weight_info(list):
    shapes = [arr.shape for arr in list]
    dtypes = [
        str(arr.dtype) for arr in list
    ]  # Gets 'float32' instead of 'numpy.float32'
    sizes = [arr.nbytes for arr in list]
    total_size = sum(sizes)
    return shapes, dtypes, sizes, total_size


def average_weights(weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    # assert this
    if len(weights) == 0:
        return []
    elif len(weights) == 1:
        return weights[0]
    else:
        ret_weights = weights[0]
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

def create_or_attach_shared_memory(name, size):
    try:
        # Try to attach to existing shared memory
        shm = shared_memory.SharedMemory(name=name, create=False)
        logger.info(f"Attached to existing shared memory: {name}")
    except FileNotFoundError:
        # If it doesn't exist, create it
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        logger.error(f"Created new shared memory: {name}")
    return shm

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
    # for model_idx, m in enumerate(model_buffer):
    #     if len(m) != L:
    #         raise ValueError(
    #             f"All models must have the same number of layers. "
    #             f"Model 0 has {L} layers, but model {model_idx} has {len(m)} layers."
    #         )
    
    # 1) Compute the plain averaged model (Model_avg)
    # Model_avg is a list of L numpy arrays, where each array is the averaged layer
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


def log_metrics(
    id: str, timestamp, acc, prec, rec, f1, epoch, iter, batch, val_acc, val_loss, test_acc, test_loss, lr=0.001, path="metrics.csv"
):
    path = prefix + path
    row = [id, epoch, batch, iter, acc, prec, rec, f1, timestamp, lr, val_acc, val_loss, test_acc, test_loss]
    line = ",".join(map(str, row))
    # This write() is atomic on Linux as long as len(line) < PIPE_BUF (≈4096)
    with open(path, "a", newline="") as f:
        f.write(line + "\n")


# ———————————————————————————————————————————————————————————————————
#  Shared Memory Functions
#  To write to shared memory
# ———————————————————————————————————————————————————————————————————


def write_metadata_to_shm(meta_bytes, size):
    """Write model weight metadata to a shared memory segment"""
    # Create metadata shared memory
    metadata_shm = shared_memory.SharedMemory(name=f"{SHM_META_NAME}", create=False)
    # Convert to bytes and write to shared memory
    metadata_shm.buf[:size] = meta_bytes
    metadata_shm.close()  # Close the metadata shared memory


def map_weights_to_shm(shm, wlist: List[List[np.ndarray]], pwlist: List[List[np.ndarray]], shapes, dtypes, sizes):
    """Copy all arrays into the shared-memory block."""
    buf = shm.buf
    offset = 0

    for w, shape, dtype, size in zip(wlist, shapes, dtypes, sizes):
        np_dtype = np.dtype(dtype)
        view = np.ndarray(shape, dtype=np_dtype, buffer=buf[offset : offset + size])
        view[:] = w  # copy data
        offset += size

    for w, shape, dtype, size in zip(pwlist, shapes, dtypes, sizes):
        np_dtype = np.dtype(dtype)
        view = np.ndarray(shape, dtype=np_dtype, buffer=buf[offset : offset + size])
        view[:] = w  # copy data
        offset += size

# ———————————————————————————————————————————————————————————————————
#  Batch update processing functions
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

# def batch_avg_normal(model_dict: dict[str, List[WeightUpdate]], ctr: int) -> bool: 
#     if incoming_counter % gossip_batch_size == 0 and incoming_counter > 0:
#         # If we have enough models, perform averaging
#         if len(model_dict) >= 1: #pretty useless otherwise
#             logger.info("Performing batch averaging of weights...")
#             models: List[List[np.ndarray]] = []
#             for node in model_dict:
#                 wupdate = model_dict[node].pop()
#                 # sum base + update, this method only cares about the total
#                 total = sum_weights(wupdate.baseweights, wupdate.update)
#                 models.append(total)

#             models.append(model.get_weights())  # Add the current model weights to the list
#             new_weights = average_weights(models)
#             model.set_weights(new_weights)
#             logger.info("Applied averaged weights to model.")
            
#             y_test_pred_probs = model.predict(x_test, verbose=0)
#             y_test_pred = y_test_pred_probs.argmax(axis=1)

#             y_test_true = y_test.argmax(axis=1)

#             # 3. Compute metrics
#             acc = accuracy_score(y_test_true, y_test_pred)
#             prec = precision_score(
#                 y_test_true, y_test_pred, average="macro", zero_division=0
#             )
#             rec = recall_score(
#                 y_test_true, y_test_pred, average="macro", zero_division=0
#             )
#             f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

#             ts = time.time()
#             test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
#             val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
#             log_metrics(node_id, ts, acc, prec, rec, f1, epoch, gossip_round, batching_counter, val_acc, val_loss, test_acc, test_loss, initial_learning_rate)
#             logger.info(f"Batch {batching_counter} accuracy: {acc:.4f}")

#             return True
#             model_list.clear()
#             gossip_list.clear()
#             batching_counter += 1
#     return False


# ———————————————————————————————————————————————————————————————————
#  Shared Memory Functions
#  To read from shared memory
# ———————————————————————————————————————————————————————————————————

def read_weights_from_shm(shm, shapes, dtypes, sizes):
    """Read weights from shared memory and reconstruct numpy arrays.
    Args:
        shm_name: Name of shared memory block
        shapes: List of array shapes
        dtypes: List of data types as strings
        sizes: List of array sizes in bytes

    Returns:
        List of numpy arrays with original shapes and values
    """
    # Reconstruct arrays from shared memory
    weights: List[np.ndarray] = []
    wupdate: List[np.ndarray] = []
    nodeid = ""
    offset = 0

    for shape, dtype, size in zip(shapes, dtypes, sizes):
        # Create numpy array view of the shared memory segment
        np_dtype = np.dtype(dtype)
        array_view = np.ndarray(
            shape=shape, dtype=np_dtype, buffer=shm.buf[offset : offset + size]
        )
        # Make a copy to get independent array
        weights.append(array_view.copy())
        offset += size

    for shape, dtype, size in zip(shapes, dtypes, sizes):
        # Create numpy array view of the shared memory segment
        np_dtype = np.dtype(dtype)
        array_view = np.ndarray(
            shape=shape, dtype=np_dtype, buffer=shm.buf[offset : offset + size]
        )
        # Make a copy to get independent array
        wupdate.append(array_view.copy())
        offset += size

    nodeid = str(bytes(shm.buf[offset : offset + 20]))

    update = WeightUpdate(nodeid, weights, wupdate)
    return update

def calc_metrics(x_test, y_test, x_val, y_val) -> Metrics:
    y_test_pred_probs = model.predict(x_test, verbose=0)
    y_test_pred = y_test_pred_probs.argmax(axis=1)

    y_test_true = y_test.argmax(axis=1)

    # 3. Compute metrics
    acc = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(
        y_test_true, y_test_pred, average="macro", zero_division=0
    )
    rec = recall_score(
        y_test_true, y_test_pred, average="macro", zero_division=0
    )
    f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    return Metrics(acc, prec, rec, f1, test_loss, test_acc, val_loss, val_acc)

# Function to call from thread with queue to pipe info through the queue
def read_weights(queue):
    # Attach to existing shared memory
    sem_go_py2go = Semaphore(
        SEM_GO_PY2GO, O_CREAT, initial_value=0
    )  # Open the existing named semaphore (block until Go posts it)
    sem_go_go2py = Semaphore(
        SEM_GO_GO2PY, O_CREAT, initial_value=0
    )  # Open the existing named semaphore (block until Go posts it)
    shm = create_or_attach_shared_memory(
        name=SHM_NAME_GO, size=2*metadata["total_size"]+20
    )  # Attach to the existing shared memory
    try:
        while True:
            sem_go_go2py.acquire()  # Wait for Go to post the signal that it's done writing weights
            weights = read_weights_from_shm(
                shm=shm,
                shapes=metadata["shapes"],
                dtypes=metadata["dtypes"],
                sizes=metadata["sizes"],
            )
            sem_go_py2go.release()  # Signal Go that Python is done reading weights
            queue.put(
                weights
            )  # Put the weights in the queue for the main thread to consume
            

    except Exception as e:
        logger.error(f"Error setting up weight reader: {e}")
    finally:
        # Cleanup
        try:
            shm.close()
        except:
            pass


class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, signum, frame):
    logger.error(f"Signal detected: {signum}")
    self.kill_now = True

try:
    #killer = GracefulKiller()
    shapes = None
    dtypes = None
    sizes = None
    total_size = None
    #################################################################
    # Create (or open) the semaphores
    #################################################################
    # Create semaphores for Py2Go
    sem_py2go = Semaphore(SEM_PY2GO, O_CREAT, initial_value=0)
    sem_go2py = Semaphore(SEM_GO2PY, O_CREAT, initial_value=0)
    sem_meta = Semaphore(SEM_META, O_CREAT, initial_value=0)

    ###################################################################
    # Initialize shared memory functions
    # with the correct use of the semaphores
    ###################################################################
    def exchange_weights_with_go(queue):
        """
        1) Python writes weights → sem_py2go.release()
        2) Python blocks on sem_go2py.acquire() until Go reads
        """
        # Attach to the existing shared memory
        try:
            shm = create_or_attach_shared_memory(
                name=SHM_NAME, size=2*total_size
            )  # Attach to the existing shared memory
            while True:
                weight_list = queue.get()
                weights = weight_list[0]
                previous_model_weights = weight_list[1]
                # Copy into shm
                logger.info(f"[Python] Acquiring semaphore to write → Go")
                map_weights_to_shm(shm, previous_model_weights, weight_diff(weights, previous_model_weights), shapes, dtypes, sizes)
                sem_py2go.release()  # signal Go: data ready
                logger.info(f"[Python] Waiting for Go to finish reading…")
                sem_go2py.acquire()  # wait for Go to ack
                logger.info(f"[Python] Go has read the weights; continuing training.")
        finally:
            try:
                if shm:
                    shm.close()
                    shm.unlink()  # Ensure shared memory is unlinked
            except Exception as e:
                logger.error(f"Error cleaning up shared memory: {e}")

    def exchange_metadata_with_go(meta_bytes):
        """
        1) Python writes metadata → sem_meta.release()
        2) Python blocks on sem_go2py.acquire() until Go reads
        """
        metadata_size = len(meta_bytes)
        # Create metadata shared memory
        metadata_shm = shared_memory.SharedMemory(
            name=f"{SHM_META_NAME}", create=True, size=metadata_size
        )
        # Copy into shm
        write_metadata_to_shm(meta_bytes, len(meta_bytes))
        sem_meta.release()  # signal Go: data ready
        logger.info(f"[Python] Waiting for Go to finish reading metadata…")
        sem_go2py.acquire()  # wait for Go to ack
        logger.info(
            f"[Python] Go has read the metadata; Unlink from shared memory block."
        )
        metadata_shm.unlink()  # unlink the metadata shared memory
        logger.info(f"[Python] Go has read the metadata; continuing training.")

    # ——————————————————————————————————————————————————————————————
    #  Build and Train Model
    # ——————————————————————————————————————————————————————————————

    # Load data
    (_, _), (x_test, y_test) = mnist.load_data()

    x_train, y_train = load_subset(f"subsets_{prefix}/subset_{node_id}")
    x_train = x_train.astype("float32") / 255.0  # scale
    x_test = x_test.astype("float32") / 255.0

    if len(x_train.shape) == 3: # Check if channel dimension is missing
        x_train = x_train.reshape(-1, 28, 28, 1)
    if len(x_test.shape) == 3: # Check if channel dimension is missing
        x_test = x_test.reshape(-1, 28, 28, 1)

    # Generate a permutation of indices
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    # Shuffle the dataset
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, num_classes=10)
    model = Sequential([
        Conv2D(
            16, (3,3),
            activation="relu",
            padding="same",
            input_shape=(28,28,1),
            kernel_initializer=initializer
        ),
        MaxPooling2D((2,2)),
        Dropout(0.2, seed=1234),          # also seed dropout
        Conv2D(
            32, (3,3),
            activation="relu",
            padding="same",
            kernel_initializer=initializer
        ),
        MaxPooling2D((2,2)),
        Dropout(0.3, seed=1234),
        Flatten(),
        Dense(32, activation="relu", kernel_initializer=initializer),
        Dropout(0.2, seed=1234),
        Dense(10, activation="softmax", kernel_initializer=initializer),
    ])
    # Define an exponential decay learning rate schedule:
    initial_learning_rate = 1e-3
    model.summary()
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=initial_learning_rate,
    #    decay_steps=10,  # number of steps after which the learning rate decays
    #    decay_rate=0.90,  # the decay factor
    #    staircase=True,
    #)  # if True, learning rate decays in discrete intervals

    # Pass the schedule to the optimizer:
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate, name="adam"
    )

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # ——————————————————————————————————————————————————————————————
    #  Get weights and metadata on the weights
    #  (and write to shared memory)
    #  This is once per model, not per epoch
    #  (unless you change the model architecture)
    # ——————————————————————————————————————————————————————————————

    # Prepare shared memory once, based on initial weights
    current_weights = model.get_weights()
    shapes, dtypes, sizes, total_size = get_weight_info(current_weights)
    metadata = {
        "total_size": total_size,
        "shapes": shapes,
        "dtypes": dtypes,  # Assuming all weights are float32
        "sizes": sizes,
    }
    logger.info(
        f"[DEBUG] total_size = {total_size}, shapes = {shapes}, dtypes = {dtypes}, sizes = {sizes}"
    )
    meta_bytes = json.dumps(metadata).encode("utf-8")
    exchange_metadata_with_go(meta_bytes)
    shm = create_or_attach_shared_memory(name=SHM_NAME, size=2*total_size+20)

    # ———————————————————————————————————————————————————————————————
    # Attach to go semaphores after metadata exchange
    # ———————————————————————————————————————————————————————————————
    sem_go_py2go = Semaphore(
        SEM_GO_PY2GO, O_CREAT, initial_value=0
    )  # Open the existing named semaphore (block until Go posts it)
    sem_go_go2py = Semaphore(
        SEM_GO_GO2PY, O_CREAT, initial_value=0
    )  # Open the existing named semaphore (block until Go posts it)
    # Print the model summary to see its architecture
    model.summary()

    # ——————————————————————————————————————————————————————————————
    #  Create a thread to read and send weights from Go
    #  Create queue to communicate between threads
    # ——————————————————————————————————————————————————————————————
    send_queue = Queue()
    receive_queue: Queue[WeightUpdate] = Queue()

    Threads = []

    t_read_weights = Thread(
        target=read_weights,
        args=(receive_queue,),
        name="read_weights_thread",
        daemon=False,
    )
    Threads.append(t_read_weights)
    t_read_weights.start()

    t_write_weights = Thread(
        target=exchange_weights_with_go,
        args=(send_queue,),
        name="write_weights_thread",
        daemon=False,
    )
    Threads.append(t_write_weights)
    t_write_weights.start()

    ###################################################################
    #  Train the model
    ###################################################################

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    num_epochs = 200
    batch_size = 64
    num_batches = math.ceil(x_train.shape[0] / batch_size)

    gossip_round = 0
    gossip_list: dict[str, List[WeightUpdate]] = {}

    model_list: List[List[np.ndarray]] = [] # List to hold models for averaging

    batching_counter = 0
    gossip_batch_size: int = 10  # Number of batches after which to perform gossip averaging

    epoch_loss = 0
    epoch_acc = 0
    previous_model_weights = model.get_weights()

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        epoch_acc = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]
            batch_loss, batch_acc = model.train_on_batch(x_batch, y_batch, True)
            epoch_loss += batch_loss
            epoch_acc += batch_acc
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        y_test_pred_probs = model.predict(x_test, verbose=0)
        y_test_pred = y_test_pred_probs.argmax(axis=1)

        y_test_true = y_test.argmax(axis=1)

        # 3. Compute metrics
        acc = accuracy_score(y_test_true, y_test_pred)
        prec = precision_score(
            y_test_true, y_test_pred, average="macro", zero_division=0
        )
        rec = recall_score(y_test_true, y_test_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

        ts = time.time()

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

        log_metrics(node_id, ts, acc, prec, rec, f1, epoch, -1, batching_counter, val_acc, val_loss, test_acc, test_loss, initial_learning_rate)

        logger.info(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
        logger.info(f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")
        logger.info(f"Epoch {epoch+1} - Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        # Extract weights at the end of epoch
        current_weights = model.get_weights()
        
        # 2. Hand them off to Go (and wait for Go to read)
        # exchange_weights_with_go(current_weights)
        send_queue.put(
            [current_weights, previous_model_weights]
        )  # Send weights to the thread for writing to shared memory
        # 3. (Optional) Here you would merge in peers’ weights from Go…
        #time.sleep(2)

        incoming_weights: List[WeightUpdate] = []
        while not receive_queue.empty():
            try:
                incoming_weight = receive_queue.get_nowait()
                incoming_weights.append(incoming_weight)
            except Empty:
                #incoming_weights = None
                pass

        ##################################################################
        # Ad hoc with variance-corrected model averaging type shi
        ##################################################################
        # if incoming_weights:
        #     incoming_counter += 1
        #     logger.info("DEBUG: received weights from Go")

        #     # 4. Average them with the current weights
        #     models = [current_weights, incoming_weights]
        #     new_weights = variance_corrected_model_averaging(models)
        #     # 5. Set the new weights in the model
        #     model.set_weights(new_weights)
        #     logger.info("[INCOMING] Weights received from Go, metrics:")
        #     y_test_pred_probs = model.predict(x_test, verbose=0)
        #     y_test_pred = y_test_pred_probs.argmax(axis=1)

        #     y_test_true = y_test.argmax(axis=1)

        #     # 3. Compute metrics
        #     acc = accuracy_score(y_test_true, y_test_pred)
        #     prec = precision_score(
        #         y_test_true, y_test_pred, average="macro", zero_division=0
        #     )
        #     rec = recall_score(
        #         y_test_true, y_test_pred, average="macro", zero_division=0
        #     )
        #     f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

        #     ts = time.time()
        #     log_metrics(node_id, ts, acc, prec, rec, f1, -1, gossip_round)

        #     loss, acc = model.evaluate(x_val, y_val, verbose=2)
        #     logger.info(f"Validation loss: {loss:.4f}, Validation accuracy: {acc:.4f}")
        #     logger.info(
        #         f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}"
        #     )
        #     gossip_round += 1

        ##################################################################
        # Batch with variance-corrected model averaging type shi
        ##################################################################
        # incoming_counter += 1
        # #if len(incoming_weights) > 0:
        # gossip_round += 1
            
        # for wupdate in incoming_weights: 
        #     #if wupdate.nodeid in gossip_list:
        #     #    gossip_list[wupdate.nodeid].append(wupdate)
        #     #else:
        #         gossip_list[wupdate.nodeid] = [wupdate]
        
        # if incoming_counter % gossip_batch_size == 0 and incoming_counter > 0:
        #     # If we have enough models, perform variance-corrected averaging
        #     if len(gossip_list) >= 1:
        #         logger.info("Performing variance-corrected model averaging...")
        #         # Add the latest received model for each node
        #         for node in gossip_list:
        #             wupdate = gossip_list[node].pop()
        #             # sum base + update, this method only cares about the total
        #             total = sum_weights(wupdate.baseweights, wupdate.update)
        #             model_list.append(total)

        #         model_list.append(model.get_weights())  # Add the current model weights to the list
        #         new_weights = variance_corrected_model_averaging(model_list)
        #         model.set_weights(new_weights)
        #         logger.info("Applied variance-corrected averaged weights to model.")
        #         model_list.clear()
        #         gossip_list.clear()
        #         y_test_pred_probs = model.predict(x_test, verbose=0)
        #         y_test_pred = y_test_pred_probs.argmax(axis=1)

        #         y_test_true = y_test.argmax(axis=1)

        #         # 3. Compute metrics
        #         acc = accuracy_score(y_test_true, y_test_pred)
        #         prec = precision_score(
        #             y_test_true, y_test_pred, average="macro", zero_division=0
        #         )
        #         rec = recall_score(
        #             y_test_true, y_test_pred, average="macro", zero_division=0
        #         )
        #         f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
                
        #         ts = time.time()
        #         test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        #         val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        #         log_metrics(node_id, ts, acc, prec, rec, f1, epoch, gossip_round, batching_counter, val_acc, val_loss, test_acc, test_loss, initial_learning_rate)
        #         logger.info(f"Batch {batching_counter} accuracy: {acc:.4f}")

        #         batching_counter += 1

        # previous_model_weights = model.get_weights()

        ##################################################################
        # Batch with running average type shi
        ##################################################################
        # if incoming_weights.count > 0:
        #     incoming_counter += 1
        #     if gossip_list == []:
        #         gossip_list = incoming_weights
        #     else:
        #         gossip_list = sum_weights(gossip_list, incoming_weights)
        #     logger.info("[INCOMING] Weights received from Go round %d", gossip_round)
        #     gossip_round += 1

        # if incoming_counter % 30 == 0 and incoming_counter > 0:
        #     # If we have enough models, perform averaging
        #     logger.info("Performing batch averaging of weights...")
        #     model_weights = model.get_weights()
        #     gossip_list = sum_weights(gossip_list, model_weights)  # Add current weights to the list
        #     new_weights = average_weights_n(gossip_list, 31)  # Average the weights
        #     model.set_weights(new_weights)
        #     logger.info("Applied averaged weights to model.")
        #     gossip_list.clear()
        #     y_test_pred_probs = model.predict(x_test, verbose=0)
        #     y_test_pred = y_test_pred_probs.argmax(axis=1)

        #     y_test_true = y_test.argmax(axis=1)

        #     # 3. Compute metrics
        #     acc = accuracy_score(y_test_true, y_test_pred)
        #     prec = precision_score(
        #         y_test_true, y_test_pred, average="macro", zero_division=0
        #     )
        #     rec = recall_score(
        #         y_test_true, y_test_pred, average="macro", zero_division=0
        #     )
        #     f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

        #     ts = time.time()
        #     test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        #     val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        #     log_metrics(node_id, ts, acc, prec, rec, f1, epoch, gossip_round, batching_counter, val_acc, val_loss, test_acc, test_loss, initial_learning_rate)

        #     logger.info(f"Batch {batching_counter} accuracy: {acc:.4f}")
        #     batching_counter += 1

        #################################################################
        # Batch with just averaging type shi
        #################################################################
        # incoming_counter += 1
        # #if len(incoming_weights) > 0:
        # gossip_round += 1
        
        # for wupdate in incoming_weights: 
        #     #if wupdate.nodeid in gossip_list:
        #     #    gossip_list[wupdate.nodeid].append(wupdate)
        #     #else:
        #         gossip_list[wupdate.nodeid] = [wupdate]

        # #success = batch_avg_normal(gossip_list, incoming_counter)
        # if incoming_counter % gossip_batch_size == 0 and incoming_counter > 0:
        #     # If we have enough models, perform averaging
        #     if len(gossip_list) >= 1: #pretty useless otherwise
        #         logger.info("Performing batch averaging of weights...")
        #         for node in gossip_list:
        #             wupdate = gossip_list[node].pop()
        #             # sum base + update, this method only cares about the total
        #             total = sum_weights(wupdate.baseweights, wupdate.update)
        #             model_list.append(total)

        #         model_list.append(model.get_weights())  # Add the current model weights to the list
        #         new_weights = average_weights(model_list)
        #         model.set_weights(new_weights)
        #         logger.info("Applied averaged weights to model.")
        #         model_list.clear()
        #         gossip_list.clear()


        #         # 3. Compute metrics
        #         metrics = calc_metrics(x_test, y_test, x_val, y_val)

        #         #for i in range(NODE_COUNT):
        #         #    metrics = calc_metrics(x_test, y_test, x_val, y_val)
                    
        #         ts = time.time()

        #         log_metrics(node_id, ts, metrics.accu, metrics.prec, metrics.rec, metrics.f1, epoch, gossip_round, batching_counter, metrics.val_acc, metrics.val_loss, metrics.test_acc, metrics.test_loss, initial_learning_rate)
        #         logger.info(f"Batch {batching_counter} accuracy: {metrics.accu:.4f} val acc {metrics.val_acc:.4f} test acc {metrics.test_acc:.4f} ")
        #         batching_counter += 1

        # previous_model_weights = model.get_weights()

        #################################################################
        # Delta corrected averaging normal
        #################################################################
        incoming_counter += 1
        #if len(incoming_weights) > 0:
        gossip_round += 1

        #but the thing is, this approach only works until every batch's training on the tiny local dataset starts erasing 
        # gossip progress (we scribbled down the math on this earlier, Other Tom, check the paper scraps)
        #so todo: implement a GROWING  learning factor for remote contributions (unlike normal learning, where the factor goes down)
        for wupdate in incoming_weights: 
            if wupdate.nodeid in gossip_list:
                gossip_list[wupdate.nodeid].append(wupdate)
            else:
                gossip_list[wupdate.nodeid] = [wupdate]

        local_diff = weight_diff(model.get_weights(), previous_model_weights)
        localupdate = WeightUpdate("local", previous_model_weights, local_diff)
        if "local" in gossip_list:
            gossip_list["local"].append(localupdate)
        else:
            gossip_list["local"] = [localupdate]

        #success = batch_avg_normal(gossip_list, incoming_counter)
        if incoming_counter == 200 or (incoming_counter % gossip_batch_size == 0 and incoming_counter > 0):
            # If we have enough models, perform averaging
            if len(gossip_list) >= 1: #pretty useless otherwise
                logger.info("Performing batch averaging of weights...")
                for node in gossip_list:
                    wupdate = gossip_list[node][0]
                    model_list.append(wupdate.baseweights)

                new_weights = average_weights(model_list)
                for node in gossip_list:
                    first = gossip_list[node][0]
                    last= gossip_list[node].pop()
                    delta = sum_weights(weight_diff(last.baseweights, first.baseweights), last.update)
                    divider = 0.15 + incoming_counter / 1000
                    delta = mult_weights(delta, divider)
                    new_weights = sum_weights(new_weights, delta)
                
                model.set_weights(new_weights)

                logger.info("Applied averaged and delta corrected weights to model.")
                model_list.clear()
                gossip_list.clear()
                y_test_pred_probs = model.predict(x_test, verbose=0)
                y_test_pred = y_test_pred_probs.argmax(axis=1)

                y_test_true = y_test.argmax(axis=1)

                # 3. Compute metrics
                acc = accuracy_score(y_test_true, y_test_pred)
                prec = precision_score(
                    y_test_true, y_test_pred, average="macro", zero_division=0
                )
                rec = recall_score(
                    y_test_true, y_test_pred, average="macro", zero_division=0
                )
                f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

                ts = time.time()
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
                log_metrics(node_id, ts, acc, prec, rec, f1, epoch, gossip_round, batching_counter, val_acc, val_loss, test_acc, test_loss, initial_learning_rate)
                logger.info(f"Batch {batching_counter} accuracy: {acc:.4f}")
                batching_counter += 1

        previous_model_weights = model.get_weights()
        
        ###############################################################
        # Ad hoc type check
        ###############################################################
        # if incoming_weights:
        #     incoming_counter += 1
        #     logger.info("DEBUG: received weights from Go")

        #     # 4. Average them with the current weights
        #     new_weights = average_weights(current_weights, incoming_weights)
        #     # 5. Set the new weights in the model
        #     model.set_weights(new_weights)
        #     logger.info("[INCOMING] Weights received from Go, metrics:")
        #     y_test_pred_probs = model.predict(x_test, verbose=0)
        #     y_test_pred = y_test_pred_probs.argmax(axis=1)

        #     y_test_true = y_test.argmax(axis=1)

        #     # 3. Compute metrics
        #     acc = accuracy_score(y_test_true, y_test_pred)
        #     prec = precision_score(
        #         y_test_true, y_test_pred, average="macro", zero_division=0
        #     )
        #     rec = recall_score(
        #         y_test_true, y_test_pred, average="macro", zero_division=0
        #     )
        #     f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

        #     ts = time.time()
        #     test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        #     val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        #     log_metrics(node_id, ts, acc, prec, rec, f1, -1, gossip_round, val_acc, val_loss, test_acc, test_loss, initial_learning_rate)
        #     # log_metrics(node_id, ts, acc, prec, rec, f1, -1, gossip_round)

        #     logger.info(f"Validation loss: {val_loss:.4f}, Validation accuracy: {acc:.4f}")
        #     logger.info(
        #         f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}"
        #     )
        #     gossip_round += 1
        time.sleep(1)  # Sleep for 1 second to simulate some processing time

    ##################################################################
    #  Final evaluation
    ##################################################################
    
    score = model.evaluate(x_test, y_test, verbose=2)
    logger.info("Test loss: %f", score[0])
    logger.info("Test accuracy: %f", score[1])

    logger.info("[PROCESSING] Weights received from Go, metrics:")

    ##################################################################
    # Batch with variance-corrected model averaging type shi
    ##################################################################
    # model_weights = model.get_weights()
    # model_list.append(model_weights)  # Add the current model weights to the list
    # new_weights = variance_corrected_model_averaging(model_list)
    # model.set_weights(new_weights)
    # model_list.clear()  # Clear the model list after averaging

    ################################################################
    # Batch with running average type shi
    ################################################################
    # model_weights = model.get_weights()
    # new_weights = average_weights(model_weights, gossip_list)
    # model.set_weights(new_weights)

    ###############################################################
    # Batch with just averaging type shi
    ###############################################################
    # model_weights = model.get_weights()
    # new_weights = average_weights(model_weights, gossip_list)
    # model.set_weights(new_weights)
    
    ####################################################################
    # Final check after batching
    ####################################################################
    y_test_pred_probs = model.predict(x_test, verbose=0)
    y_test_pred = y_test_pred_probs.argmax(axis=1)

    y_test_true = y_test.argmax(axis=1)

    # 3. Compute metrics
    acc = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(
        y_test_true, y_test_pred, average="macro", zero_division=0
    )
    rec = recall_score(
        y_test_true, y_test_pred, average="macro", zero_division=0
    )
    f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

    ts = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    log_metrics(node_id, ts, acc, prec, rec, f1, -1, 0, batching_counter, val_acc, val_loss, test_acc, test_loss, initial_learning_rate)

    loss, acc = model.evaluate(x_val, y_val, verbose=2)
    logger.info(f"Validation loss: {loss:.4f}, Validation accuracy: {acc:.4f}")
    logger.info(
        f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}"
    )

    # --- BEGIN CONTINUOUS GOSSIP AND AVERAGING PHASE ---
    logger.info("Entering continuous gossip and averaging phase...")
    try:
        while True:
            weights_to_send_to_go = None
            #new_weights_received_this_cycle = False

            incoming_weights: List[WeightUpdate] = []
            while not receive_queue.empty():
                try:
                    incoming_weight = receive_queue.get_nowait()
                    incoming_weights.append(incoming_weight)
                except Empty:
                    #incoming_weights = None
                    pass
            # 1. Try to get weights from Go (non-blocking)
            #try:
            #    incoming_weights = receive_queue.get_nowait()
            #    if incoming_weights:
            #        new_weights_received_this_cycle = True
            #except Empty:
            #    incoming_weights = None
                # logger.debug("GOSSIP PHASE: No new weights from Go in this cycle.")

            if len(incoming_weights) > 0:
                logger.info("GOSSIP PHASE: Received weights from Go.")

                for wupdate in incoming_weights: 
                    gossip_list[wupdate.nodeid] = [wupdate]

                for node in gossip_list:
                    wupdate = gossip_list[node].pop()
                    # sum base + update, this method only cares about the total
                    total = wupdate.baseweights #sum_weights(wupdate.baseweights, wupdate.update)
                    model_list.append(total)

                model_list.append(model.get_weights())  # Add the current model weights to the list
                #new_averaged_weights = average_weights(model_list)
                new_averaged_weights = average_weights(model_list)

                # Optional: Log details of incoming weights
                #if len(incoming_weights) > 0 and incoming_weights[0] is not None:
                #    logger.info(
                 #       "GOSSIP PHASE: Incoming weight array 0: Shape: %s, First element: %s",
                 #       incoming_weights[0].shape,
                 #       incoming_weights[0][0] if incoming_weights[0].size > 0 else "N/A"
                 #   )
                # 2. Average them with the current model weights
                # models = [current_model_weights, incoming_weights]
                # new_averaged_weights = variance_corrected_model_averaging(models)
                #new_averaged_weights = average_weights(current_model_weights, incoming_weights)
                #previous_model_weights = current_model_weights
                # 3. Set the new weights in the model
                model.set_weights(new_averaged_weights)
                logger.info("GOSSIP PHASE: Applied averaged weights to model.")
                weights_to_send_to_go = [new_averaged_weights, previous_model_weights]

                model_list.clear()
                gossip_list.clear()
                # 4. Re-evaluate and log metrics for this gossip round
                # Using y_test_true which was y_test.argmax(axis=1) from the training loop
                y_test_pred_probs_gossip = model.predict(x_test, verbose=0)
                y_test_pred_gossip = y_test_pred_probs_gossip.argmax(axis=1)

                acc_gossip = accuracy_score(y_test_true, y_test_pred_gossip)
                prec_gossip = precision_score(y_test_true, y_test_pred_gossip, average="macro", zero_division=0)
                rec_gossip = recall_score(y_test_true, y_test_pred_gossip, average="macro", zero_division=0)
                f1_gossip = f1_score(y_test_true, y_test_pred_gossip, average="macro", zero_division=0)
                ts_gossip = time.time()
                # Use epoch = -2 to distinguish from training epochs in metrics
                val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                log_metrics(node_id, ts_gossip, acc_gossip, prec_gossip, rec_gossip, f1_gossip, -2, gossip_round, -1, val_acc, val_loss, test_acc, test_loss, initial_learning_rate)

                #loss_val_gossip, acc_val_gossip = model.evaluate(x_val, y_val, verbose=0)
                logger.info(f"GOSSIP PHASE: Validation after averaging - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                logger.info(f"GOSSIP PHASE: Validation after averaging - Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
                gossip_round += 1
            else:
                # No new weights from Go, so we'll send our current model weights
                weights_to_send_to_go = [model.get_weights(), model.get_weights()]
                # logger.info("GOSSIP PHASE: No new weights from Go. Will send current model weights.")


            # 5. Push weights to Go
            if weights_to_send_to_go is not None:
                send_queue.put(weights_to_send_to_go)
                #if new_weights_received_this_cycle:
                logger.info("GOSSIP PHASE: Sent new averaged weights to Go.")
                #else:
                 #   logger.info("GOSSIP PHASE: Sent current model weights to Go (no new incoming).")
            
            time.sleep(5)  # Adjust sleep time as needed (e.g., 2-5 seconds)
            previous_model_weights = model.get_weights()

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
finally:
    logger.info("[Python] Script completed.")
    try:
        if shm:
            shm.close()
            shm.unlink()
        sem_py2go.release()
        sem_py2go.unlink()
        sem_go2py.unlink()
        sem_meta.unlink()
        sem_go_py2go.unlink()
        sem_go_go2py.unlink()
    except:
        pass
