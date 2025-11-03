# filepath: runner/main_emnist.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
from multiprocessing import shared_memory
import re
import sys

print(sys.path)
import numpy as np
import posix_ipc
from posix_ipc import Semaphore, O_CREAT


from threading import Thread
from queue import Queue, Empty

import time
import tensorflow as tf
tf.random.set_seed(1234)

from tensorflow.keras.datasets import mnist

# cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
try:
    tf.config.set_visible_devices([], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    logger.info(f"[DEBUG] Logical GPUs after setting visible devices: {len(logical_gpus)}")
except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    logger.error(f"[ERROR] Could not set visible devices: {e}")
    pass # Hopes CUDA_VISIBLE_DEVICES worked


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

incoming_counter = 1

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


def average_weights(current_weights, loaded_weights):
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

def sum_weights(loaded_weights, current_weights):
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

def average_weights_n(loaded_weights, n: int):
    divider = float(n)
    # Efficient vectorized averaging using numpy
    return [(w1/divider) for w1 in loaded_weights]

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


def load_subset(path):
    x_path = os.path.join(path, "x_subset.npy")
    y_path = os.path.join(path, "y_subset.npy")
    if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
        raise FileNotFoundError(f"Could not find subset files in {path}")
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y


def log_metrics(
    id: str, timestamp, acc, prec, rec, f1, epoch, iter, batch, lr=0.001, path="metrics.csv"
):
    path = prefix + path
    row = [id, epoch, batch, iter, acc, prec, rec, f1, timestamp, lr]
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


def map_weights_to_shm(shm, wlist, shapes, dtypes, sizes):
    """Copy all arrays into the shared-memory block."""
    buf = shm.buf
    offset = 0

    for w, shape, dtype, size in zip(wlist, shapes, dtypes, sizes):
        np_dtype = np.dtype(dtype)
        view = np.ndarray(shape, dtype=np_dtype, buffer=buf[offset : offset + size])
        view[:] = w  # copy data
        offset += size


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
    weights = []
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
    return weights


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
        name=SHM_NAME_GO, size=metadata["total_size"]
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


try:
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
                name=SHM_NAME, size=total_size
            )  # Attach to the existing shared memory
            while True:
                weight_list = queue.get()
                # Copy into shm
                logger.info(f"[Python] Acquiring semaphore to write → Go")
                map_weights_to_shm(shm, weight_list, shapes, dtypes, sizes)
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
    # 1. Configuration
    DS_NAME    = 'emnist/byclass'
    BATCH_SIZE = 64
    VAL_FRAC   = 0.20       # 20% of train → validation
    SEED       = 1234
    INITIAL_LR = 1e-3
    EPOCHS     = 200
    initializer = GlorotUniform(seed=SEED)

    # 2. Load EMNIST train as NumPy for splitting
    x_all, y_all = load_subset(f"subsets_emnist_byclass_{prefix}/subset_{node_id}")

    # 3. Split into train / validation via sklearn
    print(f"Splitting off {VAL_FRAC*100:.0f}% of train for validation...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all,
        test_size=VAL_FRAC,
        random_state=SEED,
        stratify=y_all
    )

    # Normalize and reshape both sets
    x_train = x_train.astype('float32') / 255.0
    x_val   = x_val.astype('float32')   / 255.0
    # Ensure channels-last shape
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val   = x_val.reshape(-1,   28, 28, 1)

    # One-hot encode labels
    ds_info     = tfds.builder(DS_NAME).info
    num_classes = ds_info.features['label'].num_classes
    y_train = to_categorical(y_train, num_classes)
    y_val   = to_categorical(y_val,   num_classes)

    # 4. Prepare test split pipeline (normalized in preprocess)
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes)
        return image, label

    print("Preparing test dataset pipeline...")
    ds_test = tfds.load(
        DS_NAME,
        split='test',
        as_supervised=True
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .cache() \
    .prefetch(tf.data.AUTOTUNE)

    # 5. Convert ds_test to NumPy arrays x_test, y_test
    print("Converting ds_test to NumPy arrays x_test and y_test...")
    x_test_list = []
    y_test_list = []

    for images_batch, labels_batch in ds_test:
        x_test_list.append(images_batch.numpy()) # Convert EagerTensor to NumPy
        y_test_list.append(labels_batch.numpy()) # Convert EagerTensor to NumPy

    if x_test_list: # Check if the list is not empty
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    else:
        print("ds_test was empty, x_test and y_test are not created.")
        x_test = np.array([]) # Or handle as an error
        y_test = np.array([])


    # 6. Build & compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)

    model = Sequential([
        Conv2D(
            16, (3,3),
            activation="relu",
            padding="same",
            input_shape=(28,28,1),
            kernel_initializer=initializer
        ),
        MaxPooling2D((2,2)),
        Dropout(0.2, seed=SEED),          # also seed dropout
        Conv2D(
            32, (3,3),
            activation="relu",
            padding="same",
            kernel_initializer=initializer
        ),
        MaxPooling2D((2,2)),
        Dropout(0.3, seed=SEED),
        Flatten(),
        Dense(32, activation="relu", kernel_initializer=initializer),
        Dropout(0.2, seed=SEED),
        Dense(num_classes, activation="softmax", kernel_initializer=initializer),
    ])

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 7. Parameters for manual ReduceLROnPlateau

    # Parameters for manual ReduceLROnPlateau
    lr_patience = 2  # From your ReduceLROnPlateau callback
    lr_factor = 0.5    # From your ReduceLROnPlateau callback
    min_lr = 1e-7      # From your  ReduceLROnPlateau callback
    lr_monitor_metric = 'accuracy' # 'accuracy' for training accuracy
    lr_mode = 'max' # For accuracy, we want to maximize it
    wait_epochs = 0 # Counter for patience
    best_metric_val = -float('inf') if lr_mode == 'max' else float('inf')
    current_lr = INITIAL_LR # Start with the initial learning rate
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
    shm = create_or_attach_shared_memory(name=SHM_NAME, size=total_size)

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
    receive_queue = Queue()

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
    num_epochs = EPOCHS
    batch_size = BATCH_SIZE
    num_batches = math.ceil(x_train.shape[0] / batch_size)

    gossip_round = 0
    gossip_list = []
    batching_counter = 0

    epoch_loss = 0
    epoch_acc = 0

    num_batches = math.ceil(len(x_train) / BATCH_SIZE)

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

        # 3. Compute metrics
        y_test_pred_probs = model.predict(x_test, verbose=0)
        y_test_pred = y_test_pred_probs.argmax(axis=1)

        y_test_true = y_test.argmax(axis=1)

        acc = accuracy_score(y_test_true, y_test_pred)
        prec = precision_score(
            y_test_true, y_test_pred, average="macro", zero_division=0
        )
        rec = recall_score(y_test_true, y_test_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

        ts = time.time()
        log_metrics(node_id, ts, acc, prec, rec, f1, epoch, -1, batching_counter, initial_learning_rate)

        loss, acc = model.evaluate(x_val, y_val, verbose=2)
        logger.info(f"Validation loss: {loss:.4f}, Validation accuracy: {acc:.4f}")
        logger.info(f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")
        # Extract weights at the end of epoch
        current_weights = model.get_weights()

        # 2. Hand them off to Go (and wait for Go to read)
        # exchange_weights_with_go(current_weights)
        send_queue.put(
            current_weights
        )  # Send weights to the thread for writing to shared memory
        # 3. (Optional) Here you would merge in peers’ weights from Go…
        try:
            incoming_weights = receive_queue.get_nowait()
        except Empty:
            incoming_weights = None
            pass

        ##################################################################
        # Batch with running average type shi
        ##################################################################
        # if incoming_weights:
        #     incoming_counter += 1
        #     logger.info("DEBUG: received weights from Go")

        #     first_element = incoming_weights[0][0]
        #     last_element = incoming_weights[0][-1]
        #     logger.info(
        #         "Read weight array %d: Shape: %s, First element: %s, Last element of first array %s",
        #         0,  # Changed 'i' to 0, as this log is about incoming_weights[0]
        #         incoming_weights[0].shape,
        #         first_element,
        #         last_element
        #     )
        #     if gossip_list == []:
        #         gossip_list = incoming_weights
        #     else:
        #         new_weights = sum_weights(gossip_list, incoming_weights)
        #     logger.info("[INCOMING] Weights received from Go round %d", gossip_round)
        #     gossip_round += 1

        #################################################################
        # Batch with just averaging type shi
        #################################################################
        if incoming_weights:
            incoming_counter += 1
            logger.info("DEBUG: received weights from Go")

            if gossip_list == []:
                gossip_list = incoming_weights
            else:
                gossip_list = average_weights(gossip_list, incoming_weights)
            logger.info("[INCOMING] Weights received from Go round %d", gossip_round)
            gossip_round += 1

        if incoming_counter % 30 == 0 and incoming_counter > 0:
            # If we have enough models, perform averaging
            logger.info("Performing batch averaging of weights...")
            new_weights = average_weights(current_weights, gossip_list)
            model.set_weights(new_weights)
            logger.info("Applied averaged weights to model.")
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
            log_metrics(node_id, ts, acc, prec, rec, f1, epoch, gossip_round, batching_counter, INITIAL_LR)
            logger.info(f"Batch {batching_counter} accuracy: {acc:.4f}")
            batching_counter += 1
        
        ###############################################################
        # Ad hoc type check
        ###############################################################
        # if incoming_weights:
        #     incoming_counter += 1
        #     logger.info("DEBUG: received weights from Go")

        #     first_element = incoming_weights[0][0]
        #     last_element = incoming_weights[0][-1]
        #     logger.info(
        #         "Read weight array %d: Shape: %s, First element: %s, Last element of first array %s",
        #         0,  # Changed 'i' to 0, as this log is about incoming_weights[0]
        #         incoming_weights[0].shape,
        #         first_element,
        #         last_element
        #     )

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
        #     log_metrics(node_id, ts, acc, prec, rec, f1, -1, gossip_round, current_lr)

        #     loss, acc = model.evaluate(x_val, y_val, verbose=2)
        #     logger.info(f"Validation loss: {loss:.4f}, Validation accuracy: {acc:.4f}")
        #     logger.info(
        #         f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}"
        #     ) 
        #     gossip_round += 1


        # time.sleep(1)  # Sleep for 1 second to simulate some processing time

    ##################################################################
    #  Final evaluation
    ##################################################################
    
    score = model.evaluate(x_test, y_test, verbose=2)
    logger.info("Test loss: %f", score[0])
    logger.info("Test accuracy: %f", score[1])

    logger.info("[PROCESSING] Weights received from Go, metrics:")

    ################################################################
    # Batch with running average type shi
    ################################################################
    # model_weights = model.get_weights()
    # new_weights = average_weights(gossip_list, model_weights)
    # model.set_weights(new_weights)

    ###############################################################
    # Batch with just averaging type shi
    ##################################################################
    
    model_weights = model.get_weights()
    new_weights = average_weights(gossip_list, model_weights)
    model.set_weights(new_weights)
    

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
    log_metrics(node_id, ts, acc, prec, rec, f1, -1, 0, batching_counter, initial_learning_rate)

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
            new_weights_received_this_cycle = False

            # 1. Try to get weights from Go (non-blocking)
            try:
                incoming_weights = receive_queue.get_nowait()
                if incoming_weights:
                    new_weights_received_this_cycle = True
            except Empty:
                incoming_weights = None
                # logger.debug("GOSSIP PHASE: No new weights from Go in this cycle.")

            if new_weights_received_this_cycle:
                logger.info("GOSSIP PHASE: Received weights from Go.")
                current_model_weights = model.get_weights()

                # Optional: Log details of incoming weights
                if len(incoming_weights) > 0 and incoming_weights[0] is not None:
                    logger.info(
                        "GOSSIP PHASE: Incoming weight array 0: Shape: %s, First element: %s",
                        incoming_weights[0].shape,
                        incoming_weights[0][0] if incoming_weights[0].size > 0 else "N/A"
                    )
                incoming_counter += 1
                # 2. Average them with the current model weights
                new_averaged_weights = average_weights(current_model_weights, incoming_weights)
                # 3. Set the new weights in the model
                model.set_weights(new_averaged_weights)
                logger.info("GOSSIP PHASE: Applied averaged weights to model.")
                weights_to_send_to_go = new_averaged_weights

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
                log_metrics(node_id, ts_gossip, acc_gossip, prec_gossip, rec_gossip, f1_gossip, -2, gossip_round)

                loss_val_gossip, acc_val_gossip = model.evaluate(x_val, y_val, verbose=0)
                logger.info(f"GOSSIP PHASE: Validation after averaging - Loss: {loss_val_gossip:.4f}, Accuracy: {acc_val_gossip:.4f}")
                
                gossip_round += 1
            else:
                # No new weights from Go, so we'll send our current model weights
                weights_to_send_to_go = model.get_weights()
                # logger.info("GOSSIP PHASE: No new weights from Go. Will send current model weights.")


            # 5. Push weights to Go
            if weights_to_send_to_go is not None:
                send_queue.put(weights_to_send_to_go)
                if new_weights_received_this_cycle:
                    logger.info("GOSSIP PHASE: Sent new averaged weights to Go.")
                else:
                    logger.info("GOSSIP PHASE: Sent current model weights to Go (no new incoming).")
            
            time.sleep(5)  # Adjust sleep time as needed (e.g., 2-5 seconds)

    except KeyboardInterrupt:
        logger.warning("GOSSIP PHASE: KeyboardInterrupt. Exiting continuous gossip phase...")

except Exception as e:
    logger.error(f"An error occurred: {e}")
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
