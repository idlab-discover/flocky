import re
import sys; print(sys.path)
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Disable GPU

import time
import tensorflow as tf
tf.random.set_seed(1234)
from tensorflow.keras.datasets import mnist
# cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import math
import logging

from tensorflow.keras.initializers import GlorotUniform
initializer = GlorotUniform(seed=1234)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', # Added %(module)s for clarity
    stream=sys.stdout  # Direct log output to standard output
)

logger.info("[Python] Starting execution...")
def extract_after_f(prefix: str) -> int:
    m = re.match(r"^f(\d+)", prefix)
    if not m:
        raise ValueError(f"No number after 'f' in {prefix!r}")
    return int(m.group(1))

prefix = sys.argv[2] if len(sys.argv) > 2 else "f1"
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
node_id = config["nodeID"]
def load_subset(path):
    x_path = os.path.join(path, "x_subset.npy")
    y_path = os.path.join(path, "y_subset.npy")
    if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
        raise FileNotFoundError(f"Could not find subset files in {path}")
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y

def log_metrics(id: str, timestamp, acc, prec, rec, f1, epoch, iter, lr=0.001, path="metrics.csv"):
    path = prefix + path
    row = [id, epoch, iter, acc, prec, rec, f1, timestamp, lr]
    line = ",".join(map(str, row))
    # This write() is atomic on Linux as long as len(line) < PIPE_BUF (≈4096)
    with open(path, "a", newline="") as f:
        f.write(line + "\n")

(_, _), (x_test, y_test) = mnist.load_data()

x_train, y_train = load_subset(f"subsets_{prefix}/subset_{node_id}")
x_train = x_train.astype('float32') / 255.0 # scale
x_test = x_test.astype('float32') / 255.0

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
y_test = to_categorical(y_test, num_classes = 10)

SEED=1234

# Build a basic 2D CNN model
model = Sequential([
        # First convolutional block
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1),
            kernel_initializer=initializer),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2, seed=SEED),
        
        # Second convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same',
            kernel_initializer=initializer),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3, seed=SEED),
        
        # Fully connected layers
        Flatten(),
        Dense(32, activation='relu',
            kernel_initializer=initializer),
        Dropout(0.2, seed=SEED),
        Dense(10, activation='softmax', 
            kernel_initializer=initializer)
    ])

    # Define an exponential decay learning rate schedule:
initial_learning_rate = 1e-3

# Pass the schedule to the optimizer:
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, name="adam")
    
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

num_epochs = 170
batch_size = 64
num_batches = math.ceil(x_train.shape[0] / batch_size)

epoch_loss = 0
epoch_acc = 0

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
    acc  = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)

    ts = time.time()
    log_metrics(node_id, ts, acc, prec, rec, f1, epoch, 0, lr=initial_learning_rate)

    loss, acc = model.evaluate(x_val, y_val, verbose=2)
    logger.info(f"Validation loss: {loss:.4f}, Validation accuracy: {acc:.4f}")
    logger.info(f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")  
    # Extract weights at the end of epoch
    # time.sleep(1)  # Sleep for 1 second to simulate some processing time

    # Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=2)
logger.info('Test loss: %d', score[0])
logger.info('Test accuracy: %d', score[1])