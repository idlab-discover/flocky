from multiprocessing import shared_memory
import sys
import numpy as np
from posix_ipc import Semaphore, O_CREAT
import os

from threading import Thread
from queue import Queue, Empty

import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input

from sklearn.model_selection import train_test_split
import math

import json
import platform

if platform.system() == "Windows":
    raise RuntimeError("This code requires a POSIX-compatible system")

# ——————————————————————————————————————————————————————————————
#  Semaphore & Shared‑Memory Setup
# ——————————————————————————————————————————————————————————————
def load_config(file="config.json"):
    """Load configuration from a JSON file."""
    with open(file, 'r') as f:
        content = f.read()
        # Replace JavaScript-style booleans and null with Python-compatible values
    content = content.replace('true', 'True').replace('false', 'False').replace('null', 'None')
    try:
        config = eval(content)
    except SyntaxError as e:
        raise ValueError(f"Failed to parse the JSON file: {file}. Error: {e}")
    print(f"Loaded config: {config}")
    return config

gossip_config = load_config("config.json")

config_file = sys.argv[1] if len(sys.argv) > 1 else "defaultConfig.json"
print("loading arg", config_file)
config = load_config(config_file)

node_id = config["nodeID"]
# Names for the IPC objects
SHM_NAME        = gossip_config["shared_memory"]["weights_name"] + node_id
SHM_META_NAME   = gossip_config["shared_memory"]["metadata_name"] + node_id
SEM_PY2GO       = gossip_config["semaphores"]["python_to_go"] + node_id      # Python → Go signal
SEM_GO2PY       = gossip_config["semaphores"]["go_to_python"] + node_id      # Go → Python signal
SEM_META        = gossip_config["semaphores"]["metadata"] + node_id

SHM_NAME_GO     = gossip_config["shared_memory_go2py"]["weights_name"] + node_id
SEM_GO_PY2GO    = gossip_config["semaphores_go2py"]["python_to_go"] + node_id      # Python → Go signal
SEM_GO_GO2PY    = gossip_config["semaphores_go2py"]["go_to_python"] + node_id      # Go → Python signal

# Names for the IPC objects
# SHM_NAME    = "/ml_weights_shm"
# SHM_META_NAME = "/ml_weights_shm_meta"
# SEM_PY2GO   = "/sem_py2go"   # Python → Go signal
# SEM_GO2PY   = "/sem_go2py"   # Go → Python signal
# SEM_META    = "/sem_meta"    # Metadata signal

# ——————————————————————————————————————————————————————————————
# Helper Functions
# ——————————————————————————————————————————————————————————————

def get_weight_info(list):
        shapes = [arr.shape for arr in list]
        dtypes = [str(arr.dtype) for arr in list]  # Gets 'float32' instead of 'numpy.float32'
        sizes  = [arr.nbytes for arr in list]
        total_size = sum(sizes)
        return shapes, dtypes, sizes, total_size

def average_weights(loaded_weights, current_weights):
        # Debug: Verify shapes match
        assert len(current_weights) == len(loaded_weights), "Weight lists have different lengths"
        for i, (w1, w2) in enumerate(zip(current_weights, loaded_weights)):
            assert w1.shape == w2.shape, f"Shape mismatch at layer {i}: {w1.shape} vs {w2.shape}"
        # Efficient vectorized averaging using numpy
        return [(w1 + w2) * 0.5 for w1, w2 in zip(current_weights, loaded_weights)]

def create_or_attach_shared_memory(name, size):
    try:
        # Try to attach to existing shared memory
        shm = shared_memory.SharedMemory(name=name, create=False)
        print(f"Attached to existing shared memory: {name}")
    except FileNotFoundError:
        # If it doesn't exist, create it
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        print(f"Created new shared memory: {name}")
    return shm

def load_subset(path):
    x_path = os.path.join(path, "x_subset.npy")
    y_path = os.path.join(path, "y_subset.npy")
    if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
        raise FileNotFoundError(f"Could not find subset files in {path}")
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y

#———————————————————————————————————————————————————————————————————
#  Shared Memory Functions
#  To write to shared memory
#———————————————————————————————————————————————————————————————————

def write_metadata_to_shm(meta_bytes, size):
    """Write model weight metadata to a shared memory segment"""
    # Create metadata shared memory
    metadata_shm = shared_memory.SharedMemory(
        name=f"{SHM_META_NAME}", 
        create=False
    )
    # Convert to bytes and write to shared memory
    metadata_shm.buf[:size] = meta_bytes
    metadata_shm.close()  # Close the metadata shared memory

def map_weights_to_shm(shm, wlist, shapes, dtypes, sizes):
    """Copy all arrays into the shared-memory block."""
    buf    = shm.buf
    offset = 0
    
    for w, shape, dtype, size in zip(wlist, shapes, dtypes, sizes):
        np_dtype = np.dtype(dtype)
        view = np.ndarray(shape, dtype=np_dtype, buffer=buf[offset:offset+size])
        view[:] = w  # copy data
        offset += size

#———————————————————————————————————————————————————————————————————
#  Shared Memory Functions
#  To read from shared memory
#———————————————————————————————————————————————————————————————————

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
            shape=shape,
            dtype=np_dtype,
            buffer=shm.buf[offset:offset+size]
        )
        # Make a copy to get independent array
        weights.append(array_view.copy())
        offset += size
    return weights

# Function to call from thread with queue to pipe info through the queue
def read_weights(queue):
    # Attach to existing shared memory
    # sem_go_py2go = Semaphore(SEM_GO_PY2GO, flags=0)  # Open the existing named semaphore (block until Go posts it)
    # sem_go_go2py = Semaphore(SEM_GO_GO2PY, flags=0)  # Open the existing named semaphore (block until Go posts it)
    shm = create_or_attach_shared_memory(name=SHM_NAME_GO, size=metadata["total_size"])  # Attach to the existing shared memory
    try:
        while True:
            sem_go_go2py.acquire()  # Wait for Go to post the signal that it's done writing weights
            weights = read_weights_from_shm(
                shm=shm,    
                shapes=metadata["shapes"],
                dtypes=metadata["dtypes"], 
                sizes=metadata["sizes"]
            )
            sem_go_py2go.release()  # Signal Go that Python is done reading weights
            queue.put(weights)  # Put the weights in the queue for the main thread to consume
    
    except Exception as e:
        print(f"Error setting up weight reader: {e}")
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
            while True:
                weight_list = queue.get()
                shm = create_or_attach_shared_memory(name=SHM_NAME, size=total_size)  # Attach to the existing shared memory
                # Copy into shm
                print(f"[Python] Acquiring semaphore to write → Go")
                map_weights_to_shm(shm, weight_list, shapes, dtypes, sizes)
                sem_py2go.release()    # signal Go: data ready
                print(f"[Python] Waiting for Go to finish reading…")
                sem_go2py.acquire()    # wait for Go to ack
                print(f"[Python] Go has read the weights; continuing training.")
        finally:
            try:
                if shm:
                    shm.close()
                    shm.unlink()  # Ensure shared memory is unlinked
            except Exception as e:
                print(f"Error cleaning up shared memory: {e}")
    
    def exchange_metadata_with_go(meta_bytes):
        """
        1) Python writes metadata → sem_meta.release()
        2) Python blocks on sem_go2py.acquire() until Go reads
        """
        metadata_size = len(meta_bytes)
        # Create metadata shared memory
        metadata_shm = shared_memory.SharedMemory(name=f"{SHM_META_NAME}", create=True, size=metadata_size)
        # Copy into shm
        write_metadata_to_shm(meta_bytes, len(meta_bytes))
        sem_meta.release()    # signal Go: data ready
        print(f"[Python] Waiting for Go to finish reading metadata…")
        sem_go2py.acquire()    # wait for Go to ack
        print(f"[Python] Go has read the metadata; Unlink from shared memory block.")
        metadata_shm.unlink()  # unlink the metadata shared memory
        print(f"[Python] Go has read the metadata; continuing training.")


    # ——————————————————————————————————————————————————————————————
    #  Build and Train Model
    # ——————————————————————————————————————————————————————————————

    # Load data
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (_, _), (x_test, y_test) = mnist.load_data()
    
    x_train, y_train = load_subset(f"subsets/subset_f{node_id}")
    x_train = x_train.astype('float32') / 255.0 # scale
    x_test = x_test.astype('float32') / 255.0

    # Generate a permutation of indices
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    # Shuffle the dataset
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, num_classes = 10)

    # Build a basic 2D CNN model
    model = Sequential([
        # First convolutional block
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Fully connected layers
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    # Define an exponential decay learning rate schedule:
    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=500,   # number of steps after which the learning rate decays
        decay_rate=0.90,      # the decay factor
        staircase=True)      # if True, learning rate decays in discrete intervals
    
    # Pass the schedule to the optimizer:
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, name="adam")
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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
        "sizes": sizes
    }
    meta_bytes = json.dumps(metadata).encode('utf-8')
    exchange_metadata_with_go(meta_bytes)
    
    #———————————————————————————————————————————————————————————————
    # Attach to go semaphores after metadata exchange
    #———————————————————————————————————————————————————————————————
    sem_go_py2go = Semaphore(SEM_GO_PY2GO, flags=0)  # Open the existing named semaphore (block until Go posts it)
    sem_go_go2py = Semaphore(SEM_GO_GO2PY, flags=0)  # Open the existing named semaphore (block until Go posts it)
    # Print the model summary to see its architecture
    model.summary()

    #——————————————————————————————————————————————————————————————
    #  Create a thread to read and send weights from Go
    #  Create queue to communicate between threads
    #——————————————————————————————————————————————————————————————
    send_queue = Queue()
    receive_queue = Queue()

    Threads = []

    t_read_weights = Thread(
        target=read_weights,
        args=(receive_queue,),
        name="read_weights_thread",
        daemon=False
    )
    Threads.append(t_read_weights)
    t_read_weights.start()

    t_write_weights = Thread(
        target=exchange_weights_with_go,
        args=(send_queue,),
        name="write_weights_thread",
        daemon=False
    )
    Threads.append(t_write_weights)
    t_write_weights.start()

    ###################################################################
    #  Train the model
    ###################################################################

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    num_epochs = 10
    batch_size = 64
    num_batches = math.ceil(x_train.shape[0] / batch_size)

    epoch_loss = 0
    epoch_acc = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
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
        
        # Extract weights at the end of epoch
        current_weights = model.get_weights()

        # 2. Hand them off to Go (and wait for Go to read)
        # exchange_weights_with_go(current_weights)
        send_queue.put(current_weights)  # Send weights to the thread for writing to shared memory
        # 3. (Optional) Here you would merge in peers’ weights from Go…
        try:
            incoming_weights = receive_queue.get_nowait()
        except Empty:
            incoming_weights = None
            pass
        if incoming_weights is not None:
            print("DEBUG: received weights from Go")
            print("WEIGHTS:")
            print(incoming_weights)
            # 4. Average them with the current weights
            new_weights = average_weights(current_weights, incoming_weights)
            # 5. Set the new weights in the model
            model.set_weights(new_weights)
        
        # model.set_weights(new_weights)
        
        # Optionally, evaluate on test set at epoch end
        loss, acc = model.evaluate(x_val, y_val, verbose=0)
        print(f"Validation loss: {loss:.4f}, Validation accuracy: {acc:.4f}")
        print(f"Epoch {epoch+1} - loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


    # Evaluate the model on the test set
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

except Exception as e:
    print(f"An error occurred: {e}")
except KeyboardInterrupt:
    # User pressed Ctrl+C
    print("KeyboardInterrupt: Exiting...")
    for thread in Threads:
        thread.join(2)
        time.sleep(1)
        if thread.is_alive():
            print(f"Thread {thread.name} is still alive after 1 second.")
        else:
            print(f"Thread {thread.name} has finished.")
except EOFError:
    print("EOFError: The end of the file was reached unexpectedly.")
except MemoryError:
    print("MemoryError: Not enough memory to allocate the shared memory block.")
finally:
    try:
        sem_py2go.release()   
        sem_py2go.unlink()
        sem_go2py.unlink()
        sem_meta.unlink()
        sem_go_py2go.unlink()
        sem_go_go2py.unlink()
    except:
        pass
