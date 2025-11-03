print("[Python] Starting execution...")
from multiprocessing import shared_memory
import sys
import numpy as np
from posix_ipc import Semaphore, O_CREAT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from threading import Thread
from queue import Queue, Empty

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
        config = json.load(f)
    print(f"Loaded config: {config}")
    return config

gossip_config = load_config("config.json")

config_file = sys.argv[1] if len(sys.argv) > 1 else "defaultConfig.json"
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
print("[Python] Starting execution...")
def get_weight_info(arrays):
    shapes = [a.shape for a in arrays]
    dtypes = [str(a.dtype) for a in arrays]
    sizes  = [a.nbytes     for a in arrays]
    total = sum(sizes)
    return shapes, dtypes, sizes, total

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
    sem2 = Semaphore(SEM_GO_PY2GO, flags=0)  # Open the existing named semaphore (block until Go posts it)
    sem1 = Semaphore(SEM_GO_GO2PY, flags=0)  # Open the existing named semaphore (block until Go posts it)
    shm = create_or_attach_shared_memory(name=SHM_NAME_GO, size=metadata["total_size"])  # Attach to the existing shared memory
    try:
        while True:
            sem1.acquire()  # Wait for Go to post the signal that it's done writing weights
            weights = read_weights_from_shm(
                shm=shm,
                shapes=metadata["shapes"],
                dtypes=metadata["dtypes"], 
                sizes=metadata["sizes"]
            )
            sem2.release()  # Signal Go that Python is done reading weights
            queue.put(weights)  # Put the weights in the queue for the main thread to consume
    
    except Exception as e:
        print(f"Error setting up weight reader: {e}")
    finally:
        # Cleanup
        try:
            shm.close()
            sem1.close()
            sem2.close()
        except:
            pass

print("[Python] Starting execution...")
try:
    shapes = None
    dtypes = None
    sizes = None
    total_size = None

    #################################################################
    # Create (or open) the semaphores
    #################################################################
    # Create semaphores for Py2Go
    print("[Python] Making semaphores")
    sem_py2go = Semaphore(SEM_PY2GO, O_CREAT, initial_value=0)
    sem_go2py = Semaphore(SEM_GO2PY, O_CREAT, initial_value=0)
    sem_meta = Semaphore(SEM_META, O_CREAT, initial_value=0)
    print("[Python] Semaphores made")
    ###################################################################
    # Initialize shared memory functions 
    # with the correct use of the semaphores
    ###################################################################
    print("[Python] Initializing functions...")
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
        metadata_shm = create_or_attach_shared_memory(SHM_META_NAME, metadata_size)
        # Copy into shm
        write_metadata_to_shm(meta_bytes, len(meta_bytes))
        sem_meta.release()    # signal Go: data ready
        print(f"[Python] Waiting for Go to finish reading metadata…")
        sem_go2py.acquire()    # wait for Go to ack
        print(f"[Python] Go has read the metadata; Unlink from shared memory block.")
        metadata_shm.unlink()  # unlink the metadata shared memory
        print(f"[Python] Go has read the metadata; continuing training.")

    print("[Python] Functions initialized")

    # ——————————————————————————————————————————————————————————————
    #  PyTorch Model Definition
    # ——————————————————————————————————————————————————————————————
    print("[Python] Defining model...")
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.pool  = nn.MaxPool2d(2,2)
            self.drop1 = nn.Dropout(0.2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.drop2 = nn.Dropout(0.3)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32*7*7, 32)
            self.drop3 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x); x = self.drop1(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x); x = self.drop2(x)
            x = self.flatten(x)
            x = F.relu(self.fc1(x)); x = self.drop3(x)
            return self.fc2(x)

    # Device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model = SimpleCNN().to(device)
    print("[Python] Done")
    # ——————————————————————————————————————————————————————————————
    #  Data Loading (NumPy subsets + torchvision MNIST)
    # ——————————————————————————————————————————————————————————————
    print("[Python] Loading subsets...")
    def load_subset(path):
        x = np.load(os.path.join(path, "x_subset.npy"))
        y = np.load(os.path.join(path, "y_subset.npy"))
        return x, y

    print("[Python] Loading training sets...")
    # Train subset
    x_t, y_t = load_subset(f"subsets/subset_{node_id}")
    x_t = (x_t.astype('float32')/255.0)[:,None,:,:]
    y_t = y_t.astype('long')

    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(x_t, y_t, test_size=0.2, random_state=42)

    def make_loader(X, Y, batch=64, shuffle=True):
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=batch, shuffle=shuffle)

    print("[Python] Make train loader and val loader...")
    train_loader = make_loader(X_tr, y_tr)
    val_loader   = make_loader(X_val, y_val, shuffle=False)
    print("[DEBUG] Before importing torchvision", flush=True)
    from torchvision import datasets, transforms
    print("[DEBUG] After importing torchvision", flush=True)

    # Debug dataset loading
    print("[DEBUG] Loading dataset...", flush=True)
    # Test loader from torchvision
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    test_ds = datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # ——————————————————————————————————————————————————————————————
    #  Optimizer, Scheduler, Loss
    # ——————————————————————————————————————————————————————————————
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    criterion = nn.CrossEntropyLoss()

    # ——————————————————————————————————————————————————————————————
    #  Get weights and metadata on the weights
    #  (and write to shared memory)
    #  This is once per model, not per epoch
    #  (unless you change the model architecture)
    # ——————————————————————————————————————————————————————————————

    # Metadata setup
    print("[Python] Getting model dict...")
    init_sd = model.state_dict()
    init_arr = [t.cpu().numpy() for t in init_sd.values()]
    sh, dt, sz, tot = get_weight_info(init_arr)
    metadata = {"shapes":sh, "dtypes":dt, "sizes":sz, "total_size":tot}
    meta_bytes = json.dumps(metadata).encode('utf-8')
    print("[Python] Encoding done metadata...")

    # Create shared memory for weights
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=tot)
    metadata = {
        "total_size": tot,
        "shapes": sh,
        "dtypes": dt,  # Assuming all weights are float32
        "sizes": sz
    }
    meta_bytes = json.dumps(metadata).encode('utf-8')
    exchange_metadata_with_go(meta_bytes)

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

    num_epochs = 10
    batch_size = 64
    num_batches = math.ceil(x_train.shape[0] / batch_size)

    epoch_loss = 0
    epoch_acc = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        epoch_acc = 0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, Yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Extract weights at the end of epoch
        sd = model.state_dict()
        curr = [t.cpu().detach().numpy() for t in sd.values()]
        send_queue.put(curr)

        # 3. (Optional) Here you would merge in peers’ weights from Go…
        try:
            incoming_weights = receive_queue.get_nowait()
        except Empty:
            incoming_weights = None
            pass
        if incoming_weights:
            print("DEBUG: received weights from Go")
            print("WEIGHTS:")
            print(incoming_weights)
            # 4. Average them with the current weights
            new_weights = average_weights(sd, incoming_weights)
            # 5. Set the new weights in the model
            model.load_state_dict(new_weights)
        
        # Evaluation
        model.eval()
    def eval_dl(dl):
        tot_l, c, tot = 0,0,0
        with torch.no_grad():
            for Xb, Yb in dl:
                Xb, Yb = Xb.to(device), Yb.to(device)
                out = model(Xb)
                tot_l += criterion(out, Yb).item()*Xb.size(0)
                c     += (out.argmax(1)==Yb).sum().item()
                tot   += Xb.size(0)
        return tot_l/tot, c/tot

    vl, va = eval_dl(val_loader)
    tl, ta = eval_dl(test_loader)
    print(f"Epoch {epoch} | Val Acc: {va:.4f} | Test Acc: {ta:.4f}")



except Exception as e:
    print(f"An error occurred: {e}")
except KeyboardInterrupt:
    # User pressed Ctrl+C
    print("KeyboardInterrupt: Exiting...")
    for thread in Threads:
        thread.join()
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
        if shm:
            shm.close()
            shm.unlink()  # Ensure shared memory is unlinked
    except Exception as e:
        print(f"Error cleaning up shared memory: {e}")
    try:
        sem_py2go.release()   
        sem_py2go.unlink()
        sem_go2py.unlink()
        sem_meta.unlink()
    except:
        pass
