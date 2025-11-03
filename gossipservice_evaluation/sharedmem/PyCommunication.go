package sharedmem

// #cgo LDFLAGS: -lrt
// #include <sys/stat.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <semaphore.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <string.h>
// // Open without creation (variadic args stripped out):
// static sem_t* sem_open_no_create(const char* name) {
//     return sem_open(name, 0);
// }
// // Open with creation, using a default mode and initial value:
// static sem_t* sem_open_create(const char* name, mode_t mode, unsigned int val) {
//     return sem_open(name, O_CREAT, mode, val);
// }
import "C"

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"test/mmap/config"
	"time"
	"unsafe"

	log "github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

var iterations int
var metrics_file_name string

var Metadata WeightMetadata

type WeightMetadata struct {
	TotalSize int      `json:"total_size"`
	Shapes    [][]int  `json:"shapes"`
	DTypes    []string `json:"dtypes"`
	Sizes     []int    `json:"sizes"`
}

/*func getPy2GoSemaphoreName() string {
	return config.Cfg.Semaphores.Py2Go + config.Cfg.NodeID
}

func getGo2PySemaphoreName() string {
	return config.Cfg.Semaphores.Go2Py + config.Cfg.NodeID
}

func getPy2GoShmSemaphoreName() string {
	return config.Cfg.SemaphoresGo2Py.Py2Go + config.Cfg.NodeID
}

func getGo2PyShmSemaphoreName() string {
	return config.Cfg.SemaphoresGo2Py.Go2Py + config.Cfg.NodeID
}

func getGo2PyShmPath() string {
	return config.Cfg.SharedMemoryGo2Py.ShmPath + config.Cfg.NodeID
}*/

// -------------------------------------------------------
// Initialize cleanup function and semaphore variables
// -------------------------------------------------------
/*var (
	sem_Go_Go2Py_name, sem_Go_Py2Go_name *C.char
	sem_Go_Go2Py, sem_Go_Py2Go           *C.sem_t
)*/

var readShmem, writeShmem *ReadWriteMemoryChannel

func Init(prefix string, readCh chan WeightUpdate, weightCh chan WeightUpdate, nCh chan int) {
	iterations = 0

	metrics_file_name = fmt.Sprintf("gossip_metrics_%s.csv", prefix)
	log.Infof("Metrics file name is %s", metrics_file_name)

	//-------------------------------------------------------
	// Make semaphores to write so python can open this
	//-------------------------------------------------------
	//sem_Go_Go2Py_name = C.CString(getGo2PySemaphoreName())
	//sem_Go_Py2Go_name = C.CString(getPy2GoSemaphoreName())
	// Create semaphores
	/*sem_Go_Go2Py = C.sem_open_create(sem_Go_Go2Py_name, 0644, 0)
	if sem_Go_Go2Py == nil {
		log.Fatal("Failed to create sem_Go_Go2Py")
	}*/
	//defer C.sem_close((*C.sem_t)(sem_Go_Go2Py))
	//defer C.sem_unlink(sem_Go_Go2Py_name)

	/*sem_Go_Py2Go = C.sem_open_create(sem_Go_Py2Go_name, 0644, 0)
	if sem_Go_Py2Go == nil {
		log.Fatal("Failed to create sem_Go_Py2Go")
	}*/
	//defer C.sem_close((*C.sem_t)(sem_Go_Py2Go))
	//defer C.sem_unlink(sem_Go_Py2Go_name)
	ReadMetadata()
	readShmem = &ReadWriteMemoryChannel{}
	readShmem.Init(config.Cfg.SharedMemory.ShmPath, config.Cfg.Semaphores.Py2Go, config.Cfg.Semaphores.Go2Py, config.Cfg.NodeID, 2*int64(Metadata.TotalSize), false)
	readShmem.StartRead(MakeReadWeights(readCh))

	writeShmem = &ReadWriteMemoryChannel{}
	writeShmem.Init(config.Cfg.SharedMemoryGo2Py.ShmPath, config.Cfg.SemaphoresGo2Py.Py2Go, config.Cfg.SemaphoresGo2Py.Go2Py, config.Cfg.NodeID, 2*int64(Metadata.TotalSize)+20, true)
	writeShmem.StartWrite(MakeCopyWeightsToShm(weightCh))
}

func Cleanup() {
	log.Infof("Cleaning up sharedmem")
	readShmem.Cleanup()
	writeShmem.Cleanup()
}

//-------------------------------------------------------------
// Functions to read from shared memory
//-------------------------------------------------------------

func ReadMetadata() {
	// Open metadata semaphore
	cNameMeta := C.CString(config.Cfg.Semaphores.Meta + config.Cfg.NodeID)
	defer C.free(unsafe.Pointer(cNameMeta))

	//var metaSemaphore *C.sem_t
	metaSemaphore := createOrAttachToSemaphore(config.Cfg.Semaphores.Meta + config.Cfg.NodeID)
	if metaSemaphore == nil {
		log.Fatal("Failed to open metadata semaphore")
	}
	defer C.sem_close((*C.sem_t)(metaSemaphore))
	defer C.sem_unlink(cNameMeta)
	cName2 := C.CString(config.Cfg.Semaphores.Go2Py + config.Cfg.NodeID)
	defer C.free(unsafe.Pointer(cName2))
	// Open semaphores
	//var sem2 *C.sem_t
	sem2 := createOrAttachToSemaphore(config.Cfg.Semaphores.Go2Py + config.Cfg.NodeID)
	if sem2 == nil {
		log.Info("Failed to open ", config.Cfg.Semaphores.Go2Py+config.Cfg.NodeID)
	}
	// Wait for Python to write metadata
	log.Info("[Go] Waiting for metadata...")
	if C.sem_wait((*C.sem_t)(metaSemaphore)) != 0 {
		log.Fatal("sem_wait failed for metadata")
	}
	log.Info("[Go] Detected new metadata; reading...")

	const retryInterval = 2 * time.Second
	var f *os.File
	var err error
	for {
		f, err = os.OpenFile(config.Cfg.SharedMemory.MetaPath+config.Cfg.NodeID, os.O_RDONLY, 0)
		if err == nil {
			log.Printf("Opened metadatafile successfully")
			break
		}
		// Log the failure and retry after a pause
		log.Printf("Failed to open metdatafile: retrying in %s...", retryInterval)
		time.Sleep(retryInterval)
	}

	// Read metadata bytes
	rawMeta, err := io.ReadAll(f)
	f.Close()
	if err != nil {
		log.Fatalf("read metadata shm: %v", err)
	}
	if C.sem_post(sem2) != 0 {
		log.Fatal("sem_post(sem_go2py) failed")
	}
	// 4 Unmarshal into struct
	var md WeightMetadata
	if err := json.Unmarshal(rawMeta, &md); err != nil {
		log.Fatalf("unmarshal metadata: %v", err)
	}
	log.Info("md:", md)
	Metadata = md
}

//type ReadChannelFunc func([]byte)

func MakeReadWeights(outCh chan WeightUpdate) func([]byte) {
	return func(data []byte) {
		weights := make([][]float32, len(Metadata.Shapes))
		update := make([][]float32, len(Metadata.Shapes))
		offset := 0

		for i, shape := range Metadata.Shapes {
			elements := 1
			for _, dim := range shape {
				elements *= dim
			}
			size := elements * 4 // float32 = 4 bytes
			sliceData := &reflect.SliceHeader{
				Data: uintptr(unsafe.Pointer(&data[offset])),
				Len:  size / 4,
				Cap:  size / 4,
			}
			weights[i] = *(*[]float32)(unsafe.Pointer(sliceData))
			offset += size
		}

		for i, shape := range Metadata.Shapes {
			elements := 1
			for _, dim := range shape {
				elements *= dim
			}
			size := elements * 4 // float32 = 4 bytes
			sliceData := &reflect.SliceHeader{
				Data: uintptr(unsafe.Pointer(&data[offset])),
				Len:  size / 4,
				Cap:  size / 4,
			}
			update[i] = *(*[]float32)(unsafe.Pointer(sliceData))
			offset += size
		}

		wupdate := WeightUpdate{
			BaseWeights: weights,
			Update:      update,
		}
		select {
		case outCh <- wupdate:
			log.Info("Weights sent to channel")
		default:
			<-outCh          // Discard weights if channel is full
			outCh <- wupdate // Send new weights
			log.Info("Weights sent to channel")
		}
	}
}

/*func ReadWeightsFromShm(weightCh chan [][]float32) {
	var py2goName = getGo2PySemaphoreName()
	var go2pyName = getPy2GoSemaphoreName()

	var (
		sem1, sem2 *C.sem_t
		data       []byte
	)
	//-------------------------------------------------------
	// Open existing semaphores
	//-------------------------------------------------------
	cName1 := C.CString(py2goName)
	cName2 := C.CString(go2pyName)

	log.Info("Semaphore names:", py2goName, cName2)
	// Open semaphores
	// sem1 = C.sem_open_no_create(cName1)
	sem1 = createOrAttachToSemaphore(py2goName)
	if sem1 == nil {
		log.Fatal("Failed to open sem_py2go")
	}

	// sem2 = C.sem_open_no_create(cName2)
	sem2 = createOrAttachToSemaphore(go2pyName)
	if sem2 == nil {
		log.Fatal("Failed to open or create %s", cName2)
	}
	//-------------------------------------------------------
	// Open shared memory file once
	//-------------------------------------------------------
	// file, err := os.OpenFile(gossipConfig.Cfg.SharedMemory.ShmPath+config.Cfg.NodeID, os.O_RDWR, 0)
	// if err != nil {
	// log.Fatalf("open shm: %v", err)
	// }
	const retryInterval = 2 * time.Second
	var file *os.File
	var err error
	for {
		file, err = os.OpenFile(config.Cfg.SharedMemory.ShmPath+config.Cfg.NodeID, os.O_RDONLY, 0)
		if err == nil {
			log.Printf("Opened metadatafile successfully")
			break
		}
		// Log the failure and retry after a pause
		log.Printf("Failed to open metdatafile: retrying in %s...", retryInterval)
		time.Sleep(retryInterval)
	}

	// After opening file, before mmap:
	fi, err := file.Stat()

	if err != nil {
		log.Fatalf("stat shm: %v", err)
	}
	if fi.Size() < int64(Metadata.TotalSize) {
		log.Fatalf("shared memory file too small: got %d, want %d", fi.Size(), Metadata.TotalSize)
	}
	//---------------------------------------------------------
	// Map memory once
	//---------------------------------------------------------
	data, err = syscall.Mmap(
		int(file.Fd()), 0, Metadata.TotalSize,
		unix.PROT_READ,
		unix.MAP_SHARED,
	)
	if err != nil {
		log.Fatalf("mmap: %v", err)
	}
	//--------------------------------------------------------
	// Setup cleanup in correct order
	//--------------------------------------------------------
	defer func() {
		// Cleanup semaphores
		C.sem_close(sem1)
		C.sem_close(sem2)
		C.sem_unlink(cName1)
		C.sem_unlink(cName2)
		// Free C strings
		C.free(unsafe.Pointer(cName1))
		C.free(unsafe.Pointer(cName2))
		// Cleanup memory mapping
		unix.Munmap(data)
		file.Close()
	}()
	//----------------------------------------------------------
	// Main loop: wait for Python, read weights, signal Python
	//----------------------------------------------------------
	for {
		if C.sem_wait(sem1) != 0 {
			fmt.Errorf("sem_wait failed")
		}

		// Move defer outside the loop to prevent stack growth
		weights := readWeights(data, Metadata)

		// Signal Python explicitly instead of using defer
		if C.sem_post(sem2) != 0 {
			fmt.Errorf("sem_post failed")
		}
		select {
		case weightCh <- weights:
			log.Info("Weights send to channel")
		default:
			<-weightCh          // Discard weights if channel is full
			weightCh <- weights // Send new weights
			log.Info("Weights send to channel")
		}
	}
}*/

// -------------------------------------------------------------
// Write to shared memory
// -------------------------------------------------------------
// Create and map new shared memory

func MakeCopyWeightsToShm(weightCh chan WeightUpdate) func([]byte) error {

	return func(data []byte) error {
		update := <-weightCh
		//update := <-updateCh
		offset := 0

		// Iterate through each weight array
		/*for i, weightArray := range weights {
			// Calculate size in bytes for this weight array
			size := len(weightArray) * int(unsafe.Sizeof(weightArray[0]))
			if offset+size > len(data) {
				return fmt.Errorf("not enough space in shared memory for weight array %d", i)
			}

			// Create a byte slice view of the float32 array
			weightBytes := (*[1 << 30]byte)(unsafe.Pointer(&weightArray[0]))[:size:size]

			// Copy the bytes to shared memory at the correct offset
			copy(data[offset:offset+size], weightBytes)

			// Update offset for next array
			offset += size
		}*/
		offset, err := matrixMemCopy(update.BaseWeights, data, 0)
		if err != nil {
			return err
		}
		offset, err = matrixMemCopy(update.Update, data, offset)
		if err != nil {
			return err
		}

		//change this shit up: go with the idea to move everything currently in python to go so we can store models + node ids here,
		// calculate the new model here even if it's slower fuck it all
		// workflow is: python sends new weights, store them as "local node"
		// receive weights from gossip: store them as "node id"
		// after x updates (epochs) from python (configurable), integrate the entire shebang and send it back to python
		// python thing should need almost no modification, since it can just keep on doing its thing if golang doesn't send
		// anything in a specific epoch
		name := update.NodeID
		if len(update.NodeID) < 20 {
			name = fmt.Sprintf("%s%s", update.NodeID, strings.Repeat(" ", 20-len(update.NodeID)))
		}

		copy(data[offset:offset+20], []byte(name)[:20])

		/*for i, weightArray := range update {
			// Calculate size in bytes for this weight array
			size := len(weightArray) * int(unsafe.Sizeof(weightArray[0]))
			if offset+size > len(data) {
				return fmt.Errorf("not enough space in shared memory for weight array %d", i)
			}

			// Create a byte slice view of the float32 array
			weightBytes := (*[1 << 30]byte)(unsafe.Pointer(&weightArray[0]))[:size:size]

			// Copy the bytes to shared memory at the correct offset
			copy(data[offset:offset+size], weightBytes)

			// Update offset for next array
			offset += size
		}*/

		// Ensure changes are written to disk
		if err := unix.Msync(data, unix.MS_SYNC); err != nil {
			return fmt.Errorf("msync failed: %v", err)
		}

		return nil
	}
}

func matrixMemCopy(matrix [][]float32, data []byte, offset int) (int, error) {
	for i, weightArray := range matrix {
		// Calculate size in bytes for this weight array
		size := len(weightArray) * int(unsafe.Sizeof(weightArray[0]))
		if offset+size > len(data) {
			return 0, fmt.Errorf("not enough space in shared memory for weight array %d", i)
		}

		// Create a byte slice view of the float32 array
		weightBytes := (*[1 << 30]byte)(unsafe.Pointer(&weightArray[0]))[:size:size]

		// Copy the bytes to shared memory at the correct offset
		copy(data[offset:offset+size], weightBytes)

		// Update offset for next array
		offset += size
	}
	return offset, nil
}

/*func WriteWeightsToShm(weightCh <-chan [][]float32, nCh <-chan int) {

	// initialize semaphore names
	var (
		semGo2Py, semPy2Go *C.sem_t
		data               []byte
		writeFile          *os.File
		n                  int
	)
	semGo2Py_name := C.CString(getGo2PyShmSemaphoreName())
	semPy2Go_name := C.CString(getPy2GoShmSemaphoreName())

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	//------------------------------------------------------------
	// Define cleanup function
	//------------------------------------------------------------
	cleanup := func() {
		log.Info("\nCleaning up resources...")

		// Close and unlink semaphores
		if semGo2Py != nil {
			C.sem_close(semGo2Py)
			// Small sleep to ensure close completes
			time.Sleep(10 * time.Millisecond)
			C.sem_unlink(semGo2Py_name)
		}
		if semPy2Go != nil {
			C.sem_close(semPy2Go)
			time.Sleep(10 * time.Millisecond)
			C.sem_unlink(semPy2Go_name)
		}

		// Free C strings
		if semGo2Py_name != nil {
			C.free(unsafe.Pointer(semGo2Py_name))
		}
		if semPy2Go_name != nil {
			C.free(unsafe.Pointer(semPy2Go_name))
		}
		// Cleanup shared memory
		if data != nil {
			unix.Munmap(data)
		}
		if writeFile != nil {
			writeFile.Close()
			// Remove the shared memory writeFile
			os.Remove(getGo2PyShmPath())
		}

		log.Info("Cleanup complete")
	}

	// Handle interrupt in goroutine
	go func() {
		<-sigChan
		cleanup()
		os.Exit(0)
	}()

	// Also use defer for normal program termination
	defer cleanup()

	// Create semaphores
	semGo2Py = createOrAttachToSemaphore(getGo2PyShmSemaphoreName())
	semPy2Go = createOrAttachToSemaphore(getGo2PyShmSemaphoreName())

	var err error
	// Usage example:
	data, writeFile, err = createSharedMemory(getGo2PyShmPath(), Metadata.TotalSize)
	if err != nil {
		log.Fatalf("Failed to create shared memory: %v", err)
	}

	var weights [][]float32
	for {
		log.Info("[Go] Write weights -> Python")
		n = <-nCh
		if n == 0 || n == 1 {
			//weights = <-weightCh
		} else {
			//weights = <-weightCh
			//weights = take_avarage(n, weights)
		}
		weights = <-weightCh
		copyWeightsToShm(data, weights, Metadata)
		if C.sem_post(semGo2Py) != 0 {
			log.Fatal("sem_post(sem_go2py) failed")
		}
		log.Info("[Go] Released semaphore to write -> Python")
		if C.sem_wait(semPy2Go) != 0 {
			log.Fatal("sem_wait(sem_py2go) failed")
		}
		//err := AppendMetrics(metrics_file_name, config.Cfg.NodeID, iterations, n, neighbour_count, time.Now())
		iterations++
		if err != nil {
			log.Errorf("Error appending metrics: %v", err)
		}
		log.Info("[Go] Acquired semaphore to read <- Python")
	}
}*/
