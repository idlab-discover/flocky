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
	"fmt"
	"os"
	"syscall"
	"time"
	"unsafe"

	log "github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

type ReadWriteMemoryChannel struct {
	//SemGo2PyName, SemPy2GoName *C.char
	ShmFilePath, Name        string
	SemWait, SemPost         *C.sem_t
	semWaitName, semPostName string
	Data                     []byte
	File                     *os.File
	shFileCreator            bool
	active                   bool
}

func (channel *ReadWriteMemoryChannel) Init(shmFilePath string, semWaitName string, semSignalName string, id string, expectedSize int64, create bool) error {
	channel.semWaitName = semWaitName + id
	channel.semPostName = semSignalName + id
	channel.shFileCreator = create

	channel.SemWait = createOrAttachToSemaphore(channel.semWaitName)
	if channel.SemWait == nil {
		log.Fatalf("Failed to create semaphore wait %s", channel.semWaitName)
	}
	channel.SemPost = createOrAttachToSemaphore(channel.semPostName)
	if channel.SemPost == nil {
		log.Fatalf("Failed to create semaphore post %s", channel.semPostName)
	}

	channel.ShmFilePath = shmFilePath
	channel.Name = id

	path := fmt.Sprintf("%s%s", shmFilePath, id)
	var err error
	if create {
		channel.File, channel.Data, err = createSharedMemory(path, int(expectedSize))
	} else {
		channel.File, channel.Data, err = openSharedMem(path, expectedSize)
	}
	if err != nil {
		log.Fatalf("Error occured opening setting up shared memory")
		return err
	}
	channel.active = true
	return nil
}

func (channel *ReadWriteMemoryChannel) StartRead(parser func([]byte)) {
	go func() {
		for channel.active {
			if C.sem_wait(channel.SemWait) != 0 {
				log.Errorf("sem_wait failed %s", channel.Name)
			}

			// Move defer outside the loop to prevent stack growth
			parser(channel.Data)

			// Signal Python explicitly instead of using defer
			if C.sem_post(channel.SemPost) != 0 {
				log.Errorf("sem_post failed %s", channel.Name)
			}
		}
	}()
}

func (channel *ReadWriteMemoryChannel) StartWrite(itemProvider func() [][]byte) {
	go func() {
		items := [][]byte{}
		for channel.active {
			// Move defer outside the loop to prevent stack growth
			newItems := itemProvider()
			items = append(items, newItems...)

			if len(items) > 0 {
				curItem := items[0]

				copy(curItem, channel.Data)

				success := true
				if C.sem_post(channel.SemPost) != 0 {
					log.Errorf("sem_post failed %s", channel.Name)
					success = false
				}

				// Signal Python explicitly instead of using defer
				if C.sem_wait(channel.SemWait) != 0 {
					log.Errorf("sem_wait failed %s", channel.Name)
					success = false
				}
				if success {
					items = items[1:]
				}
			}
		}
	}()
}

// really shifty
func (channel *ReadWriteMemoryChannel) Write(items [][]byte) {
	go func() {
		for _, curItem := range items {
			// Move defer outside the loop to prevent stack growth

			copy(curItem, channel.Data)

			success := true
			if C.sem_post(channel.SemPost) != 0 {
				log.Errorf("sem_post failed %s", channel.Name)
				success = false
			}

			// Signal Python explicitly instead of using defer
			if C.sem_wait(channel.SemWait) != 0 {
				log.Errorf("sem_wait failed %s", channel.Name)
				success = false
			}
			if !success {
				break
			}
		}
	}()
}

func openSharedMem(filename string, fileSize int64) (*os.File, []byte, error) {
	//-------------------------------------------------------
	// Open shared memory file once
	//-------------------------------------------------------
	const retryInterval = 1 * time.Second
	var file *os.File
	var err error
	for {
		file, err = os.OpenFile(filename, os.O_RDONLY, 0)
		if err == nil {
			log.Printf("Opened shared mem file successfully")
			break
		}
		// Log the failure and retry after a pause
		log.Printf("Failed to open shared mem file: retrying in %s...", retryInterval)
		time.Sleep(retryInterval)
	}

	fi, err := file.Stat()

	if err != nil {
		log.Fatalf("stat shm: %v", err)
		return nil, nil, err
	}
	if fi.Size() < fileSize {
		log.Fatalf("shared memory file too small: got %d, want %d", fi.Size(), fileSize)
		return nil, nil, err
	}

	//---------------------------------------------------------
	// Map memory once
	//---------------------------------------------------------
	data, err := syscall.Mmap(
		int(file.Fd()), 0, int(fileSize),
		unix.PROT_READ,
		unix.MAP_SHARED,
	)
	if err != nil {
		log.Fatalf("mmap: %v", err)
		return nil, nil, err
	}

	return file, data, nil
}

func createSharedMemory(name string, size int) (*os.File, []byte, error) {
	// Create shared memory writeFile
	writeFile, err := os.OpenFile(name, os.O_RDWR, 0644)
	if os.IsNotExist(err) {
		// If it doesn't exist, create it
		writeFile, err = os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create shm writeFile: %v", err)
		}

		// Set the size of the file
		if err := writeFile.Truncate(int64(size)); err != nil {
			writeFile.Close()
			return nil, nil, fmt.Errorf("failed to set shm size: %v", err)
		}
	} else if err != nil {
		return nil, nil, fmt.Errorf("failed to open shm writeFile: %v", err)
	}

	// Map the memory
	data, err := syscall.Mmap(
		int(writeFile.Fd()),
		0,
		size,
		unix.PROT_READ|unix.PROT_WRITE, // Note: Added PROT_WRITE for creation
		unix.MAP_SHARED,
	)
	if err != nil {
		writeFile.Close()
		return nil, nil, fmt.Errorf("mmap failed: %v", err)
	}

	return writeFile, data, nil
}

func (channel *ReadWriteMemoryChannel) Cleanup() {
	channel.active = false
	go2pyName := C.CString(channel.semWaitName)
	py2goName := C.CString(channel.semPostName)
	if channel.SemWait != nil {
		C.sem_close(channel.SemWait)
		time.Sleep(10 * time.Millisecond) // give kernel a moment
		C.sem_unlink(go2pyName)
	}
	if channel.SemPost != nil {
		C.sem_close(channel.SemPost)
		time.Sleep(10 * time.Millisecond)
		C.sem_unlink(py2goName)
	}
	C.free(unsafe.Pointer(go2pyName))
	C.free(unsafe.Pointer(py2goName))

	unix.Munmap(channel.Data)
	channel.File.Close()
	if channel.shFileCreator {
		// Remove the shared memory writeFile
		os.Remove(fmt.Sprintf("%s%s", channel.ShmFilePath, channel.Name))
	}
}

func createOrAttachToSemaphore(name string) *C.sem_t {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var sem *C.sem_t
	sem = C.sem_open_no_create(cName)
	if sem == nil {
		log.Infof("Failed to open semaphore: %s", name)
		log.Info("Trying to create semaphore")
		sem = C.sem_open_create(cName, 0644, 0)
	}
	return sem
}
