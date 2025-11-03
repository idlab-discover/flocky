package gossiping

import (
	"encoding/json"
	"fmt"
	"os"
)

var Cfg *Config

type Config struct {
	SharedMemory struct {
		MetadataName string `json:"metadata_name"`
		WeightsName  string `json:"weights_name"`
		MetaPath     string `json:"metaPath"`
		ShmPath      string `json:"shmPath"`
	} `json:"shared_memory"`
	Semaphores struct {
		Meta  string `json:"metadata"`
		Py2Go string `json:"python_to_go"`
		Go2Py string `json:"go_to_python"`
	} `json:"semaphores"`
	SharedMemoryGo2Py struct {
		WeightsName string `json:"weights_name"`
		ShmPath     string `json:"shmPath"`
	} `json:"shared_memory_go2py"`
	SemaphoresGo2Py struct {
		Py2Go string `json:"python_to_go"`
		Go2Py string `json:"go_to_python"`
	} `json:"semaphores_go2py"`
}

func LoadConfig(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		fmt.Printf("Load config error %v", err.Error())
		return err
	}
	decoder := json.NewDecoder(file)
	Cfg = &Config{}
	err = decoder.Decode(Cfg)
	if err != nil {
		fmt.Printf("Load config error %v", err.Error())
		return err
	}
	return err
}
