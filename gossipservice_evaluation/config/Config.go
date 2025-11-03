package config

import (
	"encoding/json"
	"os"

	log "github.com/sirupsen/logrus"
)

var Cfg *Config

type Config struct {
	DiscoAPIPort int `json:"discoAPIPort"`
	//RepoAPIPort  int `json:"repoAPIPort"`
	//MaxPing      float32 `json:"maxPing"`
	NodeID string `json:"nodeID"`
	//NodeType string `json:"nodeType"`
	//PingPeriod   int     `json:"pingPeriod"`
	//TestMode     bool    `json:"testMode"`
	//FogIP                string             `json:"fogIP"`
	//CheatyMinimalPing    float32            `json:"cheatyMinimalPing"`
	//CheatyMinimalPingMap map[string]float32 `json:"cheatyMinimalPingMap"`
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
		log.Errorf("Load config error %v", err.Error())
		return err
	}
	decoder := json.NewDecoder(file)
	Cfg = &Config{}
	err = decoder.Decode(Cfg)
	if err != nil {
		log.Errorf("Load config error %v", err.Error())
		return err
	}

	//log.Infof("NodeID check %s", Cfg.NodeID)
	if os.Getenv("NODEID") != "" {
		//log.Info("Loading nodeID from env instead")
		Cfg.NodeID = os.Getenv("NODEID")
	}

	return err
}
