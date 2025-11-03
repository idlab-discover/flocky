package config

import (
	"encoding/json"
	"os"

	log "github.com/sirupsen/logrus"
)

var Cfg *Config

type Config struct {
	RepoAPIPort        int    `json:"repoPort"`
	DiscoAPIPort       int    `json:"discoPort"`
	FledgeAPIPort      int    `json:"fledgePort"`
	DeploymentAPIPort  int    `json:"deploymentPort"`
	SwirlyAPIPort      int    `json:"swirlyPort"`
	ServiceMonitorType string `json:"serviceMonitorType"`
	ServiceLocatorType string `json:"serviceLocatorType"`
	//MaxPing            float32 `json:"maxPing"`
	NodeID   string `json:"nodeID"`
	TestMode bool   `json:"testMode"`
	//SupportComponents  map[string][]oam.Component `json:"supportServices"`

	//MonitorServices    []string `json:"monitorServices"`

	//InitialNodes      map[string]string          `json:"initialNodes"`

	//DeploymentPort int `json:"fogPort"`
	//FetchFogURL            string `json:"fetchFogUrl"`
	//FogServiceRunningURL   string `json:"fogServiceRunningUrl"`
	//AddServiceClientURL    string `json:"addServiceClientUrl"`
	//RemoveServiceClientURL string `json:"removeServiceClientUrl"`
	//ConfirmMigrateURL      string `json:"confirmMigrateUrl"`
	//FailedMigrateURL       string `json:"failedMigrateUrl"`
	//PingPeriod int `json:"pingPeriod"`
	//PingURL                string `json:"pingUrl"`

	//FledgePodURL  string `json:"fledgePodUrl"`

	//DeploymentIP            string  `json:"serverIP"`
	//CheatyMinimalServerPing float32 `json:"cheatyMinimalServerPing"`
}

func LoadConfig(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		log.Errorf("Load config error %v", err)
		return err
	}
	decoder := json.NewDecoder(file)
	Cfg = &Config{}
	err = decoder.Decode(Cfg)
	if err != nil {
		log.Errorf("Load config error %v", err)
		return err
	}

	//log.Infof("NodeID check %s", Cfg.NodeID)
	if os.Getenv("NODEID") != "" {
		//log.Infof("Loading nodeID from env instead")
		Cfg.NodeID = os.Getenv("NODEID")
	}

	return nil
}
