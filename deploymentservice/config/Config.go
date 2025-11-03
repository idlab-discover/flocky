package config

import (
	"encoding/json"
	"os"

	log "github.com/sirupsen/logrus"
)

var Cfg *Config

type Config struct {
	DeploymentAPIPort int `json:"deploymentPort"`
	RepoAPIPort       int `json:"repoPort"`
	FledgeAPIPort     int `json:"fledgePort"`
	//SwirlyAPIPort     int    `json:"swirlyPort"`
	Orchestrator string `json:"orchestrator"`
	NodeID       string `json:"nodeID"`
	//ResourceLimitsPct int    `json:"resourceLimitsPct"`

	TestMode bool `json:"testMode"`
}

func LoadConfig(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		//return err
	}
	decoder := json.NewDecoder(file)
	Cfg = &Config{}
	err = decoder.Decode(Cfg)
	if err != nil {
		log.Errorf("LoadConfig %v", err)
		//return err
	}

	//log.Infof("NodeID check %s", Cfg.NodeID)
	if os.Getenv("NODEID") != "" {
		//log.Info("Loading nodeID from env instead")
		Cfg.NodeID = os.Getenv("NODEID")
	}

	return err
}
