package config

import (
	"encoding/json"
	"oamswirly/common/nodediscovery"
	"oamswirly/common/oam"
	"os"

	log "github.com/sirupsen/logrus"
)

var Cfg *Config

type Config struct {
	DiscoAPIPort int                     `json:"discoAPIPort"`
	RepoAPIPort  int                     `json:"repoAPIPort"`
	InitialNodes []nodediscovery.SvcNode `json:"initialNodes"`
	MaxPing      float32                 `json:"maxPing"`
	NodeID       string                  `json:"nodeID"`
	NodeType     string                  `json:"nodeType"`
	PingPeriod   int                     `json:"pingPeriod"`
	TestMode     bool                    `json:"testMode"`
	//FogIP                string             `json:"fogIP"`
	CheatyMinimalPing    float32            `json:"cheatyMinimalPing"`
	CheatyMinimalPingMap map[string]float32 `json:"cheatyMinimalPingMap"`
	BasicComponentDefs   map[string][]*oam.ComponentDef
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
