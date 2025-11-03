package config

import (
	"encoding/json"
	"os"

	log "github.com/sirupsen/logrus"
)

var Cfg *Config

type Config struct {
	NodeID           string `json:"nodeID"`
	RepoAPIPort      int    `json:"repoAPIPort"`
	DiscoAPIPort     int    `json:"discoAPIPort"`
	ClusterStartPort int    `json:"clusterStartPort"`
	GossipAPIPort    int    `json:"clusterAPIPort"`
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

	return nil
}
