package config

import (
	"encoding/json"
	"os"

	log "github.com/sirupsen/logrus"
)

var Cfg *Config

type Config struct {
	DiscoAPIPort  int    `json:"discoPort"`
	MinEdgeNodes  int    `json:"minEdgeNodes"`
	MaxEdgeNodes  int    `json:"maxEdgeNodes"`
	EdgeNodeStep  int    `json:"edgeNodeStep"`
	MinFogNodes   int    `json:"minFogNodes"`
	MaxFogNodes   int    `json:"maxFogNodes"`
	FogNodeStep   int    `json:"fogNodeStep"`
	Iterations    int    `json:"iterations"`
	MonitorPeriod int64  `json:"monitorPeriod"`
	MonitorLoops  int    `json:"monitorLoops"`
	EthInterface  string `json:"ethInterface"`
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
