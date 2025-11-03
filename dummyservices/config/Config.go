package config

import (
	"encoding/json"
	"os"

	log "github.com/sirupsen/logrus"
)

var Cfg *Config

type Config struct {
	FeatherPort               int  `json:"featherPort"`
	WarrensPort               int  `json:"warrensPort"`
	DiscoAPIPort              int  `json:"discoAPIPort"`
	RepoAPIPort               int  `json:"repoAPIPort"`
	SimMultipleDummies        int  `json:"simMultipleDummies"`
	SimMemory                 int  `json:"simMemory"`
	SimCores                  int  `json:"simCores"`
	SimSecureEnclaveTrait     bool `json:"simSecureEnclaveTrait"`
	SimSecureRuntimeTrait     bool `json:"simSecureRuntimeTrait"`
	SimGreenEnergyTrait       bool `json:"simGreenEnergyTrait"`
	SimAttestationTrait       bool `json:"simAttestationTrait"`
	SimNetworkEncryptionTrait bool `json:"simNetworkEncryptionTrait"`
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
