package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/gossipservice/config"

	log "github.com/sirupsen/logrus"
)

type SvcNode struct {
	Name         string  `json:"name"`
	IP           string  `json:"ip"`
	Distance     float32 `json:"distance"`
	DiscoAPIPort int     `json:"discoAPIPort"`
	RepoAPIPort  int     `json:"repoAPIPort"`
}

func LoadNeighbours() ([]SvcNode, error) {
	endpoint := fmt.Sprintf("http://localhost:%d/%s", config.Cfg.DiscoAPIPort, "getKnownSvcNodes")
	resp, err := http.Get(endpoint)
	if err != nil {
		return nil, fmt.Errorf("fetch neighbors: %w", err)
	}
	defer resp.Body.Close()

	var nbrs []SvcNode
	if err := json.NewDecoder(resp.Body).Decode(&nbrs); err != nil {
		return nil, fmt.Errorf("decode neighbors JSON: %w", err)
	}
	return nbrs, nil
}

func RegisterGossipCapsProvider(name string, port int) {
	portStr := fmt.Sprintf("%d", port)

	client := oam.CapsProvider{
		Name:          name,
		LocalEndpoint: portStr,
	}
	clientJson, _ := json.Marshal(client)

	url := fmt.Sprintf("http://localhost:%d/registerLocalCapsProvider", config.Cfg.RepoAPIPort)
	//log.Infof("Registering caps provider %s for %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("RegisterGossipCapsProvider error %v", err)
		return
	}

	response.Body.Close()
}

func RegisterService(name oam.FlockyService, port int) {
	portStr := fmt.Sprintf("%d", port)

	client := oam.ServiceProvider{
		Name:          name,
		LocalEndpoint: portStr,
	}
	clientJson, _ := json.Marshal(client)

	url := fmt.Sprintf("http://localhost:%d/registerFlockyService", config.Cfg.RepoAPIPort)
	//log.Infof("Registering caps provider %s for %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("RegisterService error %v", err)
		return
	}

	response.Body.Close()
}

func RegisterTraitHandler(trait oam.TraitDef, endpoint string) {
	client := oam.TraitHandler{
		Trait:         trait,
		LocalEndpoint: endpoint,
	}
	clientJson, _ := json.Marshal(client)

	url := fmt.Sprintf("http://localhost:%d/registerLocalTraitHandler", config.Cfg.RepoAPIPort)
	//log.Infof("Registering caps provider %s for %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("RegisterGossipCapsProvider error %v", err)
		return
	}

	response.Body.Close()
}
