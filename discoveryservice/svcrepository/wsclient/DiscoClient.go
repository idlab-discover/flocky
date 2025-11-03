package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"oamswirly/common/nodediscovery"
	"oamswirly/discoveryservice/config"

	log "github.com/sirupsen/logrus"
)

func GetLocallyKnownSvcNodes(port int) ([]nodediscovery.SvcNode, error) {
	svcURL := fmt.Sprintf("http://localhost:%d/getKnownSvcNodes", port)
	//log.Infof("GetLocallyKnownSvcNodes %s", svcURL)
	response, err := HttpClient.Get(svcURL)

	if err != nil {
		log.Errorf("GetLocallyKnownSvcNodes error %v", err)
		return nil, err
	}

	svcNodes := []nodediscovery.SvcNode{}
	err = json.NewDecoder(response.Body).Decode(&svcNodes)
	if err != nil {
		log.Errorf("GetLocallyKnownSvcNodes JSON error %v", err)
		return nil, err
	}

	response.Body.Close()
	return svcNodes, nil
}

type NodeListener struct {
	Name string
	Port string
}

func RegisterAsNodeListener(port int, name string) {
	client := NodeListener{
		Name: name,
		Port: fmt.Sprintf("%d", config.Cfg.RepoAPIPort),
	}
	clientJson, err := json.Marshal(client)
	if err != nil {
		log.Errorf("RegisterAsNodeListener JSON error %v", err)
		return
	}

	url := fmt.Sprintf("http://localhost:%d/registerNodeListener", port)
	//log.Infof("Registering node listener at %s with %s\n", url, string(clientJson))
	response, err := HttpClient.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("RegisterAsNodeListener error %v", err)
		return
	}

	response.Body.Close()
}
