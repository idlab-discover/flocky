package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/nodediscovery"
	"oamswirly/common/oam"
	"oamswirly/swirlyservice/config"

	log "github.com/sirupsen/logrus"
)

func RegisterAsNodeUpdateListener(name string) {
	port := config.Cfg.RepoAPIPort

	client := nodediscovery.UpdateListener{
		Name: name,
		Port: fmt.Sprintf("%d", config.Cfg.SwirlyAPIPort),
	}
	clientJson, err := json.Marshal(client)
	if err != nil {
		log.Errorf("RegisterAsNodeUpdateListener JSON encode error %v", err)
	}

	url := fmt.Sprintf("http://localhost:%d/registerNodeStatusUpdatesListener", port)
	//log.Infof("Registering node status update listener at %s with %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("RegisterAsNodeUpdateListener response error %v", err)
	}

	response.Body.Close()
}

func GetConcreteComponentsFor(cd oam.Component) (map[int]oam.ComponentDef, error) {
	port := config.Cfg.RepoAPIPort
	clientJson, err := json.Marshal(cd)
	if err != nil {
		log.Errorf("GetConcreteComponentsFor JSON encode error %v", err)
	}

	url := fmt.Sprintf("http://localhost:%d/getConcreteComponentsFor", port)
	//log.Infof("Getting concrete component defs for component %s at %s", cd.Name, url)
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	cDefs := make(map[int]oam.ComponentDef)
	if err != nil {
		log.Errorf("GetConcreteComponentsFor response error %v", err)
		return cDefs, err
	}

	err = json.NewDecoder(response.Body).Decode(&cDefs)
	if err != nil {
		log.Errorf("GetConcreteComponentsFor JSON decode error %v", err)
		return cDefs, err
	}
	response.Body.Close()
	return cDefs, nil
}
