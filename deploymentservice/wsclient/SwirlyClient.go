package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/oam"

	log "github.com/sirupsen/logrus"
)

func NotifyInitTeardown(ip string, node string, component oam.Component) error {
	//port := getEdgePort(node)
	port, _ := GetFlockyService(oam.SwirlyService)
	fullURL := fmt.Sprintf("http://%s:%d/tryMigrate", ip, port)
	//log.Infof("Calling %s", fullURL)
	jsonBytes, err := json.Marshal(component)
	response, err := http.Post(fullURL, "application/json", bytes.NewBuffer(jsonBytes))

	if response.StatusCode != 200 || err != nil {
		log.Errorf("NotifyInitTeardown error %v", err)
	}

	response.Body.Close()
	return err
}

func NotifyTeardown(ip string, node string, component oam.Component) error {
	//port := getEdgePort(node)
	port, _ := GetFlockyService(oam.SwirlyService)
	fullURL := fmt.Sprintf("http://%s:%d/migrate", ip, port)
	//log.Infof("Calling %s", fullURL)
	jsonBytes, err := json.Marshal(component)
	response, err := http.Post(fullURL, "application/json", bytes.NewBuffer(jsonBytes))

	if response.StatusCode != 200 || err != nil {
		log.Errorf("NotifyTeardown error %v", err)
	}

	response.Body.Close()
	return err
}

func CancelTeardown(ip string, node string, component oam.Component) error {
	//port := getEdgePort(node)
	port, _ := GetFlockyService(oam.SwirlyService)
	fullURL := fmt.Sprintf("http://%s:%d/cancelMigrate", ip, port)
	//log.Infof("Calling %s\n", fullURL)
	jsonBytes, err := json.Marshal(component)
	response, err := http.Post(fullURL, "application/json", bytes.NewBuffer(jsonBytes))

	if response.StatusCode != 200 || err != nil {
		log.Errorf("CancelTeardown error %v", err)
	}

	response.Body.Close()
	return err
}
