package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/hwresources"
	"oamswirly/dummyservices/config"
	"oamswirly/dummyservices/ws"
	"os"
	"time"

	log "github.com/sirupsen/logrus"
)

var kubernetesHost string
var kubernetesPort string
var defaultPodFile string

var rootContext, rootContextCancel = context.WithCancel(context.Background())

var pings map[string]map[string]int

func main() {
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultconfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	config.LoadConfig(cfgFile)

	go func() {
		hwresources.CPUStatLoop()
	}()
	time.Sleep(time.Second * 1)

	/*if config.Cfg.TestMode {
		nodeNr, _ := strconv.Atoi(config.Cfg.NodeID[1:])
		port += nodeNr
	}*/
	go func() {
		router := ws.FeatherRouter()
		port := config.Cfg.FeatherPort

		//log.Infof("Hosting dummy Feather API on port %d", port)
		err := http.ListenAndServe(fmt.Sprintf(":%d", port), router)
		if err != nil {
			log.Errorf("Dummy Feather server error %v", err)
		}
	}()

	//Fake caps providers for both feather and warrens
	for i := 0; i < config.Cfg.SimMultipleDummies; i++ {
		registerProviders(i)
	}

	router := ws.WarrensRouter()
	port := config.Cfg.WarrensPort

	//log.Infof("Hosting dummy Warrens API on port %d", port)
	err := http.ListenAndServe(fmt.Sprintf(":%d", port), router)
	if err != nil {
		log.Errorf("Dummy Warrens server error %v", err)
	}

}

func registerProviders(increment int) {
	registerDummyCapsProvider("dummyFeather", config.Cfg.FeatherPort)
	registerDummyCapsProvider("dummyWarrens", config.Cfg.WarrensPort)
	//Fake a Feather apps provider
	registerDummyAppsProvider("dummyFeather", config.Cfg.FeatherPort)
	//Fake a warrens listener for new nodes
	registerDummyNodeListener("dummyWarrens", config.Cfg.WarrensPort)
}

type CapsProvider struct {
	Name          string
	LocalEndpoint string
}

func registerDummyCapsProvider(name string, port int) {
	portStr := fmt.Sprintf("%d", port)

	client := CapsProvider{
		Name:          name,
		LocalEndpoint: portStr,
	}
	clientJson, _ := json.Marshal(client)

	url := fmt.Sprintf("http://localhost:%d/registerLocalCapsProvider", config.Cfg.RepoAPIPort)
	//log.Infof("Registering caps provider %s for %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("registerDummyCapsProvider error %v", err)
		return
	}

	response.Body.Close()
}

func registerDummyAppsProvider(name string, port int) {
	portStr := fmt.Sprintf("%d", port)

	client := CapsProvider{
		Name:          name,
		LocalEndpoint: portStr,
	}
	clientJson, _ := json.Marshal(client)

	url := fmt.Sprintf("http://localhost:%d/registerLocalAppsProvider", config.Cfg.RepoAPIPort)
	//log.Infof("Registering apps provider %s for %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("registerDummyAppsProvider error %v", err)
		return
	}

	response.Body.Close()
}

type NodeListener struct {
	Name string
	Port string
}

func registerDummyNodeListener(name string, port int) {
	portStr := fmt.Sprintf("%d", port)

	client := NodeListener{
		Name: name,
		Port: portStr,
	}
	clientJson, _ := json.Marshal(client)

	url := fmt.Sprintf("http://localhost:%d/registerNodeListener", config.Cfg.DiscoAPIPort)
	//log.Infof("Registering dummy node listener at %s with %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("registerDummyNodeListener error %v", err)
		return
	}

	response.Body.Close()
}
