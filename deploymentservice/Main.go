package main

import (
	"fmt"
	"net/http"
	"oamswirly/common/hwresources"
	"oamswirly/common/oam"
	"oamswirly/deploymentservice/config"
	"oamswirly/deploymentservice/orchestration"
	"oamswirly/deploymentservice/orchestration/clients"
	"oamswirly/deploymentservice/wsserver"
	"oamswirly/gossipservice/wsclient"
	"os"
	"strconv"
	"time"

	log "github.com/sirupsen/logrus"
)

// var kubernetesHost string
// var kubernetesPort string
var defaultPodFile string

//var rootContext, rootContextCancel = context.WithCancel(context.Background())

//var pings map[string]map[string]int

func main() {
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultconfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	config.LoadConfig(cfgFile)

	orchestration.Init()
	orchType := config.Cfg.Orchestrator
	switch orchType {
	case "fledge":
		clients.Orch = (&(clients.FledgeOrchestrator{})).Init()
	}

	go func() {
		hwresources.CPUStatLoop()
	}()
	time.Sleep(time.Second * 1)
	/*defer func() {
		if r := recover(); r != nil {
			log.Error("Recovered in main.go", r)
		}
	}()*/

	wsclient.RegisterService(oam.DeploymentService, config.Cfg.DeploymentAPIPort)

	router := wsserver.DeploymentRouter()
	port := config.Cfg.DeploymentAPIPort
	if config.Cfg.TestMode {
		nodeNr, _ := strconv.Atoi(config.Cfg.NodeID[1:])
		port += nodeNr
	}
	//log.Infof("Hosting node %s on port %d\n", config.Cfg.NodeID, port)
	err := http.ListenAndServe(fmt.Sprintf(":%d", port), router)
	if err != nil {
		log.Errorf(err.Error())
	}
}
