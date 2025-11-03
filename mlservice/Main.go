package main

import (
	"context"
	"fmt"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/mlservice/config"
	goam "oamswirly/mlservice/oam"
	"oamswirly/mlservice/wsclient"
	"oamswirly/mlservice/wsserver"
	"os"
	"os/signal"
	"syscall"

	log "github.com/sirupsen/logrus"
)

var rootContext, rootContextCancel = context.WithCancel(context.Background())

var pings map[string]map[string]int

//Gossip service
//Registers as Gossip trait handler with repo (+ handle trait)
//Registers as capability provider with Gossip trait (+ get capabilities)
//Registers as GossipService with repo
//Provides registerGossipListener and pushGossipItem
//And starts the main gossip cluster...

var svcName oam.FlockyService = "discover.flocky.oam.MachineLearningSvc"

func main() {
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultconfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	config.LoadConfig(cfgFile)

	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigc
		log.Infof("Signal %v detected, shutting down", sig.String())
		wsclient.Cleanup()
		os.Exit(0)
	}()

	supportedTraits := goam.GetNodeSupportedTraits()
	for _, trait := range supportedTraits {
		wsclient.RegisterTraitHandler(trait, fmt.Sprintf("http://localhost:%s/applyMLTrait", config.Cfg.MLSvcAPIPort))
	}
	wsclient.RegisterMLCapsProvider(string(svcName), config.Cfg.MLSvcAPIPort)
	wsclient.RegisterService(svcName, config.Cfg.MLSvcAPIPort)

	router := wsserver.Router()
	port := config.Cfg.MLSvcAPIPort

	err := http.ListenAndServe(fmt.Sprintf(":%d", port), router)
	if err != nil {
		log.Errorf("ML service server error %v", err)
	}
}
