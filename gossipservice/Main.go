package main

import (
	"context"
	"fmt"
	"net/http"
	"oamswirly/gossipservice/config"
	"oamswirly/gossipservice/gossiping"
	"oamswirly/gossipservice/oam"
	goam "oamswirly/gossipservice/oam"
	"oamswirly/gossipservice/wsclient"
	"oamswirly/gossipservice/wsserver"
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

func main() {
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultconfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	config.LoadConfig(cfgFile)

	gossiping.InitGossiping()

	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigc
		log.Infof("Signal %v detected, shutting down", sig.String())
		//sharedmem.Cleanup()
		gossiping.Cleanup()
		os.Exit(0)
	}()

	wsclient.RegisterTraitHandler(goam.GetGossipTrait(), fmt.Sprintf("http://localhost:%s/applyTrait", config.Cfg.GossipAPIPort))
	wsclient.RegisterGossipCapsProvider(string(oam.SvcName), config.Cfg.GossipAPIPort)
	wsclient.RegisterService(oam.SvcName, config.Cfg.GossipAPIPort)

	router := wsserver.Router()
	port := config.Cfg.GossipAPIPort

	err := http.ListenAndServe(fmt.Sprintf(":%d", port), router)
	if err != nil {
		log.Errorf("Gossip server error %v", err)
	}
}
