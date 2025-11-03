package wsserver

import (
	"encoding/json"
	"net/http"
	"oamswirly/common/hwresources"
	"oamswirly/common/oam"
	"oamswirly/dummyservices/config"
	"oamswirly/gossipservice/gossiping"
	goam "oamswirly/mlservice/oam"
	"oamswirly/mlservice/wsclient"

	//"github.com/gorilla/mux"
	log "github.com/sirupsen/logrus"
)

func ListMLCapabilities(w http.ResponseWriter, r *http.Request) {
	caps := oam.NodeCapsContent{}

	//traits
	caps.SupportedTraits = []oam.TraitDef{}
	if config.Cfg.SimAttestationTrait {
		caps.SupportedTraits = goam.GetNodeSupportedTraits()
	}

	caps.Resources.HwStats = make(map[hwresources.ResourceType]interface{})
	caps.Resources.HwStats[goam.ResourceAccelerator] = goam.GetAcceleratorStats()

	json, err := json.Marshal(caps)
	if err != nil {
		log.Errorf("ListGossipCapabilities JSON error %v", err)
		return
	}
	_, err = w.Write(json)
	if err != nil {
		log.Errorf("ListGossipCapabilities error %v", err)
	}
}

func ApplyMLTrait(w http.ResponseWriter, r *http.Request) {
	trait := oam.Trait{}
	err := json.NewDecoder(r.Body).Decode(&trait)
	if err != nil {
		log.Errorf("ApplyTrait JSON error %v", err)
		return
	}

	switch trait.Type {
	case string(goam.MLAcceleratorTraitType):
		//Might have to ensure some GPU resources here later with the nvidia library
		break
	case string(goam.MLConfidentialComputingTraitType):
		//Definitely have to ensure some stuff here. Too bad my machine doesn't support any type of conf computing so that'll be for later.
		break
	case string(goam.MLGossipLearningTraitType):
		//The type of learning algo is implemented in Python in this version, todo implement it here later
		//In the meantime, set up gossiping shenanigans
		wsclient.SetupGossipToMLLink(trait)
		break
	}

	w.Write([]byte("OK"))
}

func ModelGossipReceived(w http.ResponseWriter, r *http.Request) {
	gossipItems := gossiping.GossipItems{}
	err := json.NewDecoder(r.Body).Decode(&gossipItems)
	if err != nil {
		log.Errorf("ApplyTrait JSON error %v", err)
		return
	}

	wsclient.ModelUpdatesReceived(gossipItems.Key, gossipItems.NodeData)

	w.Write([]byte("OK"))
}
