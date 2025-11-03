package wsserver

import (
	"encoding/json"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/dummyservices/config"
	"oamswirly/gossipservice/gossiping"
	goam "oamswirly/gossipservice/oam"

	//"github.com/gorilla/mux"
	log "github.com/sirupsen/logrus"
)

func init() {

}

func ListGossipCapabilities(w http.ResponseWriter, r *http.Request) {
	caps := oam.NodeCapsContent{}

	//traits
	caps.SupportedTraits = []oam.TraitDef{}
	if config.Cfg.SimAttestationTrait {
		caps.SupportedTraits = append(caps.SupportedTraits, goam.GetGossipTrait())
	}

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

// Implement this later, it's not needed right now but the GossipTrait will enable read/write from REST methods in a workload
// to/from the gossip cluster which might be pretty convenient
type GossipTrait struct {
	ReadEndpoints  map[string]string
	WriteEndpoints map[string]string
}

func GetGossipItem(w http.ResponseWriter, r *http.Request) {
	key := ""
	err := json.NewDecoder(r.Body).Decode(&key)
	if err != nil {
		log.Errorf("ListGossipCapabilities JSON error %v", err)
		return
	}

	gossipItems := gossiping.GetGossipForKey(key)

	json, err := json.Marshal(gossipItems)
	if err != nil {
		log.Errorf("ListGossipCapabilities JSON error %v", err)
		return
	}
	_, err = w.Write(json)
	if err != nil {
		log.Errorf("ListGossipCapabilities error %v", err)
	}
}

func PushGossipItem(w http.ResponseWriter, r *http.Request) {
	item := gossiping.GossipItem{}
	err := json.NewDecoder(r.Body).Decode(&item)
	if err != nil {
		log.Errorf("ListGossipCapabilities JSON error %v", err)
		return
	}

	gossiping.PushGossipItem(item)

	w.Write([]byte("OK"))
}

func ApplyTrait(w http.ResponseWriter, r *http.Request) {
	trait := oam.Trait{}
	err := json.NewDecoder(r.Body).Decode(&trait)
	if err != nil {
		log.Errorf("ApplyTrait JSON error %v", err)
		return
	}

	//TODO set up REST/SHM gossip channels from Trait

	w.Write([]byte("OK"))
}

func RegisterRESTGossipListener(w http.ResponseWriter, r *http.Request) {
	settings := goam.RESTSettings{}

	err := json.NewDecoder(r.Body).Decode(&settings)
	if err != nil {
		log.Errorf("RegisterRESTGossipListener JSON error %v", err)
		return
	}

	gossiping.StartRestListener(settings)
	w.Write([]byte("OK"))
}

func RegisterRESTGossipPoll(w http.ResponseWriter, r *http.Request) {
	settings := goam.RESTSettings{}

	err := json.NewDecoder(r.Body).Decode(&settings)
	if err != nil {
		log.Errorf("RegisterRESTGossipPoll JSON error %v", err)
		return
	}

	gossiping.StartRestPoller(settings)
	w.Write([]byte("OK"))
}

func RegisterShmGossipListener(w http.ResponseWriter, r *http.Request) {
	settings := goam.ShmSettings{}

	err := json.NewDecoder(r.Body).Decode(&settings)
	if err != nil {
		log.Errorf("RegisterShmGossipListener JSON error %v", err)
		return
	}

	gossiping.StartShmWriter(settings)
	w.Write([]byte("OK"))
}

func RegisterShmGossipPoll(w http.ResponseWriter, r *http.Request) {
	settings := goam.ShmSettings{}

	err := json.NewDecoder(r.Body).Decode(&settings)
	if err != nil {
		log.Errorf("RegisterShmGossipPoll JSON error %v", err)
		return
	}

	gossiping.StartShmPoller(settings)
	w.Write([]byte("OK"))
}
