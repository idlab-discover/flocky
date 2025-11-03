package wsserver

import (
	"encoding/json"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/discoveryservice/svcrepository/local"

	log "github.com/sirupsen/logrus"
)

func RegisterTraitHandler(w http.ResponseWriter, r *http.Request) {
	handler := oam.TraitHandler{}
	err := json.NewDecoder(r.Body).Decode(&handler)

	//log.Infof("RegisterLocalCapsProvider from %s", client.Name)
	if err != nil {
		log.Errorf("RegisterLocalCapsProvider JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		local.RegisterTraitHandler(handler)
		w.Write([]byte("OK"))
	}
}

func GetTraitHandler(w http.ResponseWriter, r *http.Request) {
	trait := oam.Trait{}
	err := json.NewDecoder(r.Body).Decode(&trait)

	//log.Infof("RegisterLocalAppsProvider from %s", client.Name)
	if err != nil {
		log.Errorf("RegisterLocalFlockySvc JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	handler := local.GetTraitHandler(trait.Type)
	json, err := json.Marshal(handler)
	////log.Infof("GetKnownComponentDefs Responding %s", string(json))
	if err != nil {
		log.Errorf("GetNodeSummary JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Write(json)
}

func RegisterFlockySvc(w http.ResponseWriter, r *http.Request) {
	client := oam.ServiceProvider{}
	err := json.NewDecoder(r.Body).Decode(&client)

	//log.Infof("RegisterLocalAppsProvider from %s", client.Name)
	if err != nil {
		log.Errorf("RegisterLocalFlockySvc JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		local.RegisterLocalFlockySvc(client.Name, client.LocalEndpoint)
		w.Write([]byte("OK"))
	}
}

func GetFlockySvcEndpoint(w http.ResponseWriter, r *http.Request) {
	name := oam.FlockyService("")
	err := json.NewDecoder(r.Body).Decode(&name)

	//log.Infof("RegisterLocalAppsProvider from %s", client.Name)
	if err != nil {
		log.Errorf("RegisterLocalFlockySvc JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	svc := local.GetFlockyServiceEndpoint(name)
	json, err := json.Marshal(oam.ServiceProvider{Name: name, LocalEndpoint: svc})
	////log.Infof("GetKnownComponentDefs Responding %s", string(json))
	if err != nil {
		log.Errorf("GetNodeSummary JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Write(json)
}

func ListFlockyServices(w http.ResponseWriter, r *http.Request) {
	svcs := local.GetFlockyServiceEndpoints()
	json, err := json.Marshal(svcs)
	////log.Infof("GetKnownComponentDefs Responding %s", string(json))
	if err != nil {
		log.Errorf("GetNodeSummary JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Write(json)
}
