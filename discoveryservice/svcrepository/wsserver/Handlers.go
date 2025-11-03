package wsserver

import (
	"encoding/json"
	"net/http"
	"oamswirly/common/nodediscovery"
	"oamswirly/common/oam"
	"oamswirly/discoveryservice/config"
	"oamswirly/discoveryservice/svcrepository"
	"oamswirly/discoveryservice/svcrepository/local"
	"oamswirly/discoveryservice/svcrepository/wsclient"

	log "github.com/sirupsen/logrus"
)

func AddComponentDef(w http.ResponseWriter, r *http.Request) {
	//log.Info("AddComponentDef")
	cDef := &oam.ComponentDef{}
	err := json.NewDecoder(r.Body).Decode(cDef)

	if err != nil {
		log.Errorf("AddComponentDef JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		go func() {
			svcrepository.AddComponentDef(cDef)
		}()
		w.Write([]byte("OK"))
	}
}

func RegisterLocalCapsProvider(w http.ResponseWriter, r *http.Request) {
	client := oam.CapsProvider{}
	err := json.NewDecoder(r.Body).Decode(&client)

	//log.Infof("RegisterLocalCapsProvider from %s", client.Name)
	if err != nil {
		log.Errorf("RegisterLocalCapsProvider JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		local.RegisterCapabilitiesProvider(client.Name, client.LocalEndpoint)
		w.Write([]byte("OK"))
	}
}

func RegisterLocalAppsProvider(w http.ResponseWriter, r *http.Request) {
	client := oam.CapsProvider{}
	err := json.NewDecoder(r.Body).Decode(&client)

	//log.Infof("RegisterLocalAppsProvider from %s", client.Name)
	if err != nil {
		log.Errorf("RegisterLocalAppsProvider JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		local.RegisterAppsProvider(client.Name, client.LocalEndpoint)
		w.Write([]byte("OK"))
	}
}

func RegisterNodeStatusUpdatesListener(w http.ResponseWriter, r *http.Request) {
	client := nodediscovery.UpdateListener{}
	err := json.NewDecoder(r.Body).Decode(&client)

	//log.Infof("RegisterNodeStatusUpdatesListener from %s", client.Name)
	if err != nil {
		log.Errorf("RegisterNodeStatusUpdatesListener JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		wsclient.RegisterNodeSummaryUpdateListener(client.Name, client.Port)
		w.Write([]byte("OK"))
	}
}

func GetKnownComponentDefs(w http.ResponseWriter, r *http.Request) {
	//log.Info("GetKnownComponentDefs")
	cdefs := svcrepository.GetKnownComponentDefinitions()
	json, err := json.Marshal(cdefs)
	if err != nil {
		log.Errorf("GetKnownComponentDefs JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	//log.Debugf("GetKnownComponentDefs Responding %s", string(json))

	w.Write(json)
}

func GetFullKnownComponentDefs(w http.ResponseWriter, r *http.Request) {
	//log.Info("GetKnownComponentDefs")
	names := []string{}
	err := json.NewDecoder(r.Body).Decode(&names)

	//log.Info("NodesUpdated remote node distances updated")
	if err != nil {
		log.Errorf("GetFullKnownComponentDefs JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	cdefs := svcrepository.GetFullComponentDefinitions(names)
	json, err := json.Marshal(cdefs)
	if err != nil {
		log.Errorf("GetFullKnownComponentDefs JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	//log.Debugf("GetKnownComponentDefs Responding %s", string(json))

	w.Write(json)
}

func GetKnownNodeSummaries(w http.ResponseWriter, r *http.Request) {
	//log.Info("GetKnownNodeSummaries")
	summs := svcrepository.GetKnownNodeSummaries()
	json, err := json.Marshal(summs)

	if err != nil {
		log.Errorf("GetKnownNodeSummaries JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	//log.Debugf("GetKnownNodeSummaries Responding %s", string(json))
	w.Write(json)
}

func GetConcreteComponentsFor(w http.ResponseWriter, r *http.Request) {
	component := oam.Component{}
	err := json.NewDecoder(r.Body).Decode(&component)

	if err != nil {
		log.Errorf("GetConcreteComponentsFor request JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	//log.Infof("GetConcreteComponentsFor %s", component.Name)

	cdefs := svcrepository.GetConcreteComponentsFor(component)
	json, err := json.Marshal(cdefs)
	//log.Infof("GetConcreteComponentsFor Responding %s", string(json))
	if err != nil {
		log.Errorf("GetConcreteComponentsFor response JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Write(json)
}

func GetNodeSummary(w http.ResponseWriter, r *http.Request) {
	apps, err := local.GetApplicationsSummary()
	if err != nil {
		log.Errorf("GetNodeSummary apps summary error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	caps, err := local.GetCapabilitiesSummary()
	if err != nil {
		log.Errorf("GetNodeSummary caps summary error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	summary := oam.NodeSummary{
		NodeApps: apps,
		NodeCaps: oam.NodeCaps{
			ApiVersion: "v1/beta",
			Kind:       "NodeCaps",
			Metadata: oam.Metadata{
				Name: config.Cfg.NodeID,
			},
			Caps: caps,
		},
		Name: config.Cfg.NodeID,
	}
	json, err := json.Marshal(summary)
	////log.Infof("GetKnownComponentDefs Responding %s", string(json))
	if err != nil {
		log.Errorf("GetNodeSummary JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Write(json)
}
