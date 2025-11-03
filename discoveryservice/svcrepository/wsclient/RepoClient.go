package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/nodediscovery"
	"oamswirly/common/oam"
	"oamswirly/discoveryservice/config"
	"time"

	log "github.com/sirupsen/logrus"
)

var nodeListeners map[string]string

func Init() {
	nodeListeners = make(map[string]string)

	transport := &http.Transport{
		MaxIdleConns:        150,
		MaxIdleConnsPerHost: 50,
		MaxConnsPerHost:     50,
		IdleConnTimeout:     30 * time.Second,
		//DisableCompression: true,
	}

	HttpClient = http.Client{
		Transport: transport,
		Timeout:   time.Duration(2*config.Cfg.MaxPing) * time.Millisecond,
	}
}

var HttpClient http.Client

func RegisterNodeSummaryUpdateListener(name string, endpoint string) {
	//log.Infof("RegisterNodeSummaryUpdateListener %s %s", name, endpoint)
	nodeListeners[name] = endpoint
}

func NotifyRemoteAppChangesListeners(appSummary []oam.NodeSummary) {
	//log.Info("NotifyRemoteAppChangesListeners")
	nodeJson, err := json.Marshal(appSummary)
	if err != nil {
		log.Errorf("NotifyRemoteAppChangesListeners JSON error %v", err)
		return
	}

	for _, port := range nodeListeners {
		url := fmt.Sprintf("http://localhost:%s/nodeAppsChanged", port)
		//log.Debugf("Sending node apps update to %s with %s", url, string(nodeJson))
		response, err := HttpClient.Post(url, "application/json", bytes.NewBuffer(nodeJson))

		if err != nil {
			log.Errorf("NotifyRemoteAppChangesListeners POST error %v", err)
			return
		}
		response.Body.Close()
	}
}

func NotifyRemoteCapsChangesListeners(capsSummary []oam.NodeSummary) {
	//log.Info("NotifyRemoteCapsChangesListeners")
	nodeJson, err := json.Marshal(capsSummary)
	if err != nil {
		log.Errorf("NotifyRemoteCapsChangesListeners JSON error %v", err)
		return
	}

	for _, port := range nodeListeners {
		url := fmt.Sprintf("http://localhost:%s/nodeCapsChanged", port)
		//log.Debugf("Sending node apps update to %s with %s", url, string(nodeJson))
		response, err := HttpClient.Post(url, "application/json", bytes.NewBuffer(nodeJson))

		if err != nil {
			log.Errorf("NotifyRemoteCapsChangesListeners POST error %v", err)
			return
		}
		response.Body.Close()
	}
}

/*func GetKnownComponentDefs(node nodediscovery.SvcNode) (map[string][]*oam.ComponentDef, error) {
	//log.Infof("GetKnownComponentDefs from %s", node.Name)
	response, err := http.Get(fmt.Sprintf("http://%s:%d/getKnownComponentDefs", node.IP, node.RepoAPIPort))
	if err != nil {
		log.Errorf("GetKnownComponentDefs GET error %v", err)
		return map[string][]*oam.ComponentDef{}, err
	}

	remoteKnownDefs := make(map[string][]*oam.ComponentDef)
	err = json.NewDecoder(response.Body).Decode(&remoteKnownDefs)
	if err != nil {
		log.Errorf("GetKnownComponentDefs JSON error %v", err)
		return map[string][]*oam.ComponentDef{}, err
	}
	return remoteKnownDefs, nil
}*/

func GetKnownComponentDefs(node nodediscovery.SvcNode) (map[string][]string, error) {
	//log.Infof("GetKnownComponentDefs from %s", node.Name)
	response, err := HttpClient.Get(fmt.Sprintf("http://%s:%d/getKnownComponentDefs", node.IP, node.RepoAPIPort))
	if err != nil {
		log.Errorf("GetKnownComponentDefs GET error %v", err)
		return make(map[string][]string), err
	}

	remoteKnownDefs := make(map[string][]string)
	err = json.NewDecoder(response.Body).Decode(&remoteKnownDefs)
	if err != nil {
		log.Errorf("GetKnownComponentDefs JSON error %v", err)
		return make(map[string][]string), err
	}

	response.Body.Close()
	return remoteKnownDefs, nil
}

func GetFullKnownComponentDefs(node nodediscovery.SvcNode, names []string) (map[string][]*oam.ComponentDef, error) {
	namesJson, err := json.Marshal(names)
	if err != nil {
		log.Errorf("GetFullKnownComponentDefs JSON error %v", err)
		return map[string][]*oam.ComponentDef{}, err
	}
	//log.Infof("GetKnownComponentDefs from %s", node.Name)
	url := fmt.Sprintf("http://%s:%d/getFullKnownComponentDefs", node.IP, node.RepoAPIPort)
	response, err := HttpClient.Post(url, "application/json", bytes.NewBuffer(namesJson))
	if err != nil {
		log.Errorf("GetKnownComponentDefs GET error %v", err)
		return map[string][]*oam.ComponentDef{}, err
	}

	remoteKnownDefs := make(map[string][]*oam.ComponentDef)
	err = json.NewDecoder(response.Body).Decode(&remoteKnownDefs)
	if err != nil {
		log.Errorf("GetKnownComponentDefs JSON error %v", err)
		return map[string][]*oam.ComponentDef{}, err
	}

	response.Body.Close()
	return remoteKnownDefs, nil
}

func GetNodeSummary(node nodediscovery.SvcNode) (*oam.NodeSummary, error) {
	//log.Infof("GetNodeSummary from %s", node.Name)
	response, err := HttpClient.Get(fmt.Sprintf("http://%s:%d/getNodeSummary", node.IP, node.RepoAPIPort))
	if err != nil {
		log.Errorf("GetNodeSummary GET error %v", err)
		return &oam.NodeSummary{}, err
	}

	remoteSummary := &oam.NodeSummary{}
	err = json.NewDecoder(response.Body).Decode(remoteSummary)
	if err != nil {
		log.Errorf("GetNodeSummary JSON error %v", err)
		return &oam.NodeSummary{}, err
	}

	response.Body.Close()
	return remoteSummary, nil
}
