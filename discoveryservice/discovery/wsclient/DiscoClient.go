package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/nodediscovery"
	"oamswirly/discoveryservice/config"
	"time"

	log "github.com/sirupsen/logrus"
)

/*type Client struct {
	Name         string
	Type         nodediscovery.NodeType
	RepoAPIPort  int
	DiscoAPIPort int
}*/

func NotifyNodesUpdatedListeners(port string, changedNodes []nodediscovery.SvcNode) {
	nodeJson, _ := json.Marshal(changedNodes)

	url := fmt.Sprintf("http://localhost:%s/nodesUpdated", port)
	//log.Debugf("Sending node update to %s with %s", url, string(nodeJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(nodeJson))

	if err != nil {
		log.Errorf("NotifyNodesUpdatedListeners error %v", err.Error())
		return
	}
	response.Body.Close()
}

type NodesUpdate struct {
	NewNodes     []nodediscovery.SvcNode
	DeletedNodes []nodediscovery.SvcNode
}

func NotifyNodesDiscoveredListeners(port string, newNodes []nodediscovery.SvcNode, deletedNodes []nodediscovery.SvcNode) {
	update := NodesUpdate{
		NewNodes:     newNodes,
		DeletedNodes: deletedNodes,
	}
	nodeJson, _ := json.Marshal(update)

	url := fmt.Sprintf("http://localhost:%s/nodesDiscovered", port)
	//log.Debugf("Sending node update to %s with %s", url, string(nodeJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(nodeJson))

	if err != nil {
		log.Errorf("NotifyNodesDiscoveredListeners error %v", err.Error())
		return
	}
	response.Body.Close()
}

func GetKnownSvcNodes(node nodediscovery.SvcNode) ([]nodediscovery.SvcNode, error) {
	svcURL := getKnownNodesURL(node)
	//log.Infof("GetKnownSvcNodes %s", svcURL)
	response, err := HttpClient.Get(svcURL)
	if err != nil {
		log.Errorf("GetKnownSvcNodes error %v", err.Error())
		return []nodediscovery.SvcNode{}, err
	}

	svcNodes := []nodediscovery.SvcNode{}
	err = json.NewDecoder(response.Body).Decode(&svcNodes)
	if err != nil {
		log.Errorf("GetKnownSvcNodes JSON error %v", err.Error())
		return nil, err
	}

	response.Body.Close()
	return svcNodes, nil
}

var HttpClient http.Client

func GetPing(node nodediscovery.SvcNode) (float32, error) {
	start := time.Now()

	//fullURL := fmt.Sprintf("http://%s:%d/%s", ip, config.Cfg.Port, "ping")
	url := getPingURL(node)

	client := nodediscovery.Client{
		Name:         GetNodeID(), //hostname,
		Type:         getNodeType(),
		RepoAPIPort:  config.Cfg.RepoAPIPort,
		DiscoAPIPort: config.Cfg.DiscoAPIPort,
	}

	clientJson, err := json.Marshal(client)
	if err != nil {
		log.Errorf("GetPing JSON error %v", err.Error())
		return 0, err
	}

	//log.Infof("Pinging %s with %s", url, string(clientJson))

	response, err := HttpClient.Post(url, "application/json", bytes.NewBuffer(clientJson))
	if err != nil {
		log.Errorf("GetPing post error %v", err.Error())
		return 0, err
	}

	stop := time.Now()
	newPing := float32(stop.Sub(start).Nanoseconds()) / 1000000
	if config.Cfg.TestMode {
		newPing = config.Cfg.CheatyMinimalPingMap[node.Name]
	}

	if response.StatusCode != 200 {
		//log.Infof("Status code %d", response.StatusCode)
		newPing = -1
	}

	response.Body.Close()
	return newPing, err
}

func getKnownNodesURL(node nodediscovery.SvcNode) string {
	//port := getPort(node)
	fullURL := fmt.Sprintf("http://%s:%d/getKnownSvcNodes", node.IP, node.DiscoAPIPort)
	return fullURL
}

func getPingURL(node nodediscovery.SvcNode) string {
	//port := getPort(node)
	fullURL := fmt.Sprintf("http://%s:%d/ping", node.IP, node.DiscoAPIPort)
	return fullURL
}

/*func getPort(node string) int {
	port := config.Cfg.DiscoAPIPort
	if config.Cfg.TestMode {
		nodenumber, _ := strconv.Atoi(node[1:])
		port += nodenumber
	}
	return port
}*/

func getNodeType() nodediscovery.NodeType {
	return nodediscovery.NodeType(config.Cfg.NodeType)
}

func GetNodeID() string {
	return config.Cfg.NodeID
}
