package wsserver

import (
	"encoding/json"
	"net/http"
	common "oamswirly/common/nodediscovery"
	"oamswirly/discoveryservice/config"
	"oamswirly/discoveryservice/discovery"

	"strings"

	//"github.com/gorilla/mux"
	log "github.com/sirupsen/logrus"
)

func GetKnownSvcNodes(w http.ResponseWriter, r *http.Request) {
	//log.Info("GetKnownSvcNodes")

	nodes := discovery.GetKnownNodes()
	json, err := json.Marshal(nodes)

	if err != nil {
		log.Errorf("GetKnownSvcNodes JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	//log.Debugf("Responding %s", string(json))
	w.Write(json)
}

func Ping(w http.ResponseWriter, r *http.Request) {
	//log.Info("Ping")
	client := common.Client{}
	err := json.NewDecoder(r.Body).Decode(&client)

	if err != nil {
		log.Errorf("Ping JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		//log.Infof("Ping from %s", client.Name)
		if client.Type == common.NodeTypeService {
			ip := strings.Split(r.RemoteAddr, ":")[0]
			go func() {
				discovery.AddNode(client, ip)
			}()
		}

		/*if config.Cfg.TestMode {
			sleepTime := float32(0)
			if client.Type == common.NodeTypeService {
				sleepTime = config.Cfg.CheatyMinimalPingMap[client.Name]
			} else {
				sleepTime = config.Cfg.CheatyMinimalPingMap[client.Name]
			}
			//log.Debugf("Sleep time %f for %s", sleepTime, client.Name)
			time.Sleep(time.Duration(int(sleepTime)) * time.Millisecond)
		}*/

		w.Write([]byte("OK"))
		//w.WriteHeader(200)
	}
}

func GetDiscoveredNodeStats(w http.ResponseWriter, r *http.Request) {
	//log.Info("GetDiscoveredNodeStats")
	nodes := discovery.GetKnownNodes()
	cheatNodes := config.Cfg.CheatyMinimalPingMap
	maxPing := config.Cfg.MaxPing

	numInRange := 0
	numTolerance := 0
	numOutRange := 0
	numExpected := 0
	discovered := 0

	toDiscover := make(map[string]int)
	for node, ping := range cheatNodes {
		if ping <= maxPing {
			numExpected++
			toDiscover[node] = 1
		}
	}

	for _, node := range nodes {
		if node.Distance <= maxPing+1 {
			numInRange++
		} else if node.Distance <= 2*maxPing {
			numTolerance++
		} else {
			numOutRange++
		}

		_, shouldInRange := toDiscover[node.Name]
		if shouldInRange {
			discovered++
		}
	}

	stats := common.DiscoveredNodes{
		NodesWithinRange:       numInRange,
		NodesInAcceptableRange: numTolerance,
		ExpectedInRange:        numExpected,
		OutsideRange:           numOutRange,
		Discovered:             discovered,
	}
	json, err := json.Marshal(stats)

	if err != nil {
		log.Errorf("GetDiscoveredNodeStats JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	//log.Debugf("Responding %s", string(json))
	w.Write(json)
}

type NodeListener struct {
	Name string
	Port string
}

func RegisterNodeListener(w http.ResponseWriter, r *http.Request) {
	//log.Info("RegisterNodeListener")
	client := NodeListener{}
	err := json.NewDecoder(r.Body).Decode(&client)

	if err != nil {
		log.Errorf("RegisterNodeListener JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		//log.Infof("RegisterNodeListener from %s", client.Name)
		discovery.RegisterListener(client.Name, client.Port)
		w.Write([]byte("OK"))
	}
}
