package wsserver

import (
	"encoding/json"
	"net/http"
	"oamswirly/common/nodediscovery"
	"oamswirly/discoveryservice/svcrepository"

	log "github.com/sirupsen/logrus"
)

//var NodeDiscoveredCallback func([]nodediscovery.SvcNode)

func NodesDiscovered(w http.ResponseWriter, r *http.Request) {
	//nodes := []nodediscovery.SvcNode{}
	update := nodediscovery.NodesUpdate{}
	err := json.NewDecoder(r.Body).Decode(&update)

	//log.Info("NodesDiscovered new remote nodes locally discovered/nodes deleted")

	if err != nil {
		log.Errorf("NodesDiscovered JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		go func() {
			svcrepository.AddNodes(update.NewNodes)
			svcrepository.DeleteNodes(update.DeletedNodes)
		}()
		w.Write([]byte("OK"))
	}
}

func NodesUpdated(w http.ResponseWriter, r *http.Request) {
	nodes := []nodediscovery.SvcNode{}
	err := json.NewDecoder(r.Body).Decode(&nodes)

	//log.Info("NodesUpdated remote node distances updated")
	if err != nil {
		log.Errorf("NodesUpdated JSON error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	} else {
		go func() {
			svcrepository.UpdateNodeDistances(nodes)
		}()
		w.Write([]byte("OK"))
	}
}
