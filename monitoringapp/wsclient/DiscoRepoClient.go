package wsclient

import (
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/nodediscovery"
	"oamswirly/common/oam"

	log "github.com/sirupsen/logrus"
)

func GetKnownComponentDefs(port int) (map[string][]string, error) {
	svcURL := fmt.Sprintf("http://localhost:%d/getKnownComponentDefs", port)
	//log.Infof("GetKnownComponentDefs %s", svcURL)
	response, err := http.Get(svcURL)

	if err != nil {
		log.Errorf("GetKnownComponentDefs error %v", err)
		return nil, err
	}

	components := make(map[string][]string)
	err = json.NewDecoder(response.Body).Decode(&components)
	if err != nil {
		log.Errorf("GetKnownComponentDefs JSON error %v", err)
		return nil, err
	}

	response.Body.Close()
	return components, nil
}

func GetKnownNodeSummaries(port int) ([]oam.NodeSummary, error) {
	svcURL := fmt.Sprintf("http://localhost:%d/getKnownNodeSummaries", port)
	//log.Infof("GetKnownNodeSummaries %s", svcURL)
	response, err := http.Get(svcURL)

	if err != nil {
		log.Errorf("GetKnownNodeSummaries error %v", err)
		return nil, err
	}

	svcNodes := []oam.NodeSummary{}
	err = json.NewDecoder(response.Body).Decode(&svcNodes)
	if err != nil {
		log.Errorf("GetKnownNodeSummaries JSON error %v", err)
		return nil, err
	}

	response.Body.Close()
	return svcNodes, nil
}

func GetLocallyKnownSvcNodes(port int) ([]nodediscovery.SvcNode, error) {
	svcURL := fmt.Sprintf("http://localhost:%d/getKnownSvcNodes", port)
	//log.Infof("GetLocallyKnownSvcNodes %s", svcURL)
	response, err := http.Get(svcURL)

	if err != nil {
		log.Errorf("GetLocallyKnownSvcNodes error %v", err)
		return nil, err
	}

	svcNodes := []nodediscovery.SvcNode{}
	err = json.NewDecoder(response.Body).Decode(&svcNodes)
	if err != nil {
		log.Errorf("GetLocallyKnownSvcNodes JSON error %v", err)
		return nil, err
	}

	response.Body.Close()
	return svcNodes, nil
}
