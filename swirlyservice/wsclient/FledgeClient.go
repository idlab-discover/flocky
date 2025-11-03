package wsclient

import (
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/swirlyservice/config"

	log "github.com/sirupsen/logrus"
	v1 "k8s.io/api/core/v1"
)

func GetDeployedPods() ([]v1.Pod, error) {
	fledgeURL := fmt.Sprintf("http://localhost:%d/pods", config.Cfg.FledgeAPIPort)

	//log.Infof("Fetching active pods from %s", fledgeURL)
	response, err := http.Get(fledgeURL)
	if err != nil {
		log.Errorf("GetDeployedPods response error %v", err)
	}

	fogNodes := []v1.Pod{}
	err = json.NewDecoder(response.Body).Decode(&fogNodes)
	if err != nil {
		log.Errorf("GetDeployedPods JSON decode error %v", err)
		return nil, err
	}

	return fogNodes, nil
}
