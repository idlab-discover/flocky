package clients

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/deploymentservice/config"

	log "github.com/sirupsen/logrus"
)

type FledgeOrchestrator struct {
}

func (fo *FledgeOrchestrator) Init() Orchestrator {
	//fo.clientset = getKubeClient()
	return fo
}

func (fo *FledgeOrchestrator) DeploySubApp(pod oam.SubApplication) bool {
	fullURL := fmt.Sprintf("http://localhost:%d/deployOAMPod", config.Cfg.FledgeAPIPort)
	//log.Infof("Calling %s\n", fullURL)

	/*if config.Cfg.TestMode {
		return true
	}*/

	json, err := json.Marshal(pod)
	if err != nil {
		log.Errorf("DeploySubApp JSON error %v", err)
		return false
	}
	response, err := http.Post(fullURL, "application/json", bytes.NewBuffer(json))
	if err != nil {
		log.Errorf("DeploySubApp post error %v", err)
	}
	defer response.Body.Close()
	return err == nil && response.StatusCode == http.StatusOK
}

func (fo *FledgeOrchestrator) RemoveSubApp(pod oam.SubApplication) bool {
	fullURL := fmt.Sprintf("http://localhost:%d/deleteOAMPod", config.Cfg.FledgeAPIPort)
	//log.Infof("Calling %s\n", fullURL)

	/*if config.Cfg.TestMode {
		return true
	}*/

	json, err := json.Marshal(pod)
	if err != nil {
		log.Errorf("RemoveSubApp JSON error %v", err)
		return false
	}
	response, err := http.Post(fullURL, "application/json", bytes.NewBuffer(json))
	if err != nil {
		log.Errorf("RemoveSubApp post error %v", err)
	}
	defer response.Body.Close()
	return err == nil && response.StatusCode == http.StatusOK
}
