package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/deploymentservice/config"

	log "github.com/sirupsen/logrus"
)

func GetTraitHandler(trait oam.Trait) (oam.TraitHandler, error) {
	//log.Infof("GetKnownComponentDefs from %s", node.Name)
	jsonBytes, err := json.Marshal(trait)
	response, err := http.Post(fmt.Sprintf("http://localhost:%d/getTraitHandler", config.Cfg.RepoAPIPort), "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		log.Errorf("GetTraitHandler error %v", err)
		return oam.TraitHandler{}, err
	}

	handler := oam.TraitHandler{}
	err = json.NewDecoder(response.Body).Decode(&handler)
	if err != nil {
		log.Errorf("GetTraitHandler JSON error %v", err)
		return handler, err
	}

	response.Body.Close()
	return handler, nil
}

var flockyServices map[oam.FlockyService]oam.ServiceProvider = make(map[oam.FlockyService]oam.ServiceProvider)

func GetFlockyService(svc oam.FlockyService) (oam.ServiceProvider, error) {
	_, cached := flockyServices[svc]
	if !cached {
		jsonBytes, err := json.Marshal(svc)
		response, err := http.Post(fmt.Sprintf("http://localhost:%d/getFlockyServiceEndpoint", config.Cfg.RepoAPIPort), "application/json", bytes.NewBuffer(jsonBytes))
		if err != nil {
			log.Errorf("GetFlockyService error %v", err)
			return oam.ServiceProvider{}, err
		}

		provider := oam.ServiceProvider{}
		err = json.NewDecoder(response.Body).Decode(&provider)
		if err != nil {
			log.Errorf("GetFlockyService JSON error %v", err)
			return provider, err
		}

		response.Body.Close()
		flockyServices[svc] = provider
	}
	service, _ := flockyServices[svc]
	return service, nil
}

func TryHandleTrait(localendpoint string, trait oam.Trait) (bool, error) {
	//log.Infof("GetKnownComponentDefs from %s", node.Name)
	jsonBytes, err := json.Marshal(trait)
	response, err := http.Post(localendpoint, "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		log.Errorf("TryHandleTrait error %v", err)
		return false, err
	}
	if response.StatusCode > 204 {
		return false, nil
	}

	response.Body.Close()
	return true, nil
}
