package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/nodediscovery"
	"oamswirly/common/oam"
	"oamswirly/swirlyservice/config"
	"time"

	log "github.com/sirupsen/logrus"
)

func CheckNodeAvailable(ip string, node string, subapp oam.SubApplication) bool {
	//log.Infof("Checking if node %s is available for component %s", node, subapp.Metadata.Name)
	port := config.Cfg.DeploymentAPIPort //utils.GetPort(node)

	fullURL := fmt.Sprintf("http://%s:%d/checkAvailableForServiceClient", ip, port) //"addServiceClient")

	serviceClient := nodediscovery.ServiceClient{
		Name:        config.Cfg.NodeID,
		Application: subapp,
	}
	jsonBytes, err := json.Marshal(serviceClient)
	if err != nil {
		log.Errorf("CheckNodeAvailable JSON encode error %v", err)
	}

	//log.Infof("Checking availability with %s as client for %s", fullURL, subapp.Spec.Components[0].Name)

	response, err := http.Post(fullURL, "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		log.Errorf("CheckNodeAvailable response error %v", err)
	}

	added := false
	err = json.NewDecoder(response.Body).Decode(&added)
	if err != nil {
		log.Errorf("CheckNodeAvailable JSON decode error %v", err)
		return false
	}

	//log.Infof("Node available as component client for %s? %v", subapp.Spec.Components[0].Name, added)
	return added
}

func TryRegisterClientWithNode(ip string, node string, subapp oam.SubApplication) bool {
	log.Infof("%d Trying to register client for component %s on node %s", time.Now().UnixMilli(), subapp.Spec.Components[0].Name, node)
	port := config.Cfg.DeploymentAPIPort //utils.GetPort(node)

	fullURL := fmt.Sprintf("http://%s:%d/addServiceClient", ip, port) //"addServiceClient")

	serviceClient := nodediscovery.ServiceClient{
		Name:        config.Cfg.NodeID,
		Application: subapp,
		//Component:         component,
		//ConcreteComponent: cDef,
	}
	jsonBytes, err := json.Marshal(serviceClient)
	if err != nil {
		log.Errorf("TryRegisterClient JSON encode error %v", err)
	}

	//log.Infof("Registering with %s as client for %s\n", fullURL, subapp.Spec.Components[0].Name)
	response, err := http.Post(fullURL, "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		log.Errorf("TryRegisterClient response error %v", err)
	}

	added := false
	err = json.NewDecoder(response.Body).Decode(&added)
	if err != nil {
		log.Errorf("TryRegisterClient JSON decode error %v", err)
		return false
	}

	log.Infof("%d Registered as component client for %s? %v", time.Now().UnixMilli(), subapp.Spec.Components[0].Name, added)
	return added
}

func TryRemoveFromNode(ip string, node string, subapp oam.SubApplication) bool {
	//log.Infof("Trying to remove client from component %s on node %s", subapp.Spec.Components[0].Name, node)
	port := config.Cfg.DeploymentAPIPort //utils.GetPort(node)

	fullURL := fmt.Sprintf("http://%s:%d/removeServiceClient", ip, port) // "removeServiceClient")

	serviceClient := nodediscovery.ServiceClient{
		Name:        config.Cfg.NodeID,
		Application: subapp,
		//Component:         component,
		//ConcreteComponent: cDef,
	}
	jsonBytes, err := json.Marshal(serviceClient)
	if err != nil {
		log.Errorf("TryRemoveFromNode JSON encode error %v", err)
	}

	response, err := http.Post(fullURL, "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		log.Errorf("TryRemoveFromNode response error %v", err)
	}

	removed := false
	err = json.NewDecoder(response.Body).Decode(&removed)
	if err != nil {
		log.Errorf("TryRemoveFromNode JSON decode error %v", err)
		return false
	}

	//log.Infof("Removed from component client for %s? %v", subapp.Spec.Components[0].Name, removed)
	return removed
}

func NotifyMigrateFailed(ip string, node string, subapp oam.SubApplication) bool {
	//log.Infof("Notifying node %s of failed migration for component %s", node, subapp.Spec.Components[0].Name)
	port := config.Cfg.DeploymentAPIPort //utils.GetPort(node)

	fullURL := fmt.Sprintf("http://%s:%d/migrateFailed", ip, port) // "migrateFailed")
	serviceClient := nodediscovery.ServiceClient{
		Name:        config.Cfg.NodeID,
		Application: subapp,
		//Component:         component,
		//ConcreteComponent: cDef,
	}

	json, err := json.Marshal(serviceClient)
	if err != nil {
		log.Errorf("NotifyMigrateFailed JSON encode error %v", err)
		return false
	}

	_, err = http.Post(fullURL, "application/json", bytes.NewBuffer(json))
	if err != nil {
		log.Errorf("NotifyMigrateFailed response error %v", err)
		return false
	}

	return true
}

func NotifyMigrateSuccess(ip string, node string, subapp oam.SubApplication) bool {
	//log.Infof("Notifying node %s of successful migration for component %s", node, subapp.Spec.Components[0].Name)
	port := config.Cfg.DeploymentAPIPort //utils.GetPort(node)

	fullURL := fmt.Sprintf("http://%s:%d/migrateConfirmed", ip, port) // "migrateConfirmed")

	serviceClient := nodediscovery.ServiceClient{
		Name:        config.Cfg.NodeID,
		Application: subapp,
		//Component:         component,
		//ConcreteComponent: cDef,
	}

	json, err := json.Marshal(serviceClient)
	if err != nil {
		log.Errorf("NotifyMigrateSuccess JSON encode error %v", err)
		return false
	}

	_, err = http.Post(fullURL, "application/json", bytes.NewBuffer(json))
	if err != nil {
		log.Errorf("NotifyMigrateSuccess response error %v", err)
		return false
	}

	return true
}
