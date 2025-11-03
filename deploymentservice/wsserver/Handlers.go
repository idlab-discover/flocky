package wsserver

import (
	"encoding/json"
	"net/http"
	common "oamswirly/common/nodediscovery"
	"oamswirly/deploymentservice/orchestration"
	"strings"
	//"github.com/gorilla/mux"
)

func CheckAvailableForServiceClient(w http.ResponseWriter, r *http.Request) {
	serviceClient := common.ServiceClient{}

	err := json.NewDecoder(r.Body).Decode(&serviceClient)
	if err != nil {
		w.WriteHeader(400)
	} else {
		ip := strings.Split(r.RemoteAddr, ":")[0]
		client := orchestration.ServiceClientRequest{
			IP:     ip,
			Name:   serviceClient.Name,
			SubApp: serviceClient.Application,
			//ComponentType: serviceClient.Component,
			//ComponentImpl: serviceClient.ConcreteComponent,
		}

		//log.Infof("AddServiceClient %s %s", client.Name, client.SubApp.Metadata.Name)
		success := orchestration.CheckServiceAvailableForClient(client)
		//log.Infof("AddServiceClient success? %t", success)
		jsonBytes, _ := json.Marshal(success)
		w.Write(jsonBytes)
	}
}

func AddServiceClient(w http.ResponseWriter, r *http.Request) {
	serviceClient := common.ServiceClient{}

	err := json.NewDecoder(r.Body).Decode(&serviceClient)
	if err != nil {
		w.WriteHeader(400)
	} else {
		ip := strings.Split(r.RemoteAddr, ":")[0]
		client := orchestration.ServiceClientRequest{
			IP:     ip,
			Name:   serviceClient.Name,
			SubApp: serviceClient.Application,
			//ComponentType: serviceClient.Component,
			//ComponentImpl: serviceClient.ConcreteComponent,
		}

		//log.Infof("AddServiceClient %s %s", client.Name, client.SubApp.Metadata.Name)
		success := orchestration.AddClient(client)
		//log.Infof("AddServiceClient success? %t", success)
		jsonBytes, _ := json.Marshal(success)
		w.Write(jsonBytes)
	}
}

func RemoveServiceClient(w http.ResponseWriter, r *http.Request) {
	serviceClient := common.ServiceClient{}

	err := json.NewDecoder(r.Body).Decode(&serviceClient)
	if err != nil {
		w.WriteHeader(400)
	} else {
		ip := strings.Split(r.RemoteAddr, ":")[0]
		client := orchestration.ServiceClientRequest{
			IP:     ip,
			Name:   serviceClient.Name,
			SubApp: serviceClient.Application,
			//ComponentType: serviceClient.Component,
			//ComponentImpl: serviceClient.ConcreteComponent,
		}

		orchestration.RemoveClient(client)
	}
}

func ClientMigrationConfirmed(w http.ResponseWriter, r *http.Request) {
	serviceClient := common.ServiceClient{}

	err := json.NewDecoder(r.Body).Decode(&serviceClient)
	if err != nil {
		w.WriteHeader(400)
	} else {
		ip := strings.Split(r.RemoteAddr, ":")[0]
		client := orchestration.ServiceClientRequest{
			IP:     ip,
			Name:   serviceClient.Name,
			SubApp: serviceClient.Application,
			//ComponentType: serviceClient.Component,
			//ComponentImpl: serviceClient.ConcreteComponent,
		}

		orchestration.MigrationConfirmed(client)
	}
}

func ClientMigrationDenied(w http.ResponseWriter, r *http.Request) {
	serviceClient := common.ServiceClient{}

	err := json.NewDecoder(r.Body).Decode(&serviceClient)
	if err != nil {
		w.WriteHeader(400)
	} else {
		ip := strings.Split(r.RemoteAddr, ":")[0]
		client := orchestration.ServiceClientRequest{
			IP:     ip,
			Name:   serviceClient.Name,
			SubApp: serviceClient.Application,
			//ComponentType: serviceClient.Component,
			//ComponentImpl: serviceClient.ConcreteComponent,
		}

		orchestration.MigrationDenied(client)
	}
}
