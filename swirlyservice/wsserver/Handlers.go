package wsserver

import (
	"encoding/json"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/swirlyservice/orchestration"

	//"github.com/gorilla/mux"
	log "github.com/sirupsen/logrus"
)

// GET /setFogNodes
func TryMigrate(w http.ResponseWriter, r *http.Request) {
	go func() {
		//log.Info("TryMigrate")

		component := oam.Component{}
		err := json.NewDecoder(r.Body).Decode(&component)
		if err != nil {
			log.Errorf("TryMigrate error %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		orchestration.TryMigrateComponent(component)
	}()
}

func DeployApplication(w http.ResponseWriter, r *http.Request) {

	//log.Info("DeployApplication")

	app := oam.Application{}
	err := json.NewDecoder(r.Body).Decode(&app)
	if err != nil {
		log.Errorf("DeployApplication error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	go func() {
		orchestration.DeploySupportComponentsFor(app)
	}()
}

func DeleteApplication(w http.ResponseWriter, r *http.Request) {

	//log.Info("DeleteApplication")

	app := oam.Application{}
	err := json.NewDecoder(r.Body).Decode(&app)
	if err != nil {
		log.Errorf("DeleteApplication error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	go func() {
		orchestration.RemoveSupportComponentsFor(app)
	}()
}

func Migrate(w http.ResponseWriter, r *http.Request) {
	//log.Info("Migrate")

	component := oam.Component{}
	err := json.NewDecoder(r.Body).Decode(&component)
	if err != nil {
		log.Errorf("Migrate error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	orchestration.MigrateComponent(component)
}

func CancelMigrate(w http.ResponseWriter, r *http.Request) {
	//log.Info("CancelMigrate")

	component := oam.Component{}
	err := json.NewDecoder(r.Body).Decode(&component)
	if err != nil {
		log.Errorf("CancelMigrate error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	orchestration.CancelMigrate(component)
}

func NodeCapsChanged(w http.ResponseWriter, r *http.Request) {
	//log.Info("NodeCapsChanged")

	capsSummary := []oam.NodeSummary{}
	err := json.NewDecoder(r.Body).Decode(&capsSummary)
	if err != nil {
		log.Errorf("NodeCapsChanged error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	orchestration.NodeResourcesUpdated(capsSummary)
}

func NodeAppsChanged(w http.ResponseWriter, r *http.Request) {
	//log.Info("NodeAppsChanged")

	appsSummary := []oam.NodeSummary{}
	err := json.NewDecoder(r.Body).Decode(&appsSummary)
	if err != nil {
		log.Errorf("NodeAppsChanged error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	orchestration.NodeAppsUpdated(appsSummary)
}
