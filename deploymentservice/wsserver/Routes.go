package wsserver

import (
	"github.com/gorilla/mux"

	"net/http"
)

type Route struct {
	Name        string
	Method      string
	Pattern     string
	HandlerFunc http.HandlerFunc
	Queries     []string
}

type Routes []Route

func DeploymentRouter() *mux.Router {

	router := mux.NewRouter().StrictSlash(true)
	for _, route := range routes {
		router.
			Methods(route.Method).
			Path(route.Pattern).
			Name(route.Name).
			Handler(route.HandlerFunc)
		//Queries(route.Queries)
	}

	return router
}

var routes = Routes{
	// Capability provider API
	// For now, leave it in Feather I guess, this thing would just be playing proxy

	// Orchestration service/migration API
	Route{
		Name:        "addServiceClient",
		Method:      "POST",
		Pattern:     "/addServiceClient",
		HandlerFunc: AddServiceClient,
		Queries:     []string{},
	},
	Route{
		Name:        "checkAvailableForServiceClient",
		Method:      "POST",
		Pattern:     "/checkAvailableForServiceClient",
		HandlerFunc: CheckAvailableForServiceClient,
		Queries:     []string{},
	},
	Route{
		Name:        "removeServiceClient",
		Method:      "POST",
		Pattern:     "/removeServiceClient",
		HandlerFunc: RemoveServiceClient,
		Queries:     []string{},
	},
	Route{
		Name:        "migrateConfirmed",
		Method:      "POST",
		Pattern:     "/migrateConfirmed",
		HandlerFunc: ClientMigrationConfirmed,
		Queries:     []string{},
	},
	Route{
		Name:        "migrateFailed",
		Method:      "POST",
		Pattern:     "/migrateFailed",
		HandlerFunc: ClientMigrationDenied,
		Queries:     []string{},
	},
}
