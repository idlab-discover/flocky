package ws

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

func FeatherRouter() *mux.Router {
	router := mux.NewRouter().StrictSlash(true)
	for _, route := range featherroutes {
		router.
			Methods(route.Method).
			Path(route.Pattern).
			Name(route.Name).
			Handler(route.HandlerFunc)
		//Queries(route.Queries)
	}

	return router
}

func WarrensRouter() *mux.Router {
	router := mux.NewRouter().StrictSlash(true)
	for _, route := range warrensroutes {
		router.
			Methods(route.Method).
			Path(route.Pattern).
			Name(route.Name).
			Handler(route.HandlerFunc)
		//Queries(route.Queries)
	}

	return router
}

var featherroutes = Routes{
	//Default feather API routes
	Route{
		Name:        "getDeployedComponents",
		Method:      "GET",
		Pattern:     "/getDeployedComponents",
		HandlerFunc: GetPods,
		Queries:     []string{},
	},
	Route{
		Name:        "deployPod",
		Method:      "POST",
		Pattern:     "/deployPod",
		HandlerFunc: DeployPod,
		Queries:     []string{},
	},
	Route{
		Name:        "deletePod",
		Method:      "POST",
		Pattern:     "/deletePod",
		HandlerFunc: DeletePod,
		Queries:     []string{},
	},
	Route{
		Name:        "deployOAMPod",
		Method:      "POST",
		Pattern:     "/deployOAMPod",
		HandlerFunc: DeployOAMPod,
		Queries:     []string{},
	},
	Route{
		Name:        "deleteOAMPod",
		Method:      "POST",
		Pattern:     "/deleteOAMPod",
		HandlerFunc: DeleteOAMPod,
		Queries:     []string{},
	},
	//Capabilities API routes
	Route{
		Name:        "listCapabilities",
		Method:      "GET",
		Pattern:     "/listCapabilities",
		HandlerFunc: ListFeatherCapabilities,
		Queries:     []string{},
	},
}

var warrensroutes = Routes{
	//Discovery API routes
	Route{
		Name:        "nodesDiscovered",
		Method:      "POST",
		Pattern:     "/nodesDiscovered",
		HandlerFunc: NodesDiscovered,
		Queries:     []string{},
	},
	//Capabilities API routes
	Route{
		Name:        "listCapabilities",
		Method:      "GET",
		Pattern:     "/listCapabilities",
		HandlerFunc: ListWarrensCapabilities,
		Queries:     []string{},
	},
}
