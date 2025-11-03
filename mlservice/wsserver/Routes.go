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

func Router() *mux.Router {
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
	//Default feather API routes
	Route{
		Name:        "applyMLTrait",
		Method:      "POST",
		Pattern:     "/applyMLTrait",
		HandlerFunc: ApplyMLTrait,
		Queries:     []string{},
	},
	//Capabilities API routes
	Route{
		Name:        "listCapabilities",
		Method:      "GET",
		Pattern:     "/listCapabilities",
		HandlerFunc: ListMLCapabilities,
		Queries:     []string{},
	},
	//Gossip exchange routes
	/*Route{
		Name:        "getModelGossip",
		Method:      "POST",
		Pattern:     "/getModelGossip",
		HandlerFunc: GetModelGossip,
		Queries:     []string{},
	},*/
	/*Route{
		Name:        "modelGossipReceived",
		Method:      "POST",
		Pattern:     "/modelGossipReceived",
		HandlerFunc: ModelGossipReceived,
		Queries:     []string{},
	},*/
}
