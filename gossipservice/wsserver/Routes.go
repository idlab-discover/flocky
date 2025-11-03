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
		Name:        "applyTrait",
		Method:      "POST",
		Pattern:     "/applyTrait",
		HandlerFunc: ApplyTrait,
		Queries:     []string{},
	},
	Route{
		Name:        "registerRESTGossipListener",
		Method:      "POST",
		Pattern:     "/registerRESTGossipListener",
		HandlerFunc: RegisterRESTGossipListener,
		Queries:     []string{},
	},
	Route{
		Name:        "registerRESTGossipPoll",
		Method:      "POST",
		Pattern:     "/registerRESTGossipPoll",
		HandlerFunc: RegisterRESTGossipPoll,
		Queries:     []string{},
	},
	Route{
		Name:        "pushGossipItem",
		Method:      "POST",
		Pattern:     "/pushGossipItem",
		HandlerFunc: PushGossipItem,
		Queries:     []string{},
	},
	Route{
		Name:        "getGossipItems",
		Method:      "POST",
		Pattern:     "/getGossipItems",
		HandlerFunc: GetGossipItem,
		Queries:     []string{},
	},
	Route{
		Name:        "registerShmGossipListener",
		Method:      "POST",
		Pattern:     "/registerShmGossipListener",
		HandlerFunc: RegisterShmGossipListener,
		Queries:     []string{},
	},
	Route{
		Name:        "registerShmGossipPoll",
		Method:      "POST",
		Pattern:     "/registerShmGossipPoll",
		HandlerFunc: RegisterShmGossipPoll,
		Queries:     []string{},
	},
	//Capabilities API routes
	Route{
		Name:        "listCapabilities",
		Method:      "GET",
		Pattern:     "/listCapabilities",
		HandlerFunc: ListGossipCapabilities,
		Queries:     []string{},
	},
}
