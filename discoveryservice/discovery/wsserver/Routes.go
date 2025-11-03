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

func DiscoRouter() *mux.Router {

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
	//Discovery internal API
	Route{
		Name:        "ping",
		Method:      "POST",
		Pattern:     "/ping",
		HandlerFunc: Ping,
		Queries:     []string{},
	},
	Route{
		Name:        "getKnownSvcNodes",
		Method:      "GET",
		Pattern:     "/getKnownSvcNodes",
		HandlerFunc: GetKnownSvcNodes,
		Queries:     []string{},
	},
	Route{
		Name:        "getDiscoveredNodeStats",
		Method:      "GET",
		Pattern:     "/getDiscoveredNodeStats",
		HandlerFunc: GetDiscoveredNodeStats,
		Queries:     []string{},
	},
	//Discovery event API
	Route{
		Name:        "registerNodeListener",
		Method:      "POST",
		Pattern:     "/registerNodeListener",
		HandlerFunc: RegisterNodeListener,
		Queries:     []string{},
	},
}
