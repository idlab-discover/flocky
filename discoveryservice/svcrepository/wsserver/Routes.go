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

func RepoRouter() *mux.Router {

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
	//Disco callback API
	Route{
		Name:        "nodesDiscovered",
		Method:      "POST",
		Pattern:     "/nodesDiscovered",
		HandlerFunc: NodesDiscovered,
		Queries:     []string{},
	},
	Route{
		Name:        "nodesUpdated",
		Method:      "POST",
		Pattern:     "/nodesUpdated",
		HandlerFunc: NodesUpdated,
		Queries:     []string{},
	},
	//Repo provider API
	Route{
		Name:        "registerLocalCapsProvider",
		Method:      "POST",
		Pattern:     "/registerLocalCapsProvider",
		HandlerFunc: RegisterLocalCapsProvider,
		Queries:     []string{},
	},
	Route{
		Name:        "registerTraitHandler",
		Method:      "POST",
		Pattern:     "/registerTraitHandler",
		HandlerFunc: RegisterTraitHandler,
		Queries:     []string{},
	},
	Route{
		Name:        "getTraitHandler",
		Method:      "POST",
		Pattern:     "/getTraitHandler",
		HandlerFunc: GetTraitHandler,
		Queries:     []string{},
	},
	Route{
		Name:        "registerLocalAppsProvider",
		Method:      "POST",
		Pattern:     "/registerLocalAppsProvider",
		HandlerFunc: RegisterLocalAppsProvider,
		Queries:     []string{},
	},
	//Local non-core services API
	Route{
		Name:        "registerFlockyService",
		Method:      "POST",
		Pattern:     "/registerFlockyService",
		HandlerFunc: RegisterFlockySvc,
		Queries:     []string{},
	},
	Route{
		Name:        "getFlockyServiceEndpoint",
		Method:      "GET",
		Pattern:     "/getFlockyServiceEndpoint",
		HandlerFunc: GetFlockySvcEndpoint,
		Queries:     []string{},
	},
	Route{
		Name:        "listFlockyServices",
		Method:      "GET",
		Pattern:     "/listFlockyServices",
		HandlerFunc: ListFlockyServices,
		Queries:     []string{},
	},
	//Repo listeners API
	Route{
		Name:        "registerNodeStatusUpdatesListener",
		Method:      "POST",
		Pattern:     "/registerNodeStatusUpdatesListener",
		HandlerFunc: RegisterNodeStatusUpdatesListener,
		Queries:     []string{},
	},
	Route{
		Name:        "registerLocalAppsProvider",
		Method:      "POST",
		Pattern:     "/registerLocalAppsProvider",
		HandlerFunc: RegisterLocalAppsProvider,
		Queries:     []string{},
	},
	//Repo API
	Route{
		Name:        "getKnownComponentDefs",
		Method:      "GET",
		Pattern:     "/getKnownComponentDefs",
		HandlerFunc: GetKnownComponentDefs,
		Queries:     []string{},
	},
	Route{
		Name:        "getFullKnownComponentDefs",
		Method:      "POST",
		Pattern:     "/getFullKnownComponentDefs",
		HandlerFunc: GetFullKnownComponentDefs,
		Queries:     []string{},
	},
	Route{
		Name:        "getKnownNodeSummaries",
		Method:      "GET",
		Pattern:     "/getKnownNodeSummaries",
		HandlerFunc: GetKnownNodeSummaries,
		Queries:     []string{},
	},
	Route{
		Name:        "addComponentDef",
		Method:      "POST",
		Pattern:     "/addComponentDef",
		HandlerFunc: AddComponentDef,
		Queries:     []string{},
	},
	Route{
		Name:        "getConcreteComponentsFor",
		Method:      "POST",
		Pattern:     "/getConcreteComponentsFor",
		HandlerFunc: GetConcreteComponentsFor,
		Queries:     []string{},
	},
	Route{
		//node caps and deployed stuff in one go
		Name:        "getNodeSummary",
		Method:      "GET",
		Pattern:     "/getNodeSummary",
		HandlerFunc: GetNodeSummary,
		Queries:     []string{},
	},
}
